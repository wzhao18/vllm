# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Router top-k + MegaMoE staging in one launch (SP decode MoE front).

Input: the router | latent down GEMM's FP32 rows ``y [n, E + latent]``
(``runtime.sp_mlp``: router logits in columns ``0:E``, the latent in ``E:``), or a
split-K GEMM's FP32 partials ``[splits, n, E + latent]`` summed here in split order
(the GEMM's own deterministic FP32 sum, one kernel boundary and exchange fewer).
One block per row:

* all 8 warps: stock's sigmoid + correction-bias single-group top-k on the
  logits (``_moe_C.grouped_topk`` / ``single_group_topk``, bitwise: the same
  packed ``(score, 65535 - id)`` key order, selected exactly in two passes,
  per warp then across the warps, and the same ``write_outputs`` weights on
  warp 0), written straight into the session's route buffers with the stager's rules (``megamoe_rubin.staging``:
  padding rows and out-of-range ids masked to -1 with weight 0, the sticky
  ``route_error`` flag, SF pad columns zeroed);
* before that, the same threads: the latent row rounded to BF16 (round to nearest, as a BF16-output
  GEMM rounds its FP32 accumulator), then the stager's activation quantization
  (NVFP4 block16 or MXFP8 block32), bitwise ``_stage_kernel``.

Trailing blocks write ``topk_idx = -1`` for the dead rows ``[n, capacity)``.
The launch is programmatic-dependent (PDL) on the GEMM before it and waits for
it in-kernel (``griddepcontrol.wait``); right after that wait it triggers its
dependent (token_comm, itself a PDL launch that waits for this grid before
reading any route), so the dependent's launch and prologue overlap this kernel.

Top-k (G2-F3): an exact bound + compact + rank selection (``fast_keys``) picks
the same (score, id) per rank as the two-pass warp selection (``route_block``,
kept for rows with more than 256 candidates, i.e. heavy score ties) without the
16 serial warp-max rounds, and its tail runs on the last warp under the
quantization. Outputs are bitwise those of ``route_block``.

This replaces router split-K reduce -> top-k -> (latent down join) -> stager:
four kernels and two streams on the path to MegaMoE become GEMM -> this kernel.
Built without fast math (precise ``tanhf``, IEEE division), as the stock kernels.
"""

import hashlib
import os
import threading

import torch

from ...shapes import Unsupported

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <math.h>

namespace cg = cooperative_groups;

namespace {

constexpr uint32_t kFull = 0xffffffffu;
constexpr int kWarps = 8;
constexpr int kDeadRows = 64;

__device__ __forceinline__ float sigmoid_accurate(float x) {
  return 0.5f * tanhf(0.5f * x) + 0.5f;
}

__device__ __forceinline__ void preprocess_score(float input, float bias,
                                                 float& unbiased_score,
                                                 float& selection_score) {
  unbiased_score = 0.0F;
  selection_score = -INFINITY;
  if (!isfinite(input) || !isfinite(bias)) {
    return;
  }
  float const unbiased = sigmoid_accurate(input);
  float const biased = unbiased + bias;
  unbiased_score = unbiased;
  selection_score = biased == 0.0F ? 0.0F : biased;
}

__device__ __forceinline__ uint64_t make_key(float v, int32_t idx) {
  uint32_t bits = __float_as_uint(v);
  bits ^= (bits & 0x80000000u) ? 0xffffffffu : 0x80000000u;
  return (uint64_t(bits) << 32) | uint64_t(0xFFFF & (65535 - idx));
}

__device__ __forceinline__ void unpack_key(uint64_t key, float& v,
                                           int32_t& idx) {
  idx = 65535 - int32_t(key & 0xFFFF);
  uint32_t bits = uint32_t(key >> 32);
  bits ^= (bits & 0x80000000u) ? 0x80000000u : 0xffffffffu;
  v = __uint_as_float(bits);
}

__device__ __forceinline__ uint64_t warp_max_u64(uint64_t v) {
  uint32_t hi = uint32_t(v >> 32), lo = uint32_t(v);
  uint32_t mhi = __reduce_max_sync(kFull, hi);
  uint32_t mlo = __reduce_max_sync(kFull, hi == mhi ? lo : 0u);
  return (uint64_t(mhi) << 32) | uint64_t(mlo);
}

struct Args {
  const float* y;
  int64_t ldy;
  int32_t splits;        // y holds this many split-K partials, summed in order
  int64_t split_stride;
  int32_t num_logits;
  const float* bias;
  uint8_t* x;
  int64_t x_stride;
  uint8_t* sf;
  int64_t sf_stride;
  int32_t sf_cols;
  int32_t* idx;
  float* sw;
  int64_t* err;
  const bool* pad;
  // EPLB (stock ``eplb_map_to_physical_and_record``), all null without it.
  const int64_t* l2p;
  const int64_t* replicas;
  int32_t* load;
  const bool* record;
  const int32_t* unpadded;
  int32_t map_slots;
  int32_t load_size;
  int32_t n;
  int32_t capacity;
  int32_t num_experts;
  int32_t topk;
  bool renormalize;
  float routed_scaling_factor;
  int32_t hidden;
  float scale_mult;
  float norm_const;
};

// Keys per thread in the first selection pass: 8 warps x 32 lanes x 4 = 1024
// experts at most.
constexpr int kPer = 4;

// Descending insertion sort of a lane's keys.
template <int P>
__device__ __forceinline__ void sort_desc(uint64_t (&k)[P]) {
#pragma unroll
  for (int i = 1; i < P; ++i) {
#pragma unroll
    for (int j = i; j > 0; --j) {
      uint64_t const hi = k[j] > k[j - 1] ? k[j] : k[j - 1];
      uint64_t const lo = k[j] > k[j - 1] ? k[j - 1] : k[j];
      k[j - 1] = hi;
      k[j] = lo;
    }
  }
}

// The warp's `topk` largest keys, rank r returned on lane r (0 elsewhere), from
// every lane's descending list k. Keys are unique (the expert id is in them),
// so exactly one lane holds each rank's key.
template <int P>
__device__ __forceinline__ uint64_t warp_select(uint64_t (&k)[P], int32_t lane,
                                                int32_t topk) {
  uint64_t mine = 0;
  for (int r = 0; r < topk; ++r) {
    uint64_t const best = warp_max_u64(k[0]);
    if (lane == r) mine = best;
    if (best != 0 && k[0] == best) {
#pragma unroll
      for (int j = 0; j + 1 < P; ++j) k[j] = k[j + 1];
      k[P - 1] = 0;
    }
  }
  return mine;
}

// write_outputs (verbatim) and the stager's route rules: rank r's packed key on
// lane r of the calling warp (0 on lanes >= topk).
__device__ __forceinline__ void route_write(const Args& a, int32_t row,
                                            uint64_t lane_packed,
                                            const float* s_unb) {
  int32_t const lane = threadIdx.x & 31;
  int32_t const num_experts = a.num_logits;
  int32_t const topk = a.topk;
  auto warp = cg::tiled_partition<32>(cg::this_thread_block());
  float lane_score = -INFINITY;
  int32_t lane_expert = -1;
  if (lane < topk) {
    unpack_key(lane_packed, lane_score, lane_expert);
  }
  // The selected expert's unbiased score (same expression, same value).
  float lane_unbiased = 0.0F;
  if (lane < topk && lane_expert >= 0 && lane_expert < num_experts) {
    lane_unbiased = s_unb[lane_expert];
  }
  // single_group_topk::detail::write_outputs (verbatim math).
  bool const finite_selection = lane < topk && lane_score != -INFINITY;
  lane_unbiased = finite_selection ? lane_unbiased : 0.0F;
  unsigned const finite_mask = __ballot_sync(kFull, finite_selection);
  float const sum = cg::reduce(warp, lane_unbiased, cg::plus<float>{});
  if (lane < topk) {
    float output = 0.0F;
    if (finite_mask == 0) {
      if (a.renormalize) {
        output = 1.0F / static_cast<float>(topk);
      }
    } else if (finite_selection) {
      float scale = a.routed_scaling_factor;
      if (a.renormalize) {
        scale /= sum + 1e-20F;
      }
      output = lane_unbiased * scale;
    }
    // megamoe_rubin.staging._stage_kernel route rules.
    int64_t id = lane_expert;
    if (a.pad != nullptr && a.pad[row]) id = -1;
    if (a.l2p != nullptr) {
      // Logical -> physical, the stock replica choice (Knuth hash of the row);
      // masked / out-of-range logical ids map to -1. Load recorded per route.
      int64_t phys = -1;
      if (id >= 0 && id < num_experts) {
        int64_t const reps = max(a.replicas[id], int64_t{1});
        int64_t const hashed = (int64_t{row} * 2654435769LL) & 0xFFFFFFFFLL;
        phys = a.l2p[id * a.map_slots + hashed % reps];
      }
      id = phys;
      if (*a.record && (a.unpadded == nullptr || row < *a.unpadded) &&
          phys >= 0 && phys < a.load_size) {
        atomicAdd(a.load + phys, 1);
      }
    }
    bool const valid = id >= 0 && id < a.num_experts;
    if (id != -1 && !valid) {
      a.err[0] = 1;
      a.err[1] = id;
    }
    int64_t const o = int64_t{row} * topk + lane;
    a.idx[o] = valid ? int32_t(id) : -1;
    a.sw[o] = valid ? output : 0.0F;
  }
  int32_t const vec = a.scale_mult > 0.0F ? 16 : 32;
  for (int32_t c = a.hidden / vec + lane; c < a.sf_cols; c += 32) {
    a.sf[int64_t{row} * a.sf_stride + c] = 0;
  }
}

// Block-parallel top-k of one row (the stock selection order: largest
// (score, 65535 - expert) key first, i.e. ties to the lower expert id), then
// write_outputs (verbatim) and the stager's route rules on warp 0.
// Pass 1: every warp selects its top-k of the 128 experts it scored; pass 2:
// warp 0 selects the top-k of the 8 sorted warp lists (exact:
// the global top-k is inside the union of the per-warp top-k).
__device__ __forceinline__ void route_block(const Args& a, int32_t row,
                                            const float (&lg)[kPer],
                                            const float* s_bias, float* s_unb,
                                            uint64_t* s_cand) {
  int32_t const lane = threadIdx.x & 31;
  int32_t const wid = threadIdx.x >> 5;
  int32_t const num_experts = a.num_logits;
  int32_t const topk = a.topk;
  uint64_t k1[kPer];
#pragma unroll
  for (int j = 0; j < kPer; ++j) {
    int32_t const e = j * (kWarps * 32) + threadIdx.x;
    k1[j] = 0;
    if (e < num_experts) {
      float unb, s;
      preprocess_score(lg[j], s_bias[e], unb, s);
      s_unb[e] = unb;
      k1[j] = make_key(s, e);
    }
  }
  sort_desc<kPer>(k1);
  uint64_t const w = warp_select<kPer>(k1, lane, topk);
  s_cand[wid * 32 + lane] = w;
  __syncthreads();
  if (wid != 0) return;
  // Lane l: warp l / 4's ranks (l % 4) * 8 .. + 8 (already descending).
  uint64_t k2[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) k2[j] = s_cand[(lane >> 2) * 32 + (lane & 3) * 8 + j];
  uint64_t const lane_packed = warp_select<8>(k2, lane, topk);

  route_write(a, row, lane_packed, s_unb);
}

// Fast exact top-k (G2-F3; same keys, same selected (score, id) per rank as
// route_block, so bitwise-identical outputs). Bound: L = min over the 8 warps of
// the warp's largest key hi word. The 8 warp-max keys are distinct with hi >= L,
// so for topk <= kWarps every top-k key has hi >= L. Candidates (hi >= L, real
// experts) are compacted into s_cand and each ranks itself by counting the larger
// candidates (keys are unique); rank r < topk lands in s_sel[r]. Typical rows
// keep ~10-40 candidates (no serial selection rounds); more than kWarps * 32
// candidates (heavy ties, e.g. all logits equal or non-finite) take route_block.
// Step 1 (all threads, before a barrier): scores, keys, warp max of the hi words.
__device__ __forceinline__ void fast_keys(const Args& a, const float (&lg)[kPer],
                                          const float* s_bias, float* s_unb,
                                          uint64_t (&k)[kPer], uint32_t* s_wmax,
                                          int* s_cnt, uint64_t* s_sel) {
  int32_t const lane = threadIdx.x & 31;
  int32_t const wid = threadIdx.x >> 5;
  uint32_t hmax = 0;
#pragma unroll
  for (int j = 0; j < kPer; ++j) {
    int32_t const e = j * (kWarps * 32) + threadIdx.x;
    k[j] = 0;
    if (e < a.num_logits) {
      float unb, sc;
      preprocess_score(lg[j], s_bias[e], unb, sc);
      s_unb[e] = unb;
      k[j] = make_key(sc, e);
      hmax = max(hmax, uint32_t(k[j] >> 32));
    }
  }
  uint32_t const wm = __reduce_max_sync(kFull, hmax);
  if (lane == 0) s_wmax[wid] = wm;
  if (threadIdx.x == 0) *s_cnt = 0;
  if (threadIdx.x < 32) s_sel[threadIdx.x] = 0;
}

// Step 2 (after the barrier): compact the candidates.
__device__ __forceinline__ void fast_compact(const uint64_t (&k)[kPer],
                                             const uint32_t* s_wmax, int* s_cnt,
                                             uint64_t* s_cand) {
  uint32_t lo = s_wmax[0];
#pragma unroll
  for (int w = 1; w < kWarps; ++w) lo = min(lo, s_wmax[w]);
  int c = 0;
#pragma unroll
  for (int j = 0; j < kPer; ++j) c += (k[j] != 0 && uint32_t(k[j] >> 32) >= lo) ? 1 : 0;
  if (c) {
    int base = atomicAdd(s_cnt, c);
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
      if (k[j] != 0 && uint32_t(k[j] >> 32) >= lo) {
        if (base < kWarps * 32) s_cand[base] = k[j];
        ++base;
      }
    }
  }
}

// Step 3 (after the barrier): candidate t ranks itself.
__device__ __forceinline__ void fast_rank(int t, int cnt, int topk,
                                          const uint64_t* s_cand, uint64_t* s_sel) {
  if (t < cnt) {
    uint64_t const m = s_cand[t];
    int r = 0;
#pragma unroll 4
    for (int i = 0; i < cnt; ++i) r += s_cand[i] > m ? 1 : 0;
    if (r < topk) s_sel[r] = m;
  }
}

__device__ __forceinline__ float bf16_rn(float v) {
  return __bfloat162float(__float2bfloat16_rn(v));
}

__device__ __forceinline__ uint8_t e2m1x2(float hi, float lo) {
  uint32_t r;
  asm volatile(
      "{\n .reg .b8 t;\n cvt.rn.satfinite.e2m1x2.f32 t, %1, %2;\n"
      " cvt.u32.u8 %0, t;\n}\n"
      : "=r"(r)
      : "f"(hi), "f"(lo));
  return uint8_t(r);
}

__device__ __forceinline__ uint8_t e4m3_rn(float v) {
  return uint8_t(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ float e4m3_to_float(uint8_t b) {
  __half_raw h = __nv_cvt_fp8_to_halfraw(b, __NV_E4M3);
  return __half2float(__half(h));
}

// NVFP4 block16 of the stager (one block per thread).
__device__ __forceinline__ void quant_nvfp4(const Args& a, int32_t row,
                                            int32_t blk, const float4 (&in)[8]) {
  float v[16];
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    float4 t = in[q];
    v[4 * q] = bf16_rn(t.x);
    v[4 * q + 1] = bf16_rn(t.y);
    v[4 * q + 2] = bf16_rn(t.z);
    v[4 * q + 3] = bf16_rn(t.w);
  }
  float amax = 0.0F;
#pragma unroll
  for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(v[i]));
  float const s = fminf(amax * a.scale_mult, 448.0F);
  uint8_t const s8 = e4m3_rn(s);
  float const srt = e4m3_to_float(s8);
  float const acc =
      srt > 0.0F ? __fdiv_rn(a.norm_const, fmaxf(srt, 1.1754943508222875e-38F))
                 : 0.0F;
  uint32_t w[2] = {0, 0};
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    float lo = __fmul_rn(v[2 * j], acc);
    float hi = __fmul_rn(v[2 * j + 1], acc);
    lo = lo == lo ? lo : 0.0F;
    hi = hi == hi ? hi : 0.0F;
    uint8_t p = e2m1x2(hi, lo);
    if ((p & 0x07) == 0) p &= 0xF7;
    if ((p & 0x70) == 0) p &= 0x7F;
    w[j / 4] |= uint32_t(p) << (8 * (j % 4));
  }
  *reinterpret_cast<uint2*>(a.x + int64_t{row} * a.x_stride + int64_t{blk} * 8) =
      make_uint2(w[0], w[1]);
  a.sf[int64_t{row} * a.sf_stride + blk] = s8;
}

// MXFP8 block32 of the stager (one block per thread).
__device__ __forceinline__ void quant_mxfp8(const Args& a, int32_t row,
                                            int32_t blk, const float4 (&in)[8]) {
  float v[32];
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    float4 t = in[q];
    v[4 * q] = bf16_rn(t.x);
    v[4 * q + 1] = bf16_rn(t.y);
    v[4 * q + 2] = bf16_rn(t.z);
    v[4 * q + 3] = bf16_rn(t.w);
  }
  float amax = 0.0F;
#pragma unroll
  for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(v[i]));
  float t = __fmul_rn(amax, 1.0F / 448.0F);
  t = (t == t && t < INFINITY) ? t : 0.0F;
  t = fmaxf(t, 1.1754943508222875e-38F);
  int32_t const bits = __float_as_int(t);
  int32_t e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0 ? 1 : 0);
  e = min(max(e, -126), 127);
  float const scale = __int_as_float((e + 127) << 23);
  uint32_t w[8];
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    uint32_t word = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      word |= uint32_t(e4m3_rn(__fdiv_rn(v[4 * q + i], scale))) << (8 * i);
    }
    w[q] = word;
  }
  uint4* dst = reinterpret_cast<uint4*>(a.x + int64_t{row} * a.x_stride +
                                        int64_t{blk} * 32);
  dst[0] = make_uint4(w[0], w[1], w[2], w[3]);
  dst[1] = make_uint4(w[4], w[5], w[6], w[7]);
  a.sf[int64_t{row} * a.sf_stride + blk] = uint8_t(min(max(e + 127, 1), 254));
}

template <bool NVFP4>
__global__ void __launch_bounds__(kWarps * 32) route_stage_kernel(Args a) {
  __shared__ float s_bias[kWarps * 32 * kPer];
  __shared__ float s_unb[kWarps * 32 * kPer];
  __shared__ uint64_t s_cand[kWarps * 32];
  __shared__ uint32_t s_wmax[kWarps];
  __shared__ int s_cnt;
  __shared__ uint64_t s_sel[32];
  int32_t const row = blockIdx.x;
  if (row < a.n) {
    for (int e = threadIdx.x; e < a.num_logits; e += blockDim.x) {
      s_bias[e] = a.bias[e];
    }
  }
  // The GEMM that wrote y (PDL predecessor) and every kernel before it are
  // complete past this point.
  asm volatile("griddepcontrol.wait;" ::: "memory");
  // Release the dependent (token_comm, a PDL launch that griddepcontrol.waits
  // for this grid before it reads any route) now, so its launch and prologue
  // overlap this kernel. Only after the wait: every kernel before this one is
  // complete by then, so nothing the dependent's own dependents may read early
  // is still being written upstream (G2-F3).
  asm volatile("griddepcontrol.launch_dependents;");
  if (row >= a.n) {  // dead rows [n, capacity): routes -1 only
    int64_t const base = a.n + int64_t{row - a.n} * kDeadRows;
    int64_t const end = min(base + kDeadRows, int64_t{a.capacity});
    for (int64_t i = base * a.topk + threadIdx.x; i < end * a.topk;
         i += blockDim.x) {
      a.idx[i] = -1;
    }
    return;
  }
  // Every load of the row in flight at once: this thread's logits and its
  // quantization block (one per thread: hidden / vec <= 256, host-checked).
  // A split-K GEMM's partials (splits > 1) are summed here in split order
  // ((p0 + p1) + p2 ...): the FP32 accumulator the GEMM would have stored.
  const float* yrow = a.y + int64_t{row} * a.ldy;
  float lg[kPer];
#pragma unroll
  for (int j = 0; j < kPer; ++j) {
    int32_t const e = j * (kWarps * 32) + threadIdx.x;
    lg[j] = e < a.num_logits ? yrow[e] : 0.0F;
  }
  constexpr int vec = NVFP4 ? 16 : 32;
  bool const quant = threadIdx.x < a.hidden / vec;
  float4 in[8];
  if (quant) {
    const float4* src = reinterpret_cast<const float4*>(
        yrow + a.num_logits + int64_t{threadIdx.x} * vec);
#pragma unroll
    for (int q = 0; q < vec / 4; ++q) in[q] = src[q];
  }
  if (a.splits == 2) {
    // Both partials' loads in flight together (the general loop below issues
    // split 1's loads behind split 0's); the same (p0 + p1) sums.
    const float* ys = yrow + a.split_stride;
    float l1[kPer];
    float4 p1[8];
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
      int32_t const e = j * (kWarps * 32) + threadIdx.x;
      l1[j] = e < a.num_logits ? ys[e] : 0.0F;
    }
    if (quant) {
      const float4* src = reinterpret_cast<const float4*>(
          ys + a.num_logits + int64_t{threadIdx.x} * vec);
#pragma unroll
      for (int q = 0; q < vec / 4; ++q) p1[q] = src[q];
    }
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
      int32_t const e = j * (kWarps * 32) + threadIdx.x;
      if (e < a.num_logits) lg[j] += l1[j];
    }
    if (quant) {
#pragma unroll
      for (int q = 0; q < vec / 4; ++q) {
        in[q].x += p1[q].x;
        in[q].y += p1[q].y;
        in[q].z += p1[q].z;
        in[q].w += p1[q].w;
      }
    }
  } else
  for (int s = 1; s < a.splits; ++s) {
    const float* ys = yrow + int64_t{s} * a.split_stride;
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
      int32_t const e = j * (kWarps * 32) + threadIdx.x;
      if (e < a.num_logits) lg[j] += ys[e];
    }
    if (quant) {
      const float4* src = reinterpret_cast<const float4*>(
          ys + a.num_logits + int64_t{threadIdx.x} * vec);
#pragma unroll
      for (int q = 0; q < vec / 4; ++q) {
        float4 const p = src[q];
        in[q].x += p.x;
        in[q].y += p.y;
        in[q].z += p.z;
        in[q].w += p.w;
      }
    }
  }
  __syncthreads();  // s_bias
  if (a.topk <= kWarps) {
    // Fast exact top-k (fast_keys): keys first, then the quantization on the
    // quant blocks' threads while the last warp (no quant block when
    // hidden / vec <= (kWarps - 1) * 32) ranks the candidates and writes the
    // routes, so the selection tail hides under the quantization.
    uint64_t k[kPer];
    fast_keys(a, lg, s_bias, s_unb, k, s_wmax, &s_cnt, s_sel);
    __syncthreads();
    fast_compact(k, s_wmax, &s_cnt, s_cand);
    __syncthreads();
    int const cnt = s_cnt;
    if (cnt <= kWarps * 32) {
      if (quant) {
        if (NVFP4) {
          quant_nvfp4(a, row, threadIdx.x, in);
        } else {
          quant_mxfp8(a, row, threadIdx.x, in);
        }
      }
      fast_rank(kWarps * 32 - 1 - threadIdx.x, cnt, a.topk, s_cand, s_sel);
      if (cnt > 32) __syncthreads();  // rankers beyond the last warp
      if ((threadIdx.x >> 5) == kWarps - 1) {
        __syncwarp();
        int32_t const lane = threadIdx.x & 31;
        route_write(a, row, lane < a.topk ? s_sel[lane] : 0, s_unb);
      }
      return;
    }
  }
  if (quant) {
    if (NVFP4) {
      quant_nvfp4(a, row, threadIdx.x, in);
    } else {
      quant_mxfp8(a, row, threadIdx.x, in);
    }
  }
  route_block(a, row, lg, s_bias, s_unb, s_cand);
}

}  // namespace

void route_stage(const torch::Tensor& y, int64_t num_logits,
                 const torch::Tensor& bias, torch::Tensor& x, torch::Tensor& sf,
                 torch::Tensor& idx, torch::Tensor& sw, torch::Tensor& err,
                 const c10::optional<torch::Tensor>& pad, int64_t num_experts,
                 bool renormalize, double routed_scaling_factor, bool nvfp4,
                 double norm_const, const c10::optional<torch::Tensor>& l2p,
                 const c10::optional<torch::Tensor>& replicas,
                 const c10::optional<torch::Tensor>& load,
                 const c10::optional<torch::Tensor>& record,
                 const c10::optional<torch::Tensor>& unpadded) {
  // y: FP32 rows [n, E + hidden], or split-K partials [splits, n, E + hidden].
  const int64_t d = y.dim();
  TORCH_CHECK(y.scalar_type() == at::kFloat && (d == 2 || d == 3) &&
                  y.stride(d - 1) == 1 && y.stride(d - 2) % 4 == 0 &&
                  (d == 2 || y.stride(0) % 4 == 0) &&
                  y.data_ptr<float>() != nullptr &&
                  reinterpret_cast<uintptr_t>(y.data_ptr()) % 16 == 0,
              "route_stage: y must be FP32 rows, 16-byte aligned");
  TORCH_CHECK(bias.scalar_type() == at::kFloat && bias.is_contiguous() &&
                  bias.numel() == num_logits,
              "route_stage: bias");
  TORCH_CHECK(x.scalar_type() == at::kByte && sf.scalar_type() == at::kByte &&
                  x.stride(1) == 1 && sf.stride(1) == 1 &&
                  idx.scalar_type() == at::kInt && idx.is_contiguous() &&
                  sw.scalar_type() == at::kFloat && sw.is_contiguous() &&
                  err.scalar_type() == at::kLong && err.numel() == 2,
              "route_stage: session buffers");
  const int64_t n = y.size(d - 2);
  const int64_t hidden = y.size(d - 1) - num_logits;
  const int64_t capacity = idx.size(0), topk = idx.size(1);
  const int64_t vec = nvfp4 ? 16 : 32;
  TORCH_CHECK(num_logits % 4 == 0 && num_logits <= 1024 && topk <= 32 &&
                  topk <= num_logits && hidden > 0 && hidden % vec == 0 &&
                  n <= capacity && sw.size(0) == capacity &&
                  sw.size(1) == topk && x.size(0) >= capacity &&
                  x.size(1) * (nvfp4 ? 2 : 1) >= hidden &&
                  sf.size(0) >= capacity && sf.size(1) >= hidden / vec &&
                  hidden / vec <= kWarps * 32 &&
                  x.stride(0) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "route_stage: shapes");
  const bool* pad_ptr = nullptr;
  if (pad.has_value()) {
    TORCH_CHECK(pad->scalar_type() == at::kBool && pad->dim() == 1 &&
                    pad->size(0) == n && pad->stride(0) == 1,
                "route_stage: is_padding");
    pad_ptr = pad->data_ptr<bool>();
  }
  Args a;
  a.l2p = nullptr;
  a.replicas = nullptr;
  a.load = nullptr;
  a.record = nullptr;
  a.unpadded = nullptr;
  a.map_slots = 0;
  a.load_size = 0;
  if (l2p.has_value()) {
    TORCH_CHECK(replicas.has_value() && load.has_value() && record.has_value(),
                "route_stage: EPLB needs the map, replica count, load and record tensors");
    TORCH_CHECK(l2p->scalar_type() == at::kLong && l2p->dim() == 2 &&
                    l2p->is_contiguous() && l2p->size(0) == num_logits &&
                    replicas->scalar_type() == at::kLong &&
                    replicas->is_contiguous() && replicas->numel() == num_logits &&
                    load->scalar_type() == at::kInt && load->is_contiguous() &&
                    record->scalar_type() == at::kBool && record->numel() == 1,
                "route_stage: EPLB tensors");
    a.l2p = l2p->data_ptr<int64_t>();
    a.replicas = replicas->data_ptr<int64_t>();
    a.load = load->data_ptr<int32_t>();
    a.record = record->data_ptr<bool>();
    a.map_slots = (int32_t)l2p->size(1);
    a.load_size = (int32_t)load->numel();
    if (unpadded.has_value()) {
      TORCH_CHECK(unpadded->scalar_type() == at::kInt && unpadded->numel() == 1,
                  "route_stage: EPLB unpadded token count");
      a.unpadded = unpadded->data_ptr<int32_t>();
    }
  }
  a.y = y.data_ptr<float>();
  a.ldy = y.stride(d - 2);
  a.splits = d == 3 ? (int32_t)y.size(0) : 1;
  a.split_stride = d == 3 ? y.stride(0) : 0;
  a.num_logits = (int32_t)num_logits;
  a.bias = bias.data_ptr<float>();
  a.x = x.data_ptr<uint8_t>();
  a.x_stride = x.stride(0);
  a.sf = sf.data_ptr<uint8_t>();
  a.sf_stride = sf.stride(0);
  a.sf_cols = (int32_t)sf.size(1);
  a.idx = idx.data_ptr<int32_t>();
  a.sw = sw.data_ptr<float>();
  a.err = err.data_ptr<int64_t>();
  a.pad = pad_ptr;
  a.n = (int32_t)n;
  a.capacity = (int32_t)capacity;
  a.num_experts = (int32_t)num_experts;
  a.topk = (int32_t)topk;
  a.renormalize = renormalize;
  a.routed_scaling_factor = (float)routed_scaling_factor;
  a.hidden = (int32_t)hidden;
  // NVFP4: norm_const / 6 (> 0); MXFP8 marks itself with 0.
  a.scale_mult = nvfp4 ? (float)(norm_const / 6.0) : 0.0F;
  a.norm_const = (float)norm_const;
  TORCH_CHECK(!nvfp4 || a.scale_mult > 0.0F, "route_stage: norm_const");
  const int64_t blocks = n + (capacity - n + kDeadRows - 1) / kDeadRows;
  if (blocks == 0) return;
  const c10::cuda::CUDAGuard guard(y.device());
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3((unsigned)blocks);
  config.blockDim = dim3(kWarps * 32);
  config.dynamicSmemBytes = 0;
  config.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attrs;
  config.numAttrs = 1;
  cudaError_t const rc =
      nvfp4 ? cudaLaunchKernelEx(&config, route_stage_kernel<true>, a)
            : cudaLaunchKernelEx(&config, route_stage_kernel<false>, a);
  TORCH_CHECK(rc == cudaSuccess, "route_stage launch: ", cudaGetErrorString(rc));
}
"""

_CPP_SRC = """
void route_stage(const torch::Tensor& y, int64_t num_logits,
                 const torch::Tensor& bias, torch::Tensor& x, torch::Tensor& sf,
                 torch::Tensor& idx, torch::Tensor& sw, torch::Tensor& err,
                 const c10::optional<torch::Tensor>& pad, int64_t num_experts,
                 bool renormalize, double routed_scaling_factor, bool nvfp4,
                 double norm_const, const c10::optional<torch::Tensor>& l2p,
                 const c10::optional<torch::Tensor>& replicas,
                 const c10::optional<torch::Tensor>& load,
                 const c10::optional<torch::Tensor>& record,
                 const c10::optional<torch::Tensor>& unpadded);
"""

_lock = threading.Lock()
_mod = None


def _module():
    global _mod
    if _mod is not None:
        return _mod
    with _lock:
        if _mod is None:
            from torch.utils.cpp_extension import load_inline

            import vllm.envs as envs

            h = hashlib.sha256((_CUDA_SRC + _CPP_SRC).encode())
            h.update(f"{torch.__version__} {torch.version.cuda}".encode())
            tag = h.hexdigest()[:16]
            build = os.path.join(envs.VLLM_CACHE_ROOT, "k3_route_stage", tag)
            os.makedirs(build, exist_ok=True)
            # No --use_fast_math: precise tanhf, IEEE division (stock numerics).
            _mod = load_inline(
                name=f"k3_route_stage_{tag}",
                cpp_sources=_CPP_SRC,
                cuda_sources=_CUDA_SRC,
                functions=["route_stage"],
                extra_cuda_cflags=[
                    "-O3",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    # cvt.e2m1x2 / griddepcontrol: the sm_100 family target.
                    "-gencode",
                    "arch=compute_100f,code=sm_100f",
                ],
                build_directory=build,
                verbose=False,
            )
    return _mod


def route_stage_cause(mlp) -> str | None:
    """Why ``RouteStage`` cannot reproduce this layer's stock router + MegaMoE
    staging (sigmoid + bias single-group top-k on the Rubin MegaMoE), or None."""
    from ....megamoe_rubin.experts import KimiK3MegaMoEExperts

    experts = getattr(mlp, "experts", None)
    if not isinstance(experts, KimiK3MegaMoEExperts):
        return "not the Rubin MegaMoE"
    if not getattr(mlp, "use_mega_moe", False):
        return "not a MegaMoE layer"
    if getattr(mlp, "moe_router_activation_func", None) != "sigmoid":
        return "router scoring is not sigmoid"
    if mlp.num_expert_group not in (None, 1) or mlp.topk_group not in (None, 1):
        return "grouped routing"
    bias = getattr(mlp.gate, "e_score_correction_bias", None)
    logits = mlp.gate.weight.shape[0]
    if bias is None or bias.dtype != torch.float32 or bias.numel() != logits:
        return "router correction bias"
    if not (experts.top_k <= 32 and experts.top_k <= logits <= 1024 and logits % 4 == 0):
        return f"top-{experts.top_k} of {logits}"
    if experts.problem.quant_kind == "mxfp4":
        # (stages NVFP4 or MXFP8 activations; MXFP4 ones only on the stock stager)
        return "MXFP4 activations (megamoe_quant mxfp4_mxfp4)"
    return None


class RouteStage:
    """The SP MoE front from the router | latent down FP32 GEMM rows to the
    MegaMoE launch (module docstring). ``rows``: the shard rows the GEMM is
    planned for."""

    def __init__(self, mlp, rows):
        cause = route_stage_cause(mlp)
        if cause is not None:
            raise Unsupported("moe", None, f"route_stage: {cause}")
        self.mlp, self.rows = mlp, frozenset(rows)
        self.num_logits = mlp.gate.weight.shape[0]
        self.renormalize = bool(mlp.moe_renormalize)
        self.scale = float(mlp.routed_scaling_factor)
        _module()

    def ready(self, n: int) -> bool:
        """``n`` is planned and the layer has no routing capture (it reads the
        logical ids between router and stager; such layers keep the stock path).
        An EPLB logical-to-physical map is applied in the kernel."""
        return n in self.rows and self.mlp.experts.capture_fn is None

    def stage(self, y: torch.Tensor):
        """Stage ``y``'s routes and quantized latent into the MegaMoE session of
        its bucket; returns the handle :meth:`launch` takes."""
        import vllm.envs as envs
        from vllm.forward_context import get_forward_context, is_forward_context_available

        experts = self.mlp.experts
        n = y.shape[-2]  # y [n, E + latent] or split-K partials [splits, n, E + latent]
        bucket = experts.select_bucket(n)
        rt = experts.get_runtime()
        session = rt.sessions[bucket]
        p = rt.problem
        if y.shape[-1] - self.num_logits != p.hidden or session.topk_idx.shape[1] != experts.top_k:
            raise Unsupported("moe", n, "route_stage: MegaMoE problem shape")
        pad = None
        if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
            pad = get_forward_context().is_padding
            if pad is not None:
                pad = pad[:n]
        _module().route_stage(
            y,
            self.num_logits,
            self.mlp.gate.e_score_correction_bias.data,
            session.x.view(torch.uint8),
            session.x_sf.view(torch.uint8),
            session.topk_idx,
            session.topk_weights,
            rt.route_error,
            pad,
            p.num_experts,
            self.renormalize,
            self.scale,
            p.quant_kind == "nvfp4",
            float(p.input_norm_const),
            *self._eplb_args(),
        )
        return rt, session, n

    def _eplb_args(self):
        """The layer's EPLB tensors for the in-kernel stock mapping and load
        record (``eplb_map_to_physical_and_record``), or Nones without EPLB."""
        st = self.mlp.experts.eplb_state
        if st.logical_to_physical_map is None:
            return None, None, None, None, None
        from vllm.v1.worker.ubatching import dbo_current_ubatch_id

        unpadded = None
        if st.num_unpadded_tokens_tensors is not None:
            unpadded = st.num_unpadded_tokens_tensors[dbo_current_ubatch_id()]
        return (
            st.logical_to_physical_map,
            st.logical_replica_count,
            st.expert_load_view,
            st.should_record_tensor,
            unpadded,
        )

    def launch(self, staged, output_rmsnorm=None, shared=None):
        """MegaMoE on a :meth:`stage`-d session: its BF16 ``[n, latent]`` output
        (``output_rmsnorm`` applied in its reduce), as ``experts(latent,
        topk_weights, topk_ids, output_rmsnorm=...)``. ``shared``: the layer's
        ``FusedSharedArgs`` on a fused-shared bucket; the result is then
        ``(routed, shared_out)``. A fused-shared bucket called without it runs
        its shared tiles on the warmed operands and drops that output."""
        from ....megamoe_rubin.experts import _rmsnorm_args

        rt, session, n = staged
        experts = self.mlp.experts
        weights = experts.kernel_weights
        if weights is None:
            raise RuntimeError(f"{experts.prefix}: Rubin MegaMoE not prepared.")
        out = torch.empty((n, rt.problem.hidden), dtype=torch.bfloat16, device=session.topk_idx.device)
        rmsnorm = None if output_rmsnorm is None else _rmsnorm_args(output_rmsnorm)
        fused = shared
        if session.cfg.shared is not None and shared is None:
            fused = getattr(rt, "_warm_shared", None)
            if fused is None:
                raise RuntimeError(f"{experts.prefix}: this bucket fuses the shared expert: pass shared.")
        experts.wait_scales()
        session.launch(weights, out, experts.ep_tokens(n), rmsnorm, fused)
        experts.prefetch_next_scales()
        owner = experts.prefix
        if rt._poll_owner in (None, owner):
            rt._poll_owner = owner
            rt.poll_route_errors()
        if shared is not None:
            return out, session.shared_output[:n]
        return out
