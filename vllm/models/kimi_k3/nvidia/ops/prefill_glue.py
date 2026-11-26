# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row-parallel glue kernels for Kimi K3 prefill-sized batches.

Each kernel here computes exactly what a stock kernel computes (bitwise, the
unit tests compare every output bit), with a work layout that keeps the
prefill shapes at their memory or latency floor. The stock kernels are tuned
for decode: one thread block per token row, so a prefill step of thousands of
rows runs several waves of short-lived blocks.

* :func:`router_topk`: the sigmoid + correction-bias single-group top-k of
  ``_moe_C.grouped_topk`` (csrc/moe/grouped_topk_kernels.cu,
  ``single_group_topk_block_kernel``), one warp per token. Selection is by the
  same packed ``(score, 65535 - expert)`` key, so ties pick the same experts
  in the same order; the weights are formed by the same ``write_outputs``
  code (cooperative-groups sum over the lanes in rank order).
* :func:`rms_norm`: ``_C.rms_norm`` (csrc/layernorm_kernels.cu) with one warp
  per row. The warp replays the stock reduction exactly: the per-thread sums
  of the ``block_size`` stock threads (same element order, same fused
  multiply-adds), each 32-thread group's CUB shuffle-down tree, then the
  sequential sum of the group totals; the normalization is the stock
  ``x * rsqrt(var / n + eps) * w`` in fp32, rounded once to BF16.
* :func:`gate_sigmoid_mul`: ``attn * gate.sigmoid()`` in one pass: the ATen
  BF16 sigmoid (``1 / (1 + exp(-g))`` in fp32, rounded to BF16) and the ATen
  BF16 multiply (fp32 product rounded to BF16), instead of two elementwise
  kernels and a BF16 temporary.

All three are built without fast math (precise ``expf`` / ``tanhf``, IEEE
division, no flush-to-zero), as the stock kernels are. The callers dispatch
per shape (``use_*``): the stock kernels keep decode-sized batches, where one
block per row is the faster layout.

Built on first use per source hash (torch JIT extension, cached under
``VLLM_CACHE_ROOT/k3_prefill_glue``).
"""

import hashlib
import os
import threading

import torch

from vllm.model_executor.layers.layernorm import RMSNorm

# Per-shape policy (VR200 microbenchmarks, GPU time; see use_rows_kernels):
# rows at or above which each row-parallel kernel replaces the stock one.
# * router top-k (896 experts, top 16): 23.7 vs 27.7 us at 8192 rows, even at
#   4350, slower below (the stock block kernel scores 3.5 experts per thread).
# * RMSNorm: 9.2 vs 10.6 us at 4350 x 3584, 11.3 vs 28.4 us at 17400 x 1536,
#   4.3 vs 9.9 us at 17400 x 512; slower at 1024 x 3584 and for 7168 columns
#   (4 vectors per lane), which keep the stock kernel.
# * gate sigmoid-mul: 18.5 vs 31.7 us at 8192 x 3072, 35.9 vs 61.4 us at
#   17400 x 3072.
ROUTER_TOPK_MIN_ROWS = 6144
RMS_NORM_MIN_ROWS = 2048
RMS_NORM_MAX_VECS_PER_LANE = 2
# kRmsCtaMinRows: rows from which 8-warp stock blocks run as persistent CTAs.
RMS_NORM_CTA_MIN_ROWS = 4096
GATE_MIN_ROWS = 1024

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/BFloat16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

namespace cg = cooperative_groups;

namespace {

constexpr uint32_t kFull = 0xffffffffu;

__device__ __forceinline__ float bf16_bits_to_float(uint16_t b) {
  return (float)c10::BFloat16(b, c10::BFloat16::from_bits());
}

// ---------------------------------------------------------------------------
// router_topk: csrc/moe/grouped_topk_kernels.cu single_group_topk (sigmoid,
// fp32 scores, fp32 bias, int32 ids), one warp per token.
// ---------------------------------------------------------------------------

// Verbatim: sigmoid_accurate / preprocess_score (SCORING_SIGMOID).
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

// reduce_topk::TopKRedType<float>::makeCmpVal / unpack (cub TwiddleIn/Out).
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

// Verbatim: single_group_topk::detail::write_outputs.
__device__ __forceinline__ void write_outputs(
    cg::thread_block_tile<32> const& warp, float lane_selection_score,
    float lane_unbiased, int32_t lane_expert, int32_t lane, int32_t token,
    int32_t topk, float* topk_values, int32_t* topk_indices, bool renormalize,
    float routed_scaling_factor) {
  bool const finite_selection =
      lane < topk && lane_selection_score != -INFINITY;
  lane_unbiased = finite_selection ? lane_unbiased : 0.0F;
  unsigned const finite_mask = __ballot_sync(kFull, finite_selection);
  float const sum = cg::reduce(warp, lane_unbiased, cg::plus<float>{});

  if (lane < topk) {
    float output = 0.0F;
    if (finite_mask == 0) {
      if (renormalize) {
        output = 1.0F / static_cast<float>(topk);
      }
    } else if (finite_selection) {
      float scale = routed_scaling_factor;
      if (renormalize) {
        scale /= sum + 1e-20F;
      }
      output = lane_unbiased * scale;
    }
    int64_t const output_index = int64_t{token} * topk + lane;
    topk_values[output_index] = output;
    topk_indices[output_index] = lane_expert;
  }
}

// Top-2 of this lane's keys (expert j * 32 + lane, score sel[j]) strictly
// below `limit` (0 = none).
template <int VPL>
__device__ __forceinline__ void lane_top2_regs(const float (&sel)[VPL],
                                               int32_t lane, uint64_t limit,
                                               uint64_t& h1, uint64_t& h2) {
  h1 = 0;
  h2 = 0;
#pragma unroll
  for (int j = 0; j < VPL; ++j) {
    uint64_t k = make_key(sel[j], j * 32 + lane);
    k = k < limit ? k : 0;
    if (k > h1) {
      h2 = h1;
      h1 = k;
    } else if (k > h2) {
      h2 = k;
    }
  }
}

// One token per warp, its VPL * 32 selection scores in registers; WARPS
// tokens in flight per block and a grid of resident blocks striding over the
// tokens. All warps score and select (the old one-token-per-block form left
// 3 of 4 warps idle through the serial selection, so the kernel ran latency
// bound at ~16 selecting warps per SM). The unbiased score of a selected
// expert is recomputed from its logit (same expression, same value).
template <int VPL, int WARPS>
__global__ void __launch_bounds__(WARPS * 32)
    router_topk_kernel(const float* __restrict__ scores, int64_t ld,
                       const float* __restrict__ bias, float* topk_values,
                       int32_t* topk_indices, int32_t num_tokens,
                       int32_t num_experts, int32_t topk, bool renormalize,
                       float routed_scaling_factor) {
  __shared__ float s_bias[VPL * 32];
  __shared__ uint64_t cand_keys[WARPS][32];
  auto warp = cg::tiled_partition<32>(cg::this_thread_block());
  int32_t const lane = threadIdx.x & 31;
  int32_t const wid = threadIdx.x >> 5;
  for (int e = threadIdx.x; e < VPL * 32; e += WARPS * 32) {
    s_bias[e] = e < num_experts ? bias[e] : 0.0F;
  }
  __syncthreads();
  for (int32_t token = blockIdx.x * WARPS + wid; token < num_tokens;
       token += gridDim.x * WARPS) {
    const float* row = scores + int64_t{token} * ld;
    float in[VPL];
#pragma unroll
    for (int j = 0; j < VPL; ++j) {
      int32_t const e = j * 32 + lane;
      in[j] = e < num_experts ? __ldg(row + e) : 0.0F;
    }
    float sel[VPL];
    float lane_max = -INFINITY;
#pragma unroll
    for (int j = 0; j < VPL; ++j) {
      int32_t const e = j * 32 + lane;
      float unb = 0.0F, s = -INFINITY;
      if (e < num_experts) {
        preprocess_score(in[j], s_bias[e], unb, s);
      }
      sel[j] = s;
      lane_max = fmaxf(lane_max, s);
    }

    // Prefilter: tau = the topk-th largest lane maximum. At least topk scores
    // are >= tau, so no score below tau ranks in the top k (the key orders by
    // score first); the candidates (score >= tau) are usually a few dozen.
    float m = lane_max;
#pragma unroll
    for (int k = 2; k <= 32; k <<= 1) {
#pragma unroll
      for (int j = k >> 1; j > 0; j >>= 1) {
        float const o = __shfl_xor_sync(kFull, m, j);
        bool const keep_max = ((lane & j) == 0) == ((lane & k) == 0);
        m = keep_max ? fmaxf(m, o) : fminf(m, o);
      }
    }
    float const tau = __shfl_sync(kFull, m, topk - 1);
    int32_t count = 0;
#pragma unroll
    for (int j = 0; j < VPL; ++j) count += sel[j] >= tau ? 1 : 0;
    int32_t incl = count;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      int32_t const o = __shfl_up_sync(kFull, incl, off);
      if (lane >= off) incl += o;
    }
    int32_t const num_cand = __shfl_sync(kFull, incl, 31);

    uint64_t lane_packed = 0;
    if (num_cand <= 32) {
      // Exact order of the candidates: a 32-lane bitonic sort of their keys,
      // largest first (lane r holds rank r).
      int32_t pos = incl - count;
#pragma unroll
      for (int j = 0; j < VPL; ++j) {
        if (sel[j] >= tau) cand_keys[wid][pos++] = make_key(sel[j], j * 32 + lane);
      }
      __syncwarp();
      uint64_t key = lane < num_cand ? cand_keys[wid][lane] : 0;
      __syncwarp();
#pragma unroll
      for (int k = 2; k <= 32; k <<= 1) {
#pragma unroll
        for (int j = k >> 1; j > 0; j >>= 1) {
          uint32_t const ohi = __shfl_xor_sync(kFull, uint32_t(key >> 32), j);
          uint32_t const olo = __shfl_xor_sync(kFull, uint32_t(key), j);
          uint64_t const o = (uint64_t(ohi) << 32) | olo;
          bool const keep_max = ((lane & j) == 0) == ((lane & k) == 0);
          key = keep_max ? (key > o ? key : o) : (key < o ? key : o);
        }
      }
      lane_packed = key;
    } else {
      // Many scores tied at tau: select rank by rank from every lane's keys.
      uint64_t h1, h2;
      lane_top2_regs<VPL>(sel, lane, ~uint64_t{0}, h1, h2);
      bool h2_valid = true;
      for (int kk = 0; kk < topk; ++kk) {
        uint64_t const best = warp_max_u64(h1);
        if (lane == kk) lane_packed = best;
        if (h1 == best) {  // the one lane holding it
          if (h2_valid) {
            h1 = h2;
            h2_valid = false;
          } else {
            lane_top2_regs<VPL>(sel, lane, best, h1, h2);
            h2_valid = true;
          }
        }
      }
    }

    float lane_score = -INFINITY;
    int32_t lane_expert = -1;
    if (lane < topk) {
      unpack_key(lane_packed, lane_score, lane_expert);
    }
    float lane_unbiased = 0.0F;
    if (lane < topk && lane_expert >= 0 && lane_expert < num_experts) {
      float s;
      preprocess_score(__ldg(row + lane_expert), s_bias[lane_expert],
                       lane_unbiased, s);
    }
    write_outputs(warp, lane_score, lane_unbiased, lane_expert, lane, token,
                  topk, topk_values, topk_indices, renormalize,
                  routed_scaling_factor);
  }
}

// ---------------------------------------------------------------------------
// rms_norm: csrc/layernorm_kernels.cu rms_norm_kernel<bf16, 8, 2, true>
// launched with block_size = NW * 32 threads, replayed by one warp per row.
// ---------------------------------------------------------------------------

template <int NW, int VPT>
__device__ __forceinline__ void rms_norm_rows_body(
    int32_t const blk, uint16_t* __restrict__ out,
    const uint16_t* __restrict__ input, int64_t in_stride,
    const uint16_t* __restrict__ weight, float epsilon, int32_t num_tokens,
    int32_t hidden_size) {
  constexpr int B = NW * 32;  // stock block size
  int32_t const lane = threadIdx.x & 31;
  int32_t const row = blk * 4 + (threadIdx.x >> 5);
  if (row >= num_tokens) return;  // warp-uniform
  int32_t const num_vec = hidden_size / 8;
  const uint4* v_in =
      reinterpret_cast<const uint4*>(input + int64_t{row} * in_stride);

  uint4 data[NW][VPT];
  float part[NW];
#pragma unroll
  for (int w = 0; w < NW; ++w) {
    // stock thread tid = lane + 32 * w: vectors tid, tid + B, ...
    float variance = 0.0f;
#pragma unroll
    for (int t = 0; t < VPT; ++t) {
      int32_t const i = lane + 32 * w + B * t;
      if (i < num_vec) {
        data[w][t] = v_in[i];
        const uint16_t* h = reinterpret_cast<const uint16_t*>(&data[w][t]);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          float x = bf16_bits_to_float(h[j]);
          variance += x * x;
        }
      }
    }
    part[w] = variance;
  }
  // cub::WarpReduceShfl (shuffle-down offsets 1..16) per stock warp, then
  // BlockReduceWarpReductions: warp 0 total + warp 1 total + ... in order.
  float total = 0.0f;
#pragma unroll
  for (int w = 0; w < NW; ++w) {
    float s = part[w];
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      float o = __shfl_down_sync(kFull, s, off);
      if (lane + off < 32) s = o + s;
    }
    total = w == 0 ? s : total + s;
  }
  float const s_variance =
      __shfl_sync(kFull, rsqrtf(total / hidden_size + epsilon), 0);

  const uint4* v_w = reinterpret_cast<const uint4*>(weight);
  uint4* v_out = reinterpret_cast<uint4*>(out + int64_t{row} * hidden_size);
#pragma unroll
  for (int w = 0; w < NW; ++w) {
#pragma unroll
    for (int t = 0; t < VPT; ++t) {
      int32_t const i = lane + 32 * w + B * t;
      if (i < num_vec) {
        uint4 wr = __ldg(v_w + i);
        const uint16_t* h = reinterpret_cast<const uint16_t*>(&data[w][t]);
        const uint16_t* wh = reinterpret_cast<const uint16_t*>(&wr);
        uint4 o;
        uint16_t* oh = reinterpret_cast<uint16_t*>(&o);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          float x = bf16_bits_to_float(h[j]);
          float wv = bf16_bits_to_float(wh[j]);
          oh[j] = c10::BFloat16(x * s_variance * wv).x;
        }
        v_out[i] = o;
      }
    }
  }
}

template <int NW, int VPT>
__global__ void __launch_bounds__(128)
    rms_norm_rows_kernel(uint16_t* __restrict__ out,
                         const uint16_t* __restrict__ input,
                         int64_t in_stride, const uint16_t* __restrict__ weight,
                         float epsilon, int32_t num_tokens,
                         int32_t hidden_size) {
  rms_norm_rows_body<NW, VPT>(blockIdx.x, out, input, in_stride, weight,
                              epsilon, num_tokens, hidden_size);
}

// Two independent row norms in one launch (the DSA q_a and kv_a norms of the
// same rows): CTAs [0, blocks1) replay rms_norm_rows_kernel<NW1, VPT1> on the
// first problem, the rest rms_norm_rows_kernel<NW2, VPT2> on the second. Same
// per-row code, so each output is bitwise the separate launch's.
template <int NW1, int VPT1, int NW2, int VPT2>
__global__ void __launch_bounds__(128) rms_norm_rows_pair_kernel(
    uint16_t* __restrict__ out1, const uint16_t* __restrict__ in1,
    int64_t in1_stride, const uint16_t* __restrict__ w1, int32_t n1,
    int32_t h1, uint16_t* __restrict__ out2, const uint16_t* __restrict__ in2,
    int64_t in2_stride, const uint16_t* __restrict__ w2, int32_t n2,
    int32_t h2, float epsilon, int32_t blocks1) {
  if ((int32_t)blockIdx.x < blocks1) {
    rms_norm_rows_body<NW1, VPT1>(blockIdx.x, out1, in1, in1_stride, w1,
                                  epsilon, n1, h1);
  } else {
    rms_norm_rows_body<NW2, VPT2>((int32_t)blockIdx.x - blocks1, out2, in2,
                                  in2_stride, w2, epsilon, n2, h2);
  }
}

// The same replay with one row per 4-warp CTA at a time (real warp r replays
// stock warps r * NW / 4 ..): wide rows on many tokens, where one row per warp
// leaves a partial second wave of rows (16384 rows on 212 x 64 warps).
// Persistent CTAs (one resident wave) walk rows with a stride of gridDim.x and
// load the next row before reducing / writing the current one, so HBM reads
// stay in flight across the reduction and the __syncthreads (one row per CTA
// without prefetch streamed ~6 TB/s). Per row the arithmetic and reduction
// order are unchanged (bitwise).
template <int NW, int VPT>
__global__ void __launch_bounds__(128)
    rms_norm_cta_kernel(uint16_t* __restrict__ out,
                        const uint16_t* __restrict__ input, int64_t in_stride,
                        const uint16_t* __restrict__ weight, float epsilon,
                        int32_t hidden_size, int64_t num_rows) {
  static_assert(NW % 4 == 0, "4 real warps replay NW stock warps");
  constexpr int B = NW * 32;
  constexpr int PW = NW / 4;  // stock warps per real warp
  __shared__ float s_part[NW];
  int32_t const lane = threadIdx.x & 31;
  int32_t const rw = threadIdx.x >> 5;
  int32_t const num_vec = hidden_size / 8;

  uint4 data[PW][VPT];
  auto load_row = [&](int64_t r, uint4 (&dst)[PW][VPT]) {
    const uint4* v_in = reinterpret_cast<const uint4*>(input + r * in_stride);
#pragma unroll
    for (int p = 0; p < PW; ++p) {
      int const w = rw * PW + p;
#pragma unroll
      for (int t = 0; t < VPT; ++t) {
        int32_t const i = lane + 32 * w + B * t;
        if (i < num_vec) dst[p][t] = v_in[i];
      }
    }
  };
  int64_t row = blockIdx.x;
  if (row < num_rows) load_row(row, data);
  for (; row < num_rows; row += gridDim.x) {
    uint4 ndata[PW][VPT];
    int64_t const nrow = row + gridDim.x;
    if (nrow < num_rows) load_row(nrow, ndata);
#pragma unroll
    for (int p = 0; p < PW; ++p) {
      int const w = rw * PW + p;
      float variance = 0.0f;
#pragma unroll
      for (int t = 0; t < VPT; ++t) {
        int32_t const i = lane + 32 * w + B * t;
        if (i < num_vec) {
          const uint16_t* h = reinterpret_cast<const uint16_t*>(&data[p][t]);
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            float x = bf16_bits_to_float(h[j]);
            variance += x * x;
          }
        }
      }
      float s = variance;
#pragma unroll
      for (int off = 1; off < 32; off <<= 1) {
        float o = __shfl_down_sync(kFull, s, off);
        if (lane + off < 32) s = o + s;
      }
      if (lane == 0) s_part[w] = s;
    }
    __syncthreads();
    float total = s_part[0];
#pragma unroll
    for (int w = 1; w < NW; ++w) total = total + s_part[w];
    float const s_variance = rsqrtf(total / hidden_size + epsilon);
    // s_part is rewritten by the next row's reduction.
    __syncthreads();

    const uint4* v_w = reinterpret_cast<const uint4*>(weight);
    uint4* v_out = reinterpret_cast<uint4*>(out + row * hidden_size);
#pragma unroll
    for (int p = 0; p < PW; ++p) {
      int const w = rw * PW + p;
#pragma unroll
      for (int t = 0; t < VPT; ++t) {
        int32_t const i = lane + 32 * w + B * t;
        if (i < num_vec) {
          uint4 wr = __ldg(v_w + i);
          const uint16_t* h = reinterpret_cast<const uint16_t*>(&data[p][t]);
          const uint16_t* wh = reinterpret_cast<const uint16_t*>(&wr);
          uint4 o;
          uint16_t* oh = reinterpret_cast<uint16_t*>(&o);
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            float x = bf16_bits_to_float(h[j]);
            float wv = bf16_bits_to_float(wh[j]);
            oh[j] = c10::BFloat16(x * s_variance * wv).x;
          }
          v_out[i] = o;
        }
      }
    }
#pragma unroll
    for (int p = 0; p < PW; ++p)
#pragma unroll
      for (int t = 0; t < VPT; ++t) data[p][t] = ndata[p][t];
  }
}

constexpr int64_t kRmsCtaMinRows = 4096;

// Stock blocks of SEG = 8 / 16 threads (hidden 64 / 128, e.g. per-head q / k
// norms): cub's logical warp of SEG lanes (shuffle-down offsets 1..SEG/2),
// replayed by SEG-lane segments, 32 / SEG rows per warp.
template <int SEG>
__global__ void __launch_bounds__(128)
    rms_norm_seg_kernel(uint16_t* __restrict__ out,
                        const uint16_t* __restrict__ input, int64_t in_stride,
                        const uint16_t* __restrict__ weight, float epsilon,
                        int32_t num_tokens, int32_t hidden_size,
                        int32_t heads) {
  int32_t const lane = threadIdx.x & 31;
  int32_t const seg_lane = lane & (SEG - 1);
  int32_t const row = (blockIdx.x * 128 + threadIdx.x) / SEG;
  bool const live = row < num_tokens;
  uint4 data = make_uint4(0, 0, 0, 0);
  float s = 0.0f;
  if (live) {
    data = reinterpret_cast<const uint4*>(
        input + int64_t{row / heads} * in_stride +
        int64_t{row % heads} * hidden_size)[seg_lane];
    const uint16_t* h = reinterpret_cast<const uint16_t*>(&data);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float x = bf16_bits_to_float(h[j]);
      s += x * x;
    }
  }
#pragma unroll
  for (int off = 1; off < SEG; off <<= 1) {
    float o = __shfl_down_sync(kFull, s, off);
    if (seg_lane + off < SEG) s = o + s;
  }
  float const s_variance = __shfl_sync(
      kFull, rsqrtf(s / hidden_size + epsilon), lane & ~(SEG - 1));
  if (!live) return;
  uint4 wr = __ldg(reinterpret_cast<const uint4*>(weight) + seg_lane);
  const uint16_t* h = reinterpret_cast<const uint16_t*>(&data);
  const uint16_t* wh = reinterpret_cast<const uint16_t*>(&wr);
  uint4 o;
  uint16_t* oh = reinterpret_cast<uint16_t*>(&o);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    float x = bf16_bits_to_float(h[j]);
    float wv = bf16_bits_to_float(wh[j]);
    oh[j] = c10::BFloat16(x * s_variance * wv).x;
  }
  reinterpret_cast<uint4*>(out + int64_t{row} * hidden_size)[seg_lane] = o;
}

// ---------------------------------------------------------------------------
// gate_sigmoid_mul: ATen bf16 sigmoid (1 / (1 + exp(-x)) in fp32) then ATen
// bf16 mul, 8 columns per thread.
// ---------------------------------------------------------------------------


// BF16(sigmoid(g)) by table: g is BF16, so the rounded sigmoid takes at most
// 2^16 values. Each CTA builds the entries of |g| in [2^-16, 2^16) (both
// signs, 16 KiB of BF16 bits) with the exact expression of
// gate_sigmoid_mul_vec8; zeros, subnormals, huge values, Inf and NaN take the
// expression directly. Bitwise equal; the precise expf + IEEE division per
// element (2 MUFU ops) become one shared-memory load.
constexpr int GATE_LUT_E0 = 111;
constexpr int GATE_LUT_NE = 32;
constexpr int GATE_LUT_N = 2 * GATE_LUT_NE * 128;

__device__ __forceinline__ uint16_t gate_sigmoid_bf16(float gv) {
  float const sig = 1.0f / (1.0f + ::expf(-gv));
  return c10::BFloat16(sig).x;
}

__device__ __forceinline__ uint4 gate_sigmoid_mul_vec8_lut(
    uint4 a, uint4 g, const uint16_t* __restrict__ lut) {
  const uint16_t* gh = reinterpret_cast<const uint16_t*>(&g);
  const uint32_t* aw = reinterpret_cast<const uint32_t*>(&a);
  uint4 o;
  uint32_t* ow = reinterpret_cast<uint32_t*>(&o);
  uint32_t s[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const uint32_t b = gh[j];
    const uint32_t e = ((b >> 7) & 0xffu) - (uint32_t)GATE_LUT_E0;
    s[j] = e < (uint32_t)GATE_LUT_NE
               ? lut[((b >> 15) << 12) | (e << 7) | (b & 0x7fu)]
               : gate_sigmoid_bf16(bf16_bits_to_float((uint16_t)b));
  }
  // BF16 x BF16 products, rounded by cvt.rn.bf16x2 (RNE, as c10::BFloat16 for
  // every non-NaN value) on packed pairs.
#pragma unroll
  for (int p = 0; p < 4; ++p) {
    const float2 av = make_float2(__uint_as_float(aw[p] << 16),
                                  __uint_as_float(aw[p] & 0xffff0000u));
    const float2 sv = make_float2(__uint_as_float(s[2 * p] << 16),
                                  __uint_as_float(s[2 * p + 1] << 16));
    const __nv_bfloat162 r = __floats2bfloat162_rn(av.x * sv.x, av.y * sv.y);
    ow[p] = *reinterpret_cast<const uint32_t*>(&r);
  }
  return o;
}

// CONTIG: all three tensors are dense (rows x d), indexed as one vector.
template <bool CONTIG>
__global__ void __launch_bounds__(512) gate_sigmoid_mul_kernel(uint16_t* __restrict__ out,
                                        const uint16_t* __restrict__ attn,
                                        const uint16_t* __restrict__ gate,
                                        int32_t total_vecs, int32_t vecs_per_row,
                                        int64_t attn_stride,
                                        int64_t gate_stride,
                                        int64_t out_stride) {
  constexpr int U = 4;  // loads in flight per thread
  __shared__ uint16_t lut[GATE_LUT_N];
  for (int i = threadIdx.x; i < GATE_LUT_N; i += blockDim.x) {
    const uint32_t bits = ((uint32_t)(i >> 12) << 15) |
                          ((uint32_t)(((i >> 7) & (GATE_LUT_NE - 1)) +
                                      GATE_LUT_E0) << 7) |
                          (uint32_t)(i & 0x7f);
    lut[i] = gate_sigmoid_bf16(bf16_bits_to_float((uint16_t)bits));
  }
  __syncthreads();
  int32_t const stride = gridDim.x * blockDim.x;
  for (int32_t v0 = blockIdx.x * blockDim.x + threadIdx.x; v0 < total_vecs;
       v0 += stride * U) {
    uint4 a[U], g[U];
    int64_t oo[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      int32_t const v = v0 + u * stride;
      if (v < total_vecs) {
        int64_t ao, go;
        if constexpr (CONTIG) {
          ao = go = oo[u] = int64_t{v} * 8;
        } else {
          int32_t const row = v / vecs_per_row;
          int32_t const col = (v - row * vecs_per_row) * 8;
          ao = row * attn_stride + col;
          go = row * gate_stride + col;
          oo[u] = row * out_stride + col;
        }
        a[u] = __ldg(reinterpret_cast<const uint4*>(attn + ao));
        g[u] = __ldg(reinterpret_cast<const uint4*>(gate + go));
      }
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      if (v0 + u * stride < total_vecs) {
        *reinterpret_cast<uint4*>(out + oo[u]) =
            gate_sigmoid_mul_vec8_lut(a[u], g[u], lut);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// scatter_rows: dst[index[i]] = src[i] for whole rows (the KDA final-state
// write-back), 16 bytes per thread; negative indices wrap as in index_put_.
// index may be a strided view (e.g. block_table[:, 0] of a multi-column
// spec-decode mamba block table): element i is index[i * idx_stride].
// ---------------------------------------------------------------------------

template <typename IdxT>
__global__ void scatter_rows_kernel(uint4* __restrict__ dst,
                                    const uint4* __restrict__ src,
                                    const IdxT* __restrict__ index,
                                    int64_t dst_slots, int64_t row_vecs,
                                    int64_t dst_stride_vecs,
                                    int64_t src_stride_vecs,
                                    int64_t idx_stride) {
  int64_t const i = blockIdx.y;
  int64_t slot = (int64_t)index[i * idx_stride];
  if (slot < 0) slot += dst_slots;
  const uint4* s = src + i * src_stride_vecs;
  uint4* d = dst + slot * dst_stride_vecs;
  for (int64_t v = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; v < row_vecs;
       v += (int64_t)gridDim.x * blockDim.x) {
    d[v] = s[v];
  }
}

// RoPE (NeoX, rot_dim == head_dim) on a normed context K row and the K/V cache
// insert of that row (reshape_and_cache_flash, BF16 "auto" cache): one 64-thread
// row group per context row. The RoPE arithmetic is the stock
// apply_token_rotary_embedding expression (FP32 from BF16 inputs, one rounding).
__global__ void rope_store_kv_kernel(
    const __nv_bfloat16* __restrict__ k, int64_t k_stride,
    const __nv_bfloat16* __restrict__ v, int64_t v_stride,
    const int64_t* __restrict__ positions,
    const __nv_bfloat16* __restrict__ cos_sin, int64_t cs_stride,
    const int64_t* __restrict__ slots, __nv_bfloat16* __restrict__ kc,
    __nv_bfloat16* __restrict__ vc, int64_t kbs, int64_t kps, int64_t khs,
    int64_t vbs, int64_t vps, int64_t vhs, int64_t block_size, int rows,
    int heads, int head_dim) {
  const int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= rows) return;
  const int64_t slot = slots[row];
  if (slot < 0) return;
  const int64_t blk = slot / block_size, off = slot % block_size;
  const int half = head_dim / 2, chunks = half / 8, vchunks = head_dim / 8;
  const __nv_bfloat16* cs = cos_sin + positions[row] * cs_stride;
  const __nv_bfloat16* kr = k + row * k_stride;
  __nv_bfloat16* kd = kc + blk * kbs + off * kps;
  for (int u = threadIdx.x; u < heads * chunks; u += blockDim.x) {
    const int h = u / chunks, c = (u % chunks) * 8;
    uint4 xv = *reinterpret_cast<const uint4*>(kr + h * head_dim + c);
    uint4 yv = *reinterpret_cast<const uint4*>(kr + h * head_dim + half + c);
    uint4 cv = *reinterpret_cast<const uint4*>(cs + c);
    uint4 sv = *reinterpret_cast<const uint4*>(cs + half + c);
    const __nv_bfloat16* xb = reinterpret_cast<const __nv_bfloat16*>(&xv);
    const __nv_bfloat16* yb = reinterpret_cast<const __nv_bfloat16*>(&yv);
    const __nv_bfloat16* cb = reinterpret_cast<const __nv_bfloat16*>(&cv);
    const __nv_bfloat16* sb = reinterpret_cast<const __nv_bfloat16*>(&sv);
    uint4 xo, yo;
    c10::BFloat16* xob = reinterpret_cast<c10::BFloat16*>(&xo);
    c10::BFloat16* yob = reinterpret_cast<c10::BFloat16*>(&yo);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float cos_f = __bfloat162float(cb[i]);
      const float sin_f = __bfloat162float(sb[i]);
      const float x_f = __bfloat162float(xb[i]);
      const float y_f = __bfloat162float(yb[i]);
      xob[i] = static_cast<c10::BFloat16>(x_f * cos_f - y_f * sin_f);
      yob[i] = static_cast<c10::BFloat16>(y_f * cos_f + x_f * sin_f);
    }
    *reinterpret_cast<uint4*>(kd + h * khs + c) = xo;
    *reinterpret_cast<uint4*>(kd + h * khs + half + c) = yo;
  }
  const __nv_bfloat16* vr = v + row * v_stride;
  __nv_bfloat16* vd = vc + blk * vbs + off * vps;
  for (int u = threadIdx.x; u < heads * vchunks; u += blockDim.x) {
    const int h = u / vchunks, c = (u % vchunks) * 8;
    *reinterpret_cast<uint4*>(vd + h * vhs + c) =
        *reinterpret_cast<const uint4*>(vr + h * head_dim + c);
  }
}

// ---------------------------------------------------------------------------
// ctx_kv_insert: the context K/V insert of EVERY draft layer in one launch.
// Per (context row, layer): the per-head K RMSNorm (rms_norm_seg_kernel<SEG>,
// the stock SEG-thread block replay, SEG = head_dim / 8), the NeoX RoPE of the
// normed K and the BF16 insert of K and V into the layer's cache
// (rope_store_kv_kernel): bitwise the 2 x layers launches it replaces, with
// the normed K kept in registers instead of a [rows, heads, head_dim] buffer.
// kv: [rows, layers, 2, heads, head_dim] token-major (row stride kv_stride).
// One CTA per context row walks the layers; a SEG-lane segment owns one head
// (lane j holds elements 8j..8j+7; the RoPE partner of lane j is lane j ^ SEG/2).
// ---------------------------------------------------------------------------
constexpr int kCtxKvMaxLayers = 8;
struct CtxKvLayers {
  const uint16_t* knorm_w[kCtxKvMaxLayers];
  const int64_t* slots[kCtxKvMaxLayers];
  __nv_bfloat16* kc[kCtxKvMaxLayers];
  __nv_bfloat16* vc[kCtxKvMaxLayers];
  int rows[kCtxKvMaxLayers];
};

template <int SEG>
__global__ void __launch_bounds__(256) ctx_kv_insert_kernel(
    const __nv_bfloat16* __restrict__ kv, int64_t kv_stride, int layers,
    int heads, const int64_t* __restrict__ positions,
    const __nv_bfloat16* __restrict__ cos_sin, int64_t cs_stride,
    const CtxKvLayers p, float epsilon, int64_t kbs, int64_t kps, int64_t khs,
    int64_t vbs, int64_t vps, int64_t vhs, int64_t block_size) {
  // blockDim.x = heads * SEG (full warps, host check): thread u owns the
  // 16-byte vector u of every layer's K and V row (head u / SEG).
  constexpr int HD = SEG * 8;
  constexpr int HALF = HD / 2;
  const int row = blockIdx.x;
  const int u = threadIdx.x;
  const int lane = u & 31;
  const int seg_lane = lane & (SEG - 1);
  const int h = u / SEG;
  // All loads of the row first (every layer's K and V vector, slots, cos /
  // sin), then the per-layer norm + RoPE + stores.
  int64_t slot[kCtxKvMaxLayers];
  uint4 kdat[kCtxKvMaxLayers], vdat[kCtxKvMaxLayers];
  const __nv_bfloat16* kr0 = kv + row * kv_stride;
#pragma unroll
  for (int l = 0; l < kCtxKvMaxLayers; ++l) {
    slot[l] = -1;
    if (l < layers && row < p.rows[l]) {
      slot[l] = p.slots[l][row];
      const __nv_bfloat16* kr = kr0 + (int64_t)(2 * l) * heads * HD;
      kdat[l] = reinterpret_cast<const uint4*>(kr + h * HD)[seg_lane];
      vdat[l] = reinterpret_cast<const uint4*>(kr + heads * HD + h * HD)[seg_lane];
    }
  }
  // cos / sin of this lane's 8 elements: x lanes (seg_lane < SEG / 2) hold
  // elements c = 8 seg_lane, y lanes the same c of the second half.
  const int c = 8 * (seg_lane & (SEG / 2 - 1));
  const __nv_bfloat16* cs = cos_sin + positions[row] * cs_stride;
  const uint4 cv = *reinterpret_cast<const uint4*>(cs + c);
  const uint4 sv = *reinterpret_cast<const uint4*>(cs + HALF + c);
  const __nv_bfloat16* cb = reinterpret_cast<const __nv_bfloat16*>(&cv);
  const __nv_bfloat16* sb = reinterpret_cast<const __nv_bfloat16*>(&sv);
  const bool is_x = seg_lane < SEG / 2;
#pragma unroll
  for (int l = 0; l < kCtxKvMaxLayers; ++l) {
    if (l >= layers || slot[l] < 0) continue;  // CTA-uniform
    const int64_t blk = slot[l] / block_size, off = slot[l] % block_size;
    __nv_bfloat16* kd = p.kc[l] + blk * kbs + off * kps;
    {
      // rms_norm_seg_kernel<SEG> on head h (hidden_size = HD).
      const uint4 data = kdat[l];
      const uint16_t* hb = reinterpret_cast<const uint16_t*>(&data);
      float s = 0.0f;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        float x = bf16_bits_to_float(hb[j]);
        s += x * x;
      }
#pragma unroll
      for (int o = 1; o < SEG; o <<= 1) {
        float t = __shfl_down_sync(kFull, s, o);
        if (seg_lane + o < SEG) s = t + s;
      }
      const float s_variance = __shfl_sync(
          kFull, rsqrtf(s / HD + epsilon), lane & ~(SEG - 1));
      const uint4 wr = __ldg(reinterpret_cast<const uint4*>(p.knorm_w[l]) + seg_lane);
      const uint16_t* wh = reinterpret_cast<const uint16_t*>(&wr);
      uint4 nk;
      uint16_t* nh = reinterpret_cast<uint16_t*>(&nk);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        float x = bf16_bits_to_float(hb[j]);
        float wv = bf16_bits_to_float(wh[j]);
        nh[j] = c10::BFloat16(x * s_variance * wv).x;
      }
      // NeoX RoPE of the normed head (rope_store_kv_kernel expression): the
      // partner lane holds the other half of the same 8 element pairs.
      uint4 pk;
      pk.x = __shfl_xor_sync(kFull, nk.x, SEG / 2);
      pk.y = __shfl_xor_sync(kFull, nk.y, SEG / 2);
      pk.z = __shfl_xor_sync(kFull, nk.z, SEG / 2);
      pk.w = __shfl_xor_sync(kFull, nk.w, SEG / 2);
      const __nv_bfloat16* mine = reinterpret_cast<const __nv_bfloat16*>(&nk);
      const __nv_bfloat16* other = reinterpret_cast<const __nv_bfloat16*>(&pk);
      uint4 ro;
      c10::BFloat16* rob = reinterpret_cast<c10::BFloat16*>(&ro);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float cos_f = __bfloat162float(cb[i]);
        const float sin_f = __bfloat162float(sb[i]);
        if (is_x) {
          const float x_f = __bfloat162float(mine[i]);
          const float y_f = __bfloat162float(other[i]);
          rob[i] = static_cast<c10::BFloat16>(x_f * cos_f - y_f * sin_f);
        } else {
          const float x_f = __bfloat162float(other[i]);
          const float y_f = __bfloat162float(mine[i]);
          rob[i] = static_cast<c10::BFloat16>(y_f * cos_f + x_f * sin_f);
        }
      }
      *reinterpret_cast<uint4*>(kd + h * khs + 8 * seg_lane) = ro;
    }
    __nv_bfloat16* vd = p.vc[l] + blk * vbs + off * vps;
    *reinterpret_cast<uint4*>(vd + h * vhs + 8 * seg_lane) = vdat[l];
  }
}

}  // namespace

void router_topk(torch::Tensor& topk_values, torch::Tensor& topk_indices,
                 const torch::Tensor& scores, const torch::Tensor& bias,
                 bool renormalize, double routed_scaling_factor) {
  TORCH_CHECK(scores.scalar_type() == at::kFloat &&
              bias.scalar_type() == at::kFloat &&
              topk_values.scalar_type() == at::kFloat &&
              topk_indices.scalar_type() == at::kInt);
  TORCH_CHECK(scores.dim() == 2 && scores.stride(1) == 1 &&
              bias.is_contiguous() && topk_values.is_contiguous() &&
              topk_indices.is_contiguous());
  const int64_t n = scores.size(0);
  const int64_t e = scores.size(1);
  const int64_t k = topk_values.size(1);
  TORCH_CHECK(bias.numel() == e && k <= 32 && e >= k && e <= 1024);
  if (n == 0) return;
  constexpr int kWarps = 8;
  auto stream = c10::cuda::getCurrentCUDAStream();
  const int vpl = (int)((e + 31) / 32);
  static const int sms = [] {
    int dev = 0, v = 0;
    TORCH_CHECK(cudaGetDevice(&dev) == cudaSuccess);
    TORCH_CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount,
                                       dev) == cudaSuccess);
    return v;
  }();
  const dim3 block(kWarps * 32);
  // One resident wave of blocks (grid-stride over the tokens), from the
  // occupancy of the instantiation (register bound at VPL 28-32).
#define K3_TOPK_LAUNCH(V)                                                   \
  {                                                                         \
    static int occ_##V = 0;                                                 \
    if (occ_##V == 0)                                                       \
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(                        \
          &occ_##V, router_topk_kernel<V, kWarps>, kWarps * 32, 0);         \
    const dim3 grid((unsigned)std::min<int64_t>(                            \
        (n + kWarps - 1) / kWarps, (int64_t)sms * std::max(occ_##V, 1)));   \
  router_topk_kernel<V, kWarps><<<grid, block, 0, stream>>>(                \
      scores.data_ptr<float>(), scores.stride(0), bias.data_ptr<float>(),   \
      topk_values.data_ptr<float>(), topk_indices.data_ptr<int32_t>(),      \
      (int32_t)n, (int32_t)e, (int32_t)k, renormalize,                      \
      (float)routed_scaling_factor);                                       \
  }
  if (vpl <= 8) {
    K3_TOPK_LAUNCH(8);
  } else if (vpl <= 16) {
    K3_TOPK_LAUNCH(16);
  } else if (vpl <= 28) {
    K3_TOPK_LAUNCH(28);
  } else {
    K3_TOPK_LAUNCH(32);
  }
#undef K3_TOPK_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rms_norm_rows(torch::Tensor& out, const torch::Tensor& input,
                   const torch::Tensor& weight, double epsilon,
                   int64_t block_size) {
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 &&
              out.scalar_type() == at::kBFloat16 &&
              weight.scalar_type() == at::kBFloat16);
  // 3-D [T, H, D] (per-head norms; rows = T * H, head stride D) is taken by
  // the segment kernels only.
  const bool heads3 = input.dim() == 3;
  TORCH_CHECK((input.dim() == 2 || (heads3 && input.stride(1) == input.size(2))) &&
              input.stride(-1) == 1 && out.is_contiguous() &&
              weight.is_contiguous() && weight.dim() == 1);
  const int64_t heads = heads3 ? input.size(1) : 1;
  const int64_t n = input.size(0) * heads;
  const int64_t h = input.size(-1);
  TORCH_CHECK(h % 8 == 0 && weight.numel() == h && out.numel() == n * h);
  TORCH_CHECK(input.stride(0) % 8 == 0 &&
              reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0);
  if (n == 0) return;
  if ((block_size == 8 || block_size == 16) && h == block_size * 8) {
    auto seg_stream = c10::cuda::getCurrentCUDAStream();
    const dim3 seg_grid((unsigned)((n * block_size + 127) / 128));
#define K3_RMS_SEG_LAUNCH(S)                                                \
  rms_norm_seg_kernel<S><<<seg_grid, 128, 0, seg_stream>>>(                 \
      reinterpret_cast<uint16_t*>(out.data_ptr()),                          \
      reinterpret_cast<const uint16_t*>(input.data_ptr()), input.stride(0), \
      reinterpret_cast<const uint16_t*>(weight.data_ptr()), (float)epsilon, \
      (int32_t)n, (int32_t)h, (int32_t)heads)
    if (block_size == 8) {
      K3_RMS_SEG_LAUNCH(8);
    } else {
      K3_RMS_SEG_LAUNCH(16);
    }
#undef K3_RMS_SEG_LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  TORCH_CHECK(!heads3 && block_size % 32 == 0 && block_size >= 32 &&
              block_size <= 256);
  const int nw = (int)(block_size / 32);
  const int64_t num_vec = h / 8;
  const int vpt = (int)((num_vec + block_size - 1) / block_size);
  const dim3 grid((unsigned)((n + 3) / 4)), block(128);
  auto stream = c10::cuda::getCurrentCUDAStream();
#define K3_RMS_LAUNCH(NW, VPT)                                              \
  rms_norm_rows_kernel<NW, VPT><<<grid, block, 0, stream>>>(                \
      reinterpret_cast<uint16_t*>(out.data_ptr()),                          \
      reinterpret_cast<const uint16_t*>(input.data_ptr()), input.stride(0), \
      reinterpret_cast<const uint16_t*>(weight.data_ptr()), (float)epsilon, \
      (int32_t)n, (int32_t)h)
  if (nw == 8 && (vpt == 2 || vpt == 4) && n >= kRmsCtaMinRows) {
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, input.get_device());
#define K3_RMS_CTA_LAUNCH(VPT)                                                 \
  {                                                                            \
    static int occ = 0;                                                        \
    if (occ == 0)                                                              \
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(                           \
          &occ, rms_norm_cta_kernel<8, VPT>, 128, 0);                          \
    const int64_t ctas = std::min<int64_t>(n, (int64_t)sms * std::max(occ, 1)); \
    rms_norm_cta_kernel<8, VPT><<<(unsigned)ctas, 128, 0, stream>>>(           \
        reinterpret_cast<uint16_t*>(out.data_ptr()),                           \
        reinterpret_cast<const uint16_t*>(input.data_ptr()), input.stride(0),  \
        reinterpret_cast<const uint16_t*>(weight.data_ptr()), (float)epsilon,  \
        (int32_t)h, (int64_t)n);                                               \
  }
    if (vpt == 2) {
      K3_RMS_CTA_LAUNCH(2);
    } else {
      K3_RMS_CTA_LAUNCH(4);
    }
#undef K3_RMS_CTA_LAUNCH
  } else if (nw == 2 && vpt == 1) {
    K3_RMS_LAUNCH(2, 1);
  } else if (nw == 4 && vpt == 1) {
    K3_RMS_LAUNCH(4, 1);
  } else if (nw == 6 && vpt == 1) {
    K3_RMS_LAUNCH(6, 1);
  } else if (nw == 8 && vpt == 1) {
    K3_RMS_LAUNCH(8, 1);
  } else if (nw == 8 && vpt == 2) {
    K3_RMS_LAUNCH(8, 2);
  } else if (nw == 8 && vpt == 4) {
    K3_RMS_LAUNCH(8, 4);
  } else {
    TORCH_CHECK(false, "rms_norm_rows: no instance for block ", block_size,
                " x ", vpt, " vectors per thread");
  }
#undef K3_RMS_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// rms_norm_rows of two 2-D inputs in one launch; returns false (nothing
// launched) when the pair is not the instantiated (6, 1) + (2, 1) warp-per-row
// form (q_lora 1536 + kv_lora 512 at >= 256 rows), so the caller launches the
// two norms separately.
bool rms_norm_rows_pair(torch::Tensor& out1, const torch::Tensor& in1,
                        const torch::Tensor& w1, int64_t block1,
                        torch::Tensor& out2, const torch::Tensor& in2,
                        const torch::Tensor& w2, int64_t block2,
                        double epsilon) {
  auto ok = [](const torch::Tensor& o, const torch::Tensor& x,
               const torch::Tensor& w) {
    return x.scalar_type() == at::kBFloat16 && o.scalar_type() == at::kBFloat16 &&
           w.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.stride(1) == 1 &&
           o.is_contiguous() && w.is_contiguous() && w.dim() == 1 &&
           x.size(1) % 8 == 0 && w.numel() == x.size(1) &&
           o.numel() == x.numel() && x.stride(0) % 8 == 0 &&
           reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
           reinterpret_cast<uintptr_t>(o.data_ptr()) % 16 == 0 &&
           reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0;
  };
  if (!ok(out1, in1, w1) || !ok(out2, in2, w2)) return false;
  // The stock block sizes this pair replays: hidden 1536 -> 192 threads (6
  // warps, 1 vector each), hidden 512 -> 64 threads (2 warps, 1 vector each).
  if (block1 != 192 || in1.size(1) != 1536 || block2 != 64 || in2.size(1) != 512)
    return false;
  const int64_t n1 = in1.size(0), n2 = in2.size(0);
  if (n1 == 0 || n2 == 0 || (n1 + 3) / 4 + (n2 + 3) / 4 >= (int64_t{1} << 31))
    return false;
  const int32_t blocks1 = (int32_t)((n1 + 3) / 4);
  const dim3 grid((unsigned)(blocks1 + (n2 + 3) / 4)), block(128);
  rms_norm_rows_pair_kernel<6, 1, 2, 1>
      <<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<uint16_t*>(out1.data_ptr()),
          reinterpret_cast<const uint16_t*>(in1.data_ptr()), in1.stride(0),
          reinterpret_cast<const uint16_t*>(w1.data_ptr()), (int32_t)n1,
          (int32_t)in1.size(1), reinterpret_cast<uint16_t*>(out2.data_ptr()),
          reinterpret_cast<const uint16_t*>(in2.data_ptr()), in2.stride(0),
          reinterpret_cast<const uint16_t*>(w2.data_ptr()), (int32_t)n2,
          (int32_t)in2.size(1), (float)epsilon, blocks1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return true;
}

void gate_sigmoid_mul(torch::Tensor& out, const torch::Tensor& attn,
                      const torch::Tensor& gate) {
  TORCH_CHECK(attn.scalar_type() == at::kBFloat16 &&
              gate.scalar_type() == at::kBFloat16 &&
              out.scalar_type() == at::kBFloat16);
  TORCH_CHECK(attn.dim() == 2 && gate.dim() == 2 && out.dim() == 2 &&
              attn.stride(1) == 1 && gate.stride(1) == 1 &&
              out.stride(1) == 1);
  const int64_t rows = attn.size(0);
  const int64_t d = attn.size(1);
  TORCH_CHECK(gate.size(0) == rows && gate.size(1) == d &&
              out.size(0) == rows && out.size(1) == d && d % 8 == 0);
  TORCH_CHECK(attn.stride(0) % 8 == 0 && gate.stride(0) % 8 == 0 &&
              out.stride(0) % 8 == 0 &&
              reinterpret_cast<uintptr_t>(attn.data_ptr()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(gate.data_ptr()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0);
  if (rows == 0) return;
  // 512-thread CTAs, 4 per SM: each CTA's table build is ~3% of its work.
  const int threads = 512;
  const int64_t total = rows * (d / 8);
  TORCH_CHECK(total < (int64_t{1} << 31));
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                         attn.get_device());
  const bool contig = attn.stride(0) == d && gate.stride(0) == d &&
                      out.stride(0) == d;
  // One resident wave of CTAs (grid-stride loop).
  static int occ[2] = {0, 0};
  if (occ[contig] == 0) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occ[contig],
        contig ? gate_sigmoid_mul_kernel<true> : gate_sigmoid_mul_kernel<false>,
        threads, 0);
  }
  const int64_t max_blocks = (int64_t)sms * std::max(occ[contig], 1);
  const int blocks =
      (int)std::min<int64_t>((total + threads - 1) / threads, max_blocks);
  auto stream = c10::cuda::getCurrentCUDAStream();
  auto* op = reinterpret_cast<uint16_t*>(out.data_ptr());
  auto* ap = reinterpret_cast<const uint16_t*>(attn.data_ptr());
  auto* gp = reinterpret_cast<const uint16_t*>(gate.data_ptr());
  if (contig) {
    gate_sigmoid_mul_kernel<true><<<blocks, threads, 0, stream>>>(
        op, ap, gp, (int32_t)total, (int32_t)(d / 8), attn.stride(0),
        gate.stride(0), out.stride(0));
  } else {
    gate_sigmoid_mul_kernel<false><<<blocks, threads, 0, stream>>>(
        op, ap, gp, (int32_t)total, (int32_t)(d / 8), attn.stride(0),
        gate.stride(0), out.stride(0));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void scatter_rows(torch::Tensor& dst, const torch::Tensor& index,
                  const torch::Tensor& src) {
  TORCH_CHECK(dst.scalar_type() == src.scalar_type() && index.dim() == 1 &&
              (index.scalar_type() == at::kInt || index.scalar_type() == at::kLong));
  const int64_t rows = index.numel();
  TORCH_CHECK(src.size(0) == rows && rows < 65536);
  const int64_t row_bytes = src[0].numel() * src.element_size();
  TORCH_CHECK(row_bytes % 16 == 0 && dst.stride(0) * dst.element_size() % 16 == 0 &&
              src.stride(0) * src.element_size() % 16 == 0 &&
              reinterpret_cast<uintptr_t>(dst.data_ptr()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(src.data_ptr()) % 16 == 0);
  if (rows == 0) return;
  const int64_t row_vecs = row_bytes / 16;
  const int threads = 256;
  const unsigned bx = (unsigned)std::min<int64_t>((row_vecs + threads - 1) / threads, 1024);
  const dim3 grid(bx, (unsigned)rows);
  auto stream = c10::cuda::getCurrentCUDAStream();
  const int64_t ds = dst.stride(0) * dst.element_size() / 16;
  const int64_t ss = src.stride(0) * src.element_size() / 16;
  const int64_t is = index.stride(0);
  if (index.scalar_type() == at::kInt) {
    scatter_rows_kernel<int32_t><<<grid, threads, 0, stream>>>(
        reinterpret_cast<uint4*>(dst.data_ptr()),
        reinterpret_cast<const uint4*>(src.data_ptr()), index.data_ptr<int32_t>(),
        dst.size(0), row_vecs, ds, ss, is);
  } else {
    scatter_rows_kernel<int64_t><<<grid, threads, 0, stream>>>(
        reinterpret_cast<uint4*>(dst.data_ptr()),
        reinterpret_cast<const uint4*>(src.data_ptr()), index.data_ptr<int64_t>(),
        dst.size(0), row_vecs, ds, ss, is);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rope_store_kv(const torch::Tensor& k, const torch::Tensor& v,
                   const torch::Tensor& positions, const torch::Tensor& cos_sin,
                   const torch::Tensor& slots, torch::Tensor& key_cache,
                   torch::Tensor& value_cache) {
  const int rows = slots.size(0), heads = k.size(1), head_dim = k.size(2);
  if (rows == 0) return;
  const dim3 threads(64, 4);
  const dim3 grid((rows + 3) / 4);
  auto stream = c10::cuda::getCurrentCUDAStream();
  rope_store_kv_kernel<<<grid, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()), k.stride(0),
      reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()), v.stride(0),
      positions.data_ptr<int64_t>(),
      reinterpret_cast<const __nv_bfloat16*>(cos_sin.data_ptr()),
      cos_sin.stride(0), slots.data_ptr<int64_t>(),
      reinterpret_cast<__nv_bfloat16*>(key_cache.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(value_cache.data_ptr()),
      key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
      value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
      key_cache.size(1), rows, heads, head_dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ctx_kv_insert(const torch::Tensor& kv, const std::vector<torch::Tensor>& knorm_w,
                   double epsilon, const torch::Tensor& positions,
                   const torch::Tensor& cos_sin,
                   const std::vector<torch::Tensor>& slots,
                   const std::vector<torch::Tensor>& key_caches,
                   const std::vector<torch::Tensor>& value_caches) {
  // kv [rows, layers, 2, heads, head_dim] BF16, row stride kv.stride(0); one
  // entry per layer in the lists (an undefined slots tensor skips the layer).
  TORCH_CHECK(kv.dim() == 5 && kv.size(2) == 2 && kv.stride(4) == 1 &&
              kv.stride(3) == kv.size(4) && kv.stride(2) == kv.size(3) * kv.size(4) &&
              kv.stride(1) == 2 * kv.stride(2));
  const int64_t rows = kv.size(0);
  const int layers = (int)kv.size(1), heads = (int)kv.size(3);
  const int head_dim = (int)kv.size(4);
  TORCH_CHECK(layers <= kCtxKvMaxLayers && (int)knorm_w.size() == layers &&
              (int)slots.size() == layers && (int)key_caches.size() == layers &&
              (int)value_caches.size() == layers);
  TORCH_CHECK((head_dim == 64 || head_dim == 128) &&
              (heads * (head_dim / 8)) % 32 == 0 &&
              heads * (head_dim / 8) <= 256 && rows < (1LL << 31));
  if (rows == 0) return;
  CtxKvLayers p{};
  int ref = -1;
  for (int l = 0; l < layers; ++l) {
    if (!slots[l].defined() || slots[l].numel() == 0) {
      p.rows[l] = 0;
      continue;
    }
    TORCH_CHECK(slots[l].numel() <= rows);
    p.knorm_w[l] = reinterpret_cast<const uint16_t*>(knorm_w[l].data_ptr());
    p.slots[l] = slots[l].data_ptr<int64_t>();
    p.kc[l] = reinterpret_cast<__nv_bfloat16*>(key_caches[l].data_ptr());
    p.vc[l] = reinterpret_cast<__nv_bfloat16*>(value_caches[l].data_ptr());
    p.rows[l] = (int)slots[l].numel();
    if (ref < 0) ref = l;
    // One stride set for every layer (host-checked by use_ctx_kv_insert).
    TORCH_CHECK(key_caches[l].strides() == key_caches[ref].strides() &&
                value_caches[l].strides() == value_caches[ref].strides() &&
                key_caches[l].size(1) == key_caches[ref].size(1));
  }
  if (ref < 0) return;
  const torch::Tensor& kc = key_caches[ref];
  const torch::Tensor& vc = value_caches[ref];
  auto stream = c10::cuda::getCurrentCUDAStream();
#define K3_CTX_KV_LAUNCH(S)                                                  \
  ctx_kv_insert_kernel<S><<<(unsigned)rows, heads * (head_dim / 8), 0, stream>>>( \
      reinterpret_cast<const __nv_bfloat16*>(kv.data_ptr()), kv.stride(0),   \
      layers, heads, positions.data_ptr<int64_t>(),                          \
      reinterpret_cast<const __nv_bfloat16*>(cos_sin.data_ptr()),            \
      cos_sin.stride(0), p, (float)epsilon, kc.stride(0), kc.stride(1),      \
      kc.stride(2), vc.stride(0), vc.stride(1), vc.stride(2), kc.size(1))
  if (head_dim == 64) {
    K3_CTX_KV_LAUNCH(8);
  } else {
    K3_CTX_KV_LAUNCH(16);
  }
#undef K3_CTX_KV_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_CPP_SRC = """
void router_topk(torch::Tensor& topk_values, torch::Tensor& topk_indices,
                 const torch::Tensor& scores, const torch::Tensor& bias,
                 bool renormalize, double routed_scaling_factor);
void rms_norm_rows(torch::Tensor& out, const torch::Tensor& input,
                   const torch::Tensor& weight, double epsilon,
                   int64_t block_size);
bool rms_norm_rows_pair(torch::Tensor& out1, const torch::Tensor& in1,
                        const torch::Tensor& w1, int64_t block1,
                        torch::Tensor& out2, const torch::Tensor& in2,
                        const torch::Tensor& w2, int64_t block2,
                        double epsilon);
void gate_sigmoid_mul(torch::Tensor& out, const torch::Tensor& attn,
                      const torch::Tensor& gate);
void scatter_rows(torch::Tensor& dst, const torch::Tensor& index,
                  const torch::Tensor& src);
void rope_store_kv(const torch::Tensor& k, const torch::Tensor& v,
                   const torch::Tensor& positions, const torch::Tensor& cos_sin,
                   const torch::Tensor& slots, torch::Tensor& key_cache,
                   torch::Tensor& value_cache);
void ctx_kv_insert(const torch::Tensor& kv, const std::vector<torch::Tensor>& knorm_w,
                   double epsilon, const torch::Tensor& positions,
                   const torch::Tensor& cos_sin,
                   const std::vector<torch::Tensor>& slots,
                   const std::vector<torch::Tensor>& key_caches,
                   const std::vector<torch::Tensor>& value_caches);
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
            build = os.path.join(envs.VLLM_CACHE_ROOT, "k3_prefill_glue", tag)
            os.makedirs(build, exist_ok=True)
            # No --use_fast_math: precise tanhf / expf, IEEE division,
            # subnormals kept, as the stock csrc kernels are built.
            _mod = load_inline(
                name=f"k3_prefill_glue_{tag}",
                cpp_sources=_CPP_SRC,
                cuda_sources=_CUDA_SRC,
                functions=[
                    "router_topk",
                    "rms_norm_rows",
                    "rms_norm_rows_pair",
                    "gate_sigmoid_mul",
                    "scatter_rows",
                    "rope_store_kv",
                    "ctx_kv_insert",
                ],
                extra_cuda_cflags=["-O3"],
                build_directory=build,
                verbose=False,
            )
    return _mod


def use_router_topk(
    scores: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    num_expert_group: int,
    topk_group: int,
    scoring_func: str,
) -> bool:
    """Per-shape policy + capability for :func:`router_topk`."""
    return (
        scores.dim() == 2
        and scores.shape[0] >= ROUTER_TOPK_MIN_ROWS
        and scoring_func == "sigmoid"
        and num_expert_group == 1
        and topk_group == 1
        and scores.dtype == torch.float32
        and scores.stride(1) == 1
        and bias.dtype == torch.float32
        and bias.is_contiguous()
        and bias.numel() == scores.shape[1]
        and topk <= scores.shape[1] <= 1024
        and topk <= 32
    )


def router_topk(
    scores: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``_moe_C.grouped_topk(scores, 1, 1, topk, renormalize,
    routed_scaling_factor, bias, scoring_func=1)`` (sigmoid, one group):
    fp32 weights and int32 expert ids, ``(n, topk)``, bitwise equal."""
    n = scores.shape[0]
    values = torch.empty((n, topk), dtype=torch.float32, device=scores.device)
    ids = torch.empty((n, topk), dtype=torch.int32, device=scores.device)
    _module().router_topk(
        values, ids, scores, bias, bool(renormalize), float(routed_scaling_factor)
    )
    return values, ids


def stock_rms_norm_block_size(num_rows: int, hidden: int) -> int:
    """The thread-block size ``_C.rms_norm`` launches for this shape (BF16,
    16-byte vectors): the stock reduction tree this kernel replays."""
    import vllm.envs as envs

    # csrc/core/utils.hpp vllm_is_batch_invariant(): env VLLM_BATCH_INVARIANT.
    max_block = 1024 if (envs.VLLM_BATCH_INVARIANT or num_rows < 256) else 256
    return min(hidden // 8, max_block)


def use_rms_norm(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Per-shape policy + capability for :func:`rms_norm`: a 2-D BF16 input
    whose stock launch this kernel replays (block of whole warps)."""
    if not (
        x.dim() == 2
        and x.shape[0] >= RMS_NORM_MIN_ROWS
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and weight.dim() == 1
        and weight.is_contiguous()
        and weight.numel() == x.shape[1]
        and x.stride(1) == 1
        and x.shape[1] % 8 == 0
        and x.stride(0) % 8 == 0
        and x.data_ptr() % 16 == 0
        and weight.data_ptr() % 16 == 0
    ):
        return False
    hidden = x.shape[1]
    block = stock_rms_norm_block_size(x.shape[0], hidden)
    vpt = -(-(hidden // 8) // block)
    if block in (8, 16):
        return hidden == block * 8
    if (block, vpt) == (256, 4):
        # 7168 columns: only the persistent one-row-per-CTA replay (the warp
        # per row form is slower than stock at 4 vectors per lane).
        return x.shape[0] >= RMS_NORM_CTA_MIN_ROWS
    return (
        block % 32 == 0
        and vpt <= RMS_NORM_MAX_VECS_PER_LANE
        and (block // 32, vpt) in ((2, 1), (4, 1), (6, 1), (8, 1), (8, 2))
    )


def use_head_rms_norm(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Per-shape policy + capability for :func:`rms_norm` of per-head norms,
    ``x`` = [tokens, heads, head_dim] (head stride head_dim): stock blocks of
    8 / 16 threads (head_dim 64 / 128), replayed one row per lane segment."""
    if not (
        x.dim() == 3
        and x.shape[0] * x.shape[1] >= RMS_NORM_MIN_ROWS
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and weight.is_contiguous()
        and weight.numel() == x.shape[2]
        and x.stride(2) == 1
        and x.stride(1) == x.shape[2]
        and x.stride(0) % 8 == 0
        and x.data_ptr() % 16 == 0
        and weight.data_ptr() % 16 == 0
    ):
        return False
    block = stock_rms_norm_block_size(x.shape[0] * x.shape[1], x.shape[2])
    return block in (8, 16) and x.shape[2] == block * 8


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``_C.rms_norm`` of a 2-D BF16 ``x`` (row stride allowed), bitwise equal;
    the caller checks :func:`use_rms_norm`."""
    if out is None:
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    block = stock_rms_norm_block_size(x.numel() // x.shape[-1], x.shape[-1])
    _module().rms_norm_rows(out, x, weight, float(eps), block)
    return out


def rms_norm_pair(
    x: torch.Tensor,
    wx: torch.Tensor,
    y: torch.Tensor,
    wy: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """:func:`rms_norm` of ``x`` and of ``y`` (same eps) in one launch, bitwise
    the two separate calls; None (nothing launched) for a pair of shapes the
    pair kernel does not instantiate. The caller checks :func:`use_rms_norm`
    for both."""
    bx = stock_rms_norm_block_size(x.numel() // x.shape[-1], x.shape[-1])
    by = stock_rms_norm_block_size(y.numel() // y.shape[-1], y.shape[-1])
    if not (
        x.dim() == 2
        and y.dim() == 2
        and (x.shape[-1], bx, y.shape[-1], by) == (1536, 192, 512, 64)
    ):
        return None
    out_x = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    out_y = torch.empty(y.shape, dtype=y.dtype, device=y.device)
    if not _module().rms_norm_rows_pair(out_x, x, wx, bx, out_y, y, wy, by, float(eps)):
        return None
    return out_x, out_y


def use_gate_sigmoid_mul(attn: torch.Tensor, gate: torch.Tensor) -> bool:
    """Per-shape policy + capability for :func:`gate_sigmoid_mul`."""
    return (
        attn.dim() == 2
        and attn.shape[0] >= GATE_MIN_ROWS
        and gate.shape == attn.shape
        and attn.dtype == torch.bfloat16
        and gate.dtype == torch.bfloat16
        and attn.stride(1) == 1
        and gate.stride(1) == 1
        and attn.shape[1] % 8 == 0
        and attn.stride(0) % 8 == 0
        and gate.stride(0) % 8 == 0
        and attn.data_ptr() % 16 == 0
        and gate.data_ptr() % 16 == 0
        and attn.numel() < 2**34
    )


def gate_sigmoid_mul(attn: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """``attn * gate.sigmoid()`` for 2-D BF16 tensors, bitwise equal; the
    caller checks :func:`gate_sigmoid_mul_supported`."""
    out = torch.empty(attn.shape, dtype=attn.dtype, device=attn.device)
    _module().gate_sigmoid_mul(out, attn, gate)
    return out


def eager_gate_sigmoid_mul(attn: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """``attn * gate.sigmoid()`` as eager PyTorch computes it (the BF16 sigmoid
    rounded, then the BF16 product), in one pass for prefill-sized inputs."""
    if use_gate_sigmoid_mul(attn, gate):
        return gate_sigmoid_mul(attn, gate)
    return attn * gate.sigmoid()


def write_state_rows(
    dst: torch.Tensor, index: torch.Tensor, src: torch.Tensor
) -> None:
    """``dst[index] = src`` (whole rows: dst / src rows dense past dim 0). A
    row copy kernel where it applies (same dtype, 16-byte rows; one launch, no
    index conversion); ``index_put_`` otherwise."""
    if (
        src.dtype == dst.dtype
        and index.dim() == 1
        and index.dtype in (torch.int32, torch.int64)
        and index.is_cuda
        and src.shape[0] == index.numel() < 65536
        and src.shape[1:] == dst.shape[1:]
        and dst[0].is_contiguous()
        and src[0].is_contiguous()
        and (src[0].numel() * src.element_size()) % 16 == 0
        and (dst.stride(0) * dst.element_size()) % 16 == 0
        and (src.stride(0) * src.element_size()) % 16 == 0
        and dst.data_ptr() % 16 == 0
        and src.data_ptr() % 16 == 0
    ):
        _module().scatter_rows(dst, index, src)
        return
    dst[index] = src.to(dst.dtype)


def _aligned(t: torch.Tensor, strides) -> bool:
    return t.data_ptr() % 16 == 0 and all(s % 8 == 0 for s in strides)


def use_rope_store_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    slots: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> bool:
    """Whether :func:`rope_store_kv` covers this context-K/V insert: BF16 rows
    ``[rows, heads, head_dim]`` (heads x head_dim dense per row), NeoX RoPE over
    the whole head (``cos_sin`` ``[max_pos, head_dim]`` BF16), int64 positions and
    slots, BF16 caches ``[blocks, block_size, heads, head_dim]`` dense in the head."""
    if k.dim() != 3 or v.shape != k.shape or key_cache.dim() != 4:
        return False
    rows, heads, head_dim = k.shape
    return (
        k.dtype == v.dtype == cos_sin.dtype == torch.bfloat16
        and key_cache.dtype == value_cache.dtype == torch.bfloat16
        and k.is_cuda
        and positions.dtype == slots.dtype == torch.int64
        and positions.dim() == slots.dim() == 1
        and positions.is_contiguous()
        and slots.is_contiguous()
        and positions.numel() >= slots.numel() == rows
        and rows < 2**31
        and head_dim % 16 == 0
        and cos_sin.dim() == 2
        and cos_sin.shape[1] == head_dim
        and cos_sin.stride(1) == 1
        and value_cache.shape == key_cache.shape
        and tuple(key_cache.shape[2:]) == (heads, head_dim)
        and k.stride(2) == v.stride(2) == 1
        and k.stride(1) == v.stride(1) == head_dim
        and key_cache.stride(3) == value_cache.stride(3) == 1
        and _aligned(k, (k.stride(0),))
        and _aligned(v, (v.stride(0),))
        and _aligned(cos_sin, (cos_sin.stride(0),))
        and _aligned(key_cache, key_cache.stride()[:3])
        and _aligned(value_cache, value_cache.stride()[:3])
    )


def rope_store_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    slots: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> None:
    """NeoX ``ops.rotary_embedding`` of ``k`` (out of place) followed by
    ``reshape_and_cache_flash(k_roped, v, key_cache, value_cache, slots, "auto")``
    in one pass (slots < 0 skipped): bitwise the two stock kernels. Callers check
    :func:`use_rope_store_kv`."""
    _module().rope_store_kv(k, v, positions, cos_sin, slots, key_cache, value_cache)


CTX_KV_MAX_LAYERS = 8


def use_ctx_kv_insert(
    kv: torch.Tensor,
    knorm_weights: list[torch.Tensor],
    eps: float,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    slots: list[torch.Tensor | None],
    caches: list[tuple[torch.Tensor, torch.Tensor] | None],
) -> bool:
    """Whether :func:`ctx_kv_insert` covers this context step: ``kv`` the dense
    token-major projection ``[rows, layers, 2, heads, head_dim]``, every layer
    with a cache covered by :func:`use_rope_store_kv` on the same cache strides,
    per-head K norms replayed by the SEG = head_dim / 8 segment kernel."""
    if kv.dim() != 5 or kv.shape[2] != 2 or not kv.is_contiguous():
        return False
    rows, layers, _, heads, head_dim = kv.shape
    if not (
        0 < layers <= CTX_KV_MAX_LAYERS
        and len(knorm_weights) == len(slots) == len(caches) == layers
        and head_dim in (64, 128)
        and (heads * (head_dim // 8)) % 32 == 0
        and heads * (head_dim // 8) <= 256
        and kv.dtype == torch.bfloat16
        and kv.data_ptr() % 16 == 0
    ):
        return False
    ref = None
    for w, s, c in zip(knorm_weights, slots, caches):
        if c is None or s is None:
            continue
        if not (
            w.dtype == torch.bfloat16
            and w.is_contiguous()
            and w.numel() == head_dim
            and w.data_ptr() % 16 == 0
        ):
            return False
        # The per-head norm launch this kernel replaces (stock blocks of SEG).
        if stock_rms_norm_block_size(rows * heads, head_dim) != head_dim // 8:
            return False
        k = kv[: s.numel(), 0, 0]
        if not use_rope_store_kv(k, k, positions, cos_sin, s, *c):
            return False
        if ref is None:
            ref = c
        elif (
            c[0].stride() != ref[0].stride()
            or c[1].stride() != ref[1].stride()
            or c[0].shape[1] != ref[0].shape[1]
        ):
            return False
    return ref is not None


def ctx_kv_insert(
    kv: torch.Tensor,
    knorm_weights: list[torch.Tensor],
    eps: float,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    slots: list[torch.Tensor | None],
    caches: list[tuple[torch.Tensor, torch.Tensor] | None],
) -> None:
    """Per layer ``i``: ``k = rms_norm(kv[:, i, 0], knorm_weights[i])`` (per head),
    then :func:`rope_store_kv` ``(k, kv[:, i, 1], positions, cos_sin, slots[i],
    *caches[i])`` - every layer in ONE launch, bitwise the per-layer kernels.
    Layers without slots or cache are skipped. Callers check
    :func:`use_ctx_kv_insert`."""
    none = kv.new_empty(0, dtype=torch.int64)
    live = [c is not None and s is not None for s, c in zip(slots, caches)]
    first = next(i for i, ok in enumerate(live) if ok)
    _module().ctx_kv_insert(
        kv,
        [w if ok else knorm_weights[first] for w, ok in zip(knorm_weights, live)],
        float(eps),
        positions,
        cos_sin,
        [s if ok else none for s, ok in zip(slots, live)],
        [c[0] if ok else caches[first][0] for c, ok in zip(caches, live)],
        [c[1] if ok else caches[first][1] for c, ok in zip(caches, live)],
    )


def warmup() -> None:
    """Build / load the extension before serving (first use compiles it)."""
    _module()


class KimiRowsRMSNorm(RMSNorm):
    """``RMSNorm`` whose prefill-sized inputs run :func:`rms_norm` (bitwise the
    ``vllm_c`` kernel). Only where the IR dispatch would pick ``vllm_c`` and
    the shape policy holds; every other call is the stock module."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__(hidden_size, eps=eps)
        warmup()  # build / load the JIT extension at model init, not mid-step

    def _rows_kernel(self, x: torch.Tensor) -> bool:
        if not (
            self.has_weight
            and self.variance_size_override is None
            and use_rms_norm(x, self.weight.data)
        ):
            return False
        from vllm import ir

        return (
            ir.ops.rms_norm.dispatch(
                x, self.weight.data, self.variance_epsilon, None
            ).provider
            == "vllm_c"
        )

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if residual is None and self._rows_kernel(x):
            return rms_norm(x, self.weight.data, self.variance_epsilon)
        return super().forward(x, residual)

    def forward_pair(
        self, x: torch.Tensor, other: "KimiRowsRMSNorm", y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """``(self(x), other(y))`` in one launch where both run the rows
        kernel with the instantiated pair of stock block shapes (the DSA q_a /
        kv_a norms): bitwise the two separate calls. None (nothing launched)
        otherwise; the caller then runs the two norms itself."""
        if not (
            isinstance(other, KimiRowsRMSNorm)
            and self.variance_epsilon == other.variance_epsilon
            and self._rows_kernel(x)
            and other._rows_kernel(y)
        ):
            return None
        return rms_norm_pair(
            x, self.weight.data, y, other.weight.data, self.variance_epsilon
        )
