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

// Top-2 of this lane's keys (expert j * 32 + lane, score s_sel[j * 32 + lane])
// strictly below `limit` (0 = none).
template <int VPL>
__device__ __forceinline__ void lane_top2(const float* s_sel, int32_t lane,
                                          uint64_t limit, uint64_t& h1,
                                          uint64_t& h2) {
  h1 = 0;
  h2 = 0;
#pragma unroll 4
  for (int j = 0; j < VPL; ++j) {
    uint64_t k = make_key(s_sel[j * 32 + lane], j * 32 + lane);
    k = k < limit ? k : 0;
    if (k > h1) {
      h2 = h1;
      h1 = k;
    } else if (k > h2) {
      h2 = k;
    }
  }
}

// One token per 128-thread block: the four warps score a quarter of the
// experts each (the precise sigmoid is a long dependent chain per element),
// then warp 0 selects.
template <int VPL>
__global__ void __launch_bounds__(128)
    router_topk_kernel(const float* __restrict__ scores, int64_t ld,
                       const float* __restrict__ bias, float* topk_values,
                       int32_t* topk_indices, int32_t num_tokens,
                       int32_t num_experts, int32_t topk, bool renormalize,
                       float routed_scaling_factor) {
  static_assert(VPL % 4 == 0);
  __shared__ float s_sel[VPL * 32];
  __shared__ float s_unb[VPL * 32];
  __shared__ uint64_t cand_keys[32];
  auto warp = cg::tiled_partition<32>(cg::this_thread_block());
  int32_t const lane = threadIdx.x & 31;
  int32_t const wid = threadIdx.x >> 5;
  int32_t const token = blockIdx.x;
  const float* row = scores + int64_t{token} * ld;
#pragma unroll
  for (int jj = 0; jj < VPL / 4; ++jj) {
    int32_t const e = (wid * (VPL / 4) + jj) * 32 + lane;
    float unb = 0.0F, sel = -INFINITY;
    if (e < num_experts) {
      preprocess_score(row[e], bias[e], unb, sel);
    }
    s_sel[e] = sel;
    s_unb[e] = unb;
  }
  __syncthreads();
  if (wid != 0) return;

  // Warp 0 reads the scores back from shared memory as it needs them (no
  // per-lane score array: 16 resident blocks per SM).
  float lane_max = -INFINITY;
#pragma unroll
  for (int j = 0; j < VPL; ++j) lane_max = fmaxf(lane_max, s_sel[j * 32 + lane]);

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
  for (int j = 0; j < VPL; ++j) count += s_sel[j * 32 + lane] >= tau ? 1 : 0;
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
      float const sv = s_sel[j * 32 + lane];
      if (sv >= tau) cand_keys[pos++] = make_key(sv, j * 32 + lane);
    }
    __syncwarp();
    uint64_t key = lane < num_cand ? cand_keys[lane] : 0;
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
    lane_top2<VPL>(s_sel, lane, ~uint64_t{0}, h1, h2);
    bool h2_valid = true;
    for (int kk = 0; kk < topk; ++kk) {
      uint64_t const best = warp_max_u64(h1);
      if (lane == kk) lane_packed = best;
      if (h1 == best) {  // the one lane holding it
        if (h2_valid) {
          h1 = h2;
          h2_valid = false;
        } else {
          lane_top2<VPL>(s_sel, lane, best, h1, h2);
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
  float const lane_unbiased =
      lane < topk && lane_expert >= 0 && lane_expert < num_experts
          ? s_unb[lane_expert]
          : 0.0F;
  write_outputs(warp, lane_score, lane_unbiased, lane_expert, lane, token,
                topk, topk_values, topk_indices, renormalize,
                routed_scaling_factor);
}

// ---------------------------------------------------------------------------
// rms_norm: csrc/layernorm_kernels.cu rms_norm_kernel<bf16, 8, 2, true>
// launched with block_size = NW * 32 threads, replayed by one warp per row.
// ---------------------------------------------------------------------------

template <int NW, int VPT>
__global__ void __launch_bounds__(128)
    rms_norm_rows_kernel(uint16_t* __restrict__ out,
                         const uint16_t* __restrict__ input,
                         int64_t in_stride, const uint16_t* __restrict__ weight,
                         float epsilon, int32_t num_tokens,
                         int32_t hidden_size) {
  constexpr int B = NW * 32;  // stock block size
  int32_t const lane = threadIdx.x & 31;
  int32_t const row = blockIdx.x * 4 + (threadIdx.x >> 5);
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

// ---------------------------------------------------------------------------
// gate_sigmoid_mul: ATen bf16 sigmoid (1 / (1 + exp(-x)) in fp32) then ATen
// bf16 mul, 8 columns per thread.
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint4 gate_sigmoid_mul_vec8(uint4 a, uint4 g) {
  const uint16_t* ah = reinterpret_cast<const uint16_t*>(&a);
  const uint16_t* gh = reinterpret_cast<const uint16_t*>(&g);
  uint4 o;
  uint16_t* oh = reinterpret_cast<uint16_t*>(&o);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    float const gv = bf16_bits_to_float(gh[j]);
    float const sig = 1.0f / (1.0f + ::expf(-gv));
    float const sb = (float)c10::BFloat16(sig);
    oh[j] = c10::BFloat16(bf16_bits_to_float(ah[j]) * sb).x;
  }
  return o;
}

// CONTIG: all three tensors are dense (rows x d), indexed as one vector.
template <bool CONTIG>
__global__ void gate_sigmoid_mul_kernel(uint16_t* __restrict__ out,
                                        const uint16_t* __restrict__ attn,
                                        const uint16_t* __restrict__ gate,
                                        int32_t total_vecs, int32_t vecs_per_row,
                                        int64_t attn_stride,
                                        int64_t gate_stride,
                                        int64_t out_stride) {
  constexpr int U = 4;  // loads in flight per thread
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
        *reinterpret_cast<uint4*>(out + oo[u]) = gate_sigmoid_mul_vec8(a[u], g[u]);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// scatter_rows: dst[index[i]] = src[i] for whole rows (the KDA final-state
// write-back), 16 bytes per thread; negative indices wrap as in index_put_.
// ---------------------------------------------------------------------------

template <typename IdxT>
__global__ void scatter_rows_kernel(uint4* __restrict__ dst,
                                    const uint4* __restrict__ src,
                                    const IdxT* __restrict__ index,
                                    int64_t dst_slots, int64_t row_vecs,
                                    int64_t dst_stride_vecs,
                                    int64_t src_stride_vecs) {
  int64_t const i = blockIdx.y;
  int64_t slot = (int64_t)index[i];
  if (slot < 0) slot += dst_slots;
  const uint4* s = src + i * src_stride_vecs;
  uint4* d = dst + slot * dst_stride_vecs;
  for (int64_t v = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; v < row_vecs;
       v += (int64_t)gridDim.x * blockDim.x) {
    d[v] = s[v];
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
  const dim3 grid((unsigned)n), block(128);
  auto stream = c10::cuda::getCurrentCUDAStream();
  const int vpl = (int)((e + 31) / 32);
#define K3_TOPK_LAUNCH(V)                                                   \
  router_topk_kernel<V><<<grid, block, 0, stream>>>(                        \
      scores.data_ptr<float>(), scores.stride(0), bias.data_ptr<float>(),   \
      topk_values.data_ptr<float>(), topk_indices.data_ptr<int32_t>(),      \
      (int32_t)n, (int32_t)e, (int32_t)k, renormalize,                      \
      (float)routed_scaling_factor)
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
  TORCH_CHECK(input.dim() == 2 && input.stride(1) == 1 && out.is_contiguous() &&
              weight.is_contiguous() && weight.dim() == 1);
  const int64_t n = input.size(0);
  const int64_t h = input.size(1);
  TORCH_CHECK(h % 8 == 0 && weight.numel() == h && out.size(1) == h);
  TORCH_CHECK(input.stride(0) % 8 == 0 &&
              reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0);
  TORCH_CHECK(block_size % 32 == 0 && block_size >= 32 && block_size <= 256);
  if (n == 0) return;
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
  if (nw == 2 && vpt == 1) {
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
  const int threads = 256;
  const int64_t total = rows * (d / 8);
  TORCH_CHECK(total < (int64_t{1} << 31));
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                         attn.get_device());
  const int64_t max_blocks = (int64_t)sms * 8;
  const int blocks =
      (int)std::min<int64_t>((total + threads - 1) / threads, max_blocks);
  const bool contig = attn.stride(0) == d && gate.stride(0) == d &&
                      out.stride(0) == d;
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
  if (index.scalar_type() == at::kInt) {
    scatter_rows_kernel<int32_t><<<grid, threads, 0, stream>>>(
        reinterpret_cast<uint4*>(dst.data_ptr()),
        reinterpret_cast<const uint4*>(src.data_ptr()), index.data_ptr<int32_t>(),
        dst.size(0), row_vecs, ds, ss);
  } else {
    scatter_rows_kernel<int64_t><<<grid, threads, 0, stream>>>(
        reinterpret_cast<uint4*>(dst.data_ptr()),
        reinterpret_cast<const uint4*>(src.data_ptr()), index.data_ptr<int64_t>(),
        dst.size(0), row_vecs, ds, ss);
  }
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
void gate_sigmoid_mul(torch::Tensor& out, const torch::Tensor& attn,
                      const torch::Tensor& gate);
void scatter_rows(torch::Tensor& dst, const torch::Tensor& index,
                  const torch::Tensor& src);
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
                    "gate_sigmoid_mul",
                    "scatter_rows",
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
    return (
        block % 32 == 0
        and vpt <= RMS_NORM_MAX_VECS_PER_LANE
        and (block // 32, vpt) in ((2, 1), (4, 1), (6, 1), (8, 1), (8, 2))
    )


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``_C.rms_norm`` of a 2-D BF16 ``x`` (row stride allowed), bitwise equal;
    the caller checks :func:`use_rms_norm`."""
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    block = stock_rms_norm_block_size(x.shape[0], x.shape[1])
    _module().rms_norm_rows(out, x, weight, float(eps), block)
    return out


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

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if (
            residual is None
            and self.has_weight
            and self.variance_size_override is None
            and use_rms_norm(x, self.weight.data)
        ):
            from vllm import ir

            if (
                ir.ops.rms_norm.dispatch(
                    x, self.weight.data, self.variance_epsilon, None
                ).provider
                == "vllm_c"
            ):
                return rms_norm(x, self.weight.data, self.variance_epsilon)
        return super().forward(x, residual)
