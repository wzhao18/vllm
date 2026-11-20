# SPDX-License-Identifier: Apache-2.0
"""Probe: prefill router top-k variants vs prefill_glue.router_topk (bitwise + time).

V2: fast-sigmoid prefilter (ex2/rcp approx, error-bounded), candidates compacted
per warp, the exact stock score (tanhf form) only for the compacted candidates,
then the same exact key sort. TPW tokens per warp interleaved.
"""

import argparse
import json

import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <stdint.h>
#include <math.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;
namespace {
constexpr uint32_t kFull = 0xffffffffu;

__device__ __forceinline__ float sigmoid_accurate(float x) {
  return 0.5f * tanhf(0.5f * x) + 0.5f;
}
__device__ __forceinline__ void preprocess_score(float input, float bias,
                                                 float& unbiased_score,
                                                 float& selection_score) {
  unbiased_score = 0.0F;
  selection_score = -INFINITY;
  if (!isfinite(input) || !isfinite(bias)) return;
  float const unbiased = sigmoid_accurate(input);
  float const biased = unbiased + bias;
  unbiased_score = unbiased;
  selection_score = biased == 0.0F ? 0.0F : biased;
}
__device__ __forceinline__ float ex2a(float x) {
  float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y;
}
__device__ __forceinline__ float rcpa(float x) {
  float y; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y;
}
// Fast selection score; |fast - exact| <= kEps + |v| * kRel after the bias add
// (sigmoid approx error << 1e-6; the two bias adds round independently).
__device__ __forceinline__ float fast_sel(float x, float b) {
  if (!isfinite(x) || !isfinite(b)) return -INFINITY;
  return rcpa(1.0f + ex2a(x * -1.4426950408889634f)) + b;
}
__device__ __forceinline__ uint64_t make_key(float v, int32_t idx) {
  uint32_t bits = __float_as_uint(v);
  bits ^= (bits & 0x80000000u) ? 0xffffffffu : 0x80000000u;
  return (uint64_t(bits) << 32) | uint64_t(0xFFFF & (65535 - idx));
}
__device__ __forceinline__ void unpack_key(uint64_t key, float& v, int32_t& idx) {
  idx = 65535 - int32_t(key & 0xFFFF);
  uint32_t bits = uint32_t(key >> 32);
  bits ^= (bits & 0x80000000u) ? 0x80000000u : 0xffffffffu;
  v = __uint_as_float(bits);
}
__device__ __forceinline__ uint64_t bitonic_sort_desc_u64(uint64_t key, int32_t lane) {
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
  return key;
}

template <int VPL, int WARPS>
__global__ void __launch_bounds__(WARPS * 32)
    topk_v2(const float* __restrict__ scores, int64_t ld,
            const float* __restrict__ bias, float* topk_values,
            int32_t* topk_indices, int32_t num_tokens, int32_t num_experts,
            int32_t topk, bool renormalize, float routed_scaling_factor) {
  __shared__ float s_bias[VPL * 32];
  __shared__ uint64_t cand_keys[WARPS][32];
  __shared__ int16_t cand_ids[WARPS][32];
  int32_t const lane = threadIdx.x & 31;
  int32_t const wid = threadIdx.x >> 5;
  for (int e = threadIdx.x; e < VPL * 32; e += WARPS * 32)
    s_bias[e] = e < num_experts ? bias[e] : 0.0F;
  __syncthreads();
  for (int32_t token = blockIdx.x * WARPS + wid; token < num_tokens;
       token += gridDim.x * WARPS) {
    const float* row = scores + int64_t{token} * ld;
    float f[VPL];
    float lane_max = -INFINITY;
#pragma unroll
    for (int j = 0; j < VPL; ++j) {
      int32_t const e = j * 32 + lane;
      f[j] = e < num_experts ? fast_sel(__ldg(row + e), s_bias[e]) : -INFINITY;
      lane_max = fmaxf(lane_max, f[j]);
    }
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
    float const tau_f = __shfl_sync(kFull, m, topk - 1);
    // Every exact top-k score has fast score >= tau_f - 2 eps.
    float const thr = tau_f - (4.0e-6f + 1.0e-6f * fabsf(tau_f));
    int32_t count = 0;
#pragma unroll
    for (int j = 0; j < VPL; ++j) count += f[j] >= thr ? 1 : 0;
    int32_t incl = count;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      int32_t const o = __shfl_up_sync(kFull, incl, off);
      if (lane >= off) incl += o;
    }
    int32_t const num_cand = __shfl_sync(kFull, incl, 31);
    uint64_t lane_packed = 0;
    if (num_cand <= 32) {
      int32_t pos = incl - count;
#pragma unroll
      for (int j = 0; j < VPL; ++j)
        if (f[j] >= thr) cand_ids[wid][pos++] = (int16_t)(j * 32 + lane);
      __syncwarp();
      uint64_t key = 0;
      if (lane < num_cand) {
        int32_t const e = cand_ids[wid][lane];
        float unb, s;
        preprocess_score(__ldg(row + e), s_bias[e], unb, s);
        key = make_key(s, e);
      }
      __syncwarp();
      lane_packed = bitonic_sort_desc_u64(key, lane);
    } else {
      // Rare (ties / -inf rows): exact scores of the candidates in place,
      // sorted in chunks of 32 and merged into the running top 32.
      float sel[VPL];
#pragma unroll
      for (int j = 0; j < VPL; ++j) {
        int32_t const e = j * 32 + lane;
        float unb, s = -INFINITY;
        if (f[j] >= thr && e < num_experts) preprocess_score(__ldg(row + e), s_bias[e], unb, s);
        sel[j] = s;
      }
      uint64_t best = 0;
      for (int32_t base = 0; base < num_cand; base += 32) {
        int32_t pos = incl - count;
#pragma unroll
        for (int j = 0; j < VPL; ++j) {
          if (f[j] >= thr) {
            if (pos >= base && pos < base + 32)
              cand_keys[wid][pos - base] = make_key(sel[j], j * 32 + lane);
            ++pos;
          }
        }
        __syncwarp();
        uint64_t key = lane < num_cand - base ? cand_keys[wid][lane] : 0;
        __syncwarp();
        key = bitonic_sort_desc_u64(key, lane);
        uint32_t const rhi = __shfl_sync(kFull, uint32_t(key >> 32), 31 - lane);
        uint32_t const rlo = __shfl_sync(kFull, uint32_t(key), 31 - lane);
        uint64_t const rev = (uint64_t(rhi) << 32) | rlo;
        best = best > rev ? best : rev;
#pragma unroll
        for (int j = 16; j > 0; j >>= 1) {
          uint32_t const ohi = __shfl_xor_sync(kFull, uint32_t(best >> 32), j);
          uint32_t const olo = __shfl_xor_sync(kFull, uint32_t(best), j);
          uint64_t const o = (uint64_t(ohi) << 32) | olo;
          bool const keep_max = (lane & j) == 0;
          best = keep_max ? (best > o ? best : o) : (best < o ? best : o);
        }
      }
      lane_packed = best;
    }
    float lane_score = -INFINITY;
    int32_t lane_expert = -1;
    if (lane < topk) unpack_key(lane_packed, lane_score, lane_expert);
    float lane_unbiased = 0.0F;
    if (lane < topk && lane_expert >= 0 && lane_expert < num_experts) {
      float s;
      preprocess_score(__ldg(row + lane_expert), s_bias[lane_expert], lane_unbiased, s);
    }
    // write_outputs (verbatim semantics)
    bool const finite_selection = lane < topk && lane_score != -INFINITY;
    lane_unbiased = finite_selection ? lane_unbiased : 0.0F;
    unsigned const finite_mask = __ballot_sync(kFull, finite_selection);
    float sum = lane_unbiased;
    {  // the production write_outputs sum: cg::reduce(warp, x, plus)
      auto w = cg::tiled_partition<32>(cg::this_thread_block());
      sum = cg::reduce(w, sum, cg::plus<float>{});
    }
    if (lane < topk) {
      float output = 0.0F;
      if (finite_mask == 0) {
        if (renormalize) output = 1.0F / static_cast<float>(topk);
      } else if (finite_selection) {
        float scale = routed_scaling_factor;
        if (renormalize) scale /= sum + 1e-20F;
        output = lane_unbiased * scale;
      }
      int64_t const oi = int64_t{token} * topk + lane;
      topk_values[oi] = output;
      topk_indices[oi] = lane_expert;
    }
  }
}
}  // namespace

void topk_v2_launch(torch::Tensor& vals, torch::Tensor& ids, const torch::Tensor& scores,
                    const torch::Tensor& bias, bool renorm, double scale, int64_t ctas_per_sm) {
  const int64_t n = scores.size(0), e = scores.size(1), k = vals.size(1);
  TORCH_CHECK(e <= 896 && e > 864);
  constexpr int W = 8;
  int dev = 0, sms = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  int occ = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, topk_v2<28, W>, W * 32, 0);
  if (ctas_per_sm > 0) occ = std::min<int>(occ, (int)ctas_per_sm);
  const dim3 grid((unsigned)std::min<int64_t>((n + W - 1) / W, (int64_t)sms * std::max(occ, 1)));
  topk_v2<28, W><<<grid, W * 32, 0, c10::cuda::getCurrentCUDAStream()>>>(
      scores.data_ptr<float>(), scores.stride(0), bias.data_ptr<float>(),
      vals.data_ptr<float>(), ids.data_ptr<int32_t>(), (int32_t)n, (int32_t)e,
      (int32_t)k, renorm, (float)scale);
}
"""

CPP = "void topk_v2_launch(torch::Tensor& vals, torch::Tensor& ids, const torch::Tensor& scores, const torch::Tensor& bias, bool renorm, double scale, int64_t ctas_per_sm);"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=16384)
    ap.add_argument("--clean", action="store_true", help="no NaN / -inf rows")
    ap.add_argument("--only", default="", help="comma list of distributions")
    a = ap.parse_args()
    mod = load_inline("probe_router_topk_v2", CPP, SRC, functions=["topk_v2_launch"],
                      extra_cuda_cflags=["-O3"], verbose=False)
    from vllm.models.kimi_k3.nvidia.ops import prefill_glue

    dev = "cuda"
    T, E = a.rows, 896
    res = {}

    def bench(fn, iters=50):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(iters):
                fn()
        g.replay()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(5):
            s.record()
            g.replay()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e) * 1e3 / iters)
        return sorted(ts)[2]

    torch.manual_seed(0)
    for name, std, bstd in [("n1", 1.0, 0.0), ("n3b", 3.0, 0.02), ("n12sat", 12.0, 0.0),
                            ("emu", 3.0, None), ("emu4", 3.0, "emu4")]:
        sc = torch.randn(T, E, device=dev) * std
        if bstd == "emu4":
            bias = torch.full((E,), -100.0, device=dev)
            bias[torch.arange(E, device=dev) % 56 < 4] = 0.0
        elif bstd is None:
            bias = torch.full((E,), -100.0, device=dev)
            bias[torch.arange(E, device=dev) % 224 < 56] = 0.0
        else:
            bias = torch.randn(E, device=dev) * bstd
        if a.only and name not in a.only.split(","):
            continue
        if not a.clean:
            sc[0, 5] = float("nan")
            sc[1, :] = float("-inf")
        v0, i0 = prefill_glue.router_topk(sc, bias, 16, True, 2.5)
        v2 = torch.empty_like(v0)
        i2 = torch.empty_like(i0)
        mod.topk_v2_launch(v2, i2, sc, bias, True, 2.5, 0)
        eq = torch.equal(v0, v2) and torch.equal(i0, i2)
        t0 = bench(lambda: prefill_glue.router_topk(sc, bias, 16, True, 2.5))
        t2 = bench(lambda: mod.topk_v2_launch(v2, i2, sc, bias, True, 2.5, 0))
        res[name] = (round(t0, 2), round(t2, 2), eq)
        print(f"topk {name:7s} prod {t0:7.2f} us  v2 {t2:7.2f} us  bitwise {eq}")
    print("RESULT", json.dumps(res))


if __name__ == "__main__":
    main()
