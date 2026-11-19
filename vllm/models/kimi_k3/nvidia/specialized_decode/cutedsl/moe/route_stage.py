# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Router top-k + MegaMoE staging in one launch (SP decode MoE front).

Input: the router | latent down GEMM's FP32 rows ``y [n, E + latent]``
(``runtime.sp_mlp``: router logits in columns ``0:E``, the latent in ``E:``).
One block per row:

* warp 0: stock's sigmoid + correction-bias single-group top-k on the logits
  (``_moe_C.grouped_topk`` / ``single_group_topk``, bitwise: the warp-per-token
  form of ``ops.prefill_glue.router_topk``, same packed ``(score, 65535 - id)``
  keys and the same ``write_outputs`` weights), written straight into the
  session's route buffers with the stager's rules (``megamoe_rubin.staging``:
  padding rows and out-of-range ids masked to -1 with weight 0, the sticky
  ``route_error`` flag, SF pad columns zeroed);
* warps 1-7: the latent row rounded to BF16 (round to nearest, as a BF16-output
  GEMM rounds its FP32 accumulator), then the stager's activation quantization
  (NVFP4 block16 or MXFP8 block32), bitwise ``_stage_kernel``.

Trailing blocks write ``topk_idx = -1`` for the dead rows ``[n, capacity)``.
The launch is programmatic-dependent (PDL) on the GEMM before it and waits for
it in-kernel (``griddepcontrol.wait``); it never triggers its dependents early.

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

struct Args {
  const float* y;
  int64_t ldy;
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

// Warp 0 of a live row: the top-k (prefill_glue.router_topk, one token) and
// the stager's route rules.
template <int VPL>
__device__ __forceinline__ void route_row(const Args& a, int32_t row,
                                          const float* s_bias,
                                          uint64_t* cand_keys) {
  auto warp = cg::tiled_partition<32>(cg::this_thread_block());
  int32_t const lane = threadIdx.x & 31;
  int32_t const num_experts = a.num_logits;
  int32_t const topk = a.topk;
  const float* logits = a.y + int64_t{row} * a.ldy;
  float in[VPL];
#pragma unroll
  for (int j = 0; j < VPL; ++j) {
    int32_t const e = j * 32 + lane;
    in[j] = e < num_experts ? logits[e] : 0.0F;
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
    int32_t pos = incl - count;
#pragma unroll
    for (int j = 0; j < VPL; ++j) {
      if (sel[j] >= tau) cand_keys[pos++] = make_key(sel[j], j * 32 + lane);
    }
    __syncwarp();
    uint64_t key = lane < num_cand ? cand_keys[lane] : 0;
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
    uint64_t h1, h2;
    lane_top2_regs<VPL>(sel, lane, ~uint64_t{0}, h1, h2);
    bool h2_valid = true;
    for (int kk = 0; kk < topk; ++kk) {
      uint64_t const best = warp_max_u64(h1);
      if (lane == kk) lane_packed = best;
      if (h1 == best) {
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
    preprocess_score(logits[lane_expert], s_bias[lane_expert], lane_unbiased, s);
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
                                            int32_t blk) {
  const float4* src = reinterpret_cast<const float4*>(
      a.y + int64_t{row} * a.ldy + a.num_logits + int64_t{blk} * 16);
  float v[16];
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    float4 t = src[q];
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
                                            int32_t blk) {
  const float4* src = reinterpret_cast<const float4*>(
      a.y + int64_t{row} * a.ldy + a.num_logits + int64_t{blk} * 32);
  float v[32];
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    float4 t = src[q];
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

template <int VPL, bool NVFP4>
__global__ void __launch_bounds__(kWarps * 32) route_stage_kernel(Args a) {
  __shared__ float s_bias[VPL * 32];
  __shared__ uint64_t cand_keys[32];
  int32_t const row = blockIdx.x;
  int32_t const wid = threadIdx.x >> 5;
  if (wid == 0 && row < a.n) {
    for (int e = threadIdx.x; e < VPL * 32; e += 32) {
      s_bias[e] = e < a.num_logits ? a.bias[e] : 0.0F;
    }
    __syncwarp();
  }
  // The GEMM that wrote y (PDL predecessor) and every kernel before it are
  // complete past this point.
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (row >= a.n) {  // dead rows [n, capacity): routes -1 only
    int64_t const base = a.n + int64_t{row - a.n} * kDeadRows;
    int64_t const end = min(base + kDeadRows, int64_t{a.capacity});
    for (int64_t i = base * a.topk + threadIdx.x; i < end * a.topk;
         i += blockDim.x) {
      a.idx[i] = -1;
    }
    return;
  }
  if (wid == 0) {
    route_row<VPL>(a, row, s_bias, cand_keys);
    return;
  }
  int32_t const vec = NVFP4 ? 16 : 32;
  for (int32_t blk = threadIdx.x - 32; blk < a.hidden / vec;
       blk += (kWarps - 1) * 32) {
    if (NVFP4) {
      quant_nvfp4(a, row, blk);
    } else {
      quant_mxfp8(a, row, blk);
    }
  }
}

}  // namespace

void route_stage(const torch::Tensor& y, int64_t num_logits,
                 const torch::Tensor& bias, torch::Tensor& x, torch::Tensor& sf,
                 torch::Tensor& idx, torch::Tensor& sw, torch::Tensor& err,
                 const c10::optional<torch::Tensor>& pad, int64_t num_experts,
                 bool renormalize, double routed_scaling_factor, bool nvfp4,
                 double norm_const) {
  TORCH_CHECK(y.scalar_type() == at::kFloat && y.dim() == 2 &&
                  y.stride(1) == 1 && y.stride(0) % 4 == 0 &&
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
  const int64_t n = y.size(0);
  const int64_t hidden = y.size(1) - num_logits;
  const int64_t capacity = idx.size(0), topk = idx.size(1);
  const int64_t vec = nvfp4 ? 16 : 32;
  TORCH_CHECK(num_logits % 4 == 0 && num_logits <= 1024 && topk <= 32 &&
                  topk <= num_logits && hidden > 0 && hidden % vec == 0 &&
                  n <= capacity && sw.size(0) == capacity &&
                  sw.size(1) == topk && x.size(0) >= capacity &&
                  x.size(1) * (nvfp4 ? 2 : 1) >= hidden &&
                  sf.size(0) >= capacity && sf.size(1) >= hidden / vec &&
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
  a.y = y.data_ptr<float>();
  a.ldy = y.stride(0);
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
  const int vpl = (int)((num_logits + 31) / 32);
#define K3_RS_LAUNCH(V)                                                     \
  (nvfp4 ? cudaLaunchKernelEx(&config, route_stage_kernel<V, true>, a)      \
         : cudaLaunchKernelEx(&config, route_stage_kernel<V, false>, a))
  cudaError_t rc;
  if (vpl <= 8) {
    rc = K3_RS_LAUNCH(8);
  } else if (vpl <= 16) {
    rc = K3_RS_LAUNCH(16);
  } else if (vpl <= 28) {
    rc = K3_RS_LAUNCH(28);
  } else {
    rc = K3_RS_LAUNCH(32);
  }
#undef K3_RS_LAUNCH
  TORCH_CHECK(rc == cudaSuccess, "route_stage launch: ", cudaGetErrorString(rc));
}
"""

_CPP_SRC = """
void route_stage(const torch::Tensor& y, int64_t num_logits,
                 const torch::Tensor& bias, torch::Tensor& x, torch::Tensor& sf,
                 torch::Tensor& idx, torch::Tensor& sw, torch::Tensor& err,
                 const c10::optional<torch::Tensor>& pad, int64_t num_experts,
                 bool renormalize, double routed_scaling_factor, bool nvfp4,
                 double norm_const);
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
        """``n`` is planned and the layer stages plain routes: no EPLB
        logical-to-physical map and no routing capture (both act on the top-k
        ids between router and stager; those layers keep the stock path)."""
        experts = self.mlp.experts
        return (
            n in self.rows
            and experts.eplb_state.logical_to_physical_map is None
            and experts.capture_fn is None
        )

    def run(self, y: torch.Tensor, output_rmsnorm=None) -> torch.Tensor:
        """Stage ``y``'s routes and latent, launch MegaMoE; returns its BF16
        ``[n, latent]`` output (``output_rmsnorm`` applied in its reduce), as
        ``experts(latent, topk_weights, topk_ids, output_rmsnorm=...)``."""
        import vllm.envs as envs
        from vllm.forward_context import get_forward_context, is_forward_context_available

        from ....megamoe_rubin.experts import _rmsnorm_args

        experts = self.mlp.experts
        n = y.shape[0]
        weights = experts.kernel_weights
        if weights is None:
            raise RuntimeError(f"{experts.prefix}: Rubin MegaMoE not prepared.")
        bucket = experts.select_bucket(n)
        rt = experts.get_runtime()
        session = rt.sessions[bucket]
        p = rt.problem
        if y.shape[1] - self.num_logits != p.hidden or session.topk_idx.shape[1] != experts.top_k:
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
        )
        out = torch.empty((n, p.hidden), dtype=torch.bfloat16, device=y.device)
        rmsnorm = None if output_rmsnorm is None else _rmsnorm_args(output_rmsnorm)
        session.launch(weights, out, experts.ep_tokens(n), rmsnorm)
        owner = experts.prefix
        if rt._poll_owner in (None, owner):
            rt._poll_owner = owner
            rt.poll_route_errors()
        return out
