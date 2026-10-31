# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 activation producers for Kimi K3's online-MXFP8 projections.

An online-MXFP8 linear quantizes its BF16 input with a standalone FlashInfer
kernel on every call (``mxfp8_quantize``, F8_128x4-swizzled E8M0 scales). In a
K3.1 decode step that is ~680 extra launches (~2.2 ms at r16 on VR). The
kernels here write the MXFP8 activation directly from the kernel that produces
the BF16 one, as a :class:`QuantizedActivation` the MXFP8 linear kernels
consume without re-quantizing:

* ``quantize``: plain BF16 -> MXFP8, for an input shared by several MXFP8
  linears (quantized once instead of once per consumer).
* ``situ_and_mul_quantize`` / ``silu_and_mul_quantize``: the shared-expert /
  dense MLP activation fused with the down projection's input quantization.
* ``rms_norm_quantize``: RMSNorm (the latent-MoE ``routed_expert_norm``)
  fused with the up projection's input quantization.
* ``gate_sigmoid_mul_quantize``: the DSA output gate (``attn * sigmoid(g)``)
  fused with ``o_proj``'s input quantization.

Numerics: each producer rounds its result to BF16 exactly as the unfused BF16
kernel does, then quantizes that BF16 value the way FlashInfer's CuTe DSL
``mxfp8_quantize`` does (block amax * fp32(1/448) -> E8M0 rounded up, value *
2^(127 - e) -> E4M3 RN satfinite), so a fused producer equals "BF16 producer,
then mxfp8_quantize" whenever the BF16 producer values agree (bitwise for the
activations; the RMSNorm reduction order is not FlashInfer's or vLLM's, so it
matches to the reduction's rounding).

Every producer can write into a column window of a wider MXFP8 activation
(``out=``, ``col=``): two producers then fill one K-concatenated input, e.g.
``[situ(shared gate_up) | norm(routed latent)]`` for one GEMM over the
K-stacked ``[shared down | latent up]`` weight. MXFP8 blocks are 32 wide along
K, so K-stacking two MXFP8 operands at a 128-aligned column is exact.

Built on first use per source hash (torch JIT extension, cached under
``VLLM_CACHE_ROOT/k3_mxfp8_act``).
"""

import hashlib
import os
import threading

import torch

from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic

MXFP8_BLOCK = 32

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/BFloat16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

namespace {

// ---- FlashInfer CuTe DSL MXFP8 quantization primitives (bit-exact copies of
// quantization_cute_dsl_utils.float_to_ue8m0_fast / ue8m0_to_inv_scale_fast).
__device__ __forceinline__ uint32_t float_to_ue8m0(float v) {
  uint32_t r;
  asm("{\n\t"
      ".reg .pred p_zero, p_has_mant, p_exp_zero, p_tiny_sub, p_ovf;\n\t"
      ".reg .u32 bits, exp_biased, mantissa, bump, result;\n\t"
      "setp.le.f32 p_zero, %1, 0f00000000;\n\t"
      "mov.b32 bits, %1;\n\t"
      "shr.b32 exp_biased, bits, 23;\n\t"
      "and.b32 exp_biased, exp_biased, 255;\n\t"
      "and.b32 mantissa, bits, 0x7FFFFF;\n\t"
      "setp.ne.u32 p_has_mant, mantissa, 0;\n\t"
      "selp.u32 bump, 1, 0, p_has_mant;\n\t"
      "setp.eq.u32 p_exp_zero, exp_biased, 0;\n\t"
      "setp.le.u32 p_tiny_sub, mantissa, 0x400000;\n\t"
      "and.pred p_tiny_sub, p_exp_zero, p_tiny_sub;\n\t"
      "@p_tiny_sub mov.u32 bump, 0;\n\t"
      "add.u32 result, exp_biased, bump;\n\t"
      "setp.gt.u32 p_ovf, result, 254;\n\t"
      "selp.u32 result, 254, result, p_ovf;\n\t"
      "selp.u32 %0, 0, result, p_zero;\n\t"
      "}"
      : "=r"(r)
      : "f"(v));
  return r;
}

__device__ __forceinline__ float ue8m0_to_inv_scale(uint32_t e) {
  if (e == 0) return 0.f;
  int ne = 254 - (int)e;
  ne = ne < 0 ? 0 : ne;
  return __uint_as_float(((uint32_t)ne) << 23);
}

__device__ __forceinline__ uint16_t fp8x2(float a, float b) {
  return (uint16_t)__nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE,
                                            __NV_E4M3);
}

__device__ __forceinline__ float bf16_bits_to_float(uint16_t b) {
  return __uint_as_float(((uint32_t)b) << 16);
}

__device__ __forceinline__ uint16_t float_to_bf16_bits(float f) {
  return c10::BFloat16(f).x;
}

// Byte offset of scale block `col` of `row` in the F8_128x4 swizzled layout
// with `padded_cols` (multiple of 4) scale columns.
__device__ __forceinline__ int64_t swizzled_sf_offset(int64_t row, int col,
                                                      int padded_cols) {
  return (int64_t)(col & 3) + (int64_t)(col >> 2) * 512 + (row & 31) * 16 +
         ((row & 127) >> 5) * 4 + (row >> 7) * 128 * (int64_t)padded_cols;
}

// Quantize the 8 BF16-rounded values `v` held by this thread; 4 adjacent lanes
// hold one 32-wide MXFP8 block (lane & 3 == 0 owns the scale).
__device__ __forceinline__ void quant8_store(const float (&v)[8],
                                             uint8_t* __restrict__ q,
                                             uint8_t* __restrict__ sf,
                                             int64_t row, int col8,
                                             int sf_col_base, int padded_cols,
                                             bool write) {
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(v[i]));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
  const uint32_t e = float_to_ue8m0(amax * (1.0f / 448.0f));
  const float inv = ue8m0_to_inv_scale(e);
  if (!write) return;
  uint2 packed;
  packed.x = (uint32_t)fp8x2(v[0] * inv, v[1] * inv) |
             ((uint32_t)fp8x2(v[2] * inv, v[3] * inv) << 16);
  packed.y = (uint32_t)fp8x2(v[4] * inv, v[5] * inv) |
             ((uint32_t)fp8x2(v[6] * inv, v[7] * inv) << 16);
  *reinterpret_cast<uint2*>(q) = packed;
  if ((threadIdx.x & 3) == 0) {
    sf[swizzled_sf_offset(row, sf_col_base + (col8 >> 2), padded_cols)] =
        (uint8_t)e;
  }
}

// ---- SiTU / SiLU, verbatim from csrc/activation_kernels.cu.
__device__ __forceinline__ float situ_activation(float g, float u, float beta,
                                                 float linear_beta,
                                                 bool clamp_up, float inv_beta,
                                                 float inv_linear_beta) {
  const float gate_out =
      (0.5f * beta) * tanhf(g * inv_beta) * (1.0f + tanhf(g * 0.5f));
  const float up_out = clamp_up ? linear_beta * tanhf(u * inv_linear_beta) : u;
  return gate_out * up_out;
}

// mode: 0 plain quant (x), 1 situ(x[:, :d], x[:, d:]), 2 silu-and-mul,
//       3 x * sigmoid(g)  (two inputs, both [rows, d])
template <int MODE>
__global__ void act_quant_kernel(uint8_t* __restrict__ q,
                                 uint8_t* __restrict__ sf,
                                 const uint16_t* __restrict__ x,
                                 const uint16_t* __restrict__ g,
                                 const int64_t rows, const int64_t rows_pad,
                                 const int d, const int64_t x_stride,
                                 const int64_t g_stride, const int64_t q_stride,
                                 const int sf_col_base, const int padded_cols,
                                 const float beta, const float linear_beta) {
  const int vecs_per_row = d / 8;
  const int64_t total = rows_pad * vecs_per_row;
  const bool clamp_up = linear_beta > 0.0f;
  const float inv_beta = MODE == 1 ? 1.0f / beta : 0.f;
  const float inv_linear_beta = clamp_up ? 1.0f / linear_beta : 0.0f;
  // Every lane runs the same trip count (total is a multiple of 4 vectors,
  // blockDim a multiple of 32), so the block-amax shuffles are uniform.
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  const int64_t total_round = (total + stride - 1) / stride * stride;
  for (int64_t v = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
       v < total_round; v += stride) {
    const bool live = v < total;
    const int64_t row = live ? v / vecs_per_row : 0;
    const int col = live ? (int)(v - row * vecs_per_row) * 8 : 0;
    const bool real = live && row < rows;
    float out[8];
    if (real) {
      if (MODE == 0) {
        const uint4 raw =
            __ldg(reinterpret_cast<const uint4*>(x + row * x_stride + col));
        const uint16_t* r16 = reinterpret_cast<const uint16_t*>(&raw);
#pragma unroll
        for (int i = 0; i < 8; ++i) out[i] = bf16_bits_to_float(r16[i]);
      } else if (MODE == 1 || MODE == 2) {
        const uint16_t* gp = x + row * x_stride + col;
        const uint4 graw = __ldg(reinterpret_cast<const uint4*>(gp));
        const uint4 uraw = __ldg(reinterpret_cast<const uint4*>(gp + d));
        const uint16_t* g16 = reinterpret_cast<const uint16_t*>(&graw);
        const uint16_t* u16 = reinterpret_cast<const uint16_t*>(&uraw);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const float gv = bf16_bits_to_float(g16[i]);
          const float uv = bf16_bits_to_float(u16[i]);
          float r;
          if (MODE == 1) {
            r = situ_activation(gv, uv, beta, linear_beta, clamp_up, inv_beta,
                                inv_linear_beta);
          } else {
            // csrc act_and_mul_kernel: (T)silu(x) * y, both rounded to T.
            const float s =
                bf16_bits_to_float(float_to_bf16_bits(gv / (1.0f + expf(-gv))));
            r = s * uv;
          }
          out[i] = bf16_bits_to_float(float_to_bf16_bits(r));
        }
      } else {
        // mla._gate_sigmoid_mul is torch.compiled: one FP32 expression
        // a * sigmoid(g), rounded to BF16 once.
        const uint4 araw =
            __ldg(reinterpret_cast<const uint4*>(x + row * x_stride + col));
        const uint4 graw =
            __ldg(reinterpret_cast<const uint4*>(g + row * g_stride + col));
        const uint16_t* a16 = reinterpret_cast<const uint16_t*>(&araw);
        const uint16_t* g16 = reinterpret_cast<const uint16_t*>(&graw);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const float gv = bf16_bits_to_float(g16[i]);
          const float s = 1.0f / (1.0f + expf(-gv));
          out[i] = bf16_bits_to_float(
              float_to_bf16_bits(bf16_bits_to_float(a16[i]) * s));
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i) out[i] = 0.f;
    }
    // Every lane quantizes (uniform shuffles); only real rows store data.
    quant8_store(out, q + row * q_stride + col, sf, row, col / 8, sf_col_base,
                 padded_cols, real);
    // Padding rows (rows <= row < rows_pad) get zero scales.
    if (live && !real && (threadIdx.x & 3) == 0) {
      sf[swizzled_sf_offset(row, sf_col_base + col / 32, padded_cols)] = 0;
    }
  }
}

// One block per row; K / 8 threads (K <= 8192), each owning 8 columns.
// vLLM RMSNorm (ir rms_norm as dispatched on SM10x): out = (bf16)((bf16)(x * rstd) * w).
__global__ void rms_norm_quant_kernel(uint8_t* __restrict__ q,
                                      uint8_t* __restrict__ sf,
                                      uint16_t* __restrict__ y,
                                      const uint16_t* __restrict__ x,
                                      const uint16_t* __restrict__ w,
                                      const int64_t rows, const int k,
                                      const int64_t x_stride,
                                      const int64_t q_stride,
                                      const int64_t y_stride,
                                      const int sf_col_base,
                                      const int padded_cols, const float eps) {
  __shared__ float warp_sums[32];
  const int64_t row = blockIdx.x;
  const int col = threadIdx.x * 8;
  const bool real = row < rows;
  float v[8];
  float ss = 0.f;
  if (real) {
    const uint4 raw =
        __ldg(reinterpret_cast<const uint4*>(x + row * x_stride + col));
    const uint16_t* r16 = reinterpret_cast<const uint16_t*>(&raw);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      v[i] = bf16_bits_to_float(r16[i]);
      ss += v[i] * v[i];
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0) warp_sums[warp] = ss;
  __syncthreads();
  const int nwarps = blockDim.x >> 5;
  float tot = lane < nwarps ? warp_sums[lane] : 0.f;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) tot += __shfl_xor_sync(0xffffffff, tot, o);
  const float rstd = rsqrtf(tot / (float)k + eps);
  float out[8];
  if (real) {
    const uint4 wraw = __ldg(reinterpret_cast<const uint4*>(w + col));
    const uint16_t* w16 = reinterpret_cast<const uint16_t*>(&wraw);
    uint4 yraw;
    uint16_t* y16 = reinterpret_cast<uint16_t*>(&yraw);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float n = bf16_bits_to_float(float_to_bf16_bits(v[i] * rstd));
      y16[i] = float_to_bf16_bits(n * bf16_bits_to_float(w16[i]));
      out[i] = bf16_bits_to_float(y16[i]);
    }
    if (y != nullptr) {
      *reinterpret_cast<uint4*>(y + row * y_stride + col) = yraw;
    }
  } else {
#pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = 0.f;
  }
  quant8_store(out, q + (real ? row : 0) * q_stride + col, sf, row, col / 8,
               sf_col_base, padded_cols, real);
  if (!real && (threadIdx.x & 3) == 0) {
    sf[swizzled_sf_offset(row, sf_col_base + col / 32, padded_cols)] = 0;
  }
}

void check_bf16_rows(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.scalar_type() == at::kBFloat16, name, " must be BF16");
  TORCH_CHECK(t.dim() == 2 && t.stride(1) == 1, name, " must be 2D, unit col stride");
  TORCH_CHECK(t.stride(0) % 8 == 0 &&
              reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
              name, " must be 16-byte aligned rows");
}

}  // namespace

// q: [rows, >= col + d] uint8/fp8 view at the column window; sf: swizzled
// scales of the whole (padded) activation with padded_cols scale columns.
void act_quant(torch::Tensor& q, torch::Tensor& sf, const torch::Tensor& x,
               const c10::optional<torch::Tensor>& g, int64_t mode, int64_t d,
               int64_t sf_col_base, int64_t padded_cols, double beta,
               double linear_beta) {
  check_bf16_rows(x, "x");
  TORCH_CHECK(d % 32 == 0);
  TORCH_CHECK(q.stride(1) == 1 && q.stride(0) % 8 == 0 &&
              reinterpret_cast<uintptr_t>(q.data_ptr()) % 8 == 0);
  const int64_t rows = x.size(0);
  const int64_t rows_pad = (rows + 127) / 128 * 128;
  const uint16_t* gp = nullptr;
  int64_t g_stride = 0;
  if (mode == 3) {
    TORCH_CHECK(g.has_value());
    check_bf16_rows(*g, "g");
    gp = reinterpret_cast<const uint16_t*>(g->data_ptr());
    g_stride = g->stride(0);
  }
  if (rows == 0) return;
  const int threads = 256;
  const int64_t total = rows_pad * (d / 8);
  const int blocks = (int)std::min<int64_t>((total + threads - 1) / threads, 2048);
  auto stream = c10::cuda::getCurrentCUDAStream();
#define LAUNCH(M)                                                              \
  act_quant_kernel<M><<<blocks, threads, 0, stream>>>(                         \
      reinterpret_cast<uint8_t*>(q.data_ptr()),                                \
      reinterpret_cast<uint8_t*>(sf.data_ptr()),                               \
      reinterpret_cast<const uint16_t*>(x.data_ptr()), gp, rows, rows_pad,     \
      (int)d, x.stride(0), g_stride, q.stride(0), (int)sf_col_base,            \
      (int)padded_cols, (float)beta, (float)linear_beta)
  switch (mode) {
    case 0: LAUNCH(0); break;
    case 1: LAUNCH(1); break;
    case 2: LAUNCH(2); break;
    case 3: LAUNCH(3); break;
    default: TORCH_CHECK(false, "bad mode");
  }
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rms_norm_quant(torch::Tensor& q, torch::Tensor& sf,
                    const c10::optional<torch::Tensor>& y,
                    const torch::Tensor& x, const torch::Tensor& w, double eps,
                    int64_t sf_col_base, int64_t padded_cols) {
  check_bf16_rows(x, "x");
  const int k = (int)x.size(1);
  TORCH_CHECK(k % 256 == 0 && k <= 8192, "rms_norm_quant: K % 256 == 0, K <= 8192");
  TORCH_CHECK(w.scalar_type() == at::kBFloat16 && w.is_contiguous() && w.numel() == k);
  TORCH_CHECK(q.stride(1) == 1 && q.stride(0) % 8 == 0);
  uint16_t* yp = nullptr;
  int64_t y_stride = 0;
  if (y.has_value()) {
    check_bf16_rows(*y, "y");
    yp = reinterpret_cast<uint16_t*>(y->data_ptr());
    y_stride = y->stride(0);
  }
  const int64_t rows = x.size(0);
  if (rows == 0) return;
  const int64_t rows_pad = (rows + 127) / 128 * 128;
  rms_norm_quant_kernel<<<(unsigned)rows_pad, k / 8, 0,
                          c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<uint8_t*>(q.data_ptr()),
      reinterpret_cast<uint8_t*>(sf.data_ptr()), yp,
      reinterpret_cast<const uint16_t*>(x.data_ptr()),
      reinterpret_cast<const uint16_t*>(w.data_ptr()), rows, k, x.stride(0),
      q.stride(0), y_stride, (int)sf_col_base, (int)padded_cols, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_CPP_SRC = """
void act_quant(torch::Tensor& q, torch::Tensor& sf, const torch::Tensor& x,
               const c10::optional<torch::Tensor>& g, int64_t mode, int64_t d,
               int64_t sf_col_base, int64_t padded_cols, double beta,
               double linear_beta);
void rms_norm_quant(torch::Tensor& q, torch::Tensor& sf,
                    const c10::optional<torch::Tensor>& y,
                    const torch::Tensor& x, const torch::Tensor& w, double eps,
                    int64_t sf_col_base, int64_t padded_cols);
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
            build = os.path.join(envs.VLLM_CACHE_ROOT, "k3_mxfp8_act", tag)
            os.makedirs(build, exist_ok=True)
            # No fast math: precise tanhf / expf / division as the BF16 kernels.
            _mod = load_inline(
                name=f"k3_mxfp8_act_{tag}",
                cpp_sources=_CPP_SRC,
                cuda_sources=_CUDA_SRC,
                functions=["act_quant", "rms_norm_quant"],
                extra_cuda_cflags=["-O3"],
                build_directory=build,
                verbose=False,
            )
    return _mod


def warmup() -> None:
    """Build / load the extension before serving (first use compiles it)."""
    _module()


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


class Mxfp8Buffer:
    """An MXFP8 activation ``[rows, k]`` (FP8 values + F8_128x4-swizzled E8M0
    scales) that several producers may fill column windows of."""

    def __init__(self, rows: int, k: int, device, orig_dtype=torch.bfloat16):
        assert k % (4 * MXFP8_BLOCK) == 0, k
        self.rows, self.k = rows, k
        self.padded_cols = k // MXFP8_BLOCK  # k % 128 == 0 -> multiple of 4
        self.data = torch.empty((rows, k), dtype=torch.float8_e4m3fn, device=device)
        self.scale = torch.empty(
            _round_up(max(rows, 1), 128) * self.padded_cols,
            dtype=torch.uint8,
            device=device,
        )
        self.orig_dtype = orig_dtype

    def window(self, col: int, width: int) -> torch.Tensor:
        assert col % (4 * MXFP8_BLOCK) == 0 and col + width <= self.k
        return self.data.view(torch.uint8)[:, col : col + width]

    def activation(self) -> QuantizedActivation:
        return QuantizedActivation(
            data=self.data,
            scale=self.scale,
            orig_dtype=self.orig_dtype,
            orig_shape=torch.Size((self.rows, self.k)),
            quant_key=kMxfp8Dynamic,
        )


def _act_quant(
    mode: int,
    x: torch.Tensor,
    d: int,
    g: torch.Tensor | None = None,
    beta: float = 1.0,
    linear_beta: float | None = None,
    out: Mxfp8Buffer | None = None,
    col: int = 0,
) -> QuantizedActivation | None:
    x2 = x.reshape(-1, x.shape[-1])
    rows = x2.shape[0]
    own = out is None
    if own:
        out = Mxfp8Buffer(rows, d, x.device, x.dtype)
    assert out.rows == rows
    _module().act_quant(
        out.window(col, d),
        out.scale,
        x2,
        None if g is None else g.reshape(-1, g.shape[-1]),
        mode,
        d,
        col // MXFP8_BLOCK,
        out.padded_cols,
        float(beta),
        -1.0 if linear_beta is None else float(linear_beta),
    )
    return out.activation() if own else None


def quantize(x: torch.Tensor, out: Mxfp8Buffer | None = None, col: int = 0):
    """BF16 ``[rows, k]`` -> MXFP8 (FlashInfer ``mxfp8_quantize`` semantics,
    swizzled scales)."""
    return _act_quant(0, x, x.shape[-1], out=out, col=col)


def situ_and_mul_quantize(
    x: torch.Tensor,
    beta: float,
    linear_beta: float | None,
    out: Mxfp8Buffer | None = None,
    col: int = 0,
):
    """``situ_and_mul(x)`` ([rows, 2d] -> [rows, d]) as MXFP8."""
    return _act_quant(
        1, x, x.shape[-1] // 2, beta=beta, linear_beta=linear_beta, out=out, col=col
    )


def silu_and_mul_quantize(
    x: torch.Tensor, out: Mxfp8Buffer | None = None, col: int = 0
):
    """``silu_and_mul(x)`` ([rows, 2d] -> [rows, d]) as MXFP8."""
    return _act_quant(2, x, x.shape[-1] // 2, out=out, col=col)


def gate_sigmoid_mul_quantize(
    x: torch.Tensor, g: torch.Tensor, out: Mxfp8Buffer | None = None, col: int = 0
):
    """``x * g.sigmoid()`` as MXFP8."""
    assert x.shape == g.shape
    return _act_quant(3, x, x.shape[-1], g=g, out=out, col=col)


def rms_norm_quantize(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: Mxfp8Buffer | None = None,
    col: int = 0,
    bf16_out: torch.Tensor | None = None,
):
    """``RMSNorm(x) * weight`` ([rows, k], k % 256 == 0) as MXFP8; also into
    ``bf16_out`` when given."""
    x2 = x.reshape(-1, x.shape[-1])
    rows, k = x2.shape
    own = out is None
    if own:
        out = Mxfp8Buffer(rows, k, x.device, x.dtype)
    if rows:
        _module().rms_norm_quant(
            out.window(col, k),
            out.scale,
            bf16_out,
            x2,
            weight,
            float(eps),
            col // MXFP8_BLOCK,
            out.padded_cols,
        )
    return out.activation() if own else None


def supports(k: int) -> bool:
    return k % (4 * MXFP8_BLOCK) == 0
