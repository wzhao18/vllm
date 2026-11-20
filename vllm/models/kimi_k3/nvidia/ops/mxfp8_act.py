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
  // cvt.rn.bf16.f32: RNE as c10::BFloat16 for every non-NaN value.
  return __bfloat16_as_ushort(__float2bfloat16_rn(f));
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
// SiTU with the MUFU-light form of the routed experts' MegaMoE FC1 epilogue
// (FlashInfer block_scaled_swap_ab_fc12_epilogue, vLLM S3): with
// A = e^-g, B = e^(-2g/beta), C = e^(-2u/linear_beta),
//   sigmoid(g) = 1/(1+A), tanh(g/beta) = (1-B)/(1+B), tanh(u/lbeta) = (1-C)/(1+C)
//   out = beta*lbeta * (1-B)(1-C) / ((1+A)(1+B)(1+C))
// on ex2.approx / rcp.approx (2 ex2 + 1 rcp when beta/2 is a small integer).
// Not bitwise equal to the precise-tanhf form above (test bound: BF16 noise).
__device__ __forceinline__ float ex2_approx(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float rcp_approx(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
struct Situ4Consts {
  float2 k_a, k_b, k_c, bb;  // per-element constants, duplicated for f32x2
  int a_pow;                 // A = B^a_pow when beta/2 is 1..4, else 0
  bool clamp_up;
};

__device__ __forceinline__ float2 f2mul(float2 a, float2 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  return __fmul2_rn(a, b);
#else
  return make_float2(a.x * b.x, a.y * b.y);
#endif
}
__device__ __forceinline__ float2 f2fma(float2 a, float2 b, float2 c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  return __ffma2_rn(a, b, c);
#else
  return make_float2(fmaf(a.x, b.x, c.x), fmaf(a.y, b.y, c.y));
#endif
}
__device__ __forceinline__ float2 f2add(float2 a, float2 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  return __fadd2_rn(a, b);
#else
  return make_float2(a.x + b.x, a.y + b.y);
#endif
}
__device__ __forceinline__ float2 bf16x2_to_f2(uint32_t w) {
  return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u));
}

// SiTU of two (gate, up) BF16 pairs, rounded to BF16 (the stock activation
// output dtype), returned as packed bf16x2.
__device__ __forceinline__ __nv_bfloat162 situ4_pair(uint32_t gw, uint32_t uw,
                                                     const Situ4Consts& c) {
  const float2 one = make_float2(1.0f, 1.0f);
  const float2 neg_one = make_float2(-1.0f, -1.0f);
  const float2 g = bf16x2_to_f2(gw), u = bf16x2_to_f2(uw);
  const float2 xb = f2mul(g, c.k_b);
  float2 eb = make_float2(fminf(ex2_approx(xb.x), 1.0e9f),
                          fminf(ex2_approx(xb.y), 1.0e9f));
  float2 ea;
  if (c.a_pow > 0) {
    ea = eb;
    for (int i = 1; i < c.a_pow; ++i) ea = f2mul(ea, eb);
  } else {
    const float2 xa = f2mul(g, c.k_a);
    ea = make_float2(ex2_approx(xa.x), ex2_approx(xa.y));
  }
  float2 num, den;
  if (c.clamp_up) {
    const float2 xc = f2mul(u, c.k_c);
    const float2 ec = make_float2(fminf(ex2_approx(xc.x), 1.0e9f),
                                  fminf(ex2_approx(xc.y), 1.0e9f));
    num = f2mul(f2mul(f2fma(eb, neg_one, one), f2fma(ec, neg_one, one)), c.bb);
    den = f2mul(f2mul(f2add(ea, one), f2add(eb, one)), f2add(ec, one));
  } else {
    num = f2mul(f2mul(f2fma(eb, neg_one, one), c.bb), u);
    den = f2mul(f2add(ea, one), f2add(eb, one));
  }
  const float2 r = f2mul(num, make_float2(rcp_approx(den.x), rcp_approx(den.y)));
  return __float22bfloat162_rn(r);
}

// SiTU + MXFP8 quant of one 8-column vector (a: gate, b: up): the same E8M0 /
// E4M3 steps as quant8_store, on packed operands.
__device__ __forceinline__ void situ4_quant_store(
    const uint4& a, const uint4& b, const Situ4Consts& c,
    uint8_t* __restrict__ q, uint8_t* __restrict__ sf, int64_t row, int col8,
    int sf_col_base, int padded_cols, bool real) {
  __nv_bfloat162 h[4];
  if (real) {
    h[0] = situ4_pair(a.x, b.x, c);
    h[1] = situ4_pair(a.y, b.y, c);
    h[2] = situ4_pair(a.z, b.z, c);
    h[3] = situ4_pair(a.w, b.w, c);
  } else {
#pragma unroll
    for (int i = 0; i < 4; ++i) h[i] = __float2bfloat162_rn(0.0f);
  }
  // amax on the BF16 values (exact).
  __nv_bfloat162 m = __hmax2(__habs2(h[0]), __habs2(h[1]));
  m = __hmax2(m, __hmax2(__habs2(h[2]), __habs2(h[3])));
  float amax = fmaxf(__low2float(m), __high2float(m));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
  const uint32_t e = float_to_ue8m0(amax * (1.0f / 448.0f));
  const float inv = ue8m0_to_inv_scale(e);
  if (!real) return;
  const float2 inv2 = make_float2(inv, inv);
  uint32_t w[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 v = f2mul(__bfloat1622float2(h[i]), inv2);
    w[i] = (uint32_t)fp8x2(v.x, v.y);
  }
  uint2 packed;
  packed.x = w[0] | (w[1] << 16);
  packed.y = w[2] | (w[3] << 16);
  *reinterpret_cast<uint2*>(q) = packed;
  if ((threadIdx.x & 3) == 0) {
    sf[swizzled_sf_offset(row, sf_col_base + (col8 >> 2), padded_cols)] =
        (uint8_t)e;
  }
}

// Vectors (8 columns) per thread per trip: their loads are all issued before
// any is used, so each SM keeps enough bytes in flight to stream HBM.
constexpr int ACT_QUANT_UNROLL = 4;

template <int MODE>
__device__ __forceinline__ void act_load(const uint16_t* __restrict__ x,
                                         const uint16_t* __restrict__ g,
                                         int64_t row, int col, int d,
                                         int64_t x_stride, int64_t g_stride,
                                         uint4& a, uint4& b) {
  if (MODE == 0) {
    a = __ldg(reinterpret_cast<const uint4*>(x + row * x_stride + col));
  } else if (MODE == 1 || MODE == 2 || MODE == 4) {
    const uint16_t* gp = x + row * x_stride + col;
    a = __ldg(reinterpret_cast<const uint4*>(gp));
    b = __ldg(reinterpret_cast<const uint4*>(gp + d));
  } else {
    a = __ldg(reinterpret_cast<const uint4*>(x + row * x_stride + col));
    b = __ldg(reinterpret_cast<const uint4*>(g + row * g_stride + col));
  }
}

template <int MODE>
__device__ __forceinline__ void act_compute(const uint4& a, const uint4& b,
                                            float (&out)[8], float beta,
                                            float linear_beta, bool clamp_up,
                                            float inv_beta,
                                            float inv_linear_beta) {
  if (MODE == 0) {
    const uint16_t* r16 = reinterpret_cast<const uint16_t*>(&a);
#pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = bf16_bits_to_float(r16[i]);
  } else if (MODE == 1 || MODE == 2) {
    const uint16_t* g16 = reinterpret_cast<const uint16_t*>(&a);
    const uint16_t* u16 = reinterpret_cast<const uint16_t*>(&b);
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
    const uint16_t* a16 = reinterpret_cast<const uint16_t*>(&a);
    const uint16_t* g16 = reinterpret_cast<const uint16_t*>(&b);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float gv = bf16_bits_to_float(g16[i]);
      const float s = 1.0f / (1.0f + expf(-gv));
      out[i] = bf16_bits_to_float(
          float_to_bf16_bits(bf16_bits_to_float(a16[i]) * s));
    }
  }
}

// situ4_pair with the SiTU form fixed at compile time (A_POW, CLAMP): the
// same operations in the same order, without the runtime branches.
template <int A_POW, bool CLAMP>
__device__ __forceinline__ uint32_t situ4_pair_ct(uint32_t gw, uint32_t uw,
                                                  const float2 k_a,
                                                  const float2 k_b,
                                                  const float2 k_c,
                                                  const float2 bb) {
  const float2 one = make_float2(1.0f, 1.0f);
  const float2 neg_one = make_float2(-1.0f, -1.0f);
  const float2 g = bf16x2_to_f2(gw), u = bf16x2_to_f2(uw);
  const float2 xb = f2mul(g, k_b);
  const float2 eb = make_float2(fminf(ex2_approx(xb.x), 1.0e9f),
                                fminf(ex2_approx(xb.y), 1.0e9f));
  float2 ea;
  if (A_POW > 0) {
    ea = eb;
#pragma unroll
    for (int i = 1; i < A_POW; ++i) ea = f2mul(ea, eb);
  } else {
    const float2 xa = f2mul(g, k_a);
    ea = make_float2(ex2_approx(xa.x), ex2_approx(xa.y));
  }
  float2 num, den;
  if (CLAMP) {
    const float2 xc = f2mul(u, k_c);
    const float2 ec = make_float2(fminf(ex2_approx(xc.x), 1.0e9f),
                                  fminf(ex2_approx(xc.y), 1.0e9f));
    num = f2mul(f2mul(f2fma(eb, neg_one, one), f2fma(ec, neg_one, one)), bb);
    den = f2mul(f2mul(f2add(ea, one), f2add(eb, one)), f2add(ec, one));
  } else {
    num = f2mul(f2mul(f2fma(eb, neg_one, one), bb), u);
    den = f2mul(f2add(ea, one), f2add(eb, one));
  }
  const float2 r = f2mul(num, make_float2(rcp_approx(den.x), rcp_approx(den.y)));
  const __nv_bfloat162 h = __float22bfloat162_rn(r);
  return *reinterpret_cast<const uint32_t*>(&h);
}

// Mode 4 for d % 1024 == 0, d <= 32768, 16-byte aligned rows: one CTA per
// (padded) row and one thread per 32-column MXFP8 block, so the block amax
// stays in registers and the indexing is two integer ops. Bytes equal
// act_quant_kernel<4> (same SiTU and quantization steps).
template <int A_POW, bool CLAMP>
__global__ void __launch_bounds__(1024) situ4_row_kernel(
    uint8_t* __restrict__ q, uint8_t* __restrict__ sf,
    const uint16_t* __restrict__ x, const int64_t rows, const int d,
    const int64_t x_stride, const int64_t q_stride, const int sf_col_base,
    const int padded_cols, const float beta, const float linear_beta) {
  const float log2e = 1.4426950408889634f;
  const float kb = -2.0f * log2e / beta;
  const float kc = CLAMP ? -2.0f * log2e / linear_beta : 0.0f;
  const float bbs = CLAMP ? beta * linear_beta : beta;
  const float2 k_a = make_float2(-log2e, -log2e), k_b = make_float2(kb, kb);
  const float2 k_c = make_float2(kc, kc), bb = make_float2(bbs, bbs);
  const int64_t row = blockIdx.x;
  const int blk = threadIdx.x;
  const int64_t sf_off = swizzled_sf_offset(row, sf_col_base + blk, padded_cols);
  if (row >= rows) {
    sf[sf_off] = 0;
    return;
  }
  const uint4* gp =
      reinterpret_cast<const uint4*>(x + row * x_stride + blk * 32);
  const uint4* up = reinterpret_cast<const uint4*>(x + row * x_stride + d + blk * 32);
  uint4 a[4], b[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) a[i] = __ldg(gp + i);
#pragma unroll
  for (int i = 0; i < 4; ++i) b[i] = __ldg(up + i);
  uint32_t h[16];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    h[4 * i + 0] = situ4_pair_ct<A_POW, CLAMP>(a[i].x, b[i].x, k_a, k_b, k_c, bb);
    h[4 * i + 1] = situ4_pair_ct<A_POW, CLAMP>(a[i].y, b[i].y, k_a, k_b, k_c, bb);
    h[4 * i + 2] = situ4_pair_ct<A_POW, CLAMP>(a[i].z, b[i].z, k_a, k_b, k_c, bb);
    h[4 * i + 3] = situ4_pair_ct<A_POW, CLAMP>(a[i].w, b[i].w, k_a, k_b, k_c, bb);
  }
  // amax on the BF16 values (exact).
  __nv_bfloat162 m = __habs2(*reinterpret_cast<const __nv_bfloat162*>(&h[0]));
#pragma unroll
  for (int i = 1; i < 16; ++i)
    m = __hmax2(m, __habs2(*reinterpret_cast<const __nv_bfloat162*>(&h[i])));
  const float amax = fmaxf(__low2float(m), __high2float(m));
  const uint32_t e = float_to_ue8m0(amax * (1.0f / 448.0f));
  const float inv = ue8m0_to_inv_scale(e);
  const float2 inv2 = make_float2(inv, inv);
  uint32_t w[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float2 v0 = f2mul(
        __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&h[2 * i])), inv2);
    const float2 v1 = f2mul(
        __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&h[2 * i + 1])),
        inv2);
    w[i] = (uint32_t)fp8x2(v0.x, v0.y) | ((uint32_t)fp8x2(v1.x, v1.y) << 16);
  }
  uint4* qp = reinterpret_cast<uint4*>(q + row * q_stride + blk * 32);
  qp[0] = make_uint4(w[0], w[1], w[2], w[3]);
  qp[1] = make_uint4(w[4], w[5], w[6], w[7]);
  sf[sf_off] = (uint8_t)e;
}

// mode: 0 plain quant (x), 1 situ(x[:, :d], x[:, d:]), 4 the same with the
//       routed experts' MUFU-light SiTU, 2 silu-and-mul,
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
  Situ4Consts c4;
  if (MODE == 4) {
    const float log2e = 1.4426950408889634f;
    const float kb = -2.0f * log2e / beta;
    const float kc = clamp_up ? -2.0f * log2e / linear_beta : 0.0f;
    const float bb = clamp_up ? beta * linear_beta : beta;
    const float half = 0.5f * beta;
    c4.k_a = make_float2(-log2e, -log2e);
    c4.k_b = make_float2(kb, kb);
    c4.k_c = make_float2(kc, kc);
    c4.bb = make_float2(bb, bb);
    c4.a_pow = (half == floorf(half) && half >= 1.0f && half <= 4.0f) ? (int)half : 0;
    c4.clamp_up = clamp_up;
  }
  // Every lane runs the same trip count (total is a multiple of 4 vectors,
  // blockDim a multiple of 32), so the block-amax shuffles are uniform; the
  // 4 lanes of an MXFP8 block hold consecutive vectors in every sub-trip.
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  const int64_t trip = stride * ACT_QUANT_UNROLL;
  const int64_t total_round = (total + trip - 1) / trip * trip;
  // (row, vector column) advance by (srow, scol) per sub-trip: one 64-bit
  // division per trip instead of one per vector.
  const int64_t srow = stride / vecs_per_row;
  const int scol = (int)(stride - srow * vecs_per_row);
  for (int64_t v0 = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
       v0 < total_round; v0 += trip) {
    uint4 a[ACT_QUANT_UNROLL], b[ACT_QUANT_UNROLL];
    int64_t rows_u[ACT_QUANT_UNROLL];
    int cols_u[ACT_QUANT_UNROLL];
    {
      int64_t r = v0 / vecs_per_row;
      int c = (int)(v0 - r * vecs_per_row);
#pragma unroll
      for (int u = 0; u < ACT_QUANT_UNROLL; ++u) {
        rows_u[u] = r;
        cols_u[u] = c;
        r += srow;
        c += scol;
        if (c >= vecs_per_row) {
          c -= vecs_per_row;
          ++r;
        }
      }
    }
#pragma unroll
    for (int u = 0; u < ACT_QUANT_UNROLL; ++u) {
      const int64_t v = v0 + u * stride;
      const bool live = v < total;
      const int64_t row = live ? rows_u[u] : 0;
      const int col = live ? cols_u[u] * 8 : 0;
      if (live && row < rows) {
        act_load<MODE>(x, g, row, col, d, x_stride, g_stride, a[u], b[u]);
      }
    }
#pragma unroll
    for (int u = 0; u < ACT_QUANT_UNROLL; ++u) {
      const int64_t v = v0 + u * stride;
      const bool live = v < total;
      const int64_t row = live ? rows_u[u] : 0;
      const int col = live ? cols_u[u] * 8 : 0;
      const bool real = live && row < rows;
      if (MODE == 4) {
        situ4_quant_store(a[u], b[u], c4, q + row * q_stride + col, sf, row,
                          col / 8, sf_col_base, padded_cols, real);
        if (live && !real && (threadIdx.x & 3) == 0) {
          sf[swizzled_sf_offset(row, sf_col_base + col / 32, padded_cols)] = 0;
        }
        continue;
      }
      float out[8];
      if (real) {
        act_compute<MODE>(a[u], b[u], out, beta, linear_beta, clamp_up,
                          inv_beta, inv_linear_beta);
      } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) out[i] = 0.f;
      }
      // Every lane quantizes (uniform shuffles); only real rows store data.
      quant8_store(out, q + row * q_stride + col, sf, row, col / 8,
                   sf_col_base, padded_cols, real);
      // Padding rows (rows <= row < rows_pad) get zero scales.
      if (live && !real && (threadIdx.x & 3) == 0) {
        sf[swizzled_sf_offset(row, sf_col_base + col / 32, padded_cols)] = 0;
      }
    }
  }
}

// RMS_ROWS rows per block; K / 8 threads (K <= 8192), each owning 8 columns
// of every row. All rows' loads are issued before the first reduction, so a
// block keeps RMS_ROWS x K x 2 bytes in flight (one row per block left the
// kernel latency bound at ~4 x 7 KiB per SM). Per row the arithmetic and the
// reduction order are those of the one-row form (bitwise equal).
// vLLM RMSNorm (ir rms_norm as dispatched on SM10x): out = (bf16)((bf16)(x * rstd) * w).
constexpr int RMS_ROWS = 4;
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
  __shared__ float warp_sums[RMS_ROWS][32];
  const int64_t row0 = int64_t{blockIdx.x} * RMS_ROWS;
  const int col = threadIdx.x * 8;
  uint4 raw[RMS_ROWS];
#pragma unroll
  for (int r = 0; r < RMS_ROWS; ++r) {
    raw[r] = row0 + r < rows
                 ? __ldg(reinterpret_cast<const uint4*>(
                       x + (row0 + r) * x_stride + col))
                 : make_uint4(0u, 0u, 0u, 0u);
  }
  const uint4 wraw = __ldg(reinterpret_cast<const uint4*>(w + col));
  const uint16_t* w16 = reinterpret_cast<const uint16_t*>(&wraw);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int nwarps = blockDim.x >> 5;
#pragma unroll
  for (int r = 0; r < RMS_ROWS; ++r) {
    const uint16_t* r16 = reinterpret_cast<const uint16_t*>(&raw[r]);
    float ss = 0.f;
    if (row0 + r < rows) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float v = bf16_bits_to_float(r16[i]);
        ss += v * v;
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
    if (lane == 0) warp_sums[r][warp] = ss;
  }
  __syncthreads();
#pragma unroll
  for (int r = 0; r < RMS_ROWS; ++r) {
    const int64_t row = row0 + r;
    const bool real = row < rows;
    float tot = lane < nwarps ? warp_sums[r][lane] : 0.f;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) tot += __shfl_xor_sync(0xffffffff, tot, o);
    const float rstd = rsqrtf(tot / (float)k + eps);
    float out[8];
    if (real) {
      const uint16_t* r16 = reinterpret_cast<const uint16_t*>(&raw[r]);
      uint4 yraw;
      uint16_t* y16 = reinterpret_cast<uint16_t*>(&yraw);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float n =
            bf16_bits_to_float(float_to_bf16_bits(bf16_bits_to_float(r16[i]) * rstd));
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
  if (mode == 4 && d % 1024 == 0 && d <= 32768 && q.stride(0) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0) {
    const bool clamp = linear_beta > 0.0;
    const float half = 0.5f * (float)beta;
    const int a_pow =
        (half == floorf(half) && half >= 1.0f && half <= 4.0f) ? (int)half : 0;
    auto stream = c10::cuda::getCurrentCUDAStream();
#define ROW_LAUNCH(P, C)                                                       \
  situ4_row_kernel<P, C><<<(unsigned)rows_pad, (unsigned)(d / 32), 0, stream>>>( \
      reinterpret_cast<uint8_t*>(q.data_ptr()),                                \
      reinterpret_cast<uint8_t*>(sf.data_ptr()),                               \
      reinterpret_cast<const uint16_t*>(x.data_ptr()), rows, (int)d,           \
      x.stride(0), q.stride(0), (int)sf_col_base, (int)padded_cols,            \
      (float)beta, (float)linear_beta)
    switch (a_pow * 2 + (clamp ? 1 : 0)) {
      case 0: ROW_LAUNCH(0, false); break;
      case 1: ROW_LAUNCH(0, true); break;
      case 2: ROW_LAUNCH(1, false); break;
      case 3: ROW_LAUNCH(1, true); break;
      case 4: ROW_LAUNCH(2, false); break;
      case 5: ROW_LAUNCH(2, true); break;
      case 6: ROW_LAUNCH(3, false); break;
      case 7: ROW_LAUNCH(3, true); break;
      case 8: ROW_LAUNCH(4, false); break;
      default: ROW_LAUNCH(4, true); break;
    }
#undef ROW_LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  const int threads = 256;
  const int64_t total = rows_pad * (d / 8);
  const int64_t per_block = (int64_t)threads * ACT_QUANT_UNROLL;
  const int64_t need = (total + per_block - 1) / per_block;
  auto stream = c10::cuda::getCurrentCUDAStream();
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, x.get_device());
  // One resident wave of CTAs (grid-stride loop): a partial second wave would
  // leave the SMs of the first wave idle for its whole duration.
#define LAUNCH(M)                                                              \
  {                                                                            \
    static int occ_##M = 0;                                                    \
    if (occ_##M == 0)                                                          \
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_##M,                  \
                                                    act_quant_kernel<M>,       \
                                                    threads, 0);               \
    const int blocks =                                                         \
        (int)std::min<int64_t>(need, (int64_t)sms * std::max(occ_##M, 1));     \
  act_quant_kernel<M><<<blocks, threads, 0, stream>>>(                         \
      reinterpret_cast<uint8_t*>(q.data_ptr()),                                \
      reinterpret_cast<uint8_t*>(sf.data_ptr()),                               \
      reinterpret_cast<const uint16_t*>(x.data_ptr()), gp, rows, rows_pad,     \
      (int)d, x.stride(0), g_stride, q.stride(0), (int)sf_col_base,            \
      (int)padded_cols, (float)beta, (float)linear_beta);                      \
  }
  switch (mode) {
    case 0: LAUNCH(0); break;
    case 1: LAUNCH(1); break;
    case 2: LAUNCH(2); break;
    case 3: LAUNCH(3); break;
    case 4: LAUNCH(4); break;
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
  rms_norm_quant_kernel<<<(unsigned)(rows_pad / RMS_ROWS), k / 8, 0,
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


# Rows from which the shared-expert SiTU uses the routed experts' MUFU-light
# form (prefill: the precise-tanhf form is FMA-issue bound at ~4x its HBM
# floor there; decode-sized calls are latency bound and keep the precise form).
SITU_FAST_MIN_ROWS = 2048


def situ_and_mul_quantize_auto(
    x: torch.Tensor,
    beta: float,
    linear_beta: float | None,
    out: Mxfp8Buffer | None = None,
    col: int = 0,
):
    """``situ_and_mul_quantize``, per shape: the routed experts' MUFU-light
    SiTU (MegaMoE FC1 form) from ``SITU_FAST_MIN_ROWS`` rows, else the precise
    form."""
    rows = x.numel() // x.shape[-1]
    mode = 4 if rows >= SITU_FAST_MIN_ROWS else 1
    return _act_quant(
        mode, x, x.shape[-1] // 2, beta=beta, linear_beta=linear_beta, out=out, col=col
    )


def situ_and_mul_quantize_fast(
    x: torch.Tensor,
    beta: float,
    linear_beta: float | None,
    out: Mxfp8Buffer | None = None,
    col: int = 0,
):
    """``situ_and_mul(x)`` as MXFP8 with the routed experts' SiTU form."""
    return _act_quant(
        4, x, x.shape[-1] // 2, beta=beta, linear_beta=linear_beta, out=out, col=col
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
