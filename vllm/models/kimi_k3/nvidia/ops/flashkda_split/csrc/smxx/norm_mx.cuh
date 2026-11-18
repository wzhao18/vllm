// Kimi-K3: the KDA output gate norm (FusedRMSNormGated, sigmoid, D = 128) and
// the MXFP8 quantization of the o_proj input, fused into the full-width
// recurrence epilogue. Per row (token, head): ops/kda_norm.py
// _kda_rmsnorm_gated_mxfp8_kernel's expressions with the instructions Triton
// lowers them to (approximate sqrt / full-range division / ex2-based sigmoid);
// the sum of squares runs in a different order (stock tolerance, not bitwise).
// Then the BF16 round of the gated output, UE8M0 per 32 (FlashInfer
// float_to_ue8m0_fast of amax / 448) and RN-satfinite E4M3.
#pragma once

#include <cstdint>
#include <cuda_bf16.h>

#include "conv_args.h"

namespace norm_mx {

__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ float sqrt_approx(float a) {
  float r;
  asm("sqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}

__device__ __forceinline__ float sigmoid_triton(float x) {
  float t, e;
  asm("mul.f32 %0, %1, 0fBFB8AA3B;" : "=f"(t) : "f"(x));
  asm("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(t));
  return div_full(1.0f, __fadd_rn(1.0f, e));
}

__device__ __forceinline__ uint32_t ue8m0(float amax, float& inv) {
  const float nm = __fmul_rn(amax, 0.002232142857142857f);
  const uint32_t bits = __float_as_uint(nm);
  const uint32_t exp = (bits >> 23) & 255u;
  const uint32_t mant = bits & 0x7FFFFFu;
  uint32_t bump = mant != 0 ? 1u : 0u;
  if (exp == 0 && mant <= 0x400000u) bump = 0;
  uint32_t e = min(exp + bump, 254u);
  if (nm <= 0.0f) e = 0;
  inv = e == 0 ? 0.0f : __uint_as_float((254u - e) << 23);
  return e;
}

__device__ __forceinline__ uint16_t fp8x2(float a, float b) {
  uint16_t r;
  asm("{\n.reg .b16 t;\ncvt.rn.satfinite.e4m3x2.f32 t, %2, %1;\nmov.b16 %0, t;\n}"
      : "=h"(r)
      : "f"(a), "f"(b));
  return r;
}

// 64 threads (nt 0..63): thread nt owns row nt / 4, columns 32 (nt % 4) + [0, 32)
// (one MXFP8 block). `x(row, col)` reads the bf16 output tile. Rows >= valid are
// not written; the 4 lanes of a row are adjacent (shuffle groups of 4).
template <class XTile>
__device__ __forceinline__ void tile(XTile const& x, int valid, int64_t tok0,
                                     int head, NormMxArgs const& a, int nt) {
  const int row = nt / 4, blk = nt % 4;
  const int c0 = blk * 32;
  float ss = 0.0f;
#pragma unroll 8
  for (int c = 0; c < 32; ++c) {
    const float v = float(x(row, c0 + c));
    ss = __fadd_rn(ss, __fmul_rn(v, v));
  }
  ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, 1));
  ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, 2));
  if (row >= valid) return;
  const float var = div_full(ss, 128.0f);
  const float rstd = div_full(1.0f, sqrt_approx(__fadd_rn(var, a.eps)));
  const int64_t tok = tok0 + row;
  const __nv_bfloat16* g = static_cast<const __nv_bfloat16*>(a.g) +
                           tok * a.g_ld + head * 128 + c0;
  auto y_at = [&](int c) {
    float y = __fmul_rn(__fmul_rn(float(x(row, c0 + c)), rstd), a.w[c0 + c]);
    y = __fmul_rn(y, sigmoid_triton(__bfloat162float(g[c])));
    return __bfloat162float(__float2bfloat16_rn(y));
  };
  float amax = 0.0f;
#pragma unroll 8
  for (int c = 0; c < 32; ++c) amax = fmaxf(amax, fabsf(y_at(c)));
  float inv;
  const uint32_t e = ue8m0(amax, inv);
  uint32_t words[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint16_t lo = fp8x2(__fmul_rn(y_at(4 * i), inv), __fmul_rn(y_at(4 * i + 1), inv));
    const uint16_t hi = fp8x2(__fmul_rn(y_at(4 * i + 2), inv), __fmul_rn(y_at(4 * i + 3), inv));
    words[i] = uint32_t(lo) | (uint32_t(hi) << 16);
  }
  uint4* qd = reinterpret_cast<uint4*>(static_cast<uint8_t*>(a.q) + tok * a.q_ld +
                                       head * 128 + c0);
  qd[0] = make_uint4(words[0], words[1], words[2], words[3]);
  qd[1] = make_uint4(words[4], words[5], words[6], words[7]);
  const int64_t col = int64_t(head) * 4 + blk;
  const int64_t off = (col % 4) + (col / 4) * 512 + (tok % 32) * 16 +
                      ((tok % 128) / 32) * 4 + (tok / 128) * (128 * int64_t(a.sf_cols));
  static_cast<uint8_t*>(a.sf)[off] = uint8_t(e);
}

}  // namespace norm_mx
