// Kimi-K3: per-element pieces of the KDA output gate norm (FusedRMSNormGated,
// sigmoid, D = 128) and the MXFP8 quantization of the o_proj input
// (o_norm_mx.cu). Per row (token, head): ops/kda_norm.py
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

}  // namespace norm_mx
