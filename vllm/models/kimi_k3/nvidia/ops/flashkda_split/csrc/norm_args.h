// Kimi-K3: arguments of the KDA gated o_norm + MXFP8 quant kernel
// (smxx/o_norm_mx.cu). Host-safe: no device code.
#pragma once

#include <cstdint>

// o_norm (FusedRMSNormGated, sigmoid, D = 128) of each output row and its MXFP8
// quantization (UE8M0 per 32, F8_128x4 swizzled scales): the o_proj input.
struct NormMxArgs {
  const void* g;  // bf16 output gate rows (tokens, >= H * 128), row stride g_ld
  int64_t g_ld;
  const float* w;  // [128] norm weight
  float eps;
  void* q;  // fp8 e4m3 (tokens, H * 128), row stride q_ld
  int64_t q_ld;
  void* sf;     // uint8 swizzled scales
  int sf_cols;  // H * 128 / 32
};
