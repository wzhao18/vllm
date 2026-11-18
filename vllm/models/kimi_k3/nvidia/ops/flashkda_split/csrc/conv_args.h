// Kimi-K3: arguments of the short convolution fused into FlashKDA's prepare
// kernel (smxx/conv_fused.cuh), and of the gated RMSNorm + MXFP8 quant fused
// into the recurrence epilogue (smxx/norm_mx.cuh). Host-safe: no device code.
#pragma once

#include <cstdint>

struct ConvFuse {
  const void* raw;  // bf16 (tokens, >= 3P) packed Q|K|V raw inputs
  int64_t ldx;      // raw row stride (elements)
  const float* w;   // (3P, 4) FP32 conv weights, contiguous
  void* state;      // (lines, 3P, >= 3), bf16 or FP32
  int64_t st_s0, st_s1, st_s2;
  const int32_t* cache_indices;   // [N] conv-state line per sequence
  const bool* has_initial_state;  // [N]
  void* v_out;  // bf16 (tokens, H, 128) dense conv output of the V plane
  int P;              // channels per plane (H * 128)
  int null_block_id;  // padding line (no state access), or -1
  int state_fp32;     // conv state dtype: 1 = FP32, 0 = bf16
};

// o_norm (FusedRMSNormGated, sigmoid, D = 128) of each output row and its MXFP8
// quantization (UE8M0 per 32, F8_128x4 swizzled scales): the o_proj input.
struct NormMxArgs {
  const void* g;      // bf16 output gate rows (tokens, >= H * 128), row stride g_ld
  int64_t g_ld;
  const float* w;     // [128] norm weight
  float eps;
  void* q;            // fp8 e4m3 (tokens, H * 128), row stride q_ld
  int64_t q_ld;
  void* sf;           // uint8 swizzled scales
  int sf_cols;        // H * 128 / 32
};
