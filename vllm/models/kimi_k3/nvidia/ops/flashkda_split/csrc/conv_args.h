// Kimi-K3: arguments of the short convolution fused into FlashKDA's prepare
// kernel (smxx/conv_fused.cuh). Host-safe: no device code.
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
