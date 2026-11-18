// Kimi-K3: the KDA prefill short convolution (width 4, SiLU, no bias, FP32
// weights) fused into FlashKDA's prepare kernel (K1). Arithmetic is
// ops/kda_qkv_conv/kda_qkv_conv.cu's FP32-weight path, which is the stock
// Triton causal_conv1d_fn bit for bit: an FP32 accumulator from +0, oldest tap
// first, one FMA per tap, then Triton's SiLU (ex2.approx + div.full) and a
// round-to-nearest-even bf16 store. The conv state (lines, 3P, >= 3) holds the
// last 3 raw inputs (positions -3..-1) of a sequence; the CTA of a sequence's
// first tile is its only reader and writes the new state, as kda_qkv_conv.
#pragma once

#include <cstdint>
#include <cuda_bf16.h>

#include "conv_args.h"

namespace conv_fused {

constexpr int kVec = 8;

__device__ __forceinline__ float ex2_ftz(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float rcp_ftz(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// The stock Triton SiLU, acc / (1 + exp(-acc)), as Triton lowers it for FP32.
__device__ __forceinline__ float silu_triton(float acc) {
  float t, e, d, r;
  asm("mul.f32 %0, %1, 0fBFB8AA3B;" : "=f"(t) : "f"(acc));
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(t));
  asm("add.f32 %0, %1, 0f3F800000;" : "=f"(d) : "f"(e));
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(acc), "f"(d));
  return r;
}

__device__ __forceinline__ uint32_t pack2(float a, float b) {
  __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}

// kda_qkv_conv.cu silu8: div.full.f32 equals a * rcp.approx(d) for
// 1 + e <= 2^126; larger d redo the 8 values with div.full itself.
__device__ __forceinline__ uint4 silu8(const float (&acc)[kVec]) {
  float o[kVec];
  float dmax = 0.0f;
#pragma unroll
  for (int q = 0; q < kVec / 2; ++q) {
    const float2 a = make_float2(acc[2 * q], acc[2 * q + 1]);
    const float2 t = __fmul2_rn(
        a, make_float2(-1.44269502162933349609375f, -1.44269502162933349609375f));
    const float2 d = __fadd2_rn(make_float2(ex2_ftz(t.x), ex2_ftz(t.y)),
                                make_float2(1.0f, 1.0f));
    const float2 r = __fmul2_rn(a, make_float2(rcp_ftz(d.x), rcp_ftz(d.y)));
    o[2 * q] = r.x;
    o[2 * q + 1] = r.y;
    dmax = fmaxf(dmax, fmaxf(d.x, d.y));
  }
  if (dmax > 0x1p126f) {
#pragma unroll
    for (int i = 0; i < kVec; ++i) o[i] = silu_triton(acc[i]);
  }
  return make_uint4(pack2(o[0], o[1]), pack2(o[2], o[3]), pack2(o[4], o[5]),
                    pack2(o[6], o[7]));
}

__device__ __forceinline__ uint4 conv_row_fma(const float (&h0)[kVec],
                                              const float (&h1)[kVec],
                                              const float (&h2)[kVec],
                                              const float (&cur)[kVec],
                                              const float (&w)[4][kVec]) {
  float acc[kVec];
#pragma unroll
  for (int q = 0; q < kVec / 2; ++q) {
    float2 a = __ffma2_rn(make_float2(h0[2 * q], h0[2 * q + 1]),
                          make_float2(w[0][2 * q], w[0][2 * q + 1]),
                          make_float2(0.0f, 0.0f));
    a = __ffma2_rn(make_float2(h1[2 * q], h1[2 * q + 1]),
                   make_float2(w[1][2 * q], w[1][2 * q + 1]), a);
    a = __ffma2_rn(make_float2(h2[2 * q], h2[2 * q + 1]),
                   make_float2(w[2][2 * q], w[2][2 * q + 1]), a);
    a = __ffma2_rn(make_float2(cur[2 * q], cur[2 * q + 1]),
                   make_float2(w[3][2 * q], w[3][2 * q + 1]), a);
    acc[2 * q] = a.x;
    acc[2 * q + 1] = a.y;
  }
  return silu8(acc);
}

__device__ __forceinline__ void unpack8(uint4 v, float (&f)[kVec]) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    f[2 * i] = __uint_as_float(w[i] << 16);
    f[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
  }
}

__device__ __forceinline__ uint4 ld_nc_v4(const __nv_bfloat16* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

__device__ __forceinline__ float state_ld(const ConvFuse& c, int64_t off) {
  if (c.state_fp32) return static_cast<const float*>(c.state)[off];
  return __bfloat162float(static_cast<const __nv_bfloat16*>(c.state)[off]);
}

__device__ __forceinline__ void state_st(const ConvFuse& c, int64_t off,
                                         float v) {
  if (c.state_fp32) {
    static_cast<float*>(c.state)[off] = v;
  } else {
    // exact: v is a bf16 value
    static_cast<__nv_bfloat16*>(c.state)[off] = __float2bfloat16_rn(v);
  }
}

// One 16-token tile of head `head` for the three planes: Q and K (conv + SiLU,
// bf16) into the row-major [16, 128] shared tiles, V into c.v_out. Rows past
// the sequence end are zero in the Q / K tiles and not written to v_out.
// Then, in the sequence's first tile only, the new conv state. 128 threads:
// threads 0..95 each run one (plane, 8 channels, 8 rows) stream.
template <int CHUNK>
__device__ __forceinline__ void tile_conv(const ConvFuse& c, int head,
                                          int seq_idx, int64_t bos,
                                          int seq_len, int local_t,
                                          __nv_bfloat16* q_smem,
                                          __nv_bfloat16* k_smem, int H) {
  static_assert(CHUNK == 16, "two 8-row halves per tile");
  const int tid = threadIdx.x;
  const int line = c.cache_indices[seq_idx];
  const bool skip = c.null_block_id >= 0 && line == c.null_block_id;
  const bool load_init = !skip && c.has_initial_state[seq_idx];
  if (tid < 96) {
    const int plane = tid / 32;
    const int half = (tid / 16) % 2;
    const int cv = tid % 16;
    const int ch = plane * c.P + head * 128 + cv * kVec;
    const __nv_bfloat16* raw = static_cast<const __nv_bfloat16*>(c.raw);
    const __nv_bfloat16* xcol = raw + ch;
    float w[4][kVec];
    const float4* wv = reinterpret_cast<const float4*>(c.w + int64_t(ch) * 4);
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      const float4 v = wv[i];
      w[0][i] = v.x;
      w[1][i] = v.y;
      w[2][i] = v.z;
      w[3][i] = v.w;
    }
    const int r0 = half * 8;
    const int p0 = local_t * CHUNK + r0;  // position of the first row
    float h[3][kVec];
#pragma unroll
    for (int k = 0; k < 3; ++k) {
      const int pos = p0 - 3 + k;
      if (pos >= 0) {
        unpack8(ld_nc_v4(xcol + (bos + pos) * c.ldx), h[k]);
      } else {
        const int64_t so =
            int64_t(line) * c.st_s0 + int64_t(ch) * c.st_s1 + (3 + pos) * c.st_s2;
#pragma unroll
        for (int i = 0; i < kVec; ++i)
          h[k][i] = load_init ? state_ld(c, so + i * c.st_s1) : 0.0f;
      }
    }
    __nv_bfloat16* tile = plane == 0 ? q_smem : k_smem;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      const int pos = p0 + r;
      uint4 o = make_uint4(0, 0, 0, 0);
      float cur[kVec];
      if (pos < seq_len) {
        unpack8(ld_nc_v4(xcol + (bos + pos) * c.ldx), cur);
        o = conv_row_fma(h[0], h[1], h[2], cur, w);
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
          h[0][i] = h[1][i];
          h[1][i] = h[2][i];
          h[2][i] = cur[i];
        }
        if (plane == 2 && !skip) {
          *reinterpret_cast<uint4*>(static_cast<__nv_bfloat16*>(c.v_out) + ((bos + pos) * H + head) * 128 +
                                    cv * kVec) = o;
        }
      }
      if (plane < 2) {
        *reinterpret_cast<uint4*>(tile + (r0 + r) * 128 + cv * kVec) = o;
      }
    }
  }
  if (local_t == 0 && !skip) {
    // Every history read of the old state above happens before the writes.
    __syncthreads();
    if (tid < 48) {
      const int plane = tid / 16;
      const int cv = tid % 16;
      const int ch = plane * c.P + head * 128 + cv * kVec;
      const int64_t sb = int64_t(line) * c.st_s0 + int64_t(ch) * c.st_s1;
      float old[3][kVec];
#pragma unroll
      for (int k = 0; k < 3; ++k)
#pragma unroll
        for (int i = 0; i < kVec; ++i)
          old[k][i] = load_init ? state_ld(c, sb + i * c.st_s1 + k * c.st_s2)
                                : 0.0f;
      float ns[3][kVec];
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        const int src = seq_len - 3 + j;
        if (src >= 0) {
          unpack8(ld_nc_v4(static_cast<const __nv_bfloat16*>(c.raw) + ch +
                           (bos + src) * c.ldx),
                  ns[j]);
        } else {
#pragma unroll
          for (int i = 0; i < kVec; ++i) ns[j][i] = old[j + seq_len][i];
        }
      }
#pragma unroll
      for (int j = 0; j < 3; ++j)
#pragma unroll
        for (int i = 0; i < kVec; ++i)
          state_st(c, sb + i * c.st_s1 + j * c.st_s2, ns[j][i]);
    }
  }
}

}  // namespace conv_fused
