/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 */

// Kimi K3.1 prefill AttnRes, two tokens in flight per CTA (P-ATTNRES).
//
// Same contract and math as attn_res_mxfp8.cu (online softmax over the
// num_blocks block rows + the prefix(+delta) row, RMSNorm logits, output
// RMSNorm, MXFP8 E4M3 + F8_128x4 E8M0 output, optional BF16 output); only the
// work decomposition differs (reduction order may differ from the native
// kernel, every stage keeps its precision: BF16 inputs, BF16 delta add, FP32
// statistics / weights / accumulation, BF16-rounded output quantized).
//
// Why: the shipped kernel runs ONE token at a time per SM with all 8 consumer
// warps splitting the 7168 columns, so every source chunk costs two
// 256-thread barrier round trips with only 2 warps per scheduler to hide
// them: it is latency-bound at ~10 TB/s (the 93-layer prefill forward reads
// ~6.3 rows per call). Here the 8 consumer warps form G=2 independent
// 128-thread groups; each group streams its own tokens through its own TMA
// ring (one producer lane per group), synchronises on its own named barrier,
// and caches its FP32 rows in its own TMEM columns. Two tokens are in flight
// per SM and the two groups' barrier waits overlap each other's math.
//
// Integration: H = 7168, 0 <= num_blocks <= 8, token-major block storage.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <type_traits>

using bf16_t = __nv_bfloat16;

namespace k31pf2 {

// ---- MXFP8 quantization primitives (bit-exact copies of attn_res_mxfp8.cu /
// ops/mxfp8_act.py).
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

__device__ __forceinline__ int64_t swizzled_sf_offset(int64_t row, int col,
                                                      int padded_cols) {
  return (int64_t)(col & 3) + (int64_t)(col >> 2) * 512 + (row & 31) * 16 +
         ((row & 127) >> 5) * 4 + (row >> 7) * 128 * (int64_t)padded_cols;
}

// NV BF16-rounded values of columns [col, col + NV) per lane; 32 / NV adjacent
// lanes form one 32-wide block (all 32 lanes call).
template <int NV>
__device__ __forceinline__ void quant_store(const float (&v)[NV],
                                            uint8_t* __restrict__ q_row,
                                            uint8_t* __restrict__ sf,
                                            int64_t row, int col,
                                            int padded_cols) {
  static_assert(NV == 4 || NV == 8);
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < NV; ++i) amax = fmaxf(amax, fabsf(v[i]));
#pragma unroll
  for (int m = 1; m < 32 / NV; m <<= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, m));
  }
  const uint32_t e = float_to_ue8m0(amax * (1.0f / 448.0f));
  const float inv = ue8m0_to_inv_scale(e);
  uint32_t w[NV / 4];
#pragma unroll
  for (int i = 0; i < NV / 4; ++i) {
    w[i] = (uint32_t)fp8x2(v[4 * i] * inv, v[4 * i + 1] * inv) |
           ((uint32_t)fp8x2(v[4 * i + 2] * inv, v[4 * i + 3] * inv) << 16);
  }
  if constexpr (NV == 8) {
    *reinterpret_cast<uint2*>(q_row + col) = make_uint2(w[0], w[1]);
  } else {
    *reinterpret_cast<uint32_t*>(q_row + col) = w[0];
  }
  if ((col & 31) == 0) {
    sf[swizzled_sf_offset(row, col >> 5, padded_cols)] = (uint8_t)e;
  }
}

// ---- sm_100 family primitives.
__device__ __forceinline__ void mbar_init(uint64_t& b, int count) {
  uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(&b));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(a),
               "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t& b, uint32_t bytes) {
  uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(&b));
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(a),
      "r"(bytes)
      : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t& b, int phase) {
  uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(&b));
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@p bra DONE;\n"
      "bra WAIT;\n"
      "DONE:\n"
      "}\n" ::"r"(a),
      "r"(phase)
      : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t& b) {
  uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(&b));
  asm volatile(
      "{\n"
      ".reg .b64 state;\n"
      "mbarrier.arrive.shared::cta.b64 state, [%0];\n"
      "}\n" ::"r"(a)
      : "memory");
}
__device__ __forceinline__ void cp_bulk(void* dst, const void* src,
                                        uint32_t bytes, uint64_t& b) {
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], "
      "[%1], %2, [%3];\n" ::"r"(
          static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
      "l"(src), "r"(bytes),
      "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&b)))
      : "memory");
}
__device__ __forceinline__ void group_bar(int g) {
  // Named barriers 8 + g, 128 threads (one consumer group).
  asm volatile("bar.sync %0, 128;" ::"r"(8 + g) : "memory");
}
__device__ __forceinline__ void tmem_alloc(uint32_t* dst, int cols) {
  asm volatile(
      "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(
          static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
      "r"(cols));
}
__device__ __forceinline__ void tmem_dealloc(uint32_t p, int cols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(p),
               "r"(cols));
}
__device__ __forceinline__ void tmem_relinquish() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}
__device__ __forceinline__ void tmem_st8(uint32_t addr, const float2 (&f)[4]) {
  const uint32_t* v = reinterpret_cast<const uint32_t*>(f);
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x8.b32 [%8], {%0, %1, %2, %3, %4, %5, "
      "%6, %7};\n" ::"r"(v[0]),
      "r"(v[1]), "r"(v[2]), "r"(v[3]), "r"(v[4]), "r"(v[5]), "r"(v[6]),
      "r"(v[7]), "r"(addr));
}
__device__ __forceinline__ void tmem_ld8(uint32_t addr, float2 (&f)[4]) {
  uint32_t* v = reinterpret_cast<uint32_t*>(f);
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, "
      "%7}, [%8];\n"
      : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]),
        "=r"(v[5]), "=r"(v[6]), "=r"(v[7])
      : "r"(addr));
}
__device__ __forceinline__ void tmem_st4(uint32_t addr, const float2 (&f)[4]) {
  const uint32_t* v = reinterpret_cast<const uint32_t*>(f);
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0, %1, %2, %3};\n" ::"r"(
          v[0]),
      "r"(v[1]), "r"(v[2]), "r"(v[3]), "r"(addr));
}
__device__ __forceinline__ void tmem_ld4(uint32_t addr, float2 (&f)[4]) {
  uint32_t* v = reinterpret_cast<uint32_t*>(f);
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3])
               : "r"(addr));
}
__device__ __forceinline__ float2 f2_add(float2 a, float2 b) {
  float2 r;
  asm("add.rn.f32x2 %0, %1, %2;\n"
      : "=l"(reinterpret_cast<uint64_t&>(r))
      : "l"(reinterpret_cast<uint64_t const&>(a)),
        "l"(reinterpret_cast<uint64_t const&>(b)));
  return r;
}
__device__ __forceinline__ void tmem_wait_st() {
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tmem_wait_ld() {
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ float2 f2_fma(float2 a, float2 b, float2 c) {
  float2 r;
  asm("fma.rn.f32x2 %0, %1, %2, %3;\n"
      : "=l"(reinterpret_cast<uint64_t&>(r))
      : "l"(reinterpret_cast<uint64_t const&>(a)),
        "l"(reinterpret_cast<uint64_t const&>(b)),
        "l"(reinterpret_cast<uint64_t const&>(c)));
  return r;
}
__device__ __forceinline__ float2 f2_mul(float2 a, float2 b) {
  float2 r;
  asm("mul.f32x2 %0, %1, %2;\n"
      : "=l"(reinterpret_cast<uint64_t&>(r))
      : "l"(reinterpret_cast<uint64_t const&>(a)),
        "l"(reinterpret_cast<uint64_t const&>(b)));
  return r;
}

constexpr int H = 7168;
constexpr int G = 2;                 // consumer groups = tokens in flight
constexpr int GT = 128;              // threads per group
constexpr int GW = GT / 32;          // warps per group
constexpr int VE = 28;               // columns per native (virtual) thread
constexpr int NC = 3;                // rows per ring stage
constexpr int D = 2;                 // ring stages per group
constexpr int BLK = 32 + G * GT;     // producer warp + consumer groups
constexpr int ROW_BYTES = H * 2;
constexpr int STAGE_ROWS = NC + 1;   // + the delta row
constexpr int TMEM_COLS_G = 2 * 4 * NC * 8;  // 192
constexpr int TMEM_COLS = 512;
static_assert(H == 2 * GT * VE);
static_assert(G * TMEM_COLS_G <= TMEM_COLS);

struct Plan {
  alignas(16) uint64_t ready[G][D];
  alignas(16) uint64_t consumed[G][D];
  float2 stats[G][2][2 * GW][NC];
  float ostats[G][2 * GW];
  int tok[G][D];
  uint32_t tmem_base;
};
constexpr size_t RING_BYTES = (size_t)G * D * STAGE_ROWS * ROW_BYTES;
constexpr size_t SMEM_BYTES = (RING_BYTES + sizeof(Plan) + 15) & ~size_t(15);
static_assert(SMEM_BYTES <= 232448, "AttnRes ring exceeds 227 KiB");

template <int N, bool OUT_NORM, bool QUANT, bool WRITE_BF16>
__global__ void __launch_bounds__(BLK, 1) attn_res_pf_kernel(
    bf16_t* __restrict__ block_res, bf16_t* __restrict__ layer_res,
    const bf16_t* __restrict__ delta, const bf16_t* __restrict__ res_w,
    const bf16_t* __restrict__ rms_w, bf16_t* __restrict__ output, int T,
    long long block_stride_m, long long block_stride_r, float rms_eps,
    const bf16_t* __restrict__ out_norm_w, float out_norm_eps,
    int write_idx, uint8_t* __restrict__ q_out, uint8_t* __restrict__ sf_out,
    int sf_padded_cols, const int8_t* __restrict__ sm_die,
    int* __restrict__ claim, bf16_t* prefix_out) {
  static_assert(!QUANT || OUT_NORM);
  static_assert(QUANT || WRITE_BF16);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1100
  constexpr float LOG2_E = 1.4426950408889634f;
  constexpr int NCH = (N + NC - 1) / NC;
  extern __shared__ __align__(128) char smem[];
  bf16_t* ring = reinterpret_cast<bf16_t*>(smem);
  Plan& plan = *reinterpret_cast<Plan*>(smem + RING_BYTES);

  const int tid = threadIdx.x;
  const int wid = tid >> 5;
  const int lane = tid & 31;
  const bool has_delta = delta != nullptr;
  // The updated prefix (prefix + delta) goes to prefix_out (in place when
  // null); a prefix_out other than the input also takes a delta-less prefix.
  bf16_t* const prefix_dst = prefix_out != nullptr ? prefix_out : layer_res;
  const bool copy_prefix = prefix_dst != layer_res;
  const int tstride = gridDim.x * G;

  if (tid == 0) {
#pragma unroll
    for (int g = 0; g < G; g++) {
#pragma unroll
      for (int s = 0; s < D; s++) {
        mbar_init(plan.ready[g][s], 1);
        mbar_init(plan.consumed[g][s], GW);
      }
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (wid == 1) {
    tmem_alloc(&plan.tmem_base, TMEM_COLS);
    tmem_relinquish();
  }
  __syncthreads();

  if (wid == 0) {
    // ---- producer: lane g feeds group g's ring. The token of every ring
    // stage that starts a token is published in plan.tok (-1 ends the group).
    // Static mode: token blockIdx * G + g + i * G * gridDim. Die-local mode
    // (sm_die != nullptr; the block rows of token t live on die (t >> 10) & 1):
    // the CTA claims, two at a time, tokens homed on its own SM's die and then
    // steals the other die's leftovers. Claims start after the grid
    // dependency, i.e. after the previous AttnRes reset the counters.
    if (lane < G) {
      const int g = lane;
      bool waited = false;
      int gci = 0;
      int die = 0;
      if (sm_die != nullptr) {
        uint32_t smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        die = sm_die[smid & 1023];
        asm volatile("griddepcontrol.wait;" ::: "memory");
        waited = true;
      }
      const int nsb = (T + 1023) >> 10;
      auto die_count = [&](int d) -> int {
        int full = (nsb - d + 1) >> 1;  // super-blocks k < nsb with k % 2 == d
        int c = full << 10;
        if (((nsb - 1) & 1) == d) c -= (nsb << 10) - T;
        return c;
      };
      const int cnt_own = sm_die != nullptr ? die_count(die) : 0;
      const int cnt_oth = sm_die != nullptr ? die_count(die ^ 1) : 0;
      auto die_token = [&](int d, int j) -> int {
        return ((((j >> 10) << 1) + d) << 10) | (j & 1023);
      };
      int st_next = blockIdx.x * G + g;
      int pend = -1;      // second token of the last claimed pair
      bool own_done = false;
      auto next_token = [&]() -> int {
        if (sm_die == nullptr) {
          int t = st_next;
          st_next += tstride;
          return t < T ? t : -1;
        }
        if (pend >= 0) {
          int t = pend;
          pend = -1;
          return t;
        }
        if (!own_done) {
          int j = atomicAdd(&claim[die], 2);
          if (j < cnt_own) {
            if (j + 1 < cnt_own) pend = die_token(die, j + 1);
            return die_token(die, j);
          }
          own_done = true;
        }
        int j = atomicAdd(&claim[die ^ 1], 2);
        if (j < cnt_oth) {
          if (j + 1 < cnt_oth) pend = die_token(die ^ 1, j + 1);
          return die_token(die ^ 1, j);
        }
        return -1;
      };
      for (;;) {
        const int t = next_token();
        if (t < 0) {
          const int slot = gci % D;
          mbar_wait(plan.consumed[g][slot], ((gci / D) & 1) ^ 1);
          plan.tok[g][slot] = -1;
          mbar_expect_tx(plan.ready[g][slot], 0);
          break;
        }
#pragma unroll 1
        for (int ci = 0; ci < NCH; ci++, gci++) {
          const int ns = ci * NC;
          const int an = min(NC, N - ns);
          const int slot = gci % D;
          const int par = (gci / D) & 1;
          mbar_wait(plan.consumed[g][slot], par ^ 1);
          if (ci == 0) plan.tok[g][slot] = t;
          const bool has_prefix = ns + an == N;  // the prefix is source N - 1
          uint32_t bytes = an * ROW_BYTES;
          if (has_prefix && has_delta) bytes += ROW_BYTES;
          mbar_expect_tx(plan.ready[g][slot], bytes);
          bf16_t* stage = ring + (size_t)((g * D + slot) * STAGE_ROWS) * H;
          for (int n = 0; n < an; n++) {
            const int src = ns + n;
            const bf16_t* p;
            if (src < N - 1) {
              p = block_res + (long long)t * block_stride_m +
                  (long long)src * block_stride_r;
            } else {
              if (!waited) {
                asm volatile("griddepcontrol.wait;" ::: "memory");
                waited = true;
              }
              p = layer_res + (long long)t * H;
            }
            cp_bulk(stage + (size_t)n * H, p, ROW_BYTES, plan.ready[g][slot]);
          }
          if (has_prefix && has_delta) {
            cp_bulk(stage + (size_t)NC * H, delta + (long long)t * H,
                    ROW_BYTES, plan.ready[g][slot]);
          }
        }
      }
    }
    __syncwarp();
    if (lane == 0) {
      asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
  } else {
    // ---- consumers. Group g (4 warps) processes its own tokens. Bitwise
    // parity with the native kernel: thread ct of a group does the work of
    // the native kernel's two consumer threads (native group v = 0, 1; same
    // ct): slices si = 0..2 cover columns (2 si + v) * 1024 + ct * 8 + [0, 8),
    // slice 3 covers 6144 + v * 512 + ct * 4 + [0, 4). Per virtual thread the
    // FMA order, the warp shuffle tree and the 8-warp combine tree (native
    // warp v * 4 + gw) are the native kernel's, so every statistic and output
    // is bit-identical.
    const int g = (wid - 1) / GW;
    const int ct = tid - 32 - g * GT;
    const int gw = ct >> 5;
    const uint32_t vt = plan.tmem_base + g * TMEM_COLS_G;
    auto col_of = [&](int v, int si) -> int {
      return si < 3 ? (2 * si + v) * 1024 + ct * 8 : 6144 + v * 512 + ct * 4;
    };
    auto tcol = [&](int v, int si, int n) -> uint32_t {
      return (uint32_t)(((v * 4 + si) * NC + n) * 8);
    };

    float qv[2][VE];
#pragma unroll
    for (int v = 0; v < 2; v++) {
#pragma unroll
      for (int si = 0; si < 4; si++) {
        const int col = col_of(v, si);
        const int w = si < 3 ? 8 : 4;
        int4 rv = make_int4(0, 0, 0, 0), sv = make_int4(0, 0, 0, 0);
        if (w == 8) {
          rv = *reinterpret_cast<const int4*>(rms_w + col);
          sv = *reinterpret_cast<const int4*>(res_w + col);
        } else {
          int2 r2v = *reinterpret_cast<const int2*>(rms_w + col);
          int2 s2v = *reinterpret_cast<const int2*>(res_w + col);
          rv.x = r2v.x; rv.y = r2v.y;
          sv.x = s2v.x; sv.y = s2v.y;
        }
        auto* r2 = reinterpret_cast<__nv_bfloat162*>(&rv);
        auto* s2 = reinterpret_cast<__nv_bfloat162*>(&sv);
#pragma unroll
        for (int j = 0; j < 4; j++) {
          if (2 * j >= w) continue;
          float2 rf = __bfloat1622float2(r2[j]);
          float2 sf = __bfloat1622float2(s2[j]);
          qv[v][si * 8 + 2 * j] = rf.x * sf.x;
          qv[v][si * 8 + 2 * j + 1] = rf.y * sf.y;
        }
      }
    }

    float acc[2][VE];
    int gci = 0;
    for (;;) {
      {
        const int slot0 = gci % D;
        mbar_wait(plan.ready[g][slot0], (gci / D) & 1);
      }
      const int t = plan.tok[g][gci % D];
      if (t < 0) break;
      float m_run = -FLT_MAX;
      float s_run = 0.f;
#pragma unroll
      for (int v = 0; v < 2; v++)
#pragma unroll
        for (int i = 0; i < VE; i++) acc[v][i] = 0.f;

#pragma unroll
      for (int ci = 0; ci < NCH; ci++, gci++) {
        const int ns = ci * NC;
        const int an = (N - ns) < NC ? (N - ns) : NC;
        const int slot = gci % D;
        const int par = (gci / D) & 1;
        const bf16_t* stage = ring + (size_t)((g * D + slot) * STAGE_ROWS) * H;
        if (ci > 0) mbar_wait(plan.ready[g][slot], par);

        float2 sq[2][NC], dt[2][NC];
#pragma unroll
        for (int v = 0; v < 2; v++)
#pragma unroll
          for (int n = 0; n < NC; n++) {
            sq[v][n] = make_float2(0.f, 0.f);
            dt[v][n] = make_float2(0.f, 0.f);
          }
#pragma unroll
        for (int v = 0; v < 2; v++) {
#pragma unroll
          for (int si = 0; si < 4; si++) {
            const int col = col_of(v, si);
            const int np = si < 3 ? 4 : 2;  // bf16 pairs
#pragma unroll
            for (int n = 0; n < NC; n++) {
              if (n >= an) continue;
              int4 vp = make_int4(0, 0, 0, 0);
              if (np == 4) {
                vp = *reinterpret_cast<const int4*>(stage + (size_t)n * H + col);
              } else {
                int2 h2 = *reinterpret_cast<const int2*>(stage + (size_t)n * H + col);
                vp.x = h2.x; vp.y = h2.y;
              }
              auto* v2 = reinterpret_cast<__nv_bfloat162*>(&vp);
              if (ns + n == N - 1 && (has_delta || copy_prefix || write_idx >= 0)) {
                if (has_delta) {
                  int4 dp = make_int4(0, 0, 0, 0);
                  if (np == 4) {
                    dp = *reinterpret_cast<const int4*>(stage + (size_t)NC * H + col);
                  } else {
                    int2 h2 = *reinterpret_cast<const int2*>(stage + (size_t)NC * H + col);
                    dp.x = h2.x; dp.y = h2.y;
                  }
                  auto* d2 = reinterpret_cast<const __nv_bfloat162*>(&dp);
#pragma unroll
                  for (int j = 0; j < 4; j++)
                    if (j < np) v2[j] = __hadd2(v2[j], d2[j]);
                }
                if (has_delta || copy_prefix) {
                  bf16_t* lp = prefix_dst + (long long)t * H + col;
                  if (np == 4) *reinterpret_cast<int4*>(lp) = vp;
                  else *reinterpret_cast<int2*>(lp) = make_int2(vp.x, vp.y);
                }
                if (write_idx >= 0) {
                  bf16_t* bp = block_res + (long long)t * block_stride_m +
                               (long long)write_idx * block_stride_r + col;
                  if (np == 4) *reinterpret_cast<int4*>(bp) = vp;
                  else *reinterpret_cast<int2*>(bp) = make_int2(vp.x, vp.y);
                }
              }
              float2 f[4];
#pragma unroll
              for (int j = 0; j < 4; j++)
                f[j] = j < np ? __bfloat1622float2(v2[j]) : make_float2(0.f, 0.f);
              if (np == 4) tmem_st8(vt + tcol(v, si, n), f);
              else tmem_st4(vt + tcol(v, si, n), f);
#pragma unroll
              for (int j = 0; j < 4; j++) {
                if (j >= np) continue;
                sq[v][n] = f2_fma(f[j], f[j], sq[v][n]);
                dt[v][n] = f2_fma(f[j], make_float2(qv[v][si * 8 + 2 * j],
                                                    qv[v][si * 8 + 2 * j + 1]),
                                  dt[v][n]);
              }
            }
          }
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(plan.consumed[g][slot]);

#pragma unroll
        for (int v = 0; v < 2; v++) {
          float2 red[NC];
#pragma unroll
          for (int n = 0; n < NC; n++)
            red[n] = make_float2(sq[v][n].x + sq[v][n].y, dt[v][n].x + dt[v][n].y);
#pragma unroll
          for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
            for (int n = 0; n < NC; n++) {
              uint64_t pk = reinterpret_cast<uint64_t&>(red[n]);
              pk = __shfl_xor_sync(0xffffffff, pk, off);
              red[n] = f2_add(red[n], reinterpret_cast<float2&>(pk));
            }
          }
          if (lane == 0) {
#pragma unroll
            for (int n = 0; n < NC; n++)
              plan.stats[g][gci & 1][v * GW + gw][n] = red[n];
          }
        }
        tmem_wait_st();
        group_bar(g);

        float logit[NC];
        float m_chunk = -FLT_MAX;
#pragma unroll
        for (int n = 0; n < NC; n++) {
          // Native combine: shfl_down 4, 2, 1 over the 8 warp totals.
          float2 x[8];
#pragma unroll
          for (int w = 0; w < 8; w++) x[w] = plan.stats[g][gci & 1][w][n];
#pragma unroll
          for (int w = 0; w < 4; w++) { x[w].x += x[w + 4].x; x[w].y += x[w + 4].y; }
#pragma unroll
          for (int w = 0; w < 2; w++) { x[w].x += x[w + 2].x; x[w].y += x[w + 2].y; }
          x[0].x += x[1].x;
          x[0].y += x[1].y;
          const float rsig = rsqrtf(x[0].x / H + rms_eps);
          logit[n] = x[0].y * rsig;
          if (n < an) m_chunk = fmaxf(m_chunk, logit[n]);
        }
        const float m_new = fmaxf(m_run, m_chunk);
        const float corr = exp2f((m_run - m_new) * LOG2_E);
        float wn[NC];
        float w_sum = 0.f;
#pragma unroll
        for (int n = 0; n < NC; n++) {
          wn[n] = 0.f;
          if (n < an) {
            wn[n] = exp2f((logit[n] - m_new) * LOG2_E);
            w_sum += wn[n];
          }
        }
        const float2 corr2 = make_float2(corr, corr);
#pragma unroll
        for (int v = 0; v < 2; v++) {
#pragma unroll
          for (int si = 0; si < 4; si++) {
            const int np = si < 3 ? 4 : 2;
            float2 fc[NC][4];
#pragma unroll
            for (int n = 0; n < NC; n++) {
              if (n >= an) continue;
              if (np == 4) tmem_ld8(vt + tcol(v, si, n), fc[n]);
              else tmem_ld4(vt + tcol(v, si, n), fc[n]);
            }
            tmem_wait_ld();
            float2 a[4];
#pragma unroll
            for (int j = 0; j < 4; j++) {
              if (j >= np) continue;
              a[j] = f2_mul(make_float2(acc[v][si * 8 + 2 * j],
                                        acc[v][si * 8 + 2 * j + 1]),
                            corr2);
            }
#pragma unroll
            for (int n = 0; n < NC; n++) {
              if (n >= an) continue;
              const float2 w2 = make_float2(wn[n], wn[n]);
#pragma unroll
              for (int j = 0; j < 4; j++)
                if (j < np) a[j] = f2_fma(w2, fc[n][j], a[j]);
            }
#pragma unroll
            for (int j = 0; j < 4; j++) {
              if (j >= np) continue;
              acc[v][si * 8 + 2 * j] = a[j].x;
              acc[v][si * 8 + 2 * j + 1] = a[j].y;
            }
          }
        }
        s_run = s_run * corr + w_sum;
        m_run = m_new;
      }

      // ---- epilogue
      if constexpr (OUT_NORM) {
#pragma unroll
        for (int v = 0; v < 2; v++) {
          float2 o2 = make_float2(0.f, 0.f);
#pragma unroll
          for (int si = 0; si < 4; si++) {
            const int np = si < 3 ? 4 : 2;
#pragma unroll
            for (int j = 0; j < 4; j++) {
              if (j >= np) continue;
              float2 x = make_float2(acc[v][si * 8 + 2 * j], acc[v][si * 8 + 2 * j + 1]);
              o2 = f2_fma(x, x, o2);
            }
          }
          float osq = o2.x + o2.y;
#pragma unroll
          for (int off = 16; off > 0; off >>= 1)
            osq += __shfl_xor_sync(0xffffffff, osq, off);
          if (lane == 0) plan.ostats[g][v * GW + gw] = osq;
        }
        group_bar(g);
        float x[8];
#pragma unroll
        for (int w = 0; w < 8; w++) x[w] = plan.ostats[g][w];
#pragma unroll
        for (int w = 0; w < 4; w++) x[w] += x[w + 4];
#pragma unroll
        for (int w = 0; w < 2; w++) x[w] += x[w + 2];
        x[0] += x[1];
        const float orsig = rsqrtf(x[0] / H + out_norm_eps * s_run * s_run);
#pragma unroll
        for (int v = 0; v < 2; v++) {
#pragma unroll
          for (int si = 0; si < 4; si++) {
            const int col = col_of(v, si);
            if (si < 3) {
              int4 wv = __ldg(reinterpret_cast<const int4*>(out_norm_w + col));
              const bf16_t* wb = reinterpret_cast<const bf16_t*>(&wv);
              uint4 packed;
              bf16_t* ob = reinterpret_cast<bf16_t*>(&packed);
#pragma unroll
              for (int j = 0; j < 8; j++)
                ob[j] = __float2bfloat16(acc[v][si * 8 + j] * orsig *
                                         __bfloat162float(wb[j]));
              if constexpr (WRITE_BF16)
                *reinterpret_cast<uint4*>(output + (long long)t * H + col) = packed;
              if constexpr (QUANT) {
                float qf[8];
#pragma unroll
                for (int j = 0; j < 8; j++) qf[j] = __bfloat162float(ob[j]);
                quant_store<8>(qf, q_out + (long long)t * H, sf_out, t, col,
                               sf_padded_cols);
              }
            } else {
              int2 wv = __ldg(reinterpret_cast<const int2*>(out_norm_w + col));
              const bf16_t* wb = reinterpret_cast<const bf16_t*>(&wv);
              uint2 packed;
              bf16_t* ob = reinterpret_cast<bf16_t*>(&packed);
#pragma unroll
              for (int j = 0; j < 4; j++)
                ob[j] = __float2bfloat16(acc[v][24 + j] * orsig *
                                         __bfloat162float(wb[j]));
              if constexpr (WRITE_BF16)
                *reinterpret_cast<uint2*>(output + (long long)t * H + col) = packed;
              if constexpr (QUANT) {
                float qf[4];
#pragma unroll
                for (int j = 0; j < 4; j++) qf[j] = __bfloat162float(ob[j]);
                quant_store<4>(qf, q_out + (long long)t * H, sf_out, t, col,
                               sf_padded_cols);
              }
            }
          }
        }
      } else {
        const float inv_s = 1.f / s_run;
        const float2 inv2 = make_float2(inv_s, inv_s);
#pragma unroll
        for (int v = 0; v < 2; v++) {
#pragma unroll
          for (int si = 0; si < 4; si++) {
            const int col = col_of(v, si);
            const int np = si < 3 ? 4 : 2;
            uint4 packed;
            auto* o2 = reinterpret_cast<__nv_bfloat162*>(&packed);
#pragma unroll
            for (int j = 0; j < 4; j++)
              if (j < np)
                o2[j] = __float22bfloat162_rn(f2_mul(
                    make_float2(acc[v][si * 8 + 2 * j], acc[v][si * 8 + 2 * j + 1]),
                    inv2));
            if (np == 4)
              *reinterpret_cast<uint4*>(output + (long long)t * H + col) = packed;
            else
              *reinterpret_cast<uint2*>(output + (long long)t * H + col) =
                  make_uint2(packed.x, packed.y);
          }
        }
      }
    }
  }
  __syncthreads();
  if (wid == 1) tmem_dealloc(plan.tmem_base, TMEM_COLS);
  if (sm_die != nullptr && tid == 0) {
    // The last CTA out resets the claim counters for the next call.
    __threadfence();
    if (atomicAdd(&claim[2], 1) == (int)gridDim.x - 1) {
      atomicExch(&claim[0], 0);
      atomicExch(&claim[1], 0);
      atomicExch(&claim[2], 0);
      __threadfence();
    }
  }
#else
  if (threadIdx.x == 0) printf("attn_res_pf_kernel requires sm_10x\n");
#endif
}

template <int N, bool OUT_NORM, bool QUANT, bool WRITE_BF16>
static void launch(bf16_t* blocks, bf16_t* prefix, const bf16_t* delta,
                   const bf16_t* res_w, const bf16_t* rms_w, bf16_t* out,
                   int T, long long sm_, long long sr, float eps,
                   const bf16_t* onw, float oeps, int widx, uint8_t* q,
                   uint8_t* sf, int sfc, const int8_t* sm_die, int* claim,
                   bf16_t* pout, int num_sm, cudaStream_t stream) {
  auto kernel = &attn_res_pf_kernel<N, OUT_NORM, QUANT, WRITE_BF16>;
  static bool attrs_set = false;
  if (!attrs_set) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES));
    attrs_set = true;
  }
  cudaLaunchConfig_t config{};
  config.gridDim = num_sm;
  config.blockDim = BLK;
  config.dynamicSmemBytes = SMEM_BYTES;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attrs;
  config.numAttrs = 1;
  cudaLaunchKernelEx(&config, kernel, blocks, prefix, delta, res_w, rms_w, out,
                     T, sm_, sr, eps, onw, oeps, widx, q, sf, sfc, sm_die,
                     claim, pout);
}

}  // namespace k31pf2

// Same arguments as attn_res_mxfp8.cu's attn_res_mxfp8. q/sf may be empty
// (numel 0) for a BF16-only call; output_norm_weight may be empty (no output
// RMSNorm: BF16 output only).
void attn_res_pf(torch::Tensor prefix, std::optional<torch::Tensor> delta,
                 torch::Tensor blocks, torch::Tensor norm_weight,
                 torch::Tensor qk_weight,
                 std::optional<torch::Tensor> output_norm_weight,
                 std::optional<torch::Tensor> output,
                 std::optional<torch::Tensor> q,
                 std::optional<torch::Tensor> sf, int64_t num_blocks,
                 int64_t block_write_idx, double eps, double output_norm_eps,
                 std::optional<torch::Tensor> sm_die,
                 std::optional<torch::Tensor> claim,
                 std::optional<torch::Tensor> prefix_out) {
  using namespace k31pf2;
  int const T = static_cast<int>(prefix.size(0));
  c10::cuda::CUDAGuard const device_guard(prefix.device());
  int cc_major = 0, num_sms = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &cc_major, cudaDevAttrComputeCapabilityMajor, prefix.get_device()));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &num_sms, cudaDevAttrMultiProcessorCount, prefix.get_device()));
  TORCH_CHECK(cc_major == 10, "Kimi K3 AttnRes requires the SM100 family");
  TORCH_CHECK(prefix.size(1) == H, "AttnRes requires hidden_size=7168");
  TORCH_CHECK(prefix.scalar_type() == at::kBFloat16 &&
                  blocks.scalar_type() == at::kBFloat16 &&
                  norm_weight.scalar_type() == at::kBFloat16 &&
                  qk_weight.scalar_type() == at::kBFloat16,
              "AttnRes requires BF16 tensors");
  TORCH_CHECK(prefix.is_contiguous() && blocks.stride(2) == 1 &&
                  norm_weight.is_contiguous() && qk_weight.is_contiguous() &&
                  (!delta.has_value() || delta->is_contiguous()) &&
                  (!output.has_value() || output->is_contiguous()),
              "AttnRes requires densely packed rows");
  TORCH_CHECK((blocks.stride(0) * 2) % 16 == 0 && (blocks.stride(1) * 2) % 16 == 0,
              "AttnRes block rows must be 16-byte aligned");
  TORCH_CHECK(num_blocks >= 0 && num_blocks <= 8 && num_blocks <= blocks.size(1),
              "AttnRes: num_blocks must be 0..min(8, blocks.size(1))");
  TORCH_CHECK(block_write_idx >= -1 && block_write_idx < blocks.size(1),
              "AttnRes: bad block_write_idx");
  TORCH_CHECK(blocks.size(0) >= T, "AttnRes: blocks has fewer rows than prefix");
  bool const out_norm = output_norm_weight.has_value();
  bool const quant = q.has_value() && q->numel() > 0;
  bool const write_bf16 = output.has_value();
  TORCH_CHECK(!quant || out_norm, "MXFP8 output needs the output RMSNorm");
  TORCH_CHECK(quant || write_bf16, "AttnRes needs an output");
  int sf_padded_cols = H / 32;
  if (quant) {
    TORCH_CHECK(q->size(0) == T && q->size(1) == H && q->is_contiguous() &&
                    q->element_size() == 1,
                "q must be a dense [T, 7168] byte tensor");
    TORCH_CHECK(sf.has_value() &&
                    sf->numel() >= ((T + 127) / 128) * 128 * sf_padded_cols,
                "sf too small");
  }
  if (T == 0) return;
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  if (quant) {
    int64_t const pad_rows = ((T + 127) / 128) * 128 - T;
    if (pad_rows > 0) {
      sf->view({-1})
          .narrow(0, (T / 128) * 128 * sf_padded_cols, 128 * sf_padded_cols)
          .zero_();
    }
  }
  bf16_t* bp = static_cast<bf16_t*>(blocks.data_ptr());
  bf16_t* pp = static_cast<bf16_t*>(prefix.data_ptr());
  const bf16_t* dp =
      delta.has_value() ? static_cast<const bf16_t*>(delta->data_ptr()) : nullptr;
  const bf16_t* qkp = static_cast<const bf16_t*>(qk_weight.data_ptr());
  const bf16_t* nwp = static_cast<const bf16_t*>(norm_weight.data_ptr());
  bf16_t* op = write_bf16 ? static_cast<bf16_t*>(output->data_ptr()) : nullptr;
  const bf16_t* onw =
      out_norm ? static_cast<const bf16_t*>(output_norm_weight->data_ptr())
               : nullptr;
  uint8_t* qp = quant ? static_cast<uint8_t*>(q->data_ptr()) : nullptr;
  uint8_t* sfp = quant ? static_cast<uint8_t*>(sf->data_ptr()) : nullptr;
  TORCH_CHECK(sm_die.has_value() == claim.has_value(),
              "die-local mode needs both sm_die and claim");
  const int8_t* smd = nullptr;
  int* clp = nullptr;
  if (sm_die.has_value()) {
    TORCH_CHECK(sm_die->scalar_type() == at::kChar && sm_die->numel() >= 1024 &&
                    claim->scalar_type() == at::kInt && claim->numel() >= 3,
                "sm_die must be int8[1024], claim int32[>=3]");
    TORCH_CHECK(blocks.stride(0) == blocks.size(1) * H && blocks.stride(1) == H,
                "die-local mode needs token-major dense blocks");
    smd = static_cast<const int8_t*>(sm_die->data_ptr());
    clp = static_cast<int*>(claim->data_ptr());
  }
  bf16_t* pop = nullptr;
  if (prefix_out.has_value()) {
    TORCH_CHECK(prefix_out->scalar_type() == at::kBFloat16 &&
                    prefix_out->is_contiguous() && prefix_out->size(0) == T &&
                    prefix_out->size(1) == H,
                "prefix_out must be a dense BF16 [T, 7168] tensor");
    pop = static_cast<bf16_t*>(prefix_out->data_ptr());
  }
  auto run = [&](auto n_c, auto norm_c, auto quant_c, auto bf16_c) {
    launch<decltype(n_c)::value, decltype(norm_c)::value,
           decltype(quant_c)::value, decltype(bf16_c)::value>(
        bp, pp, dp, qkp, nwp, op, T, blocks.stride(0), blocks.stride(1),
        static_cast<float>(eps), onw, static_cast<float>(output_norm_eps),
        static_cast<int>(block_write_idx), qp, sfp, sf_padded_cols, smd, clp,
        pop, num_sms, stream);
  };
  auto by_n = [&](auto norm_c, auto quant_c, auto bf16_c) {
    switch (num_blocks) {
      case 0: run(std::integral_constant<int, 1>{}, norm_c, quant_c, bf16_c); break;
      case 1: run(std::integral_constant<int, 2>{}, norm_c, quant_c, bf16_c); break;
      case 2: run(std::integral_constant<int, 3>{}, norm_c, quant_c, bf16_c); break;
      case 3: run(std::integral_constant<int, 4>{}, norm_c, quant_c, bf16_c); break;
      case 4: run(std::integral_constant<int, 5>{}, norm_c, quant_c, bf16_c); break;
      case 5: run(std::integral_constant<int, 6>{}, norm_c, quant_c, bf16_c); break;
      case 6: run(std::integral_constant<int, 7>{}, norm_c, quant_c, bf16_c); break;
      case 7: run(std::integral_constant<int, 8>{}, norm_c, quant_c, bf16_c); break;
      default: run(std::integral_constant<int, 9>{}, norm_c, quant_c, bf16_c); break;
    }
  };
  using T_ = std::true_type;
  using F_ = std::false_type;
  if (quant && write_bf16) {
    by_n(T_{}, T_{}, T_{});
  } else if (quant) {
    by_n(T_{}, T_{}, F_{});
  } else if (out_norm) {
    by_n(T_{}, F_{}, T_{});
  } else {
    by_n(F_{}, F_{}, T_{});
  }
  cudaError_t const error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, "AttnRes kernel launch failed: ",
              cudaGetErrorString(error));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("attn_res_pf", &attn_res_pf); }
