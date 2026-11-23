// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Kimi K3.1 prefill AttnRes with precomputed block logits (specialized_prefill).
//
// out = RMSNorm(softmax_i(s_i) . v_i) over the sources v_0..v_{n-1} (block
// rows) and u = bf16(prefix + delta) (the prefix source), with
// s = (v . w) * rsqrt(mean(v^2) + eps), w = norm_w * qk_w (FP32, exact).
//
// The block logits do not depend on the call's data, only on the block and on
// the call's w: the writer of block b stores rsqrt(mean(v_b^2) + eps), and one
// tensor-core GEMM per block (attn_res_lp.py) computes v_b . w_c for every
// later call c (w split exactly into BF16 hi + lo, FP32 accumulation). Here the
// block logits are n scalar loads; only the prefix source needs a row
// reduction. The block stream is a pure weighted sum (1 FFMA per element): no
// per-source reduction, no online-softmax rescale, no row buffering.
//
// Layout: 128 threads per CTA, one token at a time per CTA (persistent
// token loop); thread `tid` owns columns (k * 128 + tid) * 8 + [0, 8) for
// k = 0..6 (H = 7168). Rows stream through a per-thread cp.async ring in
// shared memory (each thread copies exactly the bytes it consumes, so the
// ring needs no CTA barrier); two CTA barriers per token (prefix statistics,
// output RMSNorm).
//
// Numerics (stock precision, order may differ): FP32 statistics and
// accumulation, BF16-rounded prefix sum, BF16 output, MXFP8 quantization of
// the BF16-rounded output with the bit-exact mxfp8_quantize primitives.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

#include <type_traits>

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include "k31_spin_guard.cuh"  // bounded mbarrier waits
K31SG_DEFINE_SET_RANK(attn_res_lp)  // rank of the timeout line (binding: set_spin_rank)

namespace k31lp {
// Internal linkage: probe / variant builds load side by side in one process.
namespace {

constexpr int H = 7168;
#ifndef K31LP_VEC
#define K31LP_VEC 8
#endif
constexpr int VEC = K31LP_VEC;       // bf16 per lane chunk
constexpr int NT = H / (7 * VEC);  // threads per token group (128 at VEC 8)
constexpr int NCH = H / (NT * VEC);  // 7 chunks per thread
constexpr int MAXB = 8;
constexpr int MAXT = 3;  // BF16 terms of a block logit (attn_res_lp.MAX_TERMS)
constexpr float LOG2_E = 1.4426950408889634f;
#ifndef K31LP_PDL
#define K31LP_PDL 1  // programmatic dependent launch (0: bench A/B builds only)
#endif
#ifndef K31LP_PROBE
#define K31LP_PROBE 0  // probe builds only (bench): 1 no block math, 2 no epilogue,
                       // 4 no output stores, 8 no CTA barriers (wrong values),
                       // 128 data-movement skeleton: every load, store, claim and
                       // barrier of the real kernel, no block / norm / quant math
#endif
#ifndef K31LP_SKEL
#define K31LP_SKEL ((K31LP_PROBE & 128) != 0)
#endif
static_assert(NCH * NT * VEC == H);
static_assert(!K31LP_SKEL || VEC == 8, "skeleton probe: VEC 8 only");

using bf16_t = __nv_bfloat16;

// ---- MXFP8 quantization primitives (bit-exact copies of
// ops/mxfp8_act.py / attn_res_mxfp8.cu).
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

// Byte offset of scale block `col` of `row` in the F8_128x4 swizzled layout
// with `padded_cols` (multiple of 4) scale columns.
__device__ __forceinline__ int64_t swizzled_sf_offset(int64_t row, int col,
                                                      int padded_cols) {
  return (int64_t)(col & 3) + (int64_t)(col >> 2) * 512 + (row & 31) * 16 +
         ((row & 127) >> 5) * 4 + (row >> 7) * 128 * (int64_t)padded_cols;
}


__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void mbar_init(uint64_t* b, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(b)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(b)),
               "r"(bytes)
               : "memory");
}
// bounded (k31_spin_guard.cuh): site = the wait's line
#define mbar_wait(b, parity) k31sg::mbar_wait(smem_u32(b), (parity), "attn_res_lp", __LINE__)
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes,
                                         uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(
          smem_u32(dst)),
      "l"(src), "r"(bytes), "r"(smem_u32(bar))
      : "memory");
}

__device__ __forceinline__ uint32_t pack2(float a, float b) {
  __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&v);
}

struct Args {
  const bf16_t* prefix;
  bf16_t* prefix_dst;  // where u is written (write_prefix)
  const bf16_t* delta;
  bf16_t* blocks;
  int64_t bsm, bsr;  // block strides (elements)
  const float* w;    // [H] FP32 norm_w * qk_w of this call
  const bf16_t* onw;  // output RMSNorm weight (NORM)
  float eps, oeps;
  // Block i's logit term rows of this call ([T] FP32 each, terms k <
  // nterms); the logit is (sum of terms) * lscale (exact power of two).
  const float* lt[MAXT][MAXB];
  int nterms;
  float lscale;
  float* rstd;  // [MAXB][rstd_stride] per-block rsqrt(mean(v^2) + eps)
  int64_t rstd_stride;
  int n, widx, T;
  int write_prefix;
  bf16_t* out;  // BF16 output (WRITE_BF16)
  uint8_t* q;   // MXFP8 values (QUANT)
  uint8_t* sf;
  int sf_padded_cols;
  long long* stamps;  // probe builds only
  const int8_t* sm_die;  // SM -> die (die-local block buffer) or nullptr
  int* claim;            // [3] claim counters (own die, other die, done)
};

// A CTA runs G token groups of NT threads (NW warps); a group processes one
// token at a time. Warp w of a group owns the contiguous columns
// [w * SEG, (w + 1) * SEG); lane l owns SEG-relative columns
// (k * 32 + l) * VEC + [0, VEC), k = 0..NCH-1. Each warp streams its own
// segment of every row with one bulk copy into its own ring (warp-local full
// barriers, slot reuse after __syncwarp). The call's query w (FP32) and
// output-norm weight live in shared memory (loaded once per CTA) and each
// warp prefetches the next token's block logits, so no global-latency stall
// sits on a token's critical path.
constexpr int NW = NT / 32;
constexpr int SEG = H / NW;
constexpr uint32_t SEG_BYTES = SEG * 2;
#ifndef K31LP_G
#define K31LP_G 4
#endif
// Token claim queue (QN entries): a group claims qa positions ahead of the
// token it processes, the fewest that keep the producer's run-ahead of D
// rows (ceil(D / R) tokens) on claimed positions: qa = max(QA, ceil(D/R) + 1).
// A short run-ahead balances the end of a call best.
#ifndef K31LP_QA
#define K31LP_QA 2
#endif
constexpr int QA = K31LP_QA;
constexpr int QN = 8;
static_assert(QA >= 2 && QA < QN && (QN & (QN - 1)) == 0);
constexpr int G = K31LP_G;
constexpr int CTA_THREADS = NT * G;
// One lane's VEC BF16 values.
using vec_t = std::conditional_t<VEC == 8, uint4, uint2>;
constexpr int VW = VEC / 2;  // 32-bit words per vec_t

// A lane's values as FP32 pairs: the element math runs as packed FP32x2
// instructions (FFMA2 / FMUL2 / FADD2: two IEEE FP32 operations per
// instruction, the same per-element rounding as FFMA / FMUL / FADD).
__device__ __forceinline__ float2 unpack2(uint32_t w) {
  return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u));
}
__device__ __forceinline__ void unpack2v(const vec_t& p, float2 (&f)[VW]) {
  const uint32_t* w = reinterpret_cast<const uint32_t*>(&p);
#pragma unroll
  for (int i = 0; i < VW; ++i) f[i] = unpack2(w[i]);
}
__device__ __forceinline__ vec_t pack2v(const float2 (&f)[VW]) {
  vec_t p;
  uint32_t* w = reinterpret_cast<uint32_t*>(&p);
#pragma unroll
  for (int i = 0; i < VW; ++i) w[i] = pack2(f[i].x, f[i].y);
  return p;
}
__device__ __forceinline__ float2 bcast2(float x) { return make_float2(x, x); }

template <int D>
struct Smem {
  vec_t slot[G][NW][D][SEG / VEC];
  float w[H];
  bf16_t onw[H];
  uint64_t full[G][NW][D];
  uint64_t cbar;  // per-call constants (w, onw) landed
  float red[G][2][NW][3];  // [.][parity][warp]: sum u^2, u.w, sum out^2
  int tq[G][QN];
  float ew[G][NW][MAXB];  // per-warp normalized block weights of the token
  const float* lt[MAXT][MAXB];  // logit term rows (no dynamic param indexing)
};

// Probe builds (K31LP_PROBE & 16): per-token phase clock stamps of group 0,
// lane 0 of warp 0, tokens 0..15 of each CTA, into the BF16 output buffer.
#define K31LP_STAMP(j)                                                       \
  do {                                                                       \
    if constexpr ((K31LP_PROBE & 16) != 0) {                                 \
      if (g == 0 && tid == 0 && it < 16)                                     \
        a.stamps[((int64_t)blockIdx.x * 16 + it) * 8 + (j)] = clock64(); \
    }                                                                        \
  } while (0)

// Probe builds (K31LP_PROBE & 32): per-CTA globaltimer stamps (entry, loop
// start, exit of group 0 / of the CTA) into stamps[blockIdx * 4 + j].
__device__ __forceinline__ long long gtime() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
#define K31LP_CTA_STAMP(j)                                                   \
  do {                                                                       \
    if constexpr ((K31LP_PROBE & 32) != 0) {                                 \
      if (threadIdx.x == 0) a.stamps[(int64_t)blockIdx.x * 4 + (j)] = gtime(); \
    }                                                                        \
  } while (0)

// Probe builds (K31LP_PROBE & 64): per-group end-of-call stats into
// stamps[4096 + (blockIdx * G + g) * 4 + j]: die, own-die tokens,
// other-die tokens, exit globaltimer.
constexpr int kGroupStatsBase = 4096;

__device__ __forceinline__ void group_sync(int g) {
  asm volatile("bar.sync %0, %1;\n" ::"r"(1 + g), "n"(NT) : "memory");
}

template <int D, bool HAS_DELTA, bool NORM, bool WRITE_BF16, bool QUANT>
__global__ void __launch_bounds__(CTA_THREADS, 1) attn_res_lp_kernel(const Args a) {
  k31sg::begin_call((long long)(a.T));  // timeout lines: call index and size
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  extern __shared__ __align__(128) unsigned char smem_raw[];
  auto& sm = *reinterpret_cast<Smem<D>*>(smem_raw);

  const int lane = threadIdx.x & 31;
  const int g = threadIdx.x / NT;
  const int tid = threadIdx.x - g * NT;  // thread in group
  const int warp = tid >> 5;
  const int n = a.n;
  constexpr int R0 = 1 + (HAS_DELTA ? 1 : 0);
  const int R = R0 + n;  // rows per token
  const int qa = max(QA, (D + R - 1) / R + 1);  // claim run-ahead (< QN)
  const int64_t seg0 = (int64_t)warp * SEG;
  K31LP_CTA_STAMP(0);

  // Token schedule: each group claims its tokens one at a time, QA positions
  // ahead, into the smem queue tq[g] (thread 0 of the group claims; the group
  // barrier publishes). With the die-local block buffer (sm_die != nullptr)
  // the block rows of token t live on die (t >> 10) & 1: a group claims
  // tokens homed on its SM's die first, then the other die's leftovers.
  int die = 0;
  if (a.sm_die != nullptr) {
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    die = a.sm_die[smid & 1023];
  }
  // Super-block k of 2^SB tokens lives on die k % 2 (attn_res_dielocal).
  constexpr int SB = 11;
  const int nsb = (a.T + (1 << SB) - 1) >> SB;
  auto die_count = [&](int d) -> int {
    if (a.sm_die == nullptr) return d == 0 ? a.T : 0;
    const int full = (nsb - d + 1) >> 1;  // super-blocks k < nsb with k % 2 == d
    int c = full << SB;
    if (((nsb - 1) & 1) == d) c -= (nsb << SB) - a.T;
    return c;
  };
  auto die_token = [&](int d, int j) -> int {
    if (a.sm_die == nullptr) return j;
    return ((((j >> SB) << 1) + d) << SB) | (j & ((1 << SB) - 1));
  };
  const int cnt_own = die_count(die);
  const int cnt_oth = die_count(die ^ 1);
  bool own_done = false;
  // Finish a claim whose own-die atomicAdd returned j.
  auto finish_claim = [&](int j) -> int {
    if (!own_done) {
      if (j < cnt_own) return die_token(die, j);
      own_done = true;
    }
    const int k = atomicAdd(&a.claim[die ^ 1], 1);
    return k < cnt_oth ? die_token(die ^ 1, k) : -1;
  };
  auto start_claim = [&]() -> int {
    return own_done ? cnt_own : atomicAdd(&a.claim[die], 1);
  };

  if (lane == 0) {
#pragma unroll
    for (int s = 0; s < D; ++s) mbar_init(&sm.full[g][warp][s], 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  if (threadIdx.x == 0) {
#pragma unroll
    for (int i = 0; i < MAXB; ++i) {
#pragma unroll
      for (int k = 0; k < MAXT; ++k) sm.lt[k][i] = a.lt[k][i];
    }
    // Per-call constants into shared memory: two bulk copies (one round
    // trip), in flight across the dependency wait below.
    mbar_init(&sm.cbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    mbar_expect_tx(&sm.cbar, H * 4 + (NORM ? H * 2 : 0));
    bulk_g2s(sm.w, a.w, H * 4, &sm.cbar);
    if constexpr (NORM) bulk_g2s(sm.onw, a.onw, H * 2, &sm.cbar);
  }
#if K31LP_PDL
  // Programmatic dependent launch: everything above reads only per-call
  // constants. The prefix / delta rows, the logits, the claim counters and
  // every output belong to this call's dependency chain: wait for the
  // preceding grid before touching any of them.
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
  // Every CTA of this persistent grid is resident: let the next grid in the
  // stream launch now. Its CTAs take the SMs this grid's CTAs leave during
  // the end of the call (launch + prologue under the tail); its own
  // griddepcontrol.wait still orders it after this grid's completion.
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
  if (tid == 0) {
    const int j = atomicAdd(&a.claim[die], qa);  // first qa positions at once
#pragma unroll 1
    for (int i = 0; i < qa; ++i) {
      sm.tq[g][i] = j + i < cnt_own ? die_token(die, j + i)
                                    : finish_claim(own_done ? 0 : cnt_own);
    }
  }
  __syncthreads();

  // Producer (lane 0 of each warp): next row to issue (token position p_it,
  // row p_r). It runs at most D rows (< qa tokens) ahead.
  int p_it = 0, p_r = 0, p_slot = 0;
  auto issue_next = [&]() {
    const int t = sm.tq[g][p_it & (QN - 1)];
    if (lane == 0 && t >= 0) {
      const bf16_t* src;
      if (p_r == 0) {
        src = a.prefix + (int64_t)t * H;
      } else if (HAS_DELTA && p_r == 1) {
        src = a.delta + (int64_t)t * H;
      } else {
        src = a.blocks + (int64_t)t * a.bsm + (int64_t)(p_r - R0) * a.bsr;
      }
      uint64_t* bar = &sm.full[g][warp][p_slot];
      mbar_expect_tx(bar, SEG_BYTES);
      bulk_g2s(&sm.slot[g][warp][p_slot][0], src + seg0, SEG_BYTES, bar);
    }
    if (t >= 0) {
      if (++p_r == R) {
        p_r = 0;
        ++p_it;
      }
      p_slot = p_slot + 1 == D ? 0 : p_slot + 1;
    }
  };
#pragma unroll
  for (int i = 0; i < D; ++i) issue_next();

  int c_slot = 0;
  uint32_t c_phase = 0;
  auto wait_row = [&]() -> const vec_t* {
    mbar_wait(&sm.full[g][warp][c_slot], c_phase);
    return &sm.slot[g][warp][c_slot][0];
  };
  auto release_row = [&]() {
    __syncwarp();
    issue_next();  // into the slot just consumed
    if (++c_slot == D) {
      c_slot = 0;
      c_phase ^= 1;
    }
  };

  // Lane i < n holds block i's logit inputs for the next token.
  float nl_t0 = 0.f, nl_t1 = 0.f, nl_t2 = 0.f, nl_r = 0.f;
  auto load_logits = [&](int it) {
    const int t = sm.tq[g][it & (QN - 1)];
    if (lane < n && t >= 0) {
      nl_t0 = __ldg(sm.lt[0][lane] + t);
      if (a.nterms > 1) nl_t1 = __ldg(sm.lt[1][lane] + t);
      if (a.nterms > 2) nl_t2 = __ldg(sm.lt[2][lane] + t);
      nl_r = __ldg(a.rstd + (int64_t)lane * a.rstd_stride + t);
    }
  };
  load_logits(0);
  mbar_wait(&sm.cbar, 0);

  K31LP_CTA_STAMP(1);
  int par = 0;
  // Thread 0: the own-die atomicAdd (in flight) for the next claimed position.
  int pending = tid == 0 ? start_claim() : 0;
  [[maybe_unused]] int p_own = 0, p_oth = 0;
  for (int it = 0;; ++it) {
    const int tt = sm.tq[g][it & (QN - 1)];
    if (tt < 0) break;
    const int64_t t = tt;
    if constexpr ((K31LP_PROBE & 64) != 0) {
      if (tid == 0) {
        if (a.sm_die == nullptr || ((tt >> SB) & 1) == die) ++p_own; else ++p_oth;
      }
    }

    K31LP_STAMP(0);
    // Block logits of this token (prefetched), then prefetch the next.
    const float s_mine = ((nl_t0 + nl_t1) + nl_t2) * a.lscale * nl_r;
    load_logits(it + 1);
    // Lane i < n: block i's softmax term (one register, not an array).
    float m_b = lane < n ? s_mine : -FLT_MAX;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m_b = fmaxf(m_b, __shfl_xor_sync(0xffffffff, m_b, o));
    float e_mine = lane < n ? exp2f((s_mine - m_b) * LOG2_E) : 0.f;
    float z_b = e_mine;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) z_b += __shfl_xor_sync(0xffffffff, z_b, o);

    // Prefix source u = bf16(prefix + delta).
    vec_t u[NCH];
    {
      const vec_t* s = wait_row();
      K31LP_STAMP(1);
#pragma unroll
      for (int k = 0; k < NCH; ++k) u[k] = s[k * 32 + lane];
      if constexpr (HAS_DELTA) {
        release_row();
        s = wait_row();
#pragma unroll
        for (int k = 0; k < NCH; ++k) {
          float2 fp[VW], fd[VW];
          unpack2v(u[k], fp);
          unpack2v(s[k * 32 + lane], fd);
#pragma unroll
          for (int j = 0; j < VW; ++j) fp[j] = __fadd2_rn(fp[j], fd[j]);
          u[k] = pack2v(fp);
        }
      }
      release_row();
    }
    K31LP_STAMP(2);
    float2 ss2 = bcast2(0.f), dw2 = bcast2(0.f);
#pragma unroll
    for (int k = 0; k < NCH; ++k) {
      const int c = (int)seg0 + (k * 32 + lane) * VEC;
      if (a.write_prefix) *reinterpret_cast<vec_t*>(a.prefix_dst + t * H + c) = u[k];
      if (a.widx >= 0)
        *reinterpret_cast<vec_t*>(a.blocks + t * a.bsm + a.widx * a.bsr + c) = u[k];
      float2 f[VW];
      unpack2v(u[k], f);
#pragma unroll
      for (int i = 0; i < VW; i += 2) {
        const float4 w4 = *reinterpret_cast<const float4*>(sm.w + c + 2 * i);
        ss2 = __ffma2_rn(f[i], f[i], ss2);
        dw2 = __ffma2_rn(f[i], make_float2(w4.x, w4.y), dw2);
        ss2 = __ffma2_rn(f[i + 1], f[i + 1], ss2);
        dw2 = __ffma2_rn(f[i + 1], make_float2(w4.z, w4.w), dw2);
      }
    }
    float ss = ss2.x + ss2.y, dw = dw2.x + dw2.y;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      ss += __shfl_xor_sync(0xffffffff, ss, o);
      dw += __shfl_xor_sync(0xffffffff, dw, o);
    }
    if (lane == 0) {
      sm.red[g][par][warp][0] = ss;
      sm.red[g][par][warp][1] = dw;
    }

    // Prefix statistics -> all softmax weights before the block stream: the
    // block weights carry the normalization and acc starts at the prefix
    // term (no combine pass after the stream).
    K31LP_STAMP(3);
    if (tid == 0) {
      sm.tq[g][(it + qa) & (QN - 1)] = finish_claim(pending);
      pending = start_claim();  // in flight for a whole token
    }
    group_sync(g);
    K31LP_STAMP(4);
    float tss = 0.f, tdw = 0.f;
#pragma unroll
    for (int wi = 0; wi < NW; ++wi) {
      tss += sm.red[g][par][wi][0];
      tdw += sm.red[g][par][wi][1];
    }
    const float r_p = rsqrtf(tss / H + a.eps);
    if (a.widx >= 0 && tid == 0) a.rstd[(int64_t)a.widx * a.rstd_stride + t] = r_p;
    float c_u = 1.f;
    if (n > 0) {
      const float s_p = tdw * r_p;
      const float M = fmaxf(m_b, s_p);
      const float sc = exp2f((m_b - M) * LOG2_E);
      const float ep = exp2f((s_p - M) * LOG2_E);
      const float inv = 1.f / (z_b * sc + ep);
      const float c_acc = sc * inv;
#pragma unroll
      e_mine *= c_acc;
      c_u = ep * inv;
    }
    if (lane < MAXB) sm.ew[g][warp][lane] = e_mine;
    __syncwarp();
    float2 acc[NCH][VW];
    {
      const float2 cu2 = bcast2(c_u);
#pragma unroll
      for (int k = 0; k < NCH; ++k) {
        float2 f[VW];
        unpack2v(u[k], f);
#pragma unroll
        for (int j = 0; j < VW; ++j) acc[k][j] = __fmul2_rn(f[j], cu2);
      }
    }

    // Block stream: acc += e_i v_i.
#pragma unroll 1
    for (int i = 0; i < n; ++i) {
      const vec_t* s = wait_row();
      const float2 ei = bcast2(sm.ew[g][warp][i]);
#pragma unroll
      for (int k = 0; k < ((K31LP_PROBE & 129) ? 0 : NCH); ++k) {
        float2 f[VW];
        unpack2v(s[k * 32 + lane], f);
#pragma unroll
        for (int j = 0; j < VW; ++j) acc[k][j] = __ffma2_rn(ei, f[j], acc[k][j]);
      }
      release_row();
    }
    K31LP_STAMP(5);
    if constexpr ((K31LP_PROBE & 2) != 0) {
      if (acc[0][0].x == 123.f) a.out[t] = __float2bfloat16(ss + dw);
      continue;
    }
    float os = 0.f;
    if constexpr (!K31LP_SKEL) {
      float2 os2 = bcast2(0.f);
#pragma unroll
      for (int k = 0; k < NCH; ++k)
#pragma unroll
        for (int j = 0; j < VW; ++j) os2 = __ffma2_rn(acc[k][j], acc[k][j], os2);
      os = os2.x + os2.y;
    }
    float orstd = 1.f;
    if constexpr (NORM && K31LP_SKEL) {
      group_sync(g);  // the norm barrier, no math
    } else if constexpr (NORM) {
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) os += __shfl_xor_sync(0xffffffff, os, o);
      if (lane == 0) sm.red[g][par][warp][2] = os;
      group_sync(g);
      float tos = 0.f;
#pragma unroll
      for (int wi = 0; wi < NW; ++wi) tos += sm.red[g][par][wi][2];
      orstd = rsqrtf(tos / H + a.oeps);
    }
#pragma unroll
    for (int k = 0; k < NCH; ++k) {
      const int c = (int)seg0 + (k * 32 + lane) * VEC;
      float2 v[VW];
      if constexpr (NORM && !K31LP_SKEL) {
        float2 wf[VW];
        unpack2v(*reinterpret_cast<const vec_t*>(sm.onw + c), wf);
        const float2 or2 = bcast2(orstd);
#pragma unroll
        for (int j = 0; j < VW; ++j) v[j] = __fmul2_rn(__fmul2_rn(acc[k][j], or2), wf[j]);
      } else {
#pragma unroll
        for (int j = 0; j < VW; ++j) v[j] = acc[k][j];
      }
      const vec_t ob = pack2v(v);
      if constexpr ((K31LP_PROBE & 4) != 0) {
        if (*reinterpret_cast<const uint32_t*>(&ob) == 0x12345678u) a.out[t] = a.onw[0];
        continue;
      }
      if constexpr (WRITE_BF16) {
        *reinterpret_cast<vec_t*>(a.out + t * H + c) = ob;
      }
      if constexpr (QUANT && K31LP_SKEL) {  // same bytes, no quant math
        *reinterpret_cast<uint2*>(a.q + t * H + c) = *reinterpret_cast<const uint2*>(&ob);
        if ((c & 31) == 0) a.sf[swizzled_sf_offset(t, c >> 5, a.sf_padded_cols)] = 0;
      } else if constexpr (QUANT) {
        float2 qv[VW];
        unpack2v(ob, qv);
        float amax = 0.f;
#pragma unroll
        for (int j = 0; j < VW; ++j) amax = fmaxf(amax, fmaxf(fabsf(qv[j].x), fabsf(qv[j].y)));
#pragma unroll
        for (int m = 1; m < 32 / VEC; m <<= 1)
          amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, m));
        const uint32_t ex = float_to_ue8m0(amax * (1.0f / 448.0f));
        const float2 inv2 = bcast2(ue8m0_to_inv_scale(ex));
        uint32_t wq[VEC / 4];
#pragma unroll
        for (int j = 0; j < VEC / 4; ++j) {
          const float2 q0 = __fmul2_rn(qv[2 * j], inv2), q1 = __fmul2_rn(qv[2 * j + 1], inv2);
          wq[j] = (uint32_t)fp8x2(q0.x, q0.y) | ((uint32_t)fp8x2(q1.x, q1.y) << 16);
        }
        if constexpr (VEC == 8) {
          *reinterpret_cast<uint2*>(a.q + t * H + c) = make_uint2(wq[0], wq[1]);
        } else {
          *reinterpret_cast<uint32_t*>(a.q + t * H + c) = wq[0];
        }
        if ((c & 31) == 0)
          a.sf[swizzled_sf_offset(t, c >> 5, a.sf_padded_cols)] = (uint8_t)ex;
      }
    }
    K31LP_STAMP(6);
    par ^= 1;
  }
  if constexpr ((K31LP_PROBE & 64) != 0) {
    if (tid == 0) {
      long long* s = a.stamps + kGroupStatsBase + ((int64_t)blockIdx.x * G + g) * 4;
      s[0] = die;
      s[1] = p_own;
      s[2] = p_oth;
      s[3] = gtime();
    }
  }
  K31LP_CTA_STAMP(2);
  // The last CTA out resets the claim counters for the next call.
  __syncthreads();
  K31LP_CTA_STAMP(3);
  if (threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(&a.claim[2], 1) == (int)gridDim.x - 1) {
      atomicExch(&a.claim[0], 0);
      atomicExch(&a.claim[1], 0);
      atomicExch(&a.claim[2], 0);
    }
  }
#endif
}

template <int D, bool HD, bool NO, bool WB, bool QU>
void launch_t(const Args& a, cudaStream_t st) {
  auto k = attn_res_lp_kernel<D, HD, NO, WB, QU>;
  const int smem = (int)sizeof(Smem<D>);
  static int occ = -1;
  static int sms = 0;
  if (occ < 0) {
    TORCH_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess,
                "attn_res_lp: ", smem, " B shared memory");
    TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k, CTA_THREADS, smem) == cudaSuccess);
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    TORCH_CHECK(occ >= 1, "attn_res_lp: does not fit an SM");
  }
  const int grid = std::min((a.T + G - 1) / G, occ * sms);  // co-resident
  if (grid <= 0) return;
  // PDL: the CTAs become resident as the preceding kernel's CTAs exit and
  // load the per-call constants before griddepcontrol.wait.
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(CTA_THREADS);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = K31LP_PDL;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k, a));
}

template <int D>
void dispatch(const Args& a, bool hd, bool no, bool wb, bool qu, cudaStream_t st) {
#define K31LP_CASE(HD, NO, WB, QU) \
  if (hd == HD && no == NO && wb == WB && qu == QU) return launch_t<D, HD, NO, WB, QU>(a, st);
  K31LP_CASE(true, true, true, true)
  K31LP_CASE(true, true, true, false)
  K31LP_CASE(true, true, false, true)
  K31LP_CASE(false, true, true, true)
  K31LP_CASE(false, true, true, false)
  K31LP_CASE(false, true, false, true)
  K31LP_CASE(true, false, true, false)
  K31LP_CASE(false, false, true, false)
#undef K31LP_CASE
  TORCH_CHECK(false, "attn_res_lp: unsupported variant");
}

}  // namespace
}  // namespace k31lp

static const void* opt_ptr(const c10::optional<torch::Tensor>& t) {
  return t.has_value() ? t->data_ptr() : nullptr;
}

static void attn_res_lp(torch::Tensor prefix, c10::optional<torch::Tensor> prefix_dst,
                 c10::optional<torch::Tensor> delta, torch::Tensor blocks,
                 torch::Tensor w, c10::optional<torch::Tensor> onw,
                 torch::Tensor lptr, int64_t nterms, double lscale,
                 torch::Tensor rstd, int64_t n, int64_t widx,
                 double eps, double oeps, c10::optional<torch::Tensor> out,
                 c10::optional<torch::Tensor> q, c10::optional<torch::Tensor> sf,
                 int64_t depth, c10::optional<torch::Tensor> sm_die,
                 torch::Tensor claim, c10::optional<torch::Tensor> stamps) {
  using namespace k31lp;
  TORCH_CHECK(prefix.dim() == 2 && prefix.size(1) == H && prefix.stride(0) == H);
  TORCH_CHECK(blocks.dim() == 3 && blocks.size(2) == H && blocks.stride(2) == 1);
  TORCH_CHECK(n >= 0 && n <= MAXB && n <= blocks.size(1));
  TORCH_CHECK(widx < blocks.size(1));
  TORCH_CHECK(w.scalar_type() == torch::kFloat32 && w.numel() == H && w.is_contiguous());
  TORCH_CHECK(!onw.has_value() || (onw->numel() == H && onw->is_contiguous()));
  TORCH_CHECK((reinterpret_cast<uintptr_t>(w.data_ptr()) & 15) == 0 &&
                  (!onw.has_value() || (reinterpret_cast<uintptr_t>(onw->data_ptr()) & 15) == 0),
              "attn_res_lp: w / output norm weight must be 16-byte aligned (bulk copies)");
  TORCH_CHECK(nterms >= 1 && nterms <= MAXT, "attn_res_lp: ", nterms, " logit terms");
  TORCH_CHECK(lptr.device().is_cpu() && lptr.scalar_type() == torch::kInt64 &&
              lptr.numel() >= nterms * n);
  TORCH_CHECK(rstd.scalar_type() == torch::kFloat32 && rstd.dim() == 2);
  if (delta.has_value()) TORCH_CHECK(delta->stride(0) == H && delta->size(0) >= prefix.size(0));
  const c10::cuda::CUDAGuard guard(prefix.device());
  Args a{};
  a.T = (int)prefix.size(0);
  a.prefix = reinterpret_cast<const bf16_t*>(prefix.data_ptr());
  a.prefix_dst = reinterpret_cast<bf16_t*>(
      prefix_dst.has_value() ? prefix_dst->data_ptr() : prefix.data_ptr());
  a.delta = reinterpret_cast<const bf16_t*>(opt_ptr(delta));
  a.blocks = reinterpret_cast<bf16_t*>(blocks.data_ptr());
  a.bsm = blocks.stride(0);
  a.bsr = blocks.stride(1);
  a.w = w.data_ptr<float>();
  a.onw = reinterpret_cast<const bf16_t*>(opt_ptr(onw));
  a.eps = (float)eps;
  a.oeps = (float)oeps;
  const int64_t* lp = lptr.data_ptr<int64_t>();
  a.nterms = (int)nterms;
  a.lscale = (float)lscale;
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < nterms; ++k)
      a.lt[k][i] = reinterpret_cast<const float*>(lp[nterms * i + k]);
  }
  a.rstd = rstd.data_ptr<float>();
  a.rstd_stride = rstd.stride(0);
  a.n = (int)n;
  a.widx = (int)widx;
  a.out = reinterpret_cast<bf16_t*>(const_cast<void*>(opt_ptr(out)));
  a.q = reinterpret_cast<uint8_t*>(const_cast<void*>(opt_ptr(q)));
  a.sf = reinterpret_cast<uint8_t*>(const_cast<void*>(opt_ptr(sf)));
  a.sf_padded_cols = H / 32;
  TORCH_CHECK(claim.scalar_type() == torch::kInt32 && claim.numel() >= 3);
  TORCH_CHECK(!sm_die.has_value() || (sm_die->scalar_type() == torch::kInt8 && sm_die->numel() >= 1024));
  TORCH_CHECK(depth + 1 < QN, "attn_res_lp: ring depth exceeds the claim queue");
  a.sm_die = sm_die.has_value() ? sm_die->data_ptr<int8_t>() : nullptr;
  a.claim = claim.data_ptr<int>();
  a.stamps = stamps.has_value() ? reinterpret_cast<long long*>(stamps->data_ptr()) : nullptr;
  // The prefix changes only with a delta or a new destination; a block-write
  // call without a delta (dataflow C: the producers added the delta) leaves
  // the in-place prefix as is (its value is the block row it writes).
  a.write_prefix = delta.has_value() || prefix_dst.has_value();
  const bool hd = delta.has_value(), no = onw.has_value();
  const bool wb = out.has_value(), qu = q.has_value();
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  if (depth == 4) return k31lp::dispatch<4>(a, hd, no, wb, qu, st);
  if (depth == 3) return k31lp::dispatch<3>(a, hd, no, wb, qu, st);
  if (depth == 2) return k31lp::dispatch<2>(a, hd, no, wb, qu, st);
  TORCH_CHECK(false, "attn_res_lp: depth must be 2, 3 or 4");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attn_res_lp", &attn_res_lp);
  m.def("set_spin_rank", [](int64_t r, int64_t diag) { k31sg_set_rank_attn_res_lp(static_cast<int>(r), diag); });
}
