// Kimi-K3: chunk-64 KDA recurrence on the tcgen05 tensor cores (dev stage of
// the fused KDA prefill kernel).
//
// FlashKDA's per-chunk math (fwd_kernel2.cuh, FlashKDA 17a037d, MIT, see
// LICENSE.FlashKDA) with 64-token chunks: 86 dependent state steps per 5.4K
// token sequence instead of 341. Per chunk c of one (sequence, head), with the
// [K, V] state S (fp32, one value row v per math thread, in registers):
//   W   = kd @ bf16(S)              Oq  = qd @ bf16(S)          (fp32 accum)
//   x   = bf16(bf16(v - bf16(W)) * beta)                        (bf16 ops)
//   U   = bf16(INV @ x)                                         (fp32 accum)
//   out = bf16(Oq + Mqk @ U)                                    (fp32 accum)
//   S   = fma(S, g_total, kr^T @ U)                             (fp32 state)
// kd / qd / kr / INV / Mqk / g_total / v come from the chunk-64 prepare (kd,
// qd, kr relative to the chunk start; INV, Mqk 64 x 64 lower triangular) as
// K-major 8x8 core-matrix byte images:
//   kd|qd [H][C][16 k-groups][128 rows: 64 kd, 64 qd][8]
//   kr^T [H][C][8 t-groups][128 k][8]    INV, Mqk [H][C][8 s-groups][64 t][8]
//   v [H][C][8 t-groups][128 v][8] (rows past the sequence end zero)
//   g_total [H][C][128] fp32
// Everything is transposed so V (128) is the MMA M dimension: S^T / x^T / U^T
// are bf16 A operands in tensor memory written row-per-thread, the images are
// B operands in shared memory, accumulators are read back row-per-thread.
//
// The state chain per chunk is W -> x -> MMA2 -> U -> MMA3 -> S: Oq (qd half
// of MMA1) and Mqk @ U (MMA4) run behind it, out(c) is read after the state
// update while W(c + 1) runs.
// out(c) is read under MMA2(c + 1), then Oq(c + 1) is issued. Tensor-memory reads go in 32-column batches
// (the fp32 state row takes 128 of the math threads' 216 registers).
//
// Shared memory ~146 KB (two CTAs per SM in the sm_107 oversized mode).
// Tensor memory (256 columns, two CTAs per SM):
//   [0, 64)    W(c); x^T then U^T (bf16) in [0, 32)
//   [64, 128)  Oq(c), += Mqk @ U
//   [128, 192) bf16 S^T (A of W and Oq), then the low half of kr^T @ U
//   [192, 256) U(c), then the high half of kr^T @ U (N = 128); the state
//              update writes bf16 S^T for chunk c + 1 behind its reads
//
// Warps: 0-3 math (thread 0 issues the MMAs), 4 loader kd|qd (3-slot
// ring), 5 loader v + beta and INV, 6 loader kr^T + Mqk + g_total (g_total
// double buffered), 7 tensor-memory owner + out stores.
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "k2_tc.cuh"
K31SG_DEFINE_SET_RANK(k64_tc)  // rank of the timeout line (binding: set_spin_rank)

namespace k64tc {

using namespace k2tc;

constexpr int kC = 64;
constexpr int kD = 128;

// kd|qd image ring (3 x 32 KB): the slot of chunk c then stages out(c), so
// the load of chunk c + 1 waits only for out(c - 2)'s bulk store.
constexpr int kKdqSlots = 3;
constexpr int kOffKdq = 0;
constexpr int kOffV = 98304;        // [8 t-groups][128 v][8] bf16
constexpr int kOffBeta = 114688;    // bf16[64]
constexpr int kOffInv = 114816;     // 8 KB
constexpr int kOffKr = 123008;      // 16 KB
constexpr int kOffMqk = 139392;     // 8 KB
constexpr int kOffGt = 147584;      // float[2][128]
constexpr int kOffBar = 148608;
// barriers: full_kdq[3], empty_kdq[3], full_vb, empty_vb, full_inv,
// empty_inv, full_krm, empty_krm, d1, d2, d3, out_full[3], d4, full_gt[2].
// Every barrier a thread waits on per chunk is either gated by that thread's
// own progress or a 2-slot ring, so no waiter can fall two phases behind.
constexpr int kNumBars = 21;
constexpr int kSmemBytes = kOffBar + 8 * kNumBars + 16;

constexpr uint32_t kColW = 0;    // W; x^T, U^T bf16 at [0, 32)
constexpr uint32_t kColO = 64;   // Oq
constexpr uint32_t kColP = 128;  // kr^T @ U; bf16 S^T at [128, 192)
constexpr uint32_t kColU = 192;  // U (S^T must survive MMA2 for Oq)

constexpr int kDbgSlots = 16;

__device__ __forceinline__ void named_sync() {
  asm volatile("bar.sync 1, 128;" ::: "memory");
}

__device__ __forceinline__ uint32_t bf2_sub_mul(uint32_t v, uint32_t w, uint32_t b) {
  __nv_bfloat162 vv = *reinterpret_cast<__nv_bfloat162*>(&v);
  __nv_bfloat162 ww = *reinterpret_cast<__nv_bfloat162*>(&w);
  __nv_bfloat162 bb = *reinterpret_cast<__nv_bfloat162*>(&b);
  __nv_bfloat162 r = __hmul2(__hsub2(vv, ww), bb);
  return *reinterpret_cast<uint32_t*>(&r);
}

__device__ __forceinline__ bool mbar_try(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n .reg .pred P1;\n mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n"
      " selp.u32 %0, 1, 0, P1;\n}"
      : "=r"(ok)
      : "r"(bar), "r"(parity)
      : "memory");
  return ok != 0;
}

// Bounded wait (k31_spin_guard.cuh): was a dev watchdog that recorded (block, warp,
// barrier id, chunk) in dbg[0] after 2^20 polls and RETURNED, letting the kernel go on
// with an incomplete stage (silent wrong data). Now the shared policy: the clock starts
// after 1024 failed polls, 900 s later one K31_SPIN_TIMEOUT line names the kernel and
// the barrier id (site), then trap. dbg / c stay in the signature (callers unchanged).
__device__ __forceinline__ void wait_wd(uint32_t bar, uint32_t parity, int id, int c,
                                        long long* dbg) {
  (void)c;
  (void)dbg;
  k31sg::Spin s;
  while (!mbar_try(bar, parity)) s.check("flashkda_k64_tc", id, parity ^ 1u, parity);
}

template <bool HasStateIn>
__global__ void __launch_bounds__(256, 2) k64_tc_kernel(
    __nv_bfloat16 const* __restrict__ kdq_img,
    __nv_bfloat16 const* __restrict__ kr_img,
    __nv_bfloat16 const* __restrict__ inv_img,
    __nv_bfloat16 const* __restrict__ mqk_img,
    float const* __restrict__ gt_ptr,            // [H][C][128]
    __nv_bfloat16 const* __restrict__ beta_ptr,  // [H][T] sigmoid(beta)
    __nv_bfloat16 const* __restrict__ v_img,     // [H][C][8][128][8]
    float const* __restrict__ init_state,        // [N][H][128 v][128 k]
    float* __restrict__ final_state,
    __nv_bfloat16* __restrict__ out_ptr,         // [T][H][128]
    int const* __restrict__ cu_seqlens, int T_total, int H, int N, int C_total,
    int flags, long long* __restrict__ dbg) {
  k31sg::begin_call((long long)(T_total));  // timeout lines: call index and size
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  extern __shared__ __align__(1024) unsigned char smem[];
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int head = blockIdx.x;
  const int seq = N - 1 - int(blockIdx.y);
  const uint32_t sbase = smem_u32(smem);
  const uint32_t bar0 = sbase + kOffBar;
  const uint32_t full_kdq0 = bar0, empty_kdq0 = bar0 + 24;
  const uint32_t full_vb = bar0 + 48, empty_vb = bar0 + 56;
  const uint32_t full_inv = bar0 + 64, empty_inv = bar0 + 72;
  const uint32_t full_krm = bar0 + 80, empty_krm = bar0 + 88;
  const uint32_t d1_bar = bar0 + 96, d2_bar = bar0 + 104, d3_bar = bar0 + 112;
  const uint32_t out_full0 = bar0 + 120, d4_bar = bar0 + 144, full_gt0 = bar0 + 152;
  uint32_t* tmem_slot = reinterpret_cast<uint32_t*>(smem + kOffBar + 8 * kNumBars);

  if (threadIdx.x == 0) {
    for (int i = 0; i < kKdqSlots; ++i) {
      mbar_init(full_kdq0 + 8 * i, 1);
      mbar_init(empty_kdq0 + 8 * i, 1);
      mbar_init(out_full0 + 8 * i, 4);
    }
    mbar_init(full_vb, 1);
    mbar_init(empty_vb, 4);
    mbar_init(full_inv, 1);
    mbar_init(empty_inv, 1);
    mbar_init(full_krm, 1);
    mbar_init(empty_krm, 1);
    mbar_init(d1_bar, 1);
    mbar_init(d2_bar, 1);
    mbar_init(d3_bar, 1);
    mbar_init(d4_bar, 1);
    mbar_init(full_gt0, 1);
    mbar_init(full_gt0 + 8, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == 7) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;" ::
            "r"(smem_u32(tmem_slot))
        : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::
                     : "memory");
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tmem_slot;

  const int bos = cu_seqlens[seq];
  const int seq_len = cu_seqlens[seq + 1] - bos;
  const int n_chunks = (seq_len + kC - 1) / kC;
  int cbase = 0;
  for (int i = 0; i < seq; ++i) cbase += (cu_seqlens[i + 1] - cu_seqlens[i] + kC - 1) / kC;
  // flags bit 1: every chunk reads chunk 0's images (L2-resident: the
  // recurrence's compute time without the workspace HBM stream).
  const bool reuse = (flags & 2) != 0;
  auto img = [&](int c) { return int64_t(head) * C_total + cbase + (reuse ? 0 : c); };
  const bool dbg_cta = (flags & 1) && blockIdx.x == 0 && blockIdx.y == 0;
  // flags bit 3: per-CTA [start ns, end ns, smid] at dbg[1024 + 4 * cta]
  const int cta_id = blockIdx.y * gridDim.x + blockIdx.x;
  auto gtimer = []() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
    return (long long)t;
  };
  if ((flags & 8) && threadIdx.x == 0) {
    unsigned sm;
    asm volatile("mov.u32 %0, %smid;" : "=r"(sm));
    dbg[1024 + 4 * cta_id] = gtimer();
    dbg[1024 + 4 * cta_id + 2] = sm;
  }
  auto kdq_off = [](int c) { return kOffKdq + 32768 * (c % kKdqSlots); };
  // (slot, parity) of chunk c's use of a kKdqSlots ring barrier
  auto ring = [](uint32_t bar, int c) { return bar + 8 * uint32_t(c % kKdqSlots); };
  auto rpar = [](int c) { return uint32_t(c / kKdqSlots) & 1u; };

  constexpr uint32_t kI64 = idesc_bf16(64, 0);
  constexpr uint32_t kI128 = idesc_bf16(128, 0);

  if (warp < 4) {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 216;");
    const int v = threadIdx.x;
    const bool issuer = threadIdx.x == 0;
    const uint32_t trow = tmem + (uint32_t(warp * 32) << 16);
    auto stamp = [&](int c, int slot) {
      if (dbg_cta && issuer && c < 64) dbg[c * kDbgSlots + slot] = clock64();
    };
    // MMA1(c) K-step j: W (kd rows) and Oq (qd rows, +1024 B), N = 64 each.
    auto mma1w = [&](int c, int j) {
      mma_ts(tmem + kColW, tmem + kColP + 8 * j,
             sdesc(sbase + kdq_off(c) + 4096 * j, 2048, 128), kI64, j != 0);
    };
    auto mma1o = [&](int c, int j) {
      mma_ts(tmem + kColO, tmem + kColP + 8 * j,
             sdesc(sbase + kdq_off(c) + 1024 + 4096 * j, 2048, 128), kI64, j != 0);
    };

    float S[kD];
    const int64_t state_off = (int64_t(seq) * H + head) * kD * kD + int64_t(v) * kD;
    if constexpr (HasStateIn) {
      float4 const* src = reinterpret_cast<float4 const*>(init_state + state_off);
#pragma unroll
      for (int i = 0; i < kD / 4; ++i) {
        float4 x = src[i];
        S[4 * i] = x.x;
        S[4 * i + 1] = x.y;
        S[4 * i + 2] = x.z;
        S[4 * i + 3] = x.w;
      }
    } else {
#pragma unroll
      for (int i = 0; i < kD; ++i) S[i] = 0.0f;
    }
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      uint32_t a[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) a[i] = pack_bf16(S[32 * q + 2 * i], S[32 * q + 2 * i + 1]);
      K2TC_ST16(trow + kColP + 16 * q, a);
    }
    tc_wait_st();
    tc_fence_before();
    named_sync();
    if (issuer && n_chunks > 0) {
      wait_wd(full_kdq0, 0, 0, -1, dbg);
      tc_fence_after();
#pragma unroll
      for (int j = 0; j < 8; ++j) mma1w(0, j);
      tc_commit(d1_bar);
#pragma unroll
      for (int j = 0; j < 8; ++j) mma1o(0, j);
    }

    // out(c) = bf16(Oq + Mqk @ U) -> shared [t][v] (chunk c's kd|qd slot)
    // -> bulk stores by warp 7. Runs under MMA2(c + 1), before Oq(c + 1).
    auto stage_out = [&](int c) {
      wait_wd(d4_bar, uint32_t(c) & 1u, 9, c, dbg);
      tc_fence_after();
      uint16_t* os = reinterpret_cast<uint16_t*>(smem + kdq_off(c)) + v;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        uint32_t o[32];
        K2TC_LD32(trow + kColO + 32 * h, o);
        tc_wait_ld();
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          const uint32_t pk = pack_bf16(__uint_as_float(o[2 * i]), __uint_as_float(o[2 * i + 1]));
          os[(32 * h + 2 * i) * kD] = uint16_t(pk & 0xFFFFu);
          os[(32 * h + 2 * i + 1) * kD] = uint16_t(pk >> 16);
        }
      }
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      __syncwarp();
      if (lane == 0) mbar_arrive(ring(out_full0, c));
    };

    for (int c = 0; c < n_chunks; ++c) {
      const uint32_t cp = uint32_t(c) & 1u;
      const bool more = c + 1 < n_chunks;
      stamp(c, 0);

      // ---- x = bf16(bf16(v - bf16(W)) * beta) -> x^T at [0, 32) ----
      wait_wd(d1_bar, cp, 1, c, dbg);
      wait_wd(full_vb, cp, 2, c, dbg);
      tc_fence_after();
      stamp(c, 1);
      {
        uint4 const* vs = reinterpret_cast<uint4 const*>(smem + kOffV) + v;
        uint4 const* bs = reinterpret_cast<uint4 const*>(smem + kOffBeta);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          uint32_t w[32], ax[16];
          K2TC_LD32(trow + kColW + 32 * h, w);
          tc_wait_ld();
#pragma unroll
          for (int g = 0; g < 4; ++g) {
            const uint4 vv = vs[(4 * h + g) * kD];
            const uint4 bb = bs[4 * h + g];
            const uint32_t vp[4] = {vv.x, vv.y, vv.z, vv.w};
            const uint32_t bp[4] = {bb.x, bb.y, bb.z, bb.w};
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const int i = 8 * g + 2 * e;
              ax[4 * g + e] = bf2_sub_mul(vp[e], pack_bf16(__uint_as_float(w[i]),
                                                           __uint_as_float(w[i + 1])), bp[e]);
            }
          }
          K2TC_ST16(trow + kColW + 16 * h, ax);
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(empty_vb);
      tc_wait_st();
      tc_fence_before();
      named_sync();
      if (issuer) {
        wait_wd(full_inv, cp, 3, c, dbg);
        tc_fence_after();
#pragma unroll
        for (int j = 0; j < 4; ++j)
          mma_ts(tmem + kColU, tmem + kColW + 8 * j, sdesc(sbase + kOffInv + 2048 * j, 1024, 128),
                 kI64, j != 0);
        tc_commit(d2_bar);
        tc_commit(empty_inv);
      }
      stamp(c, 2);
      if (c > 0) {
        stage_out(c - 1);
        tc_fence_before();
        named_sync();
        if (issuer) {
          tc_fence_after();
#pragma unroll
          for (int j = 0; j < 8; ++j) mma1o(c, j);
        }
        stamp(c, 6);
      }

      // ---- U = bf16(INV @ x) -> U^T at [0, 32); MMA3 (kr^T @ U, N = 128),
      // then MMA4 (Oq += Mqk @ U) behind it ----
      wait_wd(d2_bar, cp, 4, c, dbg);
      tc_fence_after();
      stamp(c, 3);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        uint32_t u[32], au[16];
        K2TC_LD32(trow + kColU + 32 * h, u);
        tc_wait_ld();
#pragma unroll
        for (int i = 0; i < 16; ++i)
          au[i] = pack_bf16(__uint_as_float(u[2 * i]), __uint_as_float(u[2 * i + 1]));
        K2TC_ST16(trow + kColW + 16 * h, au);
      }
      tc_wait_st();
      tc_fence_before();
      named_sync();
      if (issuer) {
        wait_wd(full_krm, cp, 5, c, dbg);
        tc_fence_after();
#pragma unroll
        for (int j = 0; j < 4; ++j)
          mma_ts(tmem + kColP, tmem + kColW + 8 * j, sdesc(sbase + kOffKr + 4096 * j, 2048, 128),
                 kI128, j != 0);
        tc_commit(d3_bar);
#pragma unroll
        for (int j = 0; j < 4; ++j)
          mma_ts(tmem + kColO, tmem + kColW + 8 * j, sdesc(sbase + kOffMqk + 2048 * j, 1024, 128),
                 kI64, 1);
        tc_commit(d4_bar);
        tc_commit(empty_krm);
      }
      stamp(c, 4);

      // ---- S = fma(S, g_total, kr^T @ U) in 32-column batches; bf16 S^T
      // lands behind the reads and feeds W(c + 1) ----
      wait_wd(d3_bar, cp, 6, c, dbg);
      wait_wd(full_gt0 + 8 * cp, uint32_t(c >> 1) & 1u, 7, c, dbg);
      tc_fence_after();
      stamp(c, 8);
      float const* gt = reinterpret_cast<float const*>(smem + kOffGt) + 128 * cp;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        uint32_t p[32], a[16];
        K2TC_LD32(trow + kColP + 32 * q, p);
        tc_wait_ld();
        if (q == 0) stamp(c, 9);
#pragma unroll
        for (int i = 0; i < 32; i += 4) {
          float4 g4 = *reinterpret_cast<float4 const*>(gt + 32 * q + i);
          float* s4 = S + 32 * q + i;
          ffma2(s4[0], s4[1], s4[0], s4[1], g4.x, g4.y, __uint_as_float(p[i]),
                __uint_as_float(p[i + 1]));
          ffma2(s4[2], s4[3], s4[2], s4[3], g4.z, g4.w, __uint_as_float(p[i + 2]),
                __uint_as_float(p[i + 3]));
          a[i / 2] = pack_bf16(s4[0], s4[1]);
          a[i / 2 + 1] = pack_bf16(s4[2], s4[3]);
        }
        K2TC_ST16(trow + kColP + 16 * q, a);
        if (q == 0) stamp(c, 10);
        if (more && (q & 1)) {
          tc_wait_st();
          tc_fence_before();
          named_sync();
          if (issuer) {
            if (q == 1) wait_wd(ring(full_kdq0, c + 1), rpar(c + 1), 8, c, dbg);
            tc_fence_after();
#pragma unroll
            for (int j = 0; j < 4; ++j) mma1w(c + 1, 2 * (q - 1) + j);
            if (q == 3) tc_commit(d1_bar);
          }
        }
        if (q == 1) stamp(c, 7);
      }
      stamp(c, 5);

    }
    if (n_chunks > 0) stage_out(n_chunks - 1);
    {
      float4* dst = reinterpret_cast<float4*>(final_state + state_off);
#pragma unroll
      for (int i = 0; i < kD / 4; ++i)
        dst[i] = make_float4(S[4 * i], S[4 * i + 1], S[4 * i + 2], S[4 * i + 3]);
    }
  } else {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    if (warp == 4) {
      for (int c = 0; c < n_chunks; ++c) {
        if (c >= kKdqSlots) wait_wd(ring(empty_kdq0, c), rpar(c - kKdqSlots), 10, c, dbg);
        if (lane == 0) {
          const uint32_t fb = ring(full_kdq0, c);
          mbar_expect_tx(fb, 32768);
          bulk_g2s(sbase + kdq_off(c), kdq_img + img(c) * (2 * kC * kD), 32768, fb);
        }
        __syncwarp();
      }
    } else if (warp == 5) {
      for (int c = 0; c < n_chunks; ++c) {
        const int valid = min(kC, seq_len - c * kC);
        const int tok0 = bos + c * kC;
        if (c >= 1) wait_wd(empty_vb, uint32_t(c - 1) & 1u, 11, c, dbg);
        __nv_bfloat16* bsm = reinterpret_cast<__nv_bfloat16*>(smem + kOffBeta);
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int t = lane + 32 * e;
          bsm[t] = t < valid ? beta_ptr[int64_t(head) * T_total + tok0 + t] : __float2bfloat16(0.0f);
        }
        __syncwarp();
        if (lane == 0) {
          mbar_expect_tx(full_vb, kC * kD * 2);
          bulk_g2s(sbase + kOffV, v_img + img(c) * (kC * kD), kC * kD * 2, full_vb);
        }
        if (c >= 1) wait_wd(empty_inv, uint32_t(c - 1) & 1u, 12, c, dbg);
        if (lane == 0) {
          mbar_expect_tx(full_inv, 8192);
          bulk_g2s(sbase + kOffInv, inv_img + img(c) * (kC * kC), 8192, full_inv);
        }
        __syncwarp();
      }
    } else if (warp == 6) {
      for (int c = 0; c < n_chunks; ++c) {
        if (c >= 1) wait_wd(empty_krm, uint32_t(c - 1) & 1u, 13, c, dbg);
        if (lane == 0) {
          mbar_expect_tx(full_krm, 16384 + 8192);
          bulk_g2s(sbase + kOffKr, kr_img + img(c) * (kC * kD), 16384, full_krm);
          bulk_g2s(sbase + kOffMqk, mqk_img + img(c) * (kC * kC), 8192, full_krm);
          mbar_expect_tx(full_gt0 + 8 * (c & 1), 512);
          bulk_g2s(sbase + kOffGt + 512 * (c & 1), gt_ptr + img(c) * kD, 512, full_gt0 + 8 * (c & 1));
        }
        __syncwarp();
      }
    } else {
      // warp 7: out(c) rows from shared memory, then release the buffer.
      for (int c = 0; c < n_chunks; ++c) {
        const int valid = min(kC, seq_len - c * kC);
        wait_wd(ring(out_full0, c), rpar(c), 14, c, dbg);
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int t = lane + 32 * e;
          if (t < valid) {
            __nv_bfloat16* dst = out_ptr + (int64_t(bos) + c * kC + t) * H * kD + int64_t(head) * kD;
            asm volatile(
                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], 256;" ::"l"(dst),
                "r"(sbase + kdq_off(c) + t * 256)
                : "memory");
          }
        }
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
        __syncwarp();
        if (lane == 0) mbar_arrive(ring(empty_kdq0, c));
      }
      asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 7) {
    tc_fence_after();
    if ((flags & 8) && lane == 0) dbg[1024 + 4 * cta_id + 1] = gtimer();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;" ::"r"(tmem)
                 : "memory");
  }
#endif
}

}  // namespace k64tc

void launch_k64_tc(void const* kdq, void const* kr, void const* inv, void const* mqk,
                   float const* gt, void const* beta, void const* v,
                   float const* init_state, float* final_state, void* out,
                   int const* cu_seqlens, int T_total, int H, int N, int C_total,
                   int flags, long long* dbg, cudaStream_t stream) {
  using k64tc::k64_tc_kernel;
  auto launch = [&](auto kernel) {
    // two ~116 KB CTAs per SM need the sm_107 oversized shared-memory mode
    // (the sm_100-compatible carveout stops at 228 KB per SM).
    cudaFuncSetAttribute(kernel, cudaFuncAttributeSharedMemoryMode,
                         cudaSharedMemoryModeAllowOversizedSharedMemory);
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, k64tc::kSmemBytes);
    kernel<<<dim3(H, N), dim3(256), k64tc::kSmemBytes, stream>>>(
        static_cast<__nv_bfloat16 const*>(kdq), static_cast<__nv_bfloat16 const*>(kr),
        static_cast<__nv_bfloat16 const*>(inv), static_cast<__nv_bfloat16 const*>(mqk), gt,
        static_cast<__nv_bfloat16 const*>(beta), static_cast<__nv_bfloat16 const*>(v),
        init_state, final_state, static_cast<__nv_bfloat16*>(out), cu_seqlens, T_total, H,
        N, C_total, flags, dbg);
  };
  if (init_state != nullptr) {
    launch(k64_tc_kernel<true>);
  } else {
    launch(k64_tc_kernel<false>);
  }
}
