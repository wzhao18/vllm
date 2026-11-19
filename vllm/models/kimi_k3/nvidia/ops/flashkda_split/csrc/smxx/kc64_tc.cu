// Kimi-K3: chunk-64 KDA state chain with the FP32 state in tensor memory
// (dev stage of the fused SOL KDA prefill kernel).
//
// Chunked delta rule in the WY form (fla chunk_kda / FlashKDA math with the
// INV products moved into the prepare), per 64-token chunk c of one
// (sequence, head), state S [K = 128 k][V = 128 v] fp32:
//   W    = w @ bf16(S)          Oq = qd @ bf16(S)              (fp32 accum)
//   vnew = bf16(u - W)                                          (fp32 sub)
//   out  = bf16(Oq + Mqk @ vnew)                                (fp32 accum)
//   S    = g_total * S + kr^T @ vnew                            (fp32 state)
// with w = bf16(INV @ (beta * kd)), u = bf16(INV @ (beta * v)), qd, kr, Mqk,
// g_total from the chunk-64 prepare (same decays and INV as prep64).
//
// Orientation: tokens / k on the tensor-memory lanes, v on the columns.
//   D1 [128 lanes, 128 cols]  lanes 0-63 W (token t), lanes 64-127 Oq -> out
//   S  [128 lanes = k, 128 cols = v] fp32: the accumulator of kr^T @ vnew,
//      pre-multiplied by g_total of its next chunk by the state pass
// All MMAs are SS (A and B in shared memory): A images K-major (w|qd rows,
// kr^T rows, Mqk rows 64-127 over zero rows 0-63), B images MN-major
// (bf16 S [k][v] and vnew [t][v], written row-per-thread by the SIMT warps).
// The FP32 state never lives in registers, so 8 SIMT warps (2 per lane
// quadrant, one per 64-column half) share every per-chunk pass and two CTAs
// fit per SM (256 tensor-memory columns, ~161 KB shared memory each).
//
// Per chunk: MMA1 (W|Oq, M128 N128 K128) -> vnew warps (lanes 0-63) ->
// MMA2 (Mqk @ vnew into D1) + MMA3 (kr^T @ vnew into S) -> out warps
// (lanes 64-127) and the state pass (all lanes: bf16 S image for MMA1(c+1),
// S *= g_total(c+1) in place).
//
// Warps: 0 loader w|qd, 1 MMA issuer + tensor-memory owner, 2 loader u,
// 3 loader kr^T + Mqk + g_total (2-slot ring), 4-11 SIMT.
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "k2_tc.cuh"

namespace kc64tc {

using namespace k2tc;

constexpr int kC = 64;
constexpr int kD = 128;
constexpr int kThreads = 384;

constexpr int kOffWq = 0;           // A [16 k-groups][128 rows][8] bf16
constexpr int kOffU = 32768;        // [64 t][128 v] bf16
constexpr int kOffKrm = 49152;      // 2 slots x {kr^T A 16 KB, Mqk A 16 KB, g_total 512 B}
constexpr int kKrmSlot = 33280;
constexpr int kOffSb = kOffKrm + 2 * kKrmSlot;   // B [16 kg][16 vg][8 k][8 v] bf16
constexpr int kOffVn = kOffSb + 32768;           // B [8 tg][16 vg][8 t][8 v] bf16
constexpr int kOffBar = kOffVn + 16384;
// barriers: full_wq, empty_wq, full_u, empty_u, full_krm[2], empty_krm[2],
// then per v-half chain h: d1, d3, vnew_rdy, sb_rdy, o_rd (at bar0 + 64 + 40 h)
constexpr int kNumBars = 18;
constexpr int kSmemBytes = kOffBar + 8 * kNumBars + 16;

constexpr uint32_t kColD1 = 0;
constexpr uint32_t kColS = 128;

__device__ __forceinline__ void mma_ss(uint32_t d, uint64_t a_desc, uint64_t b_desc,
                                       uint32_t idesc, uint32_t accumulate) {
  asm volatile(
      "{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n"
      " tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}" ::"r"(d),
      "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(accumulate)
      : "memory");
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

// Dev watchdog: a wait still pending after 2^22 polls records (block, warp,
// barrier id, chunk) in dbg[0] (bit 63) and gives up instead of hanging.
__device__ __forceinline__ void wait_wd(uint32_t bar, uint32_t parity, int id, int c,
                                        long long* dbg) {
  for (uint32_t i = 0; !mbar_try(bar, parity); ++i) {
    if (i == (1u << 20)) {
      atomicCAS(reinterpret_cast<unsigned long long*>(dbg), 0ull,
                (1ull << 63) | (uint64_t(blockIdx.y) << 40) | (uint64_t(blockIdx.x) << 24) |
                    (uint64_t(threadIdx.x >> 5) << 16) | (uint64_t(id) << 10) | uint64_t(c & 1023));
      return;
    }
  }
}

__device__ __forceinline__ long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return (long long)t;
}

template <bool HasStateIn>
__global__ void __launch_bounds__(kThreads, 2) kc64_tc_kernel(
    __nv_bfloat16 const* __restrict__ wq_img,   // [H][C][16][128][8]
    __nv_bfloat16 const* __restrict__ u_img,    // [H][C][64][128]
    __nv_bfloat16 const* __restrict__ kr_img,   // [H][C][8][128][8]
    __nv_bfloat16 const* __restrict__ mqk_img,  // [H][C][8][64][8]
    float const* __restrict__ gt_ptr,           // [H][C][128]
    float const* __restrict__ init_state,       // [N][H][128 v][128 k]
    float* __restrict__ final_state,
    __nv_bfloat16* __restrict__ out_ptr,        // [T][H][128]
    int const* __restrict__ cu_seqlens, int H, int N, int C_total, int flags,
    long long* __restrict__ dbg) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  extern __shared__ __align__(1024) unsigned char smem[];
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int head = blockIdx.x;
  const int seq = N - 1 - int(blockIdx.y);
  const uint32_t sbase = smem_u32(smem);
  const uint32_t bar0 = sbase + kOffBar;
  const uint32_t full_wq = bar0, empty_wq = bar0 + 8, full_u = bar0 + 16, empty_u = bar0 + 24;
  const uint32_t full_krm0 = bar0 + 32, empty_krm0 = bar0 + 48;
  // The two v halves of S are independent chains (v columns never mix):
  // each has its own issuer and hand-off barriers, so one half's MMAs run
  // under the other half's SIMT passes.
  auto hb = [&](int h, int i) { return bar0 + 64 + 40 * uint32_t(h) + 8 * uint32_t(i); };
  uint32_t* tmem_slot = reinterpret_cast<uint32_t*>(smem + kOffBar + 8 * kNumBars);

  // Rows 0-63 of both Mqk A images stay zero (MMA2's dead lanes 0-63).
  for (int i = threadIdx.x; i < 2 * 8 * 64; i += kThreads) {
    const int s = i >> 9, tg = (i >> 6) & 7, r = i & 63;
    *reinterpret_cast<uint4*>(smem + kOffKrm + s * kKrmSlot + 16384 + tg * 2048 + r * 16) =
        make_uint4(0, 0, 0, 0);
  }
  if (threadIdx.x == 0) {
    mbar_init(full_wq, 1);
    mbar_init(empty_wq, 2);
    mbar_init(full_u, 1);
    mbar_init(empty_u, 4);
    for (int s = 0; s < 2; ++s) {
      mbar_init(full_krm0 + 8 * s, 1);
      mbar_init(empty_krm0 + 8 * s, 2);
    }
    for (int h = 0; h < 2; ++h) {
      mbar_init(hb(h, 0), 1);
      mbar_init(hb(h, 1), 1);
      mbar_init(hb(h, 2), 2);
      mbar_init(hb(h, 3), 4);
      mbar_init(hb(h, 4), 2);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == 1) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;" ::
            "r"(smem_u32(tmem_slot))
        : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tmem_slot;

  const int bos = cu_seqlens[seq];
  const int seq_len = cu_seqlens[seq + 1] - bos;
  const int n_chunks = (seq_len + kC - 1) / kC;
  int cbase = 0;
  for (int i = 0; i < seq; ++i) cbase += (cu_seqlens[i + 1] - cu_seqlens[i] + kC - 1) / kC;
  // flags bit 1: every chunk reads chunk 0's images (L2-resident: compute
  // time without the image stream).
  const bool reuse = (flags & 2) != 0;
  auto img = [&](int c) { return int64_t(head) * C_total + cbase + (reuse ? 0 : c); };
  const int cta_id = blockIdx.y * gridDim.x + blockIdx.x;
  const bool dbg_cta = (flags & 1) && blockIdx.x == 0 && blockIdx.y == 0;
  if ((flags & 8) && threadIdx.x == 0) {
    unsigned sm;
    asm volatile("mov.u32 %0, %smid;" : "=r"(sm));
    dbg[1024 + 4 * cta_id] = gtimer();
    dbg[1024 + 4 * cta_id + 2] = sm;
  }
  auto krm = [](int s) { return kOffKrm + s * kKrmSlot; };

  if (warp == 0) {
    if (lane == 0) {
      for (int c = 0; c < n_chunks; ++c) {
        if (c >= 1) wait_wd(empty_wq, uint32_t(c - 1) & 1u, 1, c, dbg);
        mbar_expect_tx(full_wq, 32768);
        bulk_g2s(sbase + kOffWq, wq_img + img(c) * (2 * kC * kD), 32768, full_wq);
        if (c >= 1) wait_wd(empty_u, uint32_t(c - 1) & 1u, 2, c, dbg);
        mbar_expect_tx(full_u, 16384);
        bulk_g2s(sbase + kOffU, u_img + img(c) * (kC * kD), 16384, full_u);
      }
    }
  } else if (warp == 3) {
    if (lane == 0) {
      for (int c = 0; c < n_chunks; ++c) {
        const int s = c & 1;
        if (c >= 2) wait_wd(empty_krm0 + 8 * s, uint32_t((c - 2) >> 1) & 1u, 3, c, dbg);
        const uint32_t fb = full_krm0 + 8 * s;
        mbar_expect_tx(fb, 16384 + 8192 + 512);
        bulk_g2s(sbase + krm(s), kr_img + img(c) * (kC * kD), 16384, fb);
#pragma unroll
        for (int tg = 0; tg < 8; ++tg)
          bulk_g2s(sbase + krm(s) + 16384 + tg * 2048 + 1024,
                   mqk_img + img(c) * (kC * kC) + tg * 512, 1024, fb);
        bulk_g2s(sbase + krm(s) + 32768, gt_ptr + img(c) * kD, 512, fb);
      }
    }
  } else if (warp == 1 || warp == 2) {
    const int h = warp - 1;
    const uint32_t d1_bar = hb(h, 0), d3_bar = hb(h, 1), vnew_rdy = hb(h, 2);
    const uint32_t sb_rdy = hb(h, 3), o_rd = hb(h, 4);
    if (lane == 0 && n_chunks > 0) {
      constexpr uint32_t kI = idesc_bf16(64, 1);  // A K-major, B MN-major, M128 N64
      auto stamp = [&](int c, int slot) {
        if (dbg_cta && h == 0 && c < 96) dbg[16 + c * 8 + slot] = clock64();
      };
      for (int c = 0; c < n_chunks; ++c) {
        const int s = c & 1;
        stamp(c, 0);
        wait_wd(full_wq, uint32_t(c) & 1u, 4, c, dbg);
        wait_wd(sb_rdy, uint32_t(c) & 1u, 5, c, dbg);
        if (c >= 1) wait_wd(o_rd, uint32_t(c - 1) & 1u, 6, c, dbg);
        tc_fence_after();
        stamp(c, 1);
#pragma unroll
        for (int j = 0; j < 8; ++j)
          mma_ss(tmem + kColD1 + 64 * h, sdesc(sbase + kOffWq + 4096 * j, 2048, 128),
                 sdesc(sbase + kOffSb + 1024 * h + 4096 * j, 2048, 128), kI, j != 0);
        tc_commit(d1_bar);
        tc_commit(empty_wq);
        wait_wd(vnew_rdy, uint32_t(c) & 1u, 7, c, dbg);
        wait_wd(full_krm0 + 8 * s, uint32_t(c >> 1) & 1u, 8, c, dbg);
        tc_fence_after();
        stamp(c, 2);
#pragma unroll
        for (int j = 0; j < 4; ++j)
          mma_ss(tmem + kColD1 + 64 * h, sdesc(sbase + krm(s) + 16384 + 4096 * j, 2048, 128),
                 sdesc(sbase + kOffVn + 1024 * h + 4096 * j, 2048, 128), kI, 1);
#pragma unroll
        for (int j = 0; j < 4; ++j)
          mma_ss(tmem + kColS + 64 * h, sdesc(sbase + krm(s) + 4096 * j, 2048, 128),
                 sdesc(sbase + kOffVn + 1024 * h + 4096 * j, 2048, 128), kI, 1);
        tc_commit(d3_bar);
        tc_commit(empty_krm0 + 8 * s);
        stamp(c, 3);
      }
    }
  } else {
    // SIMT warps: quadrant qd (tensor-memory lanes 32 qd ..), column half hv.
    const int qd = warp & 3;
    const int hv = (warp - 4) >> 2;
    const uint32_t d1_bar = hb(hv, 0), d3_bar = hb(hv, 1), vnew_rdy = hb(hv, 2);
    const uint32_t sb_rdy = hb(hv, 3), o_rd = hb(hv, 4);
    const int r = 32 * qd + lane;  // lane row: token (D1) or k (S)
    const uint32_t trow = tmem + (uint32_t(32 * qd) << 16);
    uint4* sb = reinterpret_cast<uint4*>(smem + kOffSb);
    const int64_t st_base = (int64_t(seq) * H + head) * kD * kD;

    // bf16 S image row r (k), columns [c0, c0 + 32): 4 core-matrix rows.
    const bool sst = dbg_cta && warp == 4 && lane == 0;
    auto sstamp = [&](int c, int slot) {
      if (sst && c < 96) dbg[16 + c * 8 + slot] = clock64();
    };
    auto pstamp = [&](int c, int i) {
      if (sst && c < 48) dbg[16 + 96 * 8 + c * 4 + i] = clock64();
    };
    auto put_sb = [&](int c0, uint32_t const* pk) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int vg = (c0 >> 3) + i;
        sb[((r >> 3) * 16 + vg) * 8 + (r & 7)] =
            make_uint4(pk[4 * i], pk[4 * i + 1], pk[4 * i + 2], pk[4 * i + 3]);
      }
    };

    // Initial state: S0 (or 0) -> bf16 image, g_total(0) * S0 -> tensor memory.
    if (n_chunks > 0) {
      wait_wd(full_krm0, 0, 9, -1, dbg);
      const float g0 = reinterpret_cast<float const*>(smem + krm(0) + 32768)[r];
#pragma unroll
      for (int b = 0; b < 2; ++b) {
        const int c0 = 64 * hv + 32 * b;
        float x[32];
#pragma unroll
        for (int i = 0; i < 32; ++i)
          x[i] = HasStateIn ? init_state[st_base + int64_t(c0 + i) * kD + r] : 0.0f;
        uint32_t pk[16], sv[32];
#pragma unroll
        for (int i = 0; i < 16; ++i) pk[i] = pack_bf16(x[2 * i], x[2 * i + 1]);
        put_sb(c0, pk);
#pragma unroll
        for (int i = 0; i < 32; ++i) sv[i] = __float_as_uint(x[i] * g0);
        K2TC_ST16(trow + kColS + c0, sv);
        K2TC_ST16(trow + kColS + c0 + 16, (sv + 16));
      }
      tc_wait_st();
      tc_fence_before();
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      __syncwarp();
      if (lane == 0) mbar_arrive(sb_rdy);
    }

    for (int c = 0; c < n_chunks; ++c) {
      const uint32_t cp = uint32_t(c) & 1u;
      const bool more = c + 1 < n_chunks;
      if (qd < 2) {
        // vnew = bf16(u - W), token t = r.
        wait_wd(d1_bar, cp, 10, c, dbg);
        wait_wd(full_u, cp, 11, c, dbg);
        tc_fence_after();
        sstamp(c, 4);
        uint4 const* us = reinterpret_cast<uint4 const*>(smem + kOffU + r * 256);
        uint4* vn = reinterpret_cast<uint4*>(smem + kOffVn);
#pragma unroll
        for (int b = 0; b < 2; ++b) {
          const int c0 = 64 * hv + 32 * b;
          uint32_t w[32];
          K2TC_LD32(trow + kColD1 + c0, w);
          tc_wait_ld();
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const uint4 uu = us[(c0 >> 3) + i];
            const uint32_t up[4] = {uu.x, uu.y, uu.z, uu.w};
            uint32_t o[4];
#pragma unroll
            for (int e = 0; e < 4; ++e)
              o[e] = pack_bf16(bf16_lo(up[e]) - __uint_as_float(w[8 * i + 2 * e]),
                               bf16_hi(up[e]) - __uint_as_float(w[8 * i + 2 * e + 1]));
            vn[((r >> 3) * 16 + (c0 >> 3) + i) * 8 + (r & 7)] = make_uint4(o[0], o[1], o[2], o[3]);
          }
        }
        tc_fence_before();
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncwarp();
        sstamp(c, 5);
        if (lane == 0) {
          mbar_arrive(empty_u);
          mbar_arrive(vnew_rdy);
        }
      }
      wait_wd(d3_bar, cp, 12, c, dbg);
      tc_fence_after();
      sstamp(c, 6);
      if (qd >= 2) {
        // out row t = r - 64 (MMA2 done: d3 commits after MMA2 and MMA3).
        const int t = r - 64;
        const bool ok = c * kC + t < seq_len;
        uint4* dst = reinterpret_cast<uint4*>(out_ptr + ((int64_t(bos) + c * kC + t) * H + head) * kD);
#pragma unroll
        for (int b = 0; b < 2; ++b) {
          const int c0 = 64 * hv + 32 * b;
          uint32_t o[32];
          K2TC_LD32(trow + kColD1 + c0, o);
          tc_wait_ld();
          if (ok) {
#pragma unroll
            for (int i = 0; i < 4; ++i)
              dst[(c0 >> 3) + i] = make_uint4(
                  pack_bf16(__uint_as_float(o[8 * i]), __uint_as_float(o[8 * i + 1])),
                  pack_bf16(__uint_as_float(o[8 * i + 2]), __uint_as_float(o[8 * i + 3])),
                  pack_bf16(__uint_as_float(o[8 * i + 4]), __uint_as_float(o[8 * i + 5])),
                  pack_bf16(__uint_as_float(o[8 * i + 6]), __uint_as_float(o[8 * i + 7])));
          }
        }
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(o_rd);
      }
      // State pass, k = r: bf16 image of S(c) for MMA1(c + 1), then
      // S *= g_total(c + 1) in place (the final state stays undecayed).
      float gn = 1.0f;
      if (more) {
        wait_wd(full_krm0 + 8 * ((c + 1) & 1), uint32_t((c + 1) >> 1) & 1u, 13, c, dbg);
        gn = reinterpret_cast<float const*>(smem + krm((c + 1) & 1) + 32768)[r];
      }
#pragma unroll
      for (int b = 0; b < 2; ++b) {
        const int c0 = 64 * hv + 32 * b;
        uint32_t sv[32];
        K2TC_LD32(trow + kColS + c0, sv);
        tc_wait_ld();
        if (b == 0) pstamp(c, 0);
        if (more) {
          uint32_t pk[16];
#pragma unroll
          for (int i = 0; i < 16; ++i)
            pk[i] = pack_bf16(__uint_as_float(sv[2 * i]), __uint_as_float(sv[2 * i + 1]));
          put_sb(c0, pk);
          if (b == 0) pstamp(c, 1);
#pragma unroll
          for (int i = 0; i < 32; ++i) sv[i] = __float_as_uint(__uint_as_float(sv[i]) * gn);
          K2TC_ST16(trow + kColS + c0, sv);
          K2TC_ST16(trow + kColS + c0 + 16, (sv + 16));
        } else {
#pragma unroll
          for (int i = 0; i < 32; ++i)
            final_state[st_base + int64_t(c0 + i) * kD + r] = __uint_as_float(sv[i]);
        }
      }
      if (more) {
        pstamp(c, 2);
        tc_wait_st();
        tc_fence_before();
        pstamp(c, 3);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncwarp();
        sstamp(c, 7);
        if (lane == 0) mbar_arrive(sb_rdy);
      }
    }
    if (n_chunks == 0) {
      for (int v = 64 * hv; v < 64 * hv + 64; ++v)
        final_state[st_base + int64_t(v) * kD + r] =
            HasStateIn ? init_state[st_base + int64_t(v) * kD + r] : 0.0f;
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) {
    tc_fence_after();
    if ((flags & 8) && lane == 0) dbg[1024 + 4 * cta_id + 1] = gtimer();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;" ::"r"(tmem) : "memory");
  }
#endif
}

}  // namespace kc64tc

void launch_kc64_tc(void const* wq, void const* u, void const* kr, void const* mqk,
                    float const* gt, float const* init_state, float* final_state, void* out,
                    int const* cu_seqlens, int H, int N, int C_total, int flags, long long* dbg,
                    cudaStream_t stream) {
  using kc64tc::kc64_tc_kernel;
  auto launch = [&](auto kernel) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeSharedMemoryMode,
                         cudaSharedMemoryModeAllowOversizedSharedMemory);
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kc64tc::kSmemBytes);
    kernel<<<dim3(H, N), dim3(kc64tc::kThreads), kc64tc::kSmemBytes, stream>>>(
        static_cast<__nv_bfloat16 const*>(wq), static_cast<__nv_bfloat16 const*>(u),
        static_cast<__nv_bfloat16 const*>(kr), static_cast<__nv_bfloat16 const*>(mqk), gt,
        init_state, final_state, static_cast<__nv_bfloat16*>(out), cu_seqlens, H, N, C_total,
        flags, dbg);
  };
  if (init_state != nullptr) {
    launch(kc64_tc_kernel<true>);
  } else {
    launch(kc64_tc_kernel<false>);
  }
}
