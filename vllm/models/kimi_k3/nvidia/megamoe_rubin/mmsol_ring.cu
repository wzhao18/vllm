// SPDX-License-Identifier: Apache-2.0
// EXPERIMENT ONLY (hwdb candidate microbench, never for merge): what does the TMA-fed FULL/EMPTY ring add per stage to a
// 2-CTA K256 block-scaled NVFP4 stage on VR200 (sm_107a)? Minimal ring around the tight compile-time MMA warp of
// mmsol_stage.cu (T8: {208,96} x 2 K128 + 16 SF tcgen05.cp + commit = 549 clk). Per stage the chain is
//   producer waits EMPTY -> (expect_tx + TMA | plain arrive) -> complete_tx -> MMA warp waits FULL -> 16 cp + 4 MMA ->
//   tcgen05.commit (multicast) -> EMPTY.
// Compile-time arms: CG2 (2-CTA pair, TMA completes on the leader barrier) vs 1-CTA, ring depth D, K256 units per stage
// KS, TM (0 no ring / 1 plain arrive, no TMA / 2 TMA 16 B boxes / 3 TMA real boxes A 16K+2K SF, B 19K+6K SF per CTA),
// RINGS (2 = A and B rings as the skel / 1 = one barrier), NISS issuers per ring (alternate stages), OPS per ring stage
// (2 = data + SF, 1 = data), PF (0 = lane-0 thread, 1 = converged warp + elect.sync), HINT (try_wait suspend hint as
// the skel / none), MMA (1 = MMA consumer / 0 = consumer only waits FULL and plain-arrives EMPTY), PEER (TM 1: the peer
// CTA's producer also arrives on the leader FULL, count 2; 0 = leader only).
// Per-stage trace of block 0 (leader) for the link decomposition.
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a mmsol_ring.cu -o ring -lcuda ; run: ./ring [S] [arm]
#include "mmsol_core.cuh"
#include <cudaTypedefs.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
using namespace mm;

constexpr int W0 = 64, TRN = 256, NOUT = 8;
struct Maps { CUtensorMap a, b, sa, sb, t; };

MMD bool elect1() {
  uint32_t p; asm volatile("{.reg .pred P; elect.sync _|P, 0xffffffff; selp.u32 %0, 1, 0, P;}" : "=r"(p)); return p != 0;
}
template <int HINT> MMD void wt(uint32_t b, uint32_t ph, int what) {
  uint32_t ok;
  if (HINT) asm volatile("{.reg .pred P; mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2, 1000000; selp.u32 %0, 1, 0, P;}" : "=r"(ok) : "r"(b), "r"(ph) : "memory");
  else asm volatile("{.reg .pred P; mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2; selp.u32 %0, 1, 0, P;}" : "=r"(ok) : "r"(b), "r"(ph) : "memory");
  if (ok) return;
  const uint64_t t0 = clk64();
  for (;;) {
    if (HINT) asm volatile("{.reg .pred P; mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2, 1000000; selp.u32 %0, 1, 0, P;}" : "=r"(ok) : "r"(b), "r"(ph) : "memory");
    else asm volatile("{.reg .pred P; mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2; selp.u32 %0, 1, 0, P;}" : "=r"(ok) : "r"(b), "r"(ph) : "memory");
    if (ok) return;
    if (clk64() - t0 > 6000000000ull) { printf("RING_TIMEOUT what=%d block=%d warp=%d ph=%u\n", what, blockIdx.x, threadIdx.x / 32, ph); __trap(); }
  }
}
template <int CG2> MMD void tma3(uint32_t dst, const CUtensorMap* m, int c0, int c1, int c2, uint32_t bar) {
  if (CG2)
    asm volatile("cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
                 :: "r"(dst), "l"(m), "r"(c0), "r"(c1), "r"(c2), "r"(bar) : "memory");
  else
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
                 :: "r"(dst), "l"(m), "r"(c0), "r"(c1), "r"(c2), "r"(bar) : "memory");
}
template <int CG2> MMD void mma4(uint32_t d, uint64_t a, uint64_t b, uint32_t id, uint32_t sfa, uint32_t sfb, uint32_t acc) {
  if (CG2) mma_nvf4(d, a, b, id, sfa, sfb, acc);
  else asm volatile("{.reg .pred P, Q; elect.sync _|P, 0xffffffff; setp.ne.b32 Q, %6, 0;\n"
                    "@P tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%4], [%5], Q;}"
                    :: "r"(d), "l"(a), "l"(b), "r"(id), "r"(sfa), "r"(sfb), "r"(acc) : "memory");
}
template <int CG2> MMD void cp1(uint32_t ta, uint64_t sd) {
  if (CG2) sf_cp(ta, sd);
  else asm volatile("{.reg .pred P; elect.sync _|P, 0xffffffff; @P tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;}" :: "r"(ta), "l"(sd) : "memory");
}
template <int CG2> MMD void commit1(uint32_t bar) {
  if (CG2) commit_mc(bar, 3);
  else asm volatile("{.reg .pred P; elect.sync _|P, 0xffffffff; @P tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];}" :: "r"(bar) : "memory");
}

// smem plan (bytes per real slot): SF first (tcgen05.cp descriptors need < 256 KiB), then data.
template <int KS, int PROD = 0> struct Lay {  // PROD: production N256 tile (B 128 rows, SFB 4 KiB, 12 cps, 2 x N256 MMAs)
  static constexpr int SFA = 2048 * KS, SFB = (PROD ? 4096 : 6144) * KS, AD = 16384 * KS, BD = (PROD ? 16384 : 19456) * KS;
};

template <int CG2, int D, int KS, int TM, int RINGS, int NISS, int OPS, int PF, int HINT, int MMA, int PEER, int MASK, int PADK, int ABOX, int PROD, int ROT, int FENCE, int PING>
__global__ void __launch_bounds__(192, 1) ring_k(const __grid_constant__ Maps mp, int S, unsigned long long* out, long long* tr) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  __shared__ __align__(8) uint64_t full[2][8], empty[2][8], done;
  __shared__ uint32_t tmem_base;
  using L = Lay<KS, PROD>;
  constexpr int DR = (TM == 3 || ROT >= 1) ? D : 1;  // real operand slots (TM < 3: MMAs read slot 0)
  uint8_t* base = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  uint8_t* sfa = base; uint8_t* sfb = sfa + DR * L::SFA; uint8_t* ad = sfb + DR * L::SFB + PADK * 1024; uint8_t* bd = ad + DR * L::AD;
  uint8_t* tiny = bd + DR * L::BD;                    // TM 2: 128 B per (slot, op)
  const uint32_t cr = CG2 ? cta_rank() : 0;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (threadIdx.x == 0) {
    const uint32_t fc = (TM == 1 && CG2 && PEER) ? 2 : 1;
    for (int r = 0; r < 2; ++r) for (int i = 0; i < 8; ++i) { bar_init(sa(&full[r][i]), fc); bar_init(sa(&empty[r][i]), 1); }
    bar_init(sa(&done), 1); bar_fence_init();
  }
  if (warp == 4) {
    if (CG2) tmem_alloc576(sa(&tmem_base));
    else {
      asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(sa(&tmem_base)) : "memory");
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
  }
  tc_fence_before();
  if (CG2) cluster_sync(); else __syncthreads();
  tc_fence_after();
  const uint32_t tb = tmem_base;
  const bool trace = blockIdx.x == 0 && tr != nullptr;
  constexpr int NR = RINGS;

  // ---------------- producers: warp = ring * 2 + issuer ----------------
  if (TM > 0 && warp < 2 * NR && (warp & 1) < NISS && (cr == 0 || TM >= 2 || PEER) && (PF == 1 || lane == 0)) {
    const int r = warp >> 1, iss = warp & 1;
    const uint32_t full0 = sa(&full[r][0]), emp0 = sa(&empty[r][0]);
    const uint32_t lead0 = CG2 ? mapa(full0, 0) : full0;
    // per-CTA bytes this ring lands per stage
    // MASK bits (TM 3): 1 A data, 2 A SF, 4 B data, 8 B SF real; a cleared bit issues a 16 B box instead (same op count)
    constexpr bool RAD = TM == 3 && (MASK & 1), RAS = TM == 3 && (MASK & 2), RBD = TM == 3 && (MASK & 4), RBS = TM == 3 && (MASK & 8);
    constexpr uint32_t BA = (RAD ? (uint32_t)(ABOX * 128 * KS) : 16u) + (OPS == 2 ? (RAS ? L::SFA : 16u) : 0u);
    constexpr uint32_t BB = (RBD ? L::BD : 16u) + (OPS == 2 ? (RBS ? L::SFB : 16u) : 0u);
    const uint32_t exb = (NR == 2 ? (r == 0 ? BA : BB) : BA + BB) * (CG2 ? 2u : 1u);
    unsigned long long wsum = 0, isum = 0;
    const int g = blockIdx.x;
    if (TM == 5) {  // background TMA traffic only: self-recycled tiny (MASK 0) or real-A ops on private barriers, no ring
      for (int it = 0; it < S; ++it) {
        const int slot = it % D;
        bar_expect(full0 + 8 * slot, MASK ? 16384u : 16u);
        if (MASK) tma3<0>(sa(ad + 0), &mp.a, 0, ((it * 7 + g) & 31) * 128, 0, full0 + 8 * slot);
        else tma3<0>(sa(tiny + (slot * 8 + r * 4) * 128), &mp.t, 0, (it * 8 + g * 64) & 65535, 0, full0 + 8 * slot);
        wt<HINT>(full0 + 8 * slot, (it / D) & 1, 80);
      }
    } else
    #pragma unroll 1
    for (int s = iss; s < S; s += NISS) {
      const int slot = s % D, rr = s / D;
      const uint64_t t0 = clk64();
      if (rr > 0) wt<HINT>(emp0 + 8 * slot, (rr - 1) & 1, 10 + warp);
      if (PF == 1) __syncwarp();
      const uint64_t t1 = clk64();
      const bool me = PF == 1 ? elect1() : true;
      if (me) {
        if (TM == 1) {
          if (cr == 0) bar_arrive(full0 + 8 * slot); else bar_arrive_cluster(lead0 + 8 * slot);
        } else {
          if (cr == 0) bar_expect(full0 + 8 * slot, exb);
          const uint32_t fb = lead0 + 8 * slot;
          const int ro = (s * 7 + g) & 31;
          if (TM == 3) {
            const int ds = slot;  // DR == D
            uint8_t* tp = tiny + (slot * 8) * 128;
            const int tr0 = (ro * 8 + g * 64) & 65535;
            if (NR == 1 || r == 0) {
              if (RAD) tma3<CG2>(sa(ad + ds * L::AD), &mp.a, 0, ro * 128, 0, fb); else tma3<CG2>(sa(tp), &mp.t, 0, tr0, 0, fb);
              if (OPS == 2) { if (RAS) tma3<CG2>(sa(sfa + ds * L::SFA), &mp.sa, 0, ro * 4, 0, fb); else tma3<CG2>(sa(tp + 128), &mp.t, 0, tr0 + 1, 0, fb); }
            }
            if (NR == 1 || r == 1) {
              if (RBD) tma3<CG2>(sa(bd + ds * L::BD), &mp.b, 0, ro * (PROD ? 128 : 152), 0, fb); else tma3<CG2>(sa(tp + 256), &mp.t, 0, tr0 + 2, 0, fb);
              if (OPS == 2) { if (RBS) tma3<CG2>(sa(sfb + ds * L::SFB), &mp.sb, 0, ro * (PROD ? 8 : 12), 0, fb); else tma3<CG2>(sa(tp + 384), &mp.t, 0, tr0 + 3, 0, fb); }
            }
          } else {
            uint8_t* tp = tiny + (slot * 8 + r * 4) * 128;
            const int nop = (NR == 1 ? 2 : 1) * OPS;
            #pragma unroll
            for (int o = 0; o < nop; ++o) tma3<CG2>(sa(tp + o * 128), &mp.t, 0, (ro * 8 + o + g * 64) & 65535, 0, fb);
          }
        }
      }
      const uint64_t t2 = clk64();
      if (s >= W0) { wsum += t1 - t0; isum += t2 - t1; }
      if (trace && r == 0 && cr == 0 && lane == 0 && s >= W0 && s < W0 + TRN) {
        tr[0 * TRN + s - W0] = (long long)t0; tr[1 * TRN + s - W0] = (long long)t1; tr[2 * TRN + s - W0] = (long long)t2;
      }
    }
    if (r == 0 && iss == 0 && cr == 0 && lane == 0) { out[blockIdx.x * NOUT + 3] = wsum; out[blockIdx.x * NOUT + 4] = isum; }
  }

  // ---------------- consumer (MMA warp, leader) ----------------
  if (warp == 4 && cr == 0) {
    const uint64_t dA = desc_sw128(sa(ad)), dB = desc_sw128(sa(bd)), sA = desc_sf(sa(sfa)), sB = desc_sf(sa(sfb));
    constexpr uint32_t MM = CG2 ? 256u : 128u;
    constexpr uint32_t ida = (1u << 3) | (1u << 7) | (1u << 10) | ((uint32_t)(208 >> 3) << 17) | ((MM >> 4) << 24);
    constexpr uint32_t idb = (1u << 3) | (1u << 7) | (1u << 10) | ((uint32_t)(96 >> 3) << 17) | ((MM >> 4) << 24);
    constexpr uint32_t id256 = (1u << 3) | (1u << 7) | (1u << 10) | ((uint32_t)(256 >> 3) << 17) | ((MM >> 4) << 24);
    constexpr uint32_t CA = 0, CB = CG2 ? 416 : 256, CS0 = CG2 ? 512 : 448;
    unsigned long long fsum = 0, msum = 0; uint64_t c0 = 0, n0 = 0;
    if (PING == 2 && TM > 0 && MMA == 1 && KS == 1) {
      // software-pipelined SF: wait FULL(s+1) and issue its cps (TMEM SF image (s+1)&1) BEFORE the MMAs of stage s
      auto waitst = [&](int t) {
        const int sl = t % D; wt<HINT>(sa(&full[0][sl]), (t / D) & 1, 52);
        if (NR == 2) wt<HINT>(sa(&full[1][sl]), (t / D) & 1, 53);
        __syncwarp(); tc_fence_after();
      };
      auto opslot = [&](int t) { return (TM == 3 || ROT == 1) ? t % D : 0; };
      auto cps = [&](int t) {
        const int ds = opslot(t); const uint32_t C = (t & 1) ? 320u : CS0;
        const uint64_t xa = sA + (uint64_t)((ds * L::SFA) >> 4), xb = sB + (uint64_t)((ds * L::SFB) >> 4);
        #pragma unroll
        for (int j = 0; j < 4; ++j) cp1<CG2>(tb + C + 4 * j, xa + 32 * j);
        #pragma unroll
        for (int j = 0; j < 8; ++j) cp1<CG2>(tb + C + 16 + 4 * j, xb + 32 * j);
        if (!PROD) {
          #pragma unroll
          for (int j = 0; j < 4; ++j) cp1<CG2>(tb + C + 48 + 4 * j, xb + 256 + 32 * j);
        }
      };
      auto mmas = [&](int t) {
        const int ds = opslot(t); const uint32_t C = (t & 1) ? 320u : CS0;
        const uint64_t da = dA + (uint64_t)((ds * L::AD) >> 4), db = dB + (uint64_t)((ds * L::BD) >> 4);
        if (PROD) {
          mma4<CG2>(tb + CA, da, db, id256, tb + C, tb + C + 16, 1u);
          mma4<CG2>(tb + CA, da + 4, db + 4, id256, tb + C + 8, tb + C + 32, 1u);
        } else {
          mma4<CG2>(tb + CA, da, db, ida, tb + C, tb + C + 16, 1u);
          mma4<CG2>(tb + CA, da + 4, db + 4, ida, tb + C + 8, tb + C + 32, 1u);
          mma4<CG2>(tb + CB, da, db + 832, idb, tb + C, tb + C + 48, 1u);
          mma4<CG2>(tb + CB, da + 4, db + 836, idb, tb + C + 8, tb + C + 56, 1u);
        }
      };
      waitst(0); cps(0);
      #pragma unroll 1
      for (int s = 0; s < S; ++s) {
        if (s == W0) { c0 = clk64(); n0 = now_ns(); }
        const uint64_t m0 = clk64();
        if (s + 1 < S) waitst(s + 1);
        const uint64_t m1 = clk64();
        if (s + 1 < S) cps(s + 1);
        mmas(s);
        commit1<CG2>(sa(&empty[0][s % D]));
        if (NR == 2) commit1<CG2>(sa(&empty[1][s % D]));
        const uint64_t m2 = clk64();
        if (s >= W0) { fsum += m1 - m0; msum += m2 - m1; }
      }
    } else
    #pragma unroll 1
    for (int s = 0; s < S; ++s) {
      if (s == W0) { c0 = clk64(); n0 = now_ns(); }
      const int slot = s % D;
      const uint32_t CS = (PING && (s & 1)) ? 320u : CS0;  // PING: 2 TMEM SF images (cols 512-575 / 320-383), alternate per stage
      const uint64_t m0 = clk64();
      if (TM > 0 && TM != 5) {
        wt<HINT>(sa(&full[0][slot]), (s / D) & 1, 50);
        if (NR == 2) wt<HINT>(sa(&full[1][slot]), (s / D) & 1, 51);
        __syncwarp();
      }
      const uint64_t m1 = clk64();
      if (MMA) {
        if (FENCE == 1 ? TM > 0 : FENCE == 2) tc_fence_after();   // FENCE 1: after a ring wait (default), 0 never, 2 always
        if (FENCE == 3 && TM == 0) wt<0>(sa(&done), 1, 70);           // FENCE 3: a completed-phase mbarrier wait (no fence)
        constexpr bool DOCP = MMA == 1 || MMA == 4, DOMMA = MMA == 1 || MMA == 3;
        const int ds = ROT == 1 ? slot : ROT == 0 ? 0 : (TM == 3 ? slot : 0);  // operand slot the MMA warp reads
        const int dsm = ROT == 2 ? 0 : ROT == 3 ? slot : ds, dsc = ROT == 2 ? slot : ROT == 3 ? 0 : ds;  // 2: SF rotates only, 3: MMA operands only
        #pragma unroll
        for (int k = 0; k < KS; ++k) {
          const uint64_t da = dA + (uint64_t)((dsm * L::AD + k * 16384) >> 4), db = dB + (uint64_t)((dsm * L::BD + k * (L::BD / KS)) >> 4);
          const uint64_t xa = sA + (uint64_t)((dsc * L::SFA + k * 2048) >> 4), xb = sB + (uint64_t)((dsc * L::SFB + k * (L::SFB / KS)) >> 4);
          if (DOCP) {
          #pragma unroll
          for (int j = 0; j < 4; ++j) cp1<CG2>(tb + CS + 4 * j, xa + 32 * j);
          #pragma unroll
          for (int j = 0; j < 8; ++j) cp1<CG2>(tb + CS + 16 + 4 * j, xb + 32 * j);
          #pragma unroll
          if (!PROD) for (int j = 0; j < 4; ++j) cp1<CG2>(tb + CS + 48 + 4 * j, xb + 256 + 32 * j);
          }
          if (DOMMA && PROD) {
            mma4<CG2>(tb + CA, da, db, id256, tb + CS, tb + CS + 16, 1u);
            mma4<CG2>(tb + CA, da + 4, db + 4, id256, tb + CS + 8, tb + CS + 32, 1u);
          } else if (DOMMA) {
          mma4<CG2>(tb + CA, da, db, ida, tb + CS, tb + CS + 16, 1u);
          mma4<CG2>(tb + CA, da + 4, db + 4, ida, tb + CS + 8, tb + CS + 32, 1u);
          mma4<CG2>(tb + CB, da, db + 832, idb, tb + CS, tb + CS + 48, 1u);
          mma4<CG2>(tb + CB, da + 4, db + 836, idb, tb + CS + 8, tb + CS + 56, 1u);
          }
        }
        commit1<CG2>(sa(&empty[0][slot]));
        if (NR == 2) commit1<CG2>(sa(&empty[1][slot]));
      } else {
        if (elect1()) {
          for (int r = 0; r < NR; ++r) {
            bar_arrive(sa(&empty[r][slot]));
            if (CG2) bar_arrive_cluster(mapa(sa(&empty[r][slot]), 1));
          }
        }
        __syncwarp();
      }
      const uint64_t m2 = clk64();
      if (s >= W0) { fsum += m1 - m0; msum += m2 - m1; }
      if (trace && lane == 0 && s >= W0 && s < W0 + TRN) {
        tr[3 * TRN + s - W0] = (long long)m0; tr[4 * TRN + s - W0] = (long long)m1; tr[5 * TRN + s - W0] = (long long)m2;
      }
    }
    if (MMA) commit1<CG2>(sa(&done));
    else if (elect1()) { bar_arrive(sa(&done)); if (CG2) bar_arrive_cluster(mapa(sa(&done), 1)); }
    __syncwarp();
    if (lane == 0) {
      wt<0>(sa(&done), 0, 60);
      out[blockIdx.x * NOUT + 0] = clk64() - c0; out[blockIdx.x * NOUT + 1] = now_ns() - n0;
      out[blockIdx.x * NOUT + 2] = fsum; out[blockIdx.x * NOUT + 5] = msum;
    }
  }
  if (CG2 && cr == 1 && warp == 4 && lane == 0) wt<0>(sa(&done), 0, 61);
  tc_fence_before();
  __syncthreads();
  if (CG2) cluster_sync();
  if (warp == 4) {
    tc_fence_after();
    if (CG2) tmem_dealloc576(tb);
    else asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tb) : "memory");
  }
}

// ---------------- host ----------------
static PFN_cuTensorMapEncodeTiled_v12000 enc = nullptr;
static CUtensorMap mk(void* p, CUtensorMapDataType dt, int esz, uint64_t inner, uint64_t rows, uint32_t bi, uint32_t br, uint32_t bk,
                      CUtensorMapSwizzle sw) {
  CUtensorMap m; cuuint64_t dims[3] = {inner, rows, 2}; cuuint64_t str[2] = {inner * esz, inner * esz * rows};
  cuuint32_t box[3] = {bi, br, bk}; cuuint32_t es[3] = {1, 1, 1};
  CUresult r = enc(&m, dt, 3, p, dims, str, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, sw, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) { printf("tensor map encode failed %d (box %u %u %u)\n", (int)r, bi, br, bk); exit(1); }
  return m;
}
static void* gbuf = nullptr;
static int sms = 0;
static unsigned long long* gout = nullptr;
static long long* gtr = nullptr;

template <int CG2, int D, int KS, int TM, int RINGS, int NISS, int OPS, int PF, int HINT, int MMA, int PEER, int MASK = 15, int PADK = 0, int FORCEK = 0, int ABOX = 128, int PROD = 0, int ROT = -1, int FENCE = 1, int PING = 0>
static void run(const char* name, int S) {
  using L = Lay<KS, PROD>;
  constexpr int DR = (TM == 3 || ROT >= 1) ? D : 1;
  int smem = DR * (L::SFA + L::SFB + L::AD + L::BD) + PADK * 1024 + 8 * 8 * 128 + 1024;
  smem = std::max(smem, std::max(160, FORCEK) * 1024);  // force 1 CTA/SM
  const bool over = smem > 227 * 1024;
  auto k = ring_k<CG2, D, KS, TM, RINGS, NISS, OPS, PF, HINT, MMA, PEER, MASK, PADK, ABOX, PROD, ROT, FENCE, PING>;
  CUfunction fn; cudaGetFuncBySymbol((cudaFunction_t*)&fn, (const void*)k);
  if (over) {
    CUresult r1 = cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE, CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY);
    if (r1 != CUDA_SUCCESS) { printf("%-44s oversized attr refused %d\n", name, (int)r1); return; }
  } else cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  uint8_t* g = (uint8_t*)gbuf;
  Maps mp;
  mp.a = mk(g, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, 128, 8192, 128, ABOX, KS, CU_TENSOR_MAP_SWIZZLE_128B);
  mp.b = mk(g + (4 << 20), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, 128, 8192, 128, PROD ? 128 : 152, KS, CU_TENSOR_MAP_SWIZZLE_128B);
  mp.sa = mk(g + (8 << 20), CU_TENSOR_MAP_DATA_TYPE_UINT32, 4, 128, 2048, 128, 4, KS, CU_TENSOR_MAP_SWIZZLE_NONE);
  mp.sb = mk(g + (12 << 20), CU_TENSOR_MAP_DATA_TYPE_UINT32, 4, 128, 2048, 128, PROD ? 8 : 12, KS, CU_TENSOR_MAP_SWIZZLE_NONE);
  mp.t = mk(g + (16 << 20), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, 16, 65536, 16, 1, 1, CU_TENSOR_MAP_SWIZZLE_NONE);
  const int grid = sms / 2 * 2;
  double bc = 1e30, bn = 1e30; std::vector<double> best(NOUT, 0.0);
  std::vector<long long> h_tr(6 * TRN);
  for (int rep = 0; rep < 4; ++rep) {
    cudaMemset(gout, 0, 8 * NOUT * sms); cudaMemset(gtr, 0, 8 * 6 * TRN);
    CUlaunchConfig cfg = {}; cfg.gridDimX = grid; cfg.gridDimY = 1; cfg.gridDimZ = 1; cfg.blockDimX = 192; cfg.blockDimY = 1;
    cfg.blockDimZ = 1; cfg.sharedMemBytes = smem; cfg.hStream = 0;
    CUlaunchAttribute at[2]; at[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    at[0].value.clusterDim.x = CG2 ? 2 : 1; at[0].value.clusterDim.y = 1; at[0].value.clusterDim.z = 1;
    at[1].id = CU_LAUNCH_ATTRIBUTE_SHARED_MEMORY_MODE; at[1].value.sharedMemoryMode = CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY;
    cfg.attrs = at; cfg.numAttrs = over ? 2 : 1;
    int Sv = S; void* params[4] = {(void*)&mp, (void*)&Sv, (void*)&gout, (void*)&gtr};
    CUresult cr = cuLaunchKernelEx(&cfg, fn, params, nullptr);
    if (cr != CUDA_SUCCESS) { printf("%-44s LAUNCH FAILED %d (smem %d)\n", name, (int)cr, smem); fflush(stdout); exit(2); }
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("%-44s ERROR %s\n", name, cudaGetErrorString(e)); fflush(stdout); exit(2); }
    std::vector<unsigned long long> h(NOUT * sms); cudaMemcpy(h.data(), gout, 8 * NOUT * sms, cudaMemcpyDeviceToHost);
    std::vector<double> c, n, f, w, is, ms;
    const double SS = S - W0;
    for (int b = 0; b < grid; ++b) if (h[b * NOUT]) {
      c.push_back(h[b * NOUT] / SS); n.push_back(h[b * NOUT + 1] / SS); f.push_back(h[b * NOUT + 2] / SS);
      w.push_back(h[b * NOUT + 3] / (SS / NISS)); is.push_back(h[b * NOUT + 4] / (SS / NISS)); ms.push_back(h[b * NOUT + 5] / SS);
    }
    auto med = [](std::vector<double>& v) { if (v.empty()) return 0.0; std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
    const double mc = med(c);
    if (rep && mc < bc) {
      bc = mc; bn = med(n); best[2] = med(f); best[3] = med(w); best[4] = med(is); best[5] = med(ms);
      cudaMemcpy(h_tr.data(), gtr, 8 * 6 * TRN, cudaMemcpyDeviceToHost);
    }
  }
  // trace links (block 0 leader): P0 wait-empty start, P1 empty seen, P2 issued; M0 wait-full start, M1 full seen, M2 commit issued
  std::vector<double> iss2full, com2emp, fwait, ewait;
  auto T = [&](int k, int s) { return (double)h_tr[k * TRN + s]; };
  for (int s = 0; s < TRN; ++s) {
    if (T(2, s) && T(4, s)) iss2full.push_back(T(4, s) - T(2, s));
    if (T(3, s) && T(4, s)) fwait.push_back(T(4, s) - T(3, s));
    if (T(0, s) && T(1, s)) ewait.push_back(T(1, s) - T(0, s));
    if (s + D < TRN && T(5, s) && T(1, s + D) && T(0, s + D)) com2emp.push_back(T(1, s + D) - std::max(T(5, s), 0.0));
  }
  auto md = [](std::vector<double> v) { if (v.empty()) return -1.0; std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
  printf("%-44s %3dK %7.1f clk/st %6.1f ns/st %.2fGHz | full-wait %6.1f empty-wait %6.1f issue %5.1f mma-seg %6.1f | b0: issue->full-seen %6.1f commit->empty-seen %6.1f\n",
         name, smem / 1024, bc, bn, bc / bn, best[2], best[3], best[4], best[5], md(iss2full), md(com2emp));
  fflush(stdout);
}

int main(int argc, char** argv) {
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  const int S = argc > 1 ? atoi(argv[1]) : 2048;
  const int only = argc > 2 ? atoi(argv[2]) : -1;
  cudaDriverEntryPointQueryResult q;
  cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", (void**)&enc, 12000, cudaEnableDefault, &q);
  if (!enc) { printf("no cuTensorMapEncodeTiled\n"); return 1; }
  cudaMalloc(&gbuf, 32 << 20); cudaMemset(gbuf, 0, 32 << 20);
  cudaMalloc(&gout, 8 * NOUT * sms); cudaMalloc(&gtr, 8 * 6 * TRN);
  if (only < 0) printf("== mmsol_ring: %d stages (first %d excluded), grid %d, 1 CTA/SM; per-stage clk medians over leader CTAs\n", S, W0, sms / 2 * 2);
  int i = 0;
#define ARM(nm, ...) if (only < 0 || only == i) run<__VA_ARGS__>(nm, S); ++i;
#define ARMX(nm, ...) if (only < 0 || only == i) run<__VA_ARGS__>(nm, S); ++i;
  //   name                                              CG2 D KS TM R NI OP PF HI MMA PEER
  ARM("a00 tight no ring (2CTA {208,96}+16cp+commit)",   1, 4, 1, 0, 2, 2, 2, 0, 1, 1, 1)
  ARM("a01 skel ring: TMA real, D4, 2x2 iss, hint",      1, 4, 1, 3, 2, 2, 2, 0, 1, 1, 1)
  ARM("a02 TMA tiny 16 B",                               1, 4, 1, 2, 2, 2, 2, 0, 1, 1, 1)
  ARM("a03 no TMA, plain arrive, peer arrives (cnt 2)",  1, 4, 1, 1, 2, 2, 2, 0, 1, 1, 1)
  ARM("a04 no TMA, leader-only arrive",                  1, 4, 1, 1, 2, 2, 2, 0, 1, 1, 0)
  ARM("a05 TMA real, no hint",                           1, 4, 1, 3, 2, 2, 2, 0, 0, 1, 1)
  ARM("a06 TMA tiny, no hint",                           1, 4, 1, 2, 2, 2, 2, 0, 0, 1, 1)
  ARM("a07 TMA tiny D2",                                 1, 2, 1, 2, 2, 2, 2, 0, 1, 1, 1)
  ARM("a08 TMA tiny D8",                                 1, 8, 1, 2, 2, 2, 2, 0, 1, 1, 1)
  ARM("a09 no TMA D2",                                   1, 2, 1, 1, 2, 2, 2, 0, 1, 1, 1)
  ARM("a10 no TMA D8",                                   1, 8, 1, 1, 2, 2, 2, 0, 1, 1, 1)
  ARM("a11 TMA real D2",                                 1, 2, 1, 3, 2, 2, 2, 0, 1, 1, 1)
  ARM("a12 TMA tiny, 1 issuer/ring",                     1, 4, 1, 2, 2, 1, 2, 0, 1, 1, 1)
  ARM("a13 TMA tiny, 1 op/ring stage",                   1, 4, 1, 2, 2, 2, 1, 0, 1, 1, 1)
  ARM("a14 TMA tiny, 1 ring (4 ops/barrier), 1 iss",     1, 4, 1, 2, 1, 1, 2, 0, 1, 1, 1)
  ARM("a15 TMA tiny, elect.sync warp producer",          1, 4, 1, 2, 2, 2, 2, 1, 1, 1, 1)
  ARM("a16 TMA real, elect.sync warp producer",          1, 4, 1, 3, 2, 2, 2, 1, 1, 1, 1)
  ARM("a17 TMA tiny, no MMA (plain release)",            1, 4, 1, 2, 2, 2, 2, 0, 1, 0, 1)
  ARM("a18 TMA real, no MMA",                            1, 4, 1, 3, 2, 2, 2, 0, 1, 0, 1)
  ARM("a19 no TMA, no MMA",                              1, 4, 1, 1, 2, 2, 2, 0, 1, 0, 1)
  ARM("a20 1CTA M128 tight no ring",                     0, 4, 1, 0, 2, 2, 2, 0, 1, 1, 1)
  ARM("a21 1CTA TMA tiny D4",                            0, 4, 1, 2, 2, 2, 2, 0, 1, 1, 1)
  ARM("a22 1CTA TMA real D4",                            0, 4, 1, 3, 2, 2, 2, 0, 1, 1, 1)
  ARM("a23 1CTA no TMA D4",                              0, 4, 1, 1, 2, 2, 2, 0, 1, 1, 1)
  ARM("a24 K512/stage tight no ring",                    1, 4, 2, 0, 2, 2, 2, 0, 1, 1, 1)
  ARM("a25 K512 TMA tiny D4",                            1, 4, 2, 2, 2, 2, 2, 0, 1, 1, 1)
  ARM("a26 K512 TMA real D2",                            1, 2, 2, 3, 2, 2, 2, 0, 1, 1, 1)
  ARM("a27 K512 TMA tiny D2",                            1, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1)
  ARM("a28 K512 no TMA D4",                              1, 4, 2, 1, 2, 2, 2, 0, 1, 1, 1)
  // round 2: which bytes, which tcgen05 op, where in smem.          CG2 D KS TM R NI OP PF HI MMA PEER MASK PADK FORCEK
  ARMX("a29 real: A data only",                          1, 4, 1, 3, 2, 2, 2, 0, 1, 1, 1, 1, 0, 0)
  ARMX("a30 real: A SF only",                            1, 4, 1, 3, 2, 2, 2, 0, 1, 1, 1, 2, 0, 0)
  ARMX("a31 real: B data only",                          1, 4, 1, 3, 2, 2, 2, 0, 1, 1, 1, 4, 0, 0)
  ARMX("a32 real: B SF only",                            1, 4, 1, 3, 2, 2, 2, 0, 1, 1, 1, 8, 0, 0)
  ARMX("a33 real: SF only (A+B SF, skel NOW+NOTOK)",     1, 4, 1, 3, 2, 2, 2, 0, 1, 1, 1, 10, 0, 0)
  ARMX("a34 real: data only (A+B data)",                 1, 4, 1, 3, 2, 2, 2, 0, 1, 1, 1, 5, 0, 0)
  ARMX("a35 real, commit-only consumer (no cp/MMA)",     1, 4, 1, 3, 2, 2, 2, 0, 1, 2, 1, 15, 0, 0)
  ARMX("a36 tiny, commit-only consumer",                 1, 4, 1, 2, 2, 2, 2, 0, 1, 2, 1, 15, 0, 0)
  ARMX("a37 real, MMAs only (no cp)",                    1, 4, 1, 3, 2, 2, 2, 0, 1, 3, 1, 15, 0, 0)
  ARMX("a38 real, cps only (no MMA)",                    1, 4, 1, 3, 2, 2, 2, 0, 1, 4, 1, 15, 0, 0)
  ARMX("a39 tight, MMAs only (no cp)",                   1, 4, 1, 0, 2, 2, 2, 0, 1, 3, 1, 15, 0, 0)
  ARMX("a40 tight, cps only",                            1, 4, 1, 0, 2, 2, 2, 0, 1, 4, 1, 15, 0, 0)
  ARMX("a41 tiny, MMA operands above 227 KiB",           1, 4, 1, 2, 2, 2, 2, 0, 1, 1, 1, 15, 219, 0)
  ARMX("a42 real D2, data above 227 KiB",                1, 2, 1, 3, 2, 2, 2, 0, 1, 1, 1, 15, 211, 0)
  ARMX("a43 real D2, low, oversized launch (control)",   1, 2, 1, 3, 2, 2, 2, 0, 1, 1, 1, 15, 0, 300)
  ARMX("a44 K512 real D4 (344K, oversized)",             1, 4, 2, 3, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0)
  ARMX("a45 K512 real D3 (258K, oversized)",             1, 3, 2, 3, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0)
  ARMX("a46 real D3 (129K)",                             1, 3, 1, 3, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0)
  ARMX("a47 real D5 (215K)",                             1, 5, 1, 3, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0)
  ARMX("a48 real D6 (258K, oversized)",                  1, 6, 1, 3, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0)
  ARMX("a49 real D4, 1 ring 4 ops, 1 iss",               1, 4, 1, 3, 1, 1, 2, 0, 1, 1, 1, 15, 0, 0)
  // round 3: A-data box size sweep (other 3 ops 16 B), production N256 shape.  ... MASK PADK FORCEK ABOX PROD
  ARMX("a50 A box 128 B (1 row), others 16 B",          1, 4, 1, 3, 2, 2, 2, 0, 1, 1, 1, 1, 0, 0, 1, 0)
  ARMX("a51 A box 512 B (4 rows)",                      1, 4, 1, 3, 2, 2, 2, 0, 1, 1, 1, 1, 0, 0, 4, 0)
  ARMX("a52 A box 2 KiB (16 rows)",                     1, 4, 1, 3, 2, 2, 2, 0, 1, 1, 1, 1, 0, 0, 16, 0)
  ARMX("a53 A box 8 KiB (64 rows)",                     1, 4, 1, 3, 2, 2, 2, 0, 1, 1, 1, 1, 0, 0, 64, 0)
  ARMX("a54 A box 128 B, cps-only consumer",            1, 4, 1, 3, 2, 2, 2, 0, 1, 4, 1, 1, 0, 0, 1, 0)
  ARMX("a55 A box 2 KiB, cps-only consumer",            1, 4, 1, 3, 2, 2, 2, 0, 1, 4, 1, 1, 0, 0, 16, 0)
  ARMX("a56 A box 2 KiB, MMA-only consumer",            1, 4, 1, 3, 2, 2, 2, 0, 1, 3, 1, 1, 0, 0, 16, 0)
  ARMX("a57 PROD {256}+12cp tight",                     1, 4, 1, 0, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 1)
  ARMX("a58 PROD real D4",                              1, 4, 1, 3, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 1)
  ARMX("a59 PROD tiny D4",                              1, 4, 1, 2, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 1)
  ARMX("a60 PROD real D6, 1 iss/ring (prod issuers)",   1, 6, 1, 3, 2, 1, 2, 0, 1, 1, 1, 15, 0, 0, 128, 1)
  ARMX("a61 PROD K512 tight",                           1, 4, 2, 0, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 1)
  ARMX("a62 PROD K512 real D3",                         1, 3, 2, 3, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 1)
  ARMX("a63 PROD K512 real D3, 1 iss/ring",             1, 3, 2, 3, 2, 1, 2, 0, 1, 1, 1, 15, 0, 0, 128, 1)
  ARMX("a64 PROD K512 tiny D3",                         1, 3, 2, 2, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 1)
  ARMX("a65 PROD real D4, MMA-only (no cp)",            1, 4, 1, 3, 2, 2, 2, 0, 1, 3, 1, 15, 0, 0, 128, 1)
  // round 4: operand-address rotation vs fixed (which smem the cp / MMA reads)   ... ABOX PROD ROT
  ARMX("a66 tiny, cp/MMA read rotating slots",           1, 4, 1, 2, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 0, 1)
  ARMX("a67 real, cp/MMA read fixed slot 0",             1, 4, 1, 3, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 0, 0)
  ARMX("a68 tight no ring, rotating slots",              1, 4, 1, 0, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 0, 1)
  ARMX("a69 tight cps-only, rotating slots",             1, 4, 1, 0, 2, 2, 2, 0, 1, 4, 1, 15, 0, 0, 128, 0, 1)
  ARMX("a70 tight MMA-only, rotating slots",             1, 4, 1, 0, 2, 2, 2, 0, 1, 3, 1, 15, 0, 0, 128, 0, 1)
  ARMX("a71 tight no ring, rotating 2 slots",            1, 2, 1, 0, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 0, 1)
  ARMX("a72 PROD tight rotating",                        1, 4, 1, 0, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 1, 1)
  ARMX("a73 PROD real fixed slot 0",                     1, 4, 1, 3, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 1, 0)
  ARMX("a74 1CTA tight rotating",                        0, 4, 1, 0, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 0, 1)
  ARMX("a75 K512 tight rotating",                        1, 4, 2, 0, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 0, 1)
  // round 5: tcgen05.fence::after_thread_sync and the wait itself, rotating vs fixed operands   ... ROT FENCE
  ARMX("a76 tight rotating + fence::after per stage",    1, 4, 1, 0, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 0, 1, 2)
  ARMX("a77 tight fixed + fence::after per stage",       1, 4, 1, 0, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 0, 0, 2)
  ARMX("a78 tiny ring rotating, NO fence",               1, 4, 1, 2, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 0, 1, 0)
  ARMX("a79 real ring, NO fence (skel shape)",           1, 4, 1, 3, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 0, -1, 0)
  ARMX("a80 PROD real ring, NO fence",                   1, 4, 1, 3, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 1, -1, 0)
  ARMX("a81 tight rotating + completed mbar wait",       1, 4, 1, 0, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 0, 1, 3)
  ARMX("a82 PROD K512 real D3, NO fence",                1, 3, 2, 3, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 1, -1, 0)
  ARMX("a83 real D2, NO fence",                          1, 2, 1, 3, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 0, -1, 0)
  ARMX("a84 1CTA real D4, NO fence",                     0, 4, 1, 3, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 0, -1, 0)
  ARMX("a85 PROD tight rotating + fence",                1, 4, 1, 0, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 1, 1, 2)
  // round 6: is it the ring or TMA traffic?   ... ROT FENCE
  ARMX("a86 no-TMA ring (plain arrive), rotating",       1, 4, 1, 1, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 0, 1, 1)
  ARMX("a87 no-TMA ring, fixed",                         1, 4, 1, 1, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 0, 0, 1)
  ARMX("a88 tight rotating + background tiny TMA",       1, 4, 1, 5, 2, 2, 2, 0, 1, 1, 1, 0, 0, 0, 128, 0, 1, 1)
  ARMX("a89 tight rotating + background 16K TMA",        1, 4, 1, 5, 2, 2, 2, 0, 1, 1, 1, 1, 0, 0, 128, 0, 1, 1)
  ARMX("a90 tight fixed + background 16K TMA",           1, 4, 1, 5, 2, 2, 2, 0, 1, 1, 1, 1, 0, 0, 128, 0, 0, 1)
  ARMX("a91 PROD no-TMA ring rotating",                  1, 4, 1, 1, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 1, 1, 1)
  ARMX("a92 PROD no-TMA ring fixed",                     1, 4, 1, 1, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 1, 0, 1)
  ARMX("a93 no-TMA ring rotating, 1 ring 1 iss",         1, 4, 1, 1, 1, 1, 2, 0, 1, 1, 1, 15, 0, 0, 128, 0, 1, 1)
  // round 7: which address rotates (SF cp source vs MMA operands); K1024 amortization    ... ROT FENCE
  ARMX("a94 no-TMA ring, SF cp source rotates only",     1, 4, 1, 1, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 0, 2, 1)
  ARMX("a95 no-TMA ring, MMA operands rotate only",      1, 4, 1, 1, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 0, 3, 1)
  ARMX("a96 no-TMA ring rotating, cps-only",             1, 4, 1, 1, 2, 2, 2, 0, 1, 4, 1, 15, 0, 0, 128, 0, 1, 1)
  ARMX("a97 no-TMA ring fixed, cps-only",                1, 4, 1, 1, 2, 2, 2, 0, 1, 4, 1, 15, 0, 0, 128, 0, 0, 1)
  ARMX("a98 no-TMA ring rotating, MMA-only",             1, 4, 1, 1, 2, 2, 2, 0, 1, 3, 1, 15, 0, 0, 128, 0, 1, 1)
  ARMX("a99 no-TMA ring fixed, MMA-only",                1, 4, 1, 1, 2, 2, 2, 0, 1, 3, 1, 15, 0, 0, 128, 0, 0, 1)
  ARMX("a100 PROD K1024 tight",                          1, 2, 4, 0, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 1, -1, 1)
  ARMX("a101 PROD K1024 real D2",                        1, 2, 4, 3, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 1, -1, 1)
  ARMX("a102 PROD K512 real D4",                         1, 4, 2, 3, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 1, -1, 1)
  ARMX("a103 PROD K256 real D8",                         1, 8, 1, 3, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 1, -1, 1)
  // round 8: TMEM SF image ping-pong (cp of stage s+1 never waits on stage s MMAs reading the SF columns)  ... ROT FENCE PING
  ARMX("a104 no-TMA ring rotating, SF ping-pong",        1, 4, 1, 1, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 0, 1, 1, 1)
  ARMX("a105 real ring (skel), SF ping-pong",            1, 4, 1, 3, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 0, -1, 1, 1)
  ARMX("a106 PROD real ring, SF ping-pong",              1, 4, 1, 3, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 1, -1, 1, 1)
  ARMX("a107 tight rotating, SF ping-pong",              1, 4, 1, 0, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 0, 1, 1, 1)
  ARMX("a108 real ring 1 ring/1 iss, SF ping-pong",      1, 4, 1, 3, 1, 1, 2, 0, 1, 1, 1, 15, 0, 0, 128, 0, -1, 1, 1)
  ARMX("a109 PROD real 1 ring/1 iss",                    1, 4, 1, 3, 1, 1, 2, 0, 1, 1, 1, 15, 0, 0, 128, 1, -1, 1, 0)
  ARMX("a110 PROD K512 real D3, SF ping-pong",           1, 3, 2, 3, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 1, -1, 1, 1)
  // round 9: SF cps software-pipelined one stage ahead (PING 2)    ... ROT FENCE PING
  ARMX("a111 real ring (skel), cps 1 stage ahead",       1, 4, 1, 3, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 0, -1, 1, 2)
  ARMX("a112 PROD real ring, cps 1 stage ahead",         1, 4, 1, 3, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 1, -1, 1, 2)
  ARMX("a113 PROD real 1 ring/1 iss, cps ahead",         1, 4, 1, 3, 1, 1, 2, 0, 1, 1, 1, 15, 0, 0, 128, 1, -1, 1, 2)
  ARMX("a114 real 1 ring/1 iss, cps ahead (skel)",       1, 4, 1, 3, 1, 1, 2, 0, 1, 1, 1, 15, 0, 0, 128, 0, -1, 1, 2)
  ARMX("a115 no-TMA ring rotating, cps ahead",           1, 4, 1, 1, 2, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 0, 1, 1, 2)
  // round 10: levers combined                                    ... ROT FENCE PING
  ARMX("a116 PROD K512 real D3, 1 ring/1 iss",           1, 3, 2, 3, 1, 1, 2, 0, 1, 1, 1, 15, 0, 0, 128, 1, -1, 1, 0)
  ARMX("a117 skel K512 real D3, 1 ring/1 iss",           1, 3, 2, 3, 1, 1, 2, 0, 1, 1, 1, 15, 0, 0, 128, 0, -1, 1, 0)
  ARMX("a118 PROD K512 real D3, 1 ring/2 iss",           1, 3, 2, 3, 1, 2, 2, 0, 1, 1, 1, 15, 0, 0, 128, 1, -1, 1, 0)
  ARMX("a119 PROD real D6, 1 ring/1 iss",                1, 6, 1, 3, 1, 1, 2, 0, 1, 1, 1, 15, 0, 0, 128, 1, -1, 1, 0)
  if (only < 0) printf("DONE\n");
  return 0;
}
