// SPDX-License-Identifier: Apache-2.0
// EXPERIMENT ONLY (never for merge): mmsol_skel, the MM-SOL B2 skeleton launched in step (P1-IS arms S1 / S1-nopush,
// prefill/designs/megamoe-sol.md sec 4). One persistent launch per MoE layer: 212 CTAs, cta_group::2 pairs, 16 warps,
// exclusive 576-column TMEM (A0-verified carve), A ring 8 x 18 KiB (weights + SFA, 2 issuer warps), B ring 5 x 25 KiB
// (tokens + SFB, 2 issuer warps), MMA warp, scheduler warp (dynamic per-rank claims, FC1->FC2 / gate_up->down /
// dispatch-window dependencies), dispatch warp (window-ordered TMA bulk pushes + counted puts), 8 epilogue warps
// (FC1 SwiGLU + NVFP4 into FC1-out rows, FC2 BF16 per copy + last-copy combine pushed to the owner rank, SE gate_up
// SiTU-like + MXFP8, SE down BF16). Claim list: SE gate_up | FC1(w0) | FC1(w1) FC2(w0) | ... | FC2(w_last) | SE down.
// Real operands: this layer's routed weights + scale planes, the dispatched rows, the SE weights / input.
// CUT (named): SF warp expansion (SFA loaded expanded), deferred combines (cnt > 2), die-local homing, own-row quant
// math (its bytes are in the dispatch read), exact routing records (synthetic balanced map: half the copies are last).
// Placement B (after the stock reduce): writes only regions the stock kernel rewrites before reading (pre_reduced,
// fc1_activation, fc2_order_output); counters in the arena signal pad (< 1.2 KB).
#include "mmsol_core.cuh"
#include <cuda_bf16.h>
#ifndef MMSOL_ABL
#define MMSOL_ABL 0  // solo ablations only: 1 = one SF cp per group (prices the tcgen05.cp share of the stage)
#endif

using namespace mm;

namespace sk {
constexpr int NT = 512;                       // 16 warps
constexpr int AS = 8, BS = 5;                 // ring slots
constexpr int A_DATA = 16384, A_SF = 2048, A_SLOT = A_DATA + A_SF;          // 18 KiB
constexpr int B_DATA = 152 * 128, B_SF = 12 * 512, B_SLOT = B_DATA + B_SF;  // 25 KiB
constexpr int STG = 16384;                    // epilogue staging buffer (x2)
constexpr int DST = 8192;                     // dispatch staging (2 x 4 KiB)
constexpr int TR = 8;                         // tile ring depth
constexpr int SMEM = AS * A_SLOT + BS * B_SLOT + 2 * STG + DST + 1024;       // + alignment
constexpr int E = 112;
enum { K_SEGU = 0, K_FC1 = 1, K_FC2 = 2, K_SEDN = 3, K_END = 7 };
// counter page (uint32 words, local rank unless noted)
constexpr int C_CLAIM = 0, C_CTADONE = 1, C_FC1 = 2 /* [E*2] */, C_GU = 230 /* [16] */, C_WDONE = 250 /* [28] */;
constexpr int C_DISP = 280 /* [28], remote-written */, C_ENTRY = 310, C_DONE = 311, C_WORDS = 320;
constexpr int NSTAMP = 12, RING = 32;
}  // namespace sk

struct SkelArgs {
  CUtensorMap tmW1, tmW2, tmX1, tmX2, tmSEW1, tmSEW2, tmSEX1, tmSEX2, tmSFW, tmSFX, tmSFSE2, tmSFSE1;
  int rows_x1, rows_x2, rows_sfw, rows_sfx, rows_sesf, rows_sex1, rows_sex2;
  int n, n_u, nwin, W, se_mt, se_dt, ntiles, ngu;
  int t_u0, t_u1;                               // tokens in unit 0 / 1
  uint64_t fc1out, fc1out_span, order, order_span, se_inter_off, se_out_off;
  uint64_t dst_local; int64_t slice, comb_base, disp_per_dest; const int64_t* offs;
  uint64_t xsrc; int64_t xspan;
  uint64_t ctr;                                  // local counter page
  const int64_t* ctr_peer;                       // [world] counter page address on every rank
  uint32_t claim_base, disp_prev_lo, disp_prev_hi, epoch;
  int rank, world, push;
  int64_t* stamps; int64_t* hdr; int64_t* tstamps; uint64_t limit;
};

namespace sk {
MMD uint64_t cw(const SkelArgs& a, int w) { return a.ctr + 4ull * w; }
MMD uint64_t cpeer(const SkelArgs& a, int r, int w) { return (uint64_t)a.ctr_peer[r] + 4ull * w; }
MMD void spin_ge(uint64_t addr, uint32_t target, uint64_t limit, int rank, int what, bool sys) {
  const uint64_t t0 = now_ns(); uint32_t v;
  while ((int32_t)((v = sys ? ld_acq_sys(addr) : ld_acq_gpu(addr)) - target) < 0) {
    __nanosleep(64);
    if (now_ns() - t0 > limit) {
      printf("K31_SPIN_TIMEOUT kernel=mmsol_skel rank=%d what=%d target=%u seen=%u block=%d\n", rank, what, target, v, blockIdx.x);
      __trap();
    }
  }
}
MMD bool try_w(uint32_t b, uint32_t ph) {  // suspending wait (HW time limit), frees issue slots for the epilogue warps
  uint32_t ok;
  asm volatile("{.reg .pred P; mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2, 1000000; selp.u32 %0, 1, 0, P;}"
               : "=r"(ok) : "r"(b), "r"(ph) : "memory");
  return ok != 0;
}
MMD bool try_w_cl(uint32_t b, uint32_t ph) {
  uint32_t ok;
  asm volatile("{.reg .pred P; mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P, [%1], %2, 1000000; selp.u32 %0, 1, 0, P;}"
               : "=r"(ok) : "r"(b), "r"(ph) : "memory");
  return ok != 0;
}
MMD void bwait(uint32_t b, uint32_t ph, uint64_t limit, int what) {
  if (try_w(b, ph)) return;
  const uint64_t t0 = now_ns();
  while (!try_w(b, ph)) {
    if (now_ns() - t0 > limit) { printf("K31_SPIN_TIMEOUT kernel=mmsol_skel mbar what=%d block=%d warp=%d\n", what, blockIdx.x, threadIdx.x / 32); __trap(); }
  }
}
MMD void bwait_cl(uint32_t b, uint32_t ph, uint64_t limit, int what) {  // acquire at cluster scope (peer writes)
  if (try_w_cl(b, ph)) return;
  const uint64_t t0 = now_ns();
  while (!try_w_cl(b, ph)) {
    if (now_ns() - t0 > limit) { printf("K31_SPIN_TIMEOUT kernel=mmsol_skel tile-ring what=%d block=%d warp=%d\n", what, blockIdx.x, threadIdx.x / 32); __trap(); }
  }
}
MMD void named_bar(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }

struct Tile { int kind, e, u, m, idx; };
// stages per tile kind (K256 routed, K128 SE)
MMD int nstages(int k) { return k == K_FC1 ? 14 : k == K_FC2 ? 12 : k == K_SEGU ? 56 : 48; }
MMD int bslots(int k) { return k == K_SEGU ? 2 : 1; }

MMD Tile decode(const SkelArgs& a, int i) {
  Tile t; t.idx = i; t.e = t.u = t.m = 0;
  if (i >= a.ntiles) { t.kind = K_END; return t; }
  if (i < a.ngu) { t.kind = K_SEGU; t.e = i / 48; t.m = i % 48; return t; }
  int j = i - a.ngu;
  const int F1 = a.W * a.n_u * 24, F2 = a.W * a.n_u * 14, NR = E * a.n_u * 38;
  if (j >= NR) { j -= NR; t.kind = K_SEDN; t.e = j / 28; t.m = j % 28; return t; }
  int w, r, fc2;
  if (j < F1) { w = 0; r = j; fc2 = 0; }
  else {
    j -= F1;
    const int k = 1 + j / (F1 + F2);
    if (k <= a.nwin - 1) { r = j % (F1 + F2); if (r < F1) { w = k; fc2 = 0; } else { r -= F1; w = k - 1; fc2 = 1; } }
    else { r = j - (a.nwin - 1) * (F1 + F2); w = a.nwin - 1; fc2 = 1; }
  }
  const int mt = fc2 ? 14 : 24;
  t.kind = fc2 ? K_FC2 : K_FC1;
  t.e = w * a.W + r / (mt * a.n_u);
  const int rem = r % (mt * a.n_u);
  t.m = rem / a.n_u; t.u = rem % a.n_u;
  return t;
}
MMD int clampi(int r, int rows, int box) { const int lim = rows - box; return lim > 0 ? r % lim : 0; }

MMD uint8_t e2m1(float x) {  // round-to-nearest E2M1 code (sign | 3 bits)
  uint32_t r; asm("{.reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, %1, %2; cvt.u32.u8 %0, b;}" : "=r"(r) : "f"(0.f), "f"(x));
  return (uint8_t)(r & 0xF);
}
MMD uint32_t pack_bf16(float lo, float hi) {
  uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r;
}
MMD float silu(float x) { return __fdividef(x, 1.f + __expf(-x)); }  // 2 MUFU
}  // namespace sk

using namespace sk;

__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(NT, 1) mmsol_skel_kernel(const __grid_constant__ SkelArgs a) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  __shared__ __align__(8) uint64_t afull[AS], aempty[AS], bfull[BS], bempty[BS], tfull[TR], tempty[TR];
  __shared__ __align__(8) uint64_t accfull[2], resempty[3], dbar[2];
  __shared__ __align__(16) int4 tring[TR];
  __shared__ uint32_t tmem_base;
  uint8_t* base = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  uint8_t* aring = base;
  uint8_t* bring = aring + AS * A_SLOT;
  uint8_t* stg = bring + BS * B_SLOT;
  uint8_t* dstg = stg + 2 * STG;
  const uint32_t cr = cta_rank();
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, G = gridDim.x, g = blockIdx.x;
  const uint64_t LIM = a.limit;
  int64_t* st = a.stamps + ((int64_t)((a.epoch - 1) % RING) * G + g) * NSTAMP;
  if (threadIdx.x == 0) {
    for (int i = 0; i < NSTAMP; ++i) st[i] = 0;
    st[0] = (int64_t)now_ns(); st[10] = (int64_t)clk64(); st[9] = smid();
    for (int i = 0; i < AS; ++i) { bar_init(sa(&afull[i]), 1); bar_init(sa(&aempty[i]), 1); }
    for (int i = 0; i < BS; ++i) { bar_init(sa(&bfull[i]), 1); bar_init(sa(&bempty[i]), 1); }
    for (int i = 0; i < TR; ++i) { bar_init(sa(&tfull[i]), 1); bar_init(sa(&tempty[i]), 25); }
    bar_init(sa(&accfull[0]), 1); bar_init(sa(&accfull[1]), 1);
    for (int i = 0; i < 3; ++i) bar_init(sa(&resempty[i]), 16);
    bar_init(sa(&dbar[0]), 1); bar_init(sa(&dbar[1]), 1);
    bar_fence_init();
  }
  if (warp == 4) tmem_alloc576(sa(&tmem_base));
  tc_fence_before();
  cluster_sync();
  tc_fence_after();
  const uint32_t tb = tmem_base;
  const int Wd = a.world;

  if (warp == 5) {
    // ================= scheduler (leader CTA): claim, decode, dependency wait, publish to both CTAs =================
    if (cr == 0 && lane == 0) {
      for (int k = 0;; ++k) {
        const int slot = k % TR, round = k / TR;
        if (round > 0) bwait(sa(&tempty[slot]), (round - 1) & 1, LIM, 1);
        const uint32_t old = atom_add_gpu(cw(a, C_CLAIM), 1u);
        const int i = (int)(old - a.claim_base);
        Tile t = decode(a, i);
        if (t.kind == K_FC1) {
          const int w = t.e / a.W;
          spin_ge(cw(a, C_DISP + w), (uint32_t)Wd * ((w < 16 ? a.disp_prev_lo : a.disp_prev_hi) + 1), LIM, a.rank, 10 + w, true);
        } else if (t.kind == K_FC2) {
          spin_ge(cw(a, C_FC1 + t.e * 2 + t.u), 48u, LIM, a.rank, 3, false);
        } else if (t.kind == K_SEDN) {
          spin_ge(cw(a, C_GU + t.e / 2), 96u, LIM, a.rank, 4, false);
        }
        const int4 v = make_int4(t.kind | (t.idx << 3), t.e, t.u, t.m);
        tring[slot] = v;
        const uint32_t pslot = mapa(sa(&tring[slot]), 1);
        asm volatile("st.shared::cluster.v4.s32 [%0], {%1, %2, %3, %4};" :: "r"(pslot), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
        bar_arrive_cluster(mapa(sa(&tfull[slot]), 0));
        bar_arrive_cluster(mapa(sa(&tfull[slot]), 1));
        if (t.kind == K_END) break;
      }
    }
  } else if (warp == 6) {
    // idle (design W6 = SF expansion warp: CUT, SFA is loaded expanded)
  } else if (warp == 7) {
    // ================= dispatch: entry count, window-ordered pushes of R x (W-1) x 2080 B, counted puts =================
    if (lane == 0) {
      if (g == 0) for (int r = 0; r < Wd; ++r) red_add_sys(cpeer(a, (a.rank + r) % Wd, C_ENTRY), 1u);
      spin_ge(cw(a, C_ENTRY), a.epoch * (uint32_t)Wd, LIM, a.rank, 1, true);
      st[1] = (int64_t)now_ns();
      const int Wr = Wd - 1;
      const int64_t cb = ((a.disp_per_dest / a.nwin) + 15) / 16 * 16;
      const int PC = 4096;
      const int64_t npc = (cb + PC - 1) / PC;
      uint32_t ph[2] = {0, 0}; int k = 0; int64_t pushed = 0;
      for (int w = 0; w < a.nwin; ++w) {
        for (int64_t q = g; q < (int64_t)Wr * npc; q += G) {
          const int i = (int)(q / npc), p = (int)(q % npc);
          const int d = (a.rank + 1 + i) % Wd;
          const int64_t off = (int64_t)p * PC;
          const uint32_t len = (uint32_t)min((int64_t)PC, cb - off);
          const int s = k & 1;
          if (k >= 2) bulk_wait_read<1>();
          const uint64_t src = a.xsrc + (uint64_t)((((int64_t)a.rank * 977 + w * 131 + i * 7) * PC + off) % a.xspan) / 16 * 16;
          bar_expect(sa(&dbar[s]), len);
          asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                       :: "r"(sa(dstg + s * PC)), "l"(src), "r"(len), "r"(sa(&dbar[s])) : "memory");
          bwait(sa(&dbar[s]), ph[s], LIM, 20); ph[s] ^= 1u;
          const int dd = a.push ? d : a.rank;
          const int64_t peer = a.offs[dd];
          const uint64_t dst = a.dst_local + peer + (int64_t)(a.push ? a.rank : d) * a.slice + (w * cb + off) % (a.slice - PC);
          bulk_store(dst, sa(dstg + s * PC), len); bulk_commit();
          pushed += len; ++k;
        }
        bulk_wait<0>();
        asm volatile("fence.proxy.async.global;" ::: "memory");
        __threadfence_system();
        const uint32_t prev = atom_add_gpu(cw(a, C_WDONE + w), 1u);
        if (prev + 1u == (uint32_t)G) {
          __threadfence_system();
          for (int r = 0; r < Wd; ++r) red_add_sys(cpeer(a, (a.rank + 1 + r) % Wd, C_DISP + w), 1u);
        }
      }
      st[2] = (int64_t)now_ns(); atomicAdd((unsigned long long*)&st[8], (unsigned long long)pushed);
    }
  } else if (warp < 4) {
    // ================= TMA issuers: W0/W1 A ring (alternate stages), W2/W3 B ring (alternate slots) =================
    if (lane == 0) {
      const bool isA = warp < 2; const int par = warp & 1;
      uint32_t cnt = 0;  // ring position counter (A: stages; B: slots)
      for (int k = 0;; ++k) {
        const int ts = k % TR;
        bwait_cl(sa(&tfull[ts]), (k / TR) & 1, LIM, 30 + warp);
        const int4 v = tring[ts];
        bar_arrive_cluster(mapa(sa(&tempty[ts]), 0));
        const int kind = v.x & 7, e = v.y, u = v.z, m = v.w;
        if (kind == K_END) break;
        const int S = nstages(kind), nb = isA ? 1 : bslots(kind);
        for (int s = 0; s < S; ++s)
          for (int bb = 0; bb < nb; ++bb, ++cnt) {
            if ((int)(cnt & 1) != par) continue;
            const int R = isA ? AS : BS;
            const int slot = cnt % R; const uint32_t round = cnt / R;
            if (round > 0) bwait(sa(&(isA ? aempty : bempty)[slot]), (round - 1) & 1, LIM, 40 + warp);
            uint64_t* fb = isA ? &afull[slot] : &bfull[slot];
            const uint32_t lb = mapa(sa(fb), 0);
            uint8_t* sl = isA ? aring + slot * A_SLOT : bring + slot * B_SLOT;
            if (isA) {
              if (kind == K_FC1 || kind == K_FC2) {
                const int rowsW = (kind == K_FC1 ? 6144 : 3584);
                if (cr == 0) bar_expect(sa(fb), 2 * ((MMSOL_ABL == 3 ? 0 : A_DATA) + A_SF));
                if (MMSOL_ABL != 3) tma2d_cg2(sa(sl), kind == K_FC1 ? &a.tmW1 : &a.tmW2, s * 128, e * rowsW + m * 256 + cr * 128, lb);
                tma2d_cg2(sa(sl + A_DATA), &a.tmSFW, 0, clampi((((e * 24 + m) * 2 + cr) * 14 + s) * 4 + (kind == K_FC2) * 7777, a.rows_sfw, 4), lb);
              } else {  // SE weights: N128 half x K128 FP8 (16 KiB) + SFB (N256 x 4 UE8M0 = 2 chunks)
                if (cr == 0) bar_expect(sa(fb), 2 * (A_DATA + 1024));
                tma2d_cg2(sa(sl), kind == K_SEGU ? &a.tmSEW1 : &a.tmSEW2, s * 128, m * 256 + cr * 128, lb);
                tma2d_cg2(sa(sl + A_DATA), &a.tmSFSE2, 0, clampi((m * 56 + s) * 2 + (kind == K_SEDN) * 5555, a.rows_sesf, 2), lb);
              }
            } else {
              if (kind == K_FC1 || kind == K_FC2) {
                if (cr == 0) bar_expect(sa(fb), 2 * (MMSOL_ABL == 2 ? B_SF : B_SLOT));
                const int rb = ((e * a.n_u + u) * 304 + cr * 152);
                if (MMSOL_ABL == 2) {}
                else if (kind == K_FC1) tma2d_cg2(sa(sl), &a.tmX1, s * 128, clampi(rb, a.rows_x1, 152), lb);
                else tma2d_cg2(sa(sl), &a.tmX2, s * 128, clampi(rb, a.rows_x2, 152), lb);
                tma2d_cg2(sa(sl + B_DATA), &a.tmSFX, 0, clampi(((e * a.n_u + u) * 14 + s) * 12 + (kind == K_FC2) * 3333, a.rows_sfx, 12), lb);
              } else {  // SE tokens: M128 per CTA x K128 FP8 (16 KiB) + SFA (1 chunk)
                if (cr == 0) bar_expect(sa(fb), 2 * (A_DATA + 512));
                const int row = (kind == K_SEGU ? e * 512 + bb * 256 : e * 256) + cr * 128;
                if (kind == K_SEGU) tma2d_cg2(sa(sl), &a.tmSEX1, s * 128, clampi(row, a.rows_sex1, 128), lb);
                else tma2d_cg2(sa(sl), &a.tmSEX2, s * 128, clampi(row, a.rows_sex2, 128), lb);
                tma2d_cg2(sa(sl + A_DATA), &a.tmSFSE1, 0, clampi((row / 128) * 56 + s + (kind == K_SEDN) * 999, a.rows_sesf, 1), lb);
              }
            }
          }
      }
    }
  } else if (warp == 4) {
    // ================= MMA issuer (leader CTA, warp-uniform) =================
    if (cr == 0) {
      uint32_t acnt = 0, bcnt = 0, ntile = 0, ra = 0;  // ra: routed tile count (a-buffer parity)
      uint32_t ruse[3] = {0, 0, 0};
      const uint32_t ida = idesc_nvf4(NA), idb = idesc_nvf4(NB), idse = idesc_mx8(256);
      int64_t* ts = a.tstamps + ((int64_t)((a.epoch - 1) % RING) * G + g) * 128;
      int nts = 0;
      if (lane == 0) for (int i = 0; i < 128; ++i) ts[i] = 0;
      for (int k = 0;; ++k) {
        const int tsl = k % TR;
        bwait_cl(sa(&tfull[tsl]), (k / TR) & 1, LIM, 50);
        const int4 v = tring[tsl];
        __syncwarp();
        if (lane == 0) bar_arrive(sa(&tempty[tsl]));
        const int kind = v.x & 7;
        if (kind == K_END) break;
        const int S = nstages(kind);
        // accumulator resources: routed tile -> a[ra%2] + b; SE tile -> all three
        auto acquire = [&](int r) { if (ruse[r] > 0) bwait(sa(&resempty[r]), (ruse[r] - 1) & 1, LIM, 60 + r); ++ruse[r]; };
        const bool routed = (kind == K_FC1 || kind == K_FC2);
        const int abuf = ra & 1;
        if (routed) acquire(abuf); else { acquire(0); acquire(1); acquire(2); }
        const bool samp = (ntile % 8) == 0 && nts < 15;
        uint64_t t_first = 0, c_first = 0;
        for (int s = 0; s < S; ++s) {
          const int as_ = acnt % AS; bwait(sa(&afull[as_]), (acnt / AS) & 1, LIM, 70);
          const int bs0 = bcnt % BS; bwait(sa(&bfull[bs0]), (bcnt / BS) & 1, LIM, 71);
          int bs1 = 0;
          if (kind == K_SEGU) { bs1 = (bcnt + 1) % BS; bwait(sa(&bfull[bs1]), ((bcnt + 1) / BS) & 1, LIM, 72); }
          tc_fence_after();
          if (s == 0 && st[3] == 0 && lane == 0) st[3] = (int64_t)now_ns();
          if (samp && s == 1) { t_first = now_ns(); c_first = clk64(); }  // stage 1 .. S-1: excludes the b turnaround
          uint8_t* A = aring + as_ * A_SLOT; uint8_t* B = bring + bs0 * B_SLOT;
          if (routed) {
            const int ncp = MMSOL_ABL == 1 ? 1 : (MMSOL_ABL == 6 || (MMSOL_ABL == 8 && s > 0)) ? 0 : 4;
            const uint32_t sfo = (MMSOL_ABL == 9 && (s & 1)) ? (uint32_t)(352 - COL_SFA) : 0u;  // image 2 offset
            for (int j = 0; j < ncp; ++j) sf_cp(tb + sfo + COL_SFA + 4 * j, desc_sf(sa(A + A_DATA) + 512 * j));
            for (int j = 0; j < 2 * ncp; ++j) sf_cp(tb + sfo + COL_SFBA + 4 * j, desc_sf(sa(B + B_DATA) + 512 * j));
            for (int j = 0; j < ncp; ++j) sf_cp(tb + sfo + COL_SFBB + 4 * j, desc_sf(sa(B + B_DATA) + 4096 + 512 * j));
            const uint64_t da = desc_sw128(sa(A)), dba = desc_sw128(sa(B)), dbb = desc_sw128(sa(B) + 104 * 128);
            const uint32_t acc_a = tb + (abuf ? COL_A1 : COL_A0);
            for (int i = 0; i < 2 && MMSOL_ABL != 7; ++i)
              mma_nvf4(acc_a, da + 4 * i, dba + 4 * i, ida, tb + sfo + COL_SFA + 8 * i, tb + sfo + COL_SFBA + 16 * i, (s > 0 || i > 0));
            if (s == 0) acquire(2);  // b turnaround: previous tile's b drained (only the stage-0 a-MMAs overlap it)
            for (int i = 0; i < 2 && MMSOL_ABL != 7; ++i)
              mma_nvf4(tb + COL_B, da + 4 * i, dbb + 4 * i, idb, tb + sfo + COL_SFA + 8 * i, tb + sfo + COL_SFBB + 8 * i, (s > 0 || i > 0));
          } else {
            uint8_t* B1 = bring + bs1 * B_SLOT;
            sf_cp(tb + COL_SE_SFB, desc_sf(sa(A + A_DATA))); sf_cp(tb + COL_SE_SFB + 4, desc_sf(sa(A + A_DATA) + 512));
            sf_cp(tb + COL_SE_SFA, desc_sf(sa(B + A_DATA)));
            if (kind == K_SEGU) sf_cp(tb + COL_SE_SFA + 4, desc_sf(sa(B1 + A_DATA)));
            const uint64_t dw = desc_sw128(sa(A)), dt0 = desc_sw128(sa(B)), dt1 = desc_sw128(sa(B1));
            for (int i = 0; i < 2; ++i) {
              mma_mx8(tb + COL_SE0, dt0 + 4 * i, dw + 4 * i, idse, tb + COL_SE_SFA, tb + COL_SE_SFB, (s > 0 || i > 0));
              if (kind == K_SEGU)
                mma_mx8(tb + COL_SE1, dt1 + 4 * i, dw + 4 * i, idse, tb + COL_SE_SFA + 4, tb + COL_SE_SFB, (s > 0 || i > 0));
            }
          }
          commit_mc(sa(&aempty[as_]), 3); commit_mc(sa(&bempty[bs0]), 3);
          if (kind == K_SEGU) commit_mc(sa(&bempty[bs1]), 3);
          ++acnt; bcnt += bslots(kind);
          if (samp && s == S - 1 && lane == 0) {
            ts[nts * 4 + 0] = kind | ((S - 1) << 8); ts[nts * 4 + 1] = (int64_t)(now_ns() - t_first);
            ts[nts * 4 + 2] = (int64_t)(clk64() - c_first); ts[nts * 4 + 3] = v.x >> 3; ++nts;
          }
        }
        commit_mc(sa(&accfull[ntile & 1]), 3);
        if (lane == 0) st[4] = (int64_t)now_ns();
        ++ntile; if (routed) ++ra;
      }
    }
  } else {
    // ================= epilogue warps 8..15: quadrant q = warp % 4, half h = (warp - 8) / 4 =================
    const int q = warp % 4, h = (warp - 8) / 4, et = threadIdx.x - 256;  // et 0..255
    const uint32_t trow = tb + ((uint32_t)(q * 32) << 16);
    const uint32_t res_lead[3] = {mapa(sa(&resempty[0]), 0), mapa(sa(&resempty[1]), 0), mapa(sa(&resempty[2]), 0)};
    uint32_t ntile = 0, ra = 0, fbuf = 0, cbuf = 0; int64_t pushed = 0; uint64_t t_lastpush = 0;
    for (int k = 0;; ++k) {
      const int tsl = k % TR;
      bwait_cl(sa(&tfull[tsl]), (k / TR) & 1, LIM, 80);
      const int4 v = tring[tsl];
      __syncwarp();
      if (lane == 0) bar_arrive_cluster(mapa(sa(&tempty[tsl]), 0));
      const int kind = v.x & 7, e = v.y, u = v.z, m = v.w;
      if (kind == K_END) break;
      const uint64_t te0 = now_ns();
      bwait(sa(&accfull[ntile & 1]), (ntile >> 1) & 1, LIM, 81);
      tc_fence_after();
      const uint64_t te1 = now_ns();
      const int T = u ? a.t_u1 : a.t_u0;
      if (MMSOL_ABL >= 5) {  // ablation: epilogue does nothing but release (prices epilogue/TMEM-read interference)
        tc_fence_before(); __syncwarp();
        if (lane == 0) {
          if (kind == K_FC1 || kind == K_FC2) { bar_arrive_cluster(res_lead[ra & 1]); bar_arrive_cluster(res_lead[2]); }
          else { bar_arrive_cluster(res_lead[0]); bar_arrive_cluster(res_lead[1]); bar_arrive_cluster(res_lead[2]); }
        }
        if (kind == K_FC1 || kind == K_SEGU) {
          named_bar(1, 256);
          if (et == 0) red_add_gpu(cw(a, kind == K_FC1 ? C_FC1 + e * 2 + u : C_GU + e), 1u);
        }
      } else if (kind == K_FC1) {
        // ---- SwiGLU (16-row gate/up stripes: lanes 0-15 gate, 16-31 up) + NVFP4 (16 features, UE4M3) -> staging ----
        uint8_t* sb = stg + (fbuf & 1) * STG;
        for (int pc = 0; pc < 2; ++pc) {  // pc 0 = b piece first (fast b turnaround), 1 = a piece
          const int N = pc ? NA : NB, col0 = pc ? ((ra & 1) ? COL_A1 : COL_A0) : COL_B, tok0 = pc ? 0 : NA;
          const int nch = N / 16;
          uint32_t r[16];
          for (int ch = h; ch < nch; ch += 2) {
            MM_LD16(trow + col0 + ch * 16, r); tmem_ld_wait();
            if (ch + 2 >= nch) { tc_fence_before(); __syncwarp(); if (lane == 0) bar_arrive_cluster(res_lead[pc ? (ra & 1) : 2]); }
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
              const int tok = tok0 + ch * 16 + j;
              const float x = __uint_as_float(r[j]);
              const float up = __shfl_down_sync(0xffffffff, x, 16);
              float act = silu(x) * up;
              float am = fabsf(act);
              am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, 8)); am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, 4));
              am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, 2)); am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, 1));
              const float sc = fmaxf(am * (1.f / 6.f), 1e-12f);
              const uint8_t c = e2m1(__fdividef(act, sc));
              const uint8_t c1 = (uint8_t)__shfl_xor_sync(0xffffffff, (int)c, 1);
              if (tok < T && lane < 16 && !(lane & 1)) sb[tok * 36 + q * 8 + lane / 2] = (uint8_t)(c | (c1 << 4));
              if (tok < T && lane == 0) sb[tok * 36 + 32 + q] = (uint8_t)__float2int_rn(sc);
            }
          }
          if (N == NB && h == 1 && nch % 2 == 1) {}  // nch even for both pieces (13 / 6): nothing
        }
        // a piece: 13 chunks -> half 0 takes 7, half 1 takes 6; resource release handled at its last chunk above
        named_bar(1, 256);
        if (et == 0) {
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          const uint64_t off = ((uint64_t)((e * a.n_u + u) * 24 + m) * 2 + cr) * (304 * 36) % (a.fc1out_span - STG);
          bulk_store(a.fc1out + off / 16 * 16, sa(sb), (uint32_t)((T * 36 + 15) / 16 * 16)); bulk_commit();
          bulk_wait<0>();
          asm volatile("fence.proxy.async.global;" ::: "memory");
          red_add_gpu(cw(a, C_FC1 + e * 2 + u), 1u);
        }
        ++fbuf;  // double-buffered staging: tile t+2 reuses this buffer only after et 0 passed tile t+1's barrier
      } else if (kind == K_FC2) {
        // ---- per-copy BF16; even tokens -> order buffer, odd tokens = last copy: + lower copy (L2), push to owner ----
        uint32_t r[16];
        for (int pc = 0; pc < 2; ++pc) {
          const int N = pc ? NB : NA, col0 = pc ? COL_B : ((ra & 1) ? COL_A1 : COL_A0), tok0 = pc ? NA : 0;
          const int nch = N / 16;
          for (int ch0 = 0; ch0 < nch; ch0 += 2) {
            const int ch = ch0 + h;
            uint8_t* sb = stg + (cbuf & 1) * STG + h * 4096;
            const bool have = ch < nch;
            if (have) {
              MM_LD16(trow + col0 + ch * 16, r); tmem_ld_wait();
              uint16_t lv[8];  // the 8 last copies' lower-copy BF16 values (one L2 round, loads in flight together)
              #pragma unroll
              for (int jj = 0; jj < 8; ++jj) {
                const int tok = tok0 + ch * 16 + 2 * jj + 1;
                const uint64_t lo = a.order + ((((uint64_t)(e * a.n_u + u) * 304 + tok - 1) * 14 + (m + 13) % 14) * 512 + cr * 256 + (q * 32 + lane) * 2) % (a.order_span - 512);
                lv[jj] = __ldcg((const unsigned short*)(lo & ~1ull));
              }
              #pragma unroll
              for (int j = 0; j < 16; ++j) {
                float x = __bfloat162float(__float2bfloat16_rn(__uint_as_float(r[j])));
                if (j & 1) x = __bfloat162float(__float2bfloat16_rn(fmaf(__uint_as_float((uint32_t)lv[j / 2] << 16), 1.f, x)));
                ((__nv_bfloat16*)(sb + ((j & 1) * 8 + j / 2) * 256))[q * 32 + lane] = __float2bfloat16_rn(x);  // even | odd
              }
            }
            named_bar(2 + h, 128);
            if (q == 0 && lane == 0 && have) {
              asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
              {  // non-last copies of the chunk: one 2 KiB bulk store (tile-major order-buffer layout)
                const int nv = min(8, max(0, (T - (tok0 + ch * 16) + 1) / 2));
                if (nv > 0)
                  bulk_store((a.order + ((((uint64_t)(e * a.n_u + u) * 304 + tok0 + ch * 16) * 14 + m) * 512 + cr * 4096) % (a.order_span - 4096)) & ~15ull,
                             sa(sb), (uint32_t)nv * 256);
              }
              for (int jj = 0; jj < 8; ++jj) {  // last copies: combine row slice pushed to the owner rank
                const int tok = tok0 + ch * 16 + 2 * jj + 1;
                if (tok >= T) break;
                const int gtok = (e * 293 + tok) * 7 + m;   // synthetic owner map (balanced over ranks)
                const int d = a.push ? (gtok % Wd) : a.rank;
                const uint64_t dst = a.dst_local + a.offs[d] + (int64_t)(a.push ? a.rank : gtok % Wd) * a.slice + a.comb_base +
                      ((((int64_t)(e * a.n_u + u) * 304 + tok) * 14 + m) * 512 + cr * 256) % (a.slice - a.comb_base - 512);
                pushed += 256; t_lastpush = now_ns();
                bulk_store(dst, sa(sb + (8 + jj) * 256), 256);
              }
              bulk_commit(); bulk_wait_read<1>();
            }
            ++cbuf;
            named_bar(2 + h, 128);
            if (ch0 + 2 >= nch) {  // this piece fully read by this warp
              if (pc == 0 || true) {}
            }
          }
          tc_fence_before(); __syncwarp();
          if (lane == 0) bar_arrive_cluster(res_lead[pc ? 2 : (ra & 1)]);
        }
      } else if (kind == K_SEGU) {
        // ---- SE gate_up: lane = token row, acc SE[h] 256 cols (16-col gate/up stripes) -> act -> MXFP8 rows ----
        const uint32_t col0 = h ? COL_SE1 : COL_SE0;
        const int row = e * 512 + h * 256 + cr * 128 + q * 32 + lane;
        uint8_t* dst = (uint8_t*)(a.order + (a.se_inter_off + ((uint64_t)row * 6144 + m * 128)) % (a.order_span - 256));
        dst = (uint8_t*)((uint64_t)dst & ~15ull);
        for (int g32 = 0; g32 < 4; ++g32) {  // 4 groups of 32 acts = 4 x (gate 16 | up 16) x 2
          float act[32]; uint32_t r[16], s2[16];
          for (int hh = 0; hh < 2; ++hh) {
            MM_LD16(trow + col0 + (g32 * 2 + hh) * 32, r); MM_LD16(trow + col0 + (g32 * 2 + hh) * 32 + 16, s2); tmem_ld_wait();
            #pragma unroll
            for (int j = 0; j < 16; ++j) act[hh * 16 + j] = silu(__uint_as_float(r[j])) * __uint_as_float(s2[j]);
          }
          float am = 0.f;
          #pragma unroll
          for (int j = 0; j < 32; ++j) am = fmaxf(am, fabsf(act[j]));
          const float inv = __fdividef(448.f, fmaxf(am, 1e-30f));
          uint32_t pk[8];
          #pragma unroll
          for (int j = 0; j < 8; ++j) {
            uint16_t lo, hi;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(act[4 * j + 1] * inv), "f"(act[4 * j] * inv));
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(act[4 * j + 3] * inv), "f"(act[4 * j + 2] * inv));
            pk[j] = (uint32_t)lo | ((uint32_t)hi << 16);
          }
          *(int4*)(dst + g32 * 32) = make_int4(pk[0], pk[1], pk[2], pk[3]);
          *(int4*)(dst + g32 * 32 + 16) = make_int4(pk[4], pk[5], pk[6], pk[7]);
        }
        tc_fence_before(); __syncwarp();
        if (lane == 0) { bar_arrive_cluster(res_lead[0]); bar_arrive_cluster(res_lead[1]); bar_arrive_cluster(res_lead[2]); }
        __threadfence();
        named_bar(1, 256);
        if (et == 0) red_add_gpu(cw(a, C_GU + e), 1u);
      } else {  // K_SEDN
        // ---- SE down: lane = token row, acc SE0 256 cols; half h takes cols h*128.. -> BF16 rows of the SE output ----
        const int row = e * 256 + cr * 128 + q * 32 + lane;
        uint8_t* dst = (uint8_t*)((a.order + (a.se_out_off + ((uint64_t)row * 7168 + m * 256 + h * 128) * 2) % (a.order_span - 512)) & ~15ull);
        for (int c = 0; c < 8; ++c) {
          uint32_t r[16]; MM_LD16(trow + COL_SE0 + h * 128 + c * 16, r); tmem_ld_wait();
          uint32_t p[8];
          #pragma unroll
          for (int j = 0; j < 8; ++j) p[j] = pack_bf16(__uint_as_float(r[2 * j]), __uint_as_float(r[2 * j + 1]));
          *(int4*)(dst + c * 32) = make_int4(p[0], p[1], p[2], p[3]);
          *(int4*)(dst + c * 32 + 16) = make_int4(p[4], p[5], p[6], p[7]);
        }
        tc_fence_before(); __syncwarp();
        if (lane == 0) { bar_arrive_cluster(res_lead[0]); bar_arrive_cluster(res_lead[1]); bar_arrive_cluster(res_lead[2]); }
      }
      if (et == 0 && (ntile % 4) == 0 && ntile / 4 < 16) {
        int64_t* es = a.tstamps + ((int64_t)((a.epoch - 1) % RING) * G + g) * 128 + 64 + (ntile / 4) * 4;
        es[0] = kind | 0x100; es[1] = (int64_t)(te1 - te0); es[2] = (int64_t)(now_ns() - te1); es[3] = v.x >> 3;
      }
      ++ntile; if (kind == K_FC1 || kind == K_FC2) ++ra;
    }
    if (q == 0 && lane == 0) {  // FC2 push issuers: all pushes complete before the done count
      bulk_wait<0>();
      asm volatile("fence.proxy.async.global;" ::: "memory");
      __threadfence_system();
      if (pushed) { atomicAdd((unsigned long long*)&st[8], (unsigned long long)pushed); }
      if (t_lastpush) atomicMax((unsigned long long*)&st[5], (unsigned long long)t_lastpush);
    }
  }
  // ================= end of layer: CTA done count; last CTA of the rank -> done counts, wait incoming, reset =================
  tc_fence_before();
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    const uint32_t prev = atom_add_gpu(cw(a, C_CTADONE), 1u);
    if (prev + 1u == (uint32_t)G) {
      __threadfence_system();
      for (int r = 0; r < Wd; ++r) red_add_sys(cpeer(a, (a.rank + 1 + r) % Wd, C_DONE), 1u);
      spin_ge(cw(a, C_DONE), a.epoch * (uint32_t)Wd, LIM, a.rank, 2, true);
      st[6] = (int64_t)now_ns();
      // reset the local per-call counters (every CTA of this rank is past its last use)
      volatile uint32_t* c = (volatile uint32_t*)a.ctr;
      c[C_CLAIM] = 0; c[C_CTADONE] = 0;
      for (int i = 0; i < E * 2; ++i) c[C_FC1 + i] = 0;
      for (int i = 0; i < 16; ++i) c[C_GU + i] = 0;
      for (int i = 0; i < 28; ++i) c[C_WDONE + i] = 0;
      __threadfence_system();
      int64_t* hd = a.hdr + (int64_t)((a.epoch - 1) % RING) * 8;
      hd[0] = a.epoch; hd[1] = a.n; hd[2] = a.push; hd[3] = (int64_t)now_ns(); hd[4] = G; hd[5] = a.ntiles; hd[6] = g; hd[7] = a.W;
    }
    st[7] = (int64_t)now_ns(); st[11] = (int64_t)clk64();
  }
  cluster_sync();
  if (warp == 4) { tc_fence_after(); tmem_dealloc576(tb); }
}

// ---------------- host side (torch extension) ----------------
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cudaTypedefs.h>

static PFN_cuTensorMapEncodeTiled_v12000 g_enc = nullptr;
static CUfunction g_fn = nullptr;
static void enc_init() {
  if (g_enc) return;
  cudaDriverEntryPointQueryResult q;
  void* f = nullptr;
  C10_CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &f, 12000, cudaEnableDefault, &q));
  TORCH_CHECK(f && q == cudaDriverEntryPointSuccess, "mmsol_skel: no cuTensorMapEncodeTiled");
  g_enc = (PFN_cuTensorMapEncodeTiled_v12000)f;
}
// 2D map over `bytes` at p, rows of `rowb` bytes (u8, SW128, box {128 B, br}) or u32 SF rows of 512 B (box {128, br})
static CUtensorMap mk(uint64_t p, uint64_t bytes, uint64_t rowb, uint32_t br, bool sf, int* rows_out) {
  enc_init();
  CUtensorMap m;
  const uint64_t rows = bytes / rowb;
  TORCH_CHECK(rows >= br, "mmsol_skel: region too small for its box");
  cuuint64_t dims[2] = {sf ? rowb / 4 : rowb, rows}; cuuint64_t str[1] = {rowb};
  cuuint32_t box[2] = {sf ? 128u : 128u, br}; cuuint32_t es[2] = {1, 1};
  CUresult r = g_enc(&m, sf ? CU_TENSOR_MAP_DATA_TYPE_UINT32 : CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)p, dims, str, box, es,
                     CU_TENSOR_MAP_INTERLEAVE_NONE, sf ? CU_TENSOR_MAP_SWIZZLE_NONE : CU_TENSOR_MAP_SWIZZLE_128B,
                     CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "mmsol_skel: cuTensorMapEncodeTiled failed ", (int)r);
  if (rows_out) *rows_out = (int)std::min<uint64_t>(rows, 0x7fffffff);
  return m;
}

// regions: [ptr, bytes] for W1, W2, SFW, X1 (dispatched rows: the local pre_reduced region), X2 (FC1-out), SFX,
// SEW1, SEW2, SFSE, SEX1, SEX2 (SE intermediate inside order)
torch::Tensor mmsol_skel_build(std::vector<int64_t> P, std::vector<int64_t> I) {
  // P: W1, W1b, W2, W2b, SFW, SFWb, X1, X1b, X2, X2b, SFX, SFXb, SEW1, SEW1b, SEW2, SEW2b, SFSE, SFSEb, SEX1, SEX1b, SEX2, SEX2b
  SkelArgs a; memset(&a, 0, sizeof(a));
  a.tmW1 = mk(P[0], P[1], 1792, 128, false, nullptr);
  a.tmW2 = mk(P[2], P[3], 1536, 128, false, nullptr);
  a.tmSFW = mk(P[4], P[5], 512, 4, true, &a.rows_sfw);
  a.tmX1 = mk(P[6], P[7], 1792, 152, false, &a.rows_x1);
  a.tmX2 = mk(P[8], P[9], 1536, 152, false, &a.rows_x2);
  a.tmSFX = mk(P[10], P[11], 512, 12, true, &a.rows_sfx);
  a.tmSEW1 = mk(P[12], P[13], 7168, 128, false, nullptr);
  a.tmSEW2 = mk(P[14], P[15], 6144, 128, false, nullptr);
  a.tmSFSE2 = mk(P[16], P[17], 512, 2, true, &a.rows_sesf);
  a.tmSFSE1 = mk(P[16], P[17], 512, 1, true, nullptr);
  a.tmSEX1 = mk(P[18], P[19], 7168, 128, false, &a.rows_sex1);
  a.tmSEX2 = mk(P[20], P[21], 6144, 128, false, &a.rows_sex2);
  (void)I;
  auto t = torch::empty({(int64_t)sizeof(SkelArgs)}, torch::dtype(torch::kUInt8));
  memcpy(t.data_ptr(), &a, sizeof(a));
  return t;
}

// I: n, n_u, nwin, W, se_mt, se_dt, ntiles, ngu, t_u0, t_u1, rank, world, push, claim_base, disp_prev_lo, disp_prev_hi, epoch, grid, smem
// Q: fc1out, fc1out_span, order, order_span, se_inter_off, se_out_off, dst_local, slice, comb_base, disp_per_dest, xsrc, xspan, ctr, limit
void mmsol_skel_launch(torch::Tensor maps, torch::Tensor offs, torch::Tensor ctr_peer, torch::Tensor stamps, torch::Tensor hdr,
                       torch::Tensor tstamps, std::vector<int64_t> I, std::vector<int64_t> Q) {
  SkelArgs a; memcpy(&a, maps.data_ptr(), sizeof(a));
  a.n = I[0]; a.n_u = I[1]; a.nwin = I[2]; a.W = I[3]; a.se_mt = I[4]; a.se_dt = I[5]; a.ntiles = I[6]; a.ngu = I[7];
  a.t_u0 = I[8]; a.t_u1 = I[9]; a.rank = I[10]; a.world = I[11]; a.push = I[12];
  a.claim_base = (uint32_t)I[13]; a.disp_prev_lo = (uint32_t)I[14]; a.disp_prev_hi = (uint32_t)I[15]; a.epoch = (uint32_t)I[16];
  a.fc1out = Q[0]; a.fc1out_span = Q[1]; a.order = Q[2]; a.order_span = Q[3]; a.se_inter_off = Q[4]; a.se_out_off = Q[5];
  a.dst_local = Q[6]; a.slice = Q[7]; a.comb_base = Q[8]; a.disp_per_dest = Q[9]; a.xsrc = Q[10]; a.xspan = Q[11];
  a.ctr = Q[12]; a.limit = Q[13];
  a.offs = offs.data_ptr<int64_t>(); a.ctr_peer = ctr_peer.data_ptr<int64_t>();
  a.stamps = stamps.data_ptr<int64_t>(); a.hdr = hdr.data_ptr<int64_t>(); a.tstamps = tstamps.data_ptr<int64_t>();
  const int grid = (int)I[17], smem = (int)I[18];
  CUlaunchConfig cfg = {}; cfg.gridDimX = grid; cfg.gridDimY = 1; cfg.gridDimZ = 1; cfg.blockDimX = NT; cfg.blockDimY = 1;
  cfg.blockDimZ = 1; cfg.sharedMemBytes = smem; cfg.hStream = c10::cuda::getCurrentCUDAStream().stream();
  CUlaunchAttribute at[1]; at[0].id = CU_LAUNCH_ATTRIBUTE_SHARED_MEMORY_MODE;
  at[0].value.sharedMemoryMode = CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY;
  cfg.attrs = at; cfg.numAttrs = 1;
  void* params[1] = {(void*)&a};
  CUresult r = cuLaunchKernelEx(&cfg, g_fn, params, nullptr);
  TORCH_CHECK(r == CUDA_SUCCESS, "mmsol_skel: launch failed ", (int)r);
}

int64_t mmsol_skel_smem() { return SMEM; }
void mmsol_skel_warm(int64_t smem) {  // module load + oversized-smem attributes at init (lazy-loading lesson)
  cudaFuncAttributes fa; C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, mmsol_skel_kernel));
  C10_CUDA_CHECK(cudaGetFuncBySymbol((cudaFunction_t*)&g_fn, (const void*)mmsol_skel_kernel));
  CUresult r1 = cuFuncSetAttribute(g_fn, CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE, CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY);
  CUresult r2 = CUDA_SUCCESS;  // as FI: oversized = mode attribute + launch attribute only (MAX_DYNAMIC refuses > opt-in)
  printf("MMSOL skel warm: static %zu regs %d dyn %lld: driver mode rc %d, maxdyn rc %d\n", fa.sharedSizeBytes, fa.numRegs,
         (long long)smem, (int)r1, (int)r2);
  TORCH_CHECK(r1 == CUDA_SUCCESS && r2 == CUDA_SUCCESS, "mmsol_skel: oversized smem attributes refused");
}
void mmsol_skel_zero(int64_t p, int64_t bytes) { C10_CUDA_CHECK(cudaMemsetAsync((void*)p, 0, bytes, c10::cuda::getCurrentCUDAStream().stream())); }
int64_t mmsol_skel_args_bytes() { return sizeof(SkelArgs); }
