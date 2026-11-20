// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Kimi K3.1 DSA decode under DCP: exact global top-K on a single path (included by
// dcp_exchange.cu after topk_v2.cuh; same parameters, mailboxes and epochs as topk_v2).
//
// Same selection and output as topk_merge: every key above the K-th key tau plus the
// `need` keys equal to tau with the smallest global ids; this rank's tokens above tau in
// position order, then its selected ties in position order (KV slots), -1 padding, count.
//
// One CTA per row; the row is bulk-copied to SMEM (its first kCapG 8-key groups; longer
// rows read the rest from global memory) and every pass is one compact loop over it:
//  1. a 1/8 group sample: its max, then a histogram of its keys within 8 octaves of the
//     max -> theta (the sample's (q + 256)-th key, q = ceil(K * L / G)) and e_est;
//  2. one pass: keys >= theta -> fine histogram (2048 keys), valid keys;
//  3. e = the q-th key; message = header + exact counts of the 256 keys around e + the
//     positions of every key in [e - d, e + d] (<= kSpBand, position order);
//  4. one round (multicast tagged words): tau, need and the tie cut by global id;
//  5. selection into position bitmaps (quads OR their words), one scan, the write.
// The only other path (decided alike on every rank from the exchanged words): radix rounds
// over the 40-bit composite (key << 24 | 0xFFFFFF - gid), 8 bits per round, from the key
// digits, or from the id digits when round 1 fixed tau. No second kernel.

namespace k31dcpx {

constexpr int kSpHdr = 8;
constexpr int kSpWin = 256;
constexpr int kSpBand = 256;   // band keys per rank
constexpr int kSpBandH = 128;  // band half-widths tried
constexpr int kSpWantX = 768;   // theta admits ~q + kSpWantX keys (window / band room below e)
constexpr int kSpMsg = kSpHdr + kSpWin / 2 + kSpBand;  // 392 words
static_assert(kSpMsg <= kV2Msg && kSpMsg % 4 == 0, "round-1 slot");
static_assert(kV2Tie <= kSpBand, "tie lists in tg");
constexpr int kSpFine = 2048;
constexpr int kSpBt = 1024;   // block-table entries cached per row
constexpr int kSpRs = 257;    // per-rank window suffix stride

template <int T>
struct SpCfg {
  static constexpr int kWarps = T / 32;
  // SMEM row: the first kCapG 8-key groups (longer rows stream the rest from global
  // memory and select in chunks of kCapG groups); the candidate pass gives thread t the
  // kNR contiguous groups [kNR t, kNR t + kNR) (odd 16-byte pitch: conflict-free reads)
  static constexpr int kNR = 9;
  static constexpr int kCapG = kNR * T;
  static constexpr int kBmW = kCapG / 4;  // bitmap words per chunk (slot 8g + j)
  static constexpr int kCand = 2 * kBmW;  // candidate list (aliases the bitmaps)
  static constexpr int kFB = kSpFine / T;
  static_assert(kFB * T == kSpFine && kV2MaxW * kSpRs <= kSpFine + 64, "SMEM aliasing");
  static_assert(kCapG * 8 < (1 << 20), "candidate positions take 20 bits");
};

struct SpTh {
  uint32_t th, t2;
  bool kx;
};
__device__ __forceinline__ SpTh sp_th(int th) {
  return SpTh{(uint32_t)th, v2_th_bf16(th), th == 0x8000};
}
// 0xFFFF in each 16-bit half whose key is >= th
__device__ __forceinline__ uint32_t sp_ge(uint32_t w, const SpTh& t) {
  if (!t.kx) return v2_ge2(w, t.t2);
  const uint32_t kw = keys2(w);
  return ((kw & 0xFFFFu) >= t.th ? 0xFFFFu : 0u) | ((kw >> 16) >= t.th ? 0xFFFF0000u : 0u);
}
// 8-bit mask (bit j: key j) from four half masks
__device__ __forceinline__ uint32_t sp_bits(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  return (a & 1u) | ((a >> 30) & 2u) | ((b & 1u) << 2) | ((b >> 28) & 8u) | ((c & 1u) << 4) |
         ((c >> 26) & 32u) | ((d & 1u) << 6) | ((d >> 24) & 128u);
}
__device__ __forceinline__ uint32_t sp_mask(const uint4& r, const SpTh& t) {
  return sp_bits(sp_ge(r.x, t), sp_ge(r.y, t), sp_ge(r.z, t), sp_ge(r.w, t));
}

// keys of 8 raw bf16 values as 4 packed words; key j of them (runtime j: selects, no
// local memory)
__device__ __forceinline__ uint4 sp_keys(const uint4& r) {
  return make_uint4(keys2(r.x), keys2(r.y), keys2(r.z), keys2(r.w));
}
__device__ __forceinline__ uint32_t sp_key_at(const uint4& kw, int j) {
  const uint32_t w = (j & 4) ? ((j & 2) ? kw.w : kw.z) : ((j & 2) ? kw.y : kw.x);
  return (j & 1) ? (w >> 16) : (w & 0xFFFFu);
}
__device__ __forceinline__ uint32_t sp_max_nan2(uint32_t a, uint32_t b) {
  uint32_t m;
  asm("max.NaN.bf16x2 %0, %1, %2;" : "=r"(m) : "r"(a), "r"(b));
  return m;
}
__device__ __forceinline__ uint32_t sp_nvalid(const uint4& r) {
  const uint4 kw = sp_keys(r);
  uint32_t n = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint32_t w = j == 0 ? kw.x : j == 1 ? kw.y : j == 2 ? kw.z : kw.w;
    n += ((w & 0xFFFFu) != 0u) + ((w >> 16) != 0u);
  }
  return n;
}

template <int T>
__global__ void __launch_bounds__(T, T >= 1024 ? 1 : 2) topk_sp_kernel(V2Params p) {
  using C = SpCfg<T>;
  constexpr int CG = C::kCapG, FB = C::kFB;
  __shared__ __align__(128) uint4 srow[CG];
  __shared__ __align__(8) uint64_t s_bar;
  __shared__ __align__(16) uint32_t bm[2 * C::kBmW];
  __shared__ __align__(16) uint32_t fine[kSpFine + 64];  // later: window suffixes / radix rx
  __shared__ __align__(16) uint32_t coarse[256];         // later: radix histogram
  __shared__ __align__(16) uint32_t msg[kSpMsg];
  __shared__ __align__(16) uint2 bnd[kSpBand];
  __shared__ __align__(16) uint32_t suf[T];
  __shared__ __align__(16) uint32_t tg[kV2MaxW * kSpBand];
  __shared__ int btab[kSpBt];
  __shared__ uint32_t warp_sums[64];
  __shared__ int s_i[16];
  __shared__ uint32_t s_u[16];
  __shared__ int s_tn[kV2MaxW];
  __shared__ long long s_dbg[32];
  uint32_t* const bmg = bm;
  // round-1 slots received: in the SMEM row (dead after the candidate list; paths that read the
  // row after the round re-land it)
  uint32_t* const rx = reinterpret_cast<uint32_t*>(srow);
  static_assert(kV2MaxW * kSpMsg * 4 <= C::kCapG * 16, "rx in the SMEM row");
  uint32_t* const bme = bm + C::kBmW;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  uint32_t ws_par = 0;
#define SPWS (warp_sums + 32 * (ws_par ^= 1u))
  // s_u: 0 max, 1 nc, 2 beyond, 3 valid, 4 nst, 5 nb, 6 sat, 7 gstar, 8 A, 9 bcnt, 10 rem
  // s_i: 0 theta, 1 e_est, 2 e, 3 lo, 4 bl, 5 bh, 6 hifail, 7 tau, 8 lev0, 9 dstar, 10 hd
  auto zero_row_smem = [&]() {
    for (int i = tid; i < kSpFine + 64; i += T) fine[i] = 0;
    for (int i = tid; i < 256; i += T) coarse[i] = 0;
    if (tid < 16) { s_u[tid] = 0; s_i[tid] = 0; }
  };
  zero_row_smem();
  if (tid == 0) {
    mbar_init(&s_bar, kV2Cp);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  uint32_t bar_phase = 0;
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const int W = p.W, K = p.K, IL = p.interleave;
  auto log2_or = [](int x) -> int { return (x > 0 && (x & (x - 1)) == 0) ? __ffs(x) - 1 : -1; };
  const int bs_sh = log2_or(p.block_size), il_sh = log2_or(IL), w_sh = log2_or(W);
  const int gid_sh = (il_sh >= 0 && w_sh >= 0) ? il_sh + w_sh : -1;
  auto gid_of = [&](int pos, int r) -> uint32_t {
    if (gid_sh >= 0) return (uint32_t)((((pos >> il_sh) << gid_sh) | (r << il_sh)) | (pos & (IL - 1)));
    return (uint32_t)(((pos / IL) * W + r) * IL + pos % IL);
  };
#define SPDBG(i)                                                                     \
  if (p.dbg && tid == 0) {                                                           \
    if ((i) == 0) {                                                                  \
      for (int z = 1; z < 30; ++z) s_dbg[z] = 0;                                     \
      p.dbg[(int64_t)row * kV2Dbg] = now_ns();                                       \
    }                                                                                \
    if (*reinterpret_cast<volatile int*>(&s_i[15]) == 0x7FFFFFF1) __trap();         \
    s_dbg[i] = (long long)clock64();                                                 \
    if ((i) == 8) {                                                                  \
      p.dbg[(int64_t)row * kV2Dbg + 8] = now_ns();                                   \
      for (int z = 0; z < 30; ++z) p.dbg[(int64_t)row * kV2Dbg + 32 + z] = s_dbg[z]; \
    }                                                                                \
  }
#define SPSTAT(i) \
  if (p.stats && tid == 0) atomicAdd(&p.stats[i], 1);

  for (int row = blockIdx.x; row < p.T; row += gridDim.x) {
    SPDBG(0);
    const uint16_t* lrow = p.logits + (int64_t)row * p.lstride;
    const int L = p.local_len[row];
    const int Gl = p.global_len[row];
    const int ng = (L + 7) >> 3;
    // group g = positions [8g, 8g + 8) (16-byte aligned rows, lcols % 8 == 0: the host
    // checks); groups [0, CG) land in SMEM (bulk copies by kV2Cp warps), later groups are
    // read from global memory; positions >= L read as NaN
    const int nsm = ::min(ng, CG);
    if ((tid & 31) == 0 && (tid >> 5) < kV2Cp) {
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      const int c = tid >> 5, a = nsm * c / kV2Cp, b = nsm * (c + 1) / kV2Cp;
      if (b > a) bulk_g2s(srow + a, reinterpret_cast<const uint4*>(lrow) + a, (uint32_t)(b - a) * 16u, &s_bar);
      else asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(&s_bar)) : "memory");
    }
    auto grp = [&](int g) -> uint4 {
      uint4 r;
      if (K31_LIKELY(g < CG)) {
        r = srow[g];
      } else {  // volatile: never speculated onto the SMEM path
        asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                     : "l"(reinterpret_cast<const uint4*>(lrow) + g));
      }
      const int nv = L - 8 * g;
      if (K31_RARE(nv < 8)) {
        r.x |= (nv > 0 ? 0u : 0xFFFFu) | (nv > 1 ? 0u : 0xFFFF0000u);
        r.y |= (nv > 2 ? 0u : 0xFFFFu) | (nv > 3 ? 0u : 0xFFFF0000u);
        r.z |= (nv > 4 ? 0u : 0xFFFFu) | (nv > 5 ? 0u : 0xFFFF0000u);
        r.w |= (nv > 6 ? 0u : 0xFFFFu) | (nv > 7 ? 0u : 0xFFFF0000u);
      }
      return r;
    };
    int* orow = p.out + (int64_t)row * p.ostride;
    const int req = p.req_of_row != nullptr ? p.req_of_row[row] : row / p.next_n;
    const int* btrow = p.block_table != nullptr ? p.block_table + (int64_t)req * p.bt_stride : nullptr;
    if (K31_RARE(Gl <= K)) {  // the whole context (every rank sees the same global length)
      for (int i = tid; i < L; i += T)
        orow[i] = btrow == nullptr ? i : btrow[i / p.block_size] * p.block_stride + i % p.block_size;
      for (int i = L + tid; i < K; i += T) orow[i] = -1;
      if (tid == 0) p.out_count[row] = L;
      mbar_wait(&s_bar, bar_phase);
      bar_phase ^= 1u;
      __syncthreads();
      continue;
    }
    if (tid == 0 && Gl >= (1 << 24)) spin_fail(p.err, 50);
    const int ep = p.epoch[row];
    const uint32_t tag = (uint32_t)(ep % 255);
    const int par = ep & 1;
    SPSTAT(0);
    const unsigned long long kl = (unsigned long long)K * (unsigned)L + (unsigned)(Gl - 1);
    const uint32_t q = max(1u, (uint32_t)(kl / (unsigned)Gl));
    mbar_wait(&s_bar, bar_phase);
    bar_phase ^= 1u;
    SPDBG(9);
    // ---- 1. sample (one group in 8): its max, then its keys within 8 octaves of it
    {
      uint32_t m = 1;
#pragma unroll 1
      for (int g = tid * 8 + (tid & 7); g < ng; g += T * 8) m = ::max(m, max_key_bound(grp(g)));
      m = __reduce_max_sync(0xffffffffu, m);
      if (lane == 0) atomicMax(&s_u[0], m);
    }
    __syncthreads();
    const uint32_t mb = s_u[0];
    // theta from the sample: bins of 4 keys over the 8 octaves below its max; a sample whose share
    // of q + kSpWantX keys lies further down (a few far larger keys) is histogrammed again with bins
    // of 64 keys over 128 octaves (the bf16 positive range), so theta never stops above the bulk
    uint32_t base = mb > 1024u ? mb - 1024u : 1u, bsh = 2;
#pragma unroll 1
    for (int pass = 0;; ++pass) {
#pragma unroll 1
      for (int g = tid * 8 + (tid & 7); g < ng; g += T * 8) {
        const uint4 kw = sp_keys(grp(g));
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const uint32_t k = sp_key_at(kw, j);
          if (k >= base) atomicAdd(&coarse[(k - base) >> bsh], 1u);
        }
      }
      __syncthreads();
      SPDBG(10);
      if (warp == 0) {  // theta: the bin of the sample's (q + kSpWantX)-th key
        const uint32_t want = (q + (uint32_t)kSpWantX + 7) / 8, want_e = max(1u, q / 8);
        uint32_t c[8], s = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) { c[j] = coarse[255 - 8 * lane - j]; s += c[j]; }
        uint32_t x = s;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
          const uint32_t y = __shfl_up_sync(0xffffffffu, x, o);
          if (lane >= o) x += y;
        }
        const uint32_t ex = x - s;
        const bool ft = ex < want && ex + s >= want, fe = ex < want_e && ex + s >= want_e;
        if (ft || fe) {
          uint32_t run = ex;
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const uint32_t bb = 255 - 8 * lane - j;
            const bool a = run < want, ae = run < want_e;
            run += c[j];
            if (ft && a && run >= want) s_i[0] = (int)(base + (bb << bsh));
            if (fe && ae && run >= want_e) s_i[1] = (int)(base + (bb << bsh));
          }
        }
        const uint32_t bt = __ballot_sync(0xffffffffu, ft), be = __ballot_sync(0xffffffffu, fe);
        if (lane == 0) {
          if (!bt) s_i[0] = (int)base;
          if (!be) s_i[1] = (int)base;
          s_i[12] = bt ? 0 : 1;
          s_i[2] = -1;
        }
      }
      __syncthreads();
      break;
    }
    SPDBG(1);
    int theta = s_i[0];
    // ---- 2. keys >= theta. Rows in SMEM: thread t tests its kNR contiguous groups (masks
    // in registers, no atomics), one scan places every candidate in position order
    // (cand[i] = position << 12 | min(key - theta, 4095)), then the fine histogram from
    // the list; the band and the selection read the list, not the row. Longer rows,
    // theta = key(+0) and lists over kCand take the full passes.
    uint32_t* const cand = bm;  // the selection bitmaps are unused on the list path
    int nc = 0;
    bool ovf = ng > CG;
    if (!ovf) {
      constexpr int NR = C::kNR;
      const int tg0 = tid * NR;
      auto group = [&](int g) -> uint4 {  // positions >= L read as NaN
        uint4 r = srow[g];
        const int nvk = L - 8 * g;
        if (K31_RARE(nvk < 8)) {
          r.x |= (nvk > 0 ? 0u : 0xFFFFu) | (nvk > 1 ? 0u : 0xFFFF0000u);
          r.y |= (nvk > 2 ? 0u : 0xFFFFu) | (nvk > 3 ? 0u : 0xFFFF0000u);
          r.z |= (nvk > 4 ? 0u : 0xFFFFu) | (nvk > 5 ? 0u : 0xFFFF0000u);
          r.w |= (nvk > 6 ? 0u : 0xFFFFu) | (nvk > 7 ? 0u : 0xFFFF0000u);
        }
        return r;
      };
      // per group: 4 packed compares (0xFFFF per key >= theta) summed as words (IADD3: the halves'
      // counts come back from the sum, below), a has-candidate bit, a NaN-propagating max. A
      // sample bound that admits fewer than q keys is lowered (1, 2, 4 octaves) and the pass redone.
      constexpr int kMW = (NR + 3) / 4;
      uint32_t mk[kMW], nv, tot, t2;
      int c;
#pragma unroll 1
      for (int it = 0;; ++it) {
      if (theta == 0x8000) {  // key(+0): -0 passes the packed compare
        ovf = true;
        break;
      }
      t2 = v2_th_bf16(theta);
      uint32_t cnt = 0, nm0 = 0, nm1 = 0;
#pragma unroll
      for (int w = 0; w < kMW; ++w) mk[w] = 0u;
#pragma unroll
      for (int i = 0; i < NR; ++i) {
        const int g = tg0 + i;
        if (g < ng) {
          const uint4 r = group(g);
          // 0xFFFF / 0 per key -> one byte per key (prmt) -> one bit per key (mask, mul)
          const uint32_t lo = __byte_perm(v2_ge2(r.x, t2), v2_ge2(r.y, t2), 0x6420);
          const uint32_t hi = __byte_perm(v2_ge2(r.z, t2), v2_ge2(r.w, t2), 0x6420);
          const uint32_t m8 = (((lo & 0x08040201u) * 0x01010101u) >> 24) |
                              ((((hi & 0x08040201u) * 0x01010101u) >> 20) & 0xF0u);
          cnt += __popc(m8);
          mk[i / 4] |= m8 << (8 * (i % 4));
          nm0 = sp_max_nan2(nm0, sp_max_nan2(r.x, r.y));
          nm1 = sp_max_nan2(nm1, sp_max_nan2(r.z, r.w));
        }
      }
      nv = 8u * (uint32_t)::max(0, ::min(NR, ng - tg0));
      if (K31_RARE(v2_is_nan2(nm0) || v2_is_nan2(nm1))) {  // NaN keys (and the tail group): recount
        nv = 0;
        for (int i = 0; i < NR; ++i)
          if (tg0 + i < ng) nv += sp_nvalid(group(tg0 + i));
      }
      c = (int)(v2_scan<T>(cnt | (nv << 16), SPWS, &tot) & 0xFFFFu);
      nc = (int)(tot & 0xFFFFu);
      if (K31_LIKELY(nc >= (int)q || it == 2)) break;
      SPSTAT(14);
      if (it == 0 && theta > (int)base) {  // sample noise: one step down inside the sample's range
        theta = ::max((int)base, theta - 256);
        continue;
      }
      // the q-th key lies below the 8 octaves under the sample max (a few far larger keys, or
      // a bulk of negative logits): sample histogram over the whole key range (bins of 256 keys);
      // theta = the bin of the sample's (q + kSpWantX)-th key, kept within the fine range below
      // the bin of its q-th key
      SPSTAT(19);
      __syncthreads();
      for (int i = tid; i < 256; i += T) coarse[i] = 0;
      __syncthreads();
#pragma unroll 1
      for (int g = tid * 8 + (tid & 7); g < ng; g += T * 8) {
        const uint4 kw = sp_keys(grp(g));
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const uint32_t k = sp_key_at(kw, j);
          if (k) atomicAdd(&coarse[k >> 8], 1u);
        }
      }
      __syncthreads();
      if (warp == 0) {
        const uint32_t want = (q + (uint32_t)kSpWantX + 7) / 8, want_e = max(1u, q / 8);
        uint32_t cb[8], sb = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) { cb[j] = coarse[255 - 8 * lane - j]; sb += cb[j]; }
        uint32_t x = sb;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
          const uint32_t y = __shfl_up_sync(0xffffffffu, x, o);
          if (lane >= o) x += y;
        }
        const uint32_t ex = x - sb;
        const bool ft = ex < want && ex + sb >= want, fe = ex < want_e && ex + sb >= want_e;
        uint32_t run = ex;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const uint32_t bb = 255 - 8 * lane - j;
          const bool a = run < want, ae = run < want_e;
          run += cb[j];
          if (ft && a && run >= want) s_i[13] = (int)(bb << 8);
          if (fe && ae && run >= want_e) s_i[14] = (int)(bb << 8);
        }
        const uint32_t bt = __ballot_sync(0xffffffffu, ft), be = __ballot_sync(0xffffffffu, fe);
        if (lane == 0) {
          if (!bt) s_i[13] = 1;
          if (!be) s_i[14] = 1;
        }
      }
      __syncthreads();
      theta = ::max(1, ::max(s_i[13], s_i[14] - (kSpFine - 512)));
      }
      SPDBG(11);
      ovf = ovf || nc > C::kCand;
      if (!ovf) {
        const uint16_t* s16 = reinterpret_cast<const uint16_t*>(srow);
#pragma unroll
        for (int w = 0; w < kMW; ++w) {
          uint32_t m = mk[w];
          while (m) {
            const int b = __ffs(m) - 1;
            m &= m - 1;
            const int pos = 8 * (tg0 + 4 * w + (b >> 3)) + (b & 7);
            const uint32_t d = (keys2(s16[pos]) & 0xFFFFu) - (uint32_t)theta;
            cand[c++] = ((uint32_t)pos << 12) | ::min(d, 4095u);
          }
        }
        if (tid == 0) {
          s_u[1] = (uint32_t)nc;
          s_u[3] = tot >> 16;
        }
      }
      __syncthreads();
      SPDBG(12);
      if (!ovf) {
        uint32_t bey = 0;
        for (int i = tid; i < nc; i += T) {
          const uint32_t d = cand[i] & 0xFFFu;
          if (d < (uint32_t)kSpFine) atomicAdd(&fine[d], 1u);
          else ++bey;
        }
        bey = __reduce_add_sync(0xffffffffu, bey);
        if (lane == 0 && bey) atomicAdd(&s_u[2], bey);
      }
    }
    if (ovf) {
      const SpTh tt = sp_th(theta);
      uint32_t cnt = 0, bey = 0, nv = 0;
      auto visit = [&](int g, const uint4& r) {
        if (g >= ng) return;
        nv += K31_RARE(v2_is_nan2(v2_max_nan(r))) ? sp_nvalid(r) : 8u;
        uint32_t mk = sp_mask(r, tt);
        if (mk) {
          cnt += __popc(mk);
          const uint4 kw = sp_keys(r);
#pragma unroll 1
          while (mk) {
            const int j = __ffs(mk) - 1;
            mk &= mk - 1;
            const int d = (int)sp_key_at(kw, j) - theta;
            if (d < kSpFine) atomicAdd(&fine[d], 1u);
            else ++bey;
          }
        }
      };
#pragma unroll 1
      for (int g = tid; g < ng; g += T) visit(g, grp(g));
      cnt = __reduce_add_sync(0xffffffffu, cnt);
      bey = __reduce_add_sync(0xffffffffu, bey);
      nv = __reduce_add_sync(0xffffffffu, nv);
      if (lane == 0) {
        atomicAdd(&s_u[1], cnt);
        atomicAdd(&s_u[2], bey);
        atomicAdd(&s_u[3], nv);
      }
    }
    __syncthreads();
    SPDBG(2);
    // ---- 3. e, window, count above it, band
    const uint32_t bey = s_u[2];
    {
      uint32_t v = 0;
#pragma unroll
      for (int j = 0; j < FB; ++j) v += fine[kSpFine - 1 - FB * tid - j];
      uint32_t tot;
      const uint32_t pre = v2_scan<T>(v, SPWS, &tot);
      const uint32_t above = bey + pre;
      suf[tid] = above;
      if (above < q && above + v >= q) {
        uint32_t run = above;
        for (int j = 0; j < FB; ++j) {
          run += fine[kSpFine - 1 - FB * tid - j];
          if (run >= q) { s_i[2] = theta + kSpFine - 1 - FB * tid - j; break; }
        }
      }
    }
    __syncthreads();
    SPDBG(13);
    // keys >= x, for x in [theta, theta + kSpFine]
    auto cloc = [&](int x) -> uint32_t {
      const int d = x - theta;
      if (d >= kSpFine) return bey;
      const int t = (kSpFine - 1 - d) / FB, top = kSpFine - 1 - FB * t;
      uint32_t c = suf[t];
      for (int y = top; y >= d; --y) c += fine[y];
      return c;
    };
    const int e = s_i[2];
    if (p.dbg && tid == 0) {
      long long* d = p.dbg + (int64_t)row * kV2Dbg;
      d[31] = s_u[1]; d[29] = e - theta; d[28] = q; d[27] = bey; d[26] = (long long)theta; d[25] = mb;
    }
    const int lo = e < 0 ? 0 : ::min(::max(theta, e - kSpWin / 2), theta + kSpFine - kSpWin);
    if (tid < kSpWin / 2) {
      const uint32_t c0 = e < 0 ? 0u : fine[lo - theta + 2 * tid], c1 = e < 0 ? 0u : fine[lo - theta + 2 * tid + 1];
      if (c0 >= kCountMax || c1 >= kCountMax) {  // a 12-bit count saturates: the exact count of ONE
        s_u[6] = 1;                                // such key goes in the header (words 5-6)
        if ((c0 >= kCountMax && lo + 2 * tid == 0x8000) || (c1 >= kCountMax && lo + 2 * tid + 1 == 0x8000)) s_u[14] = 1;
        atomicAdd(&s_u[12], (c0 >= kCountMax ? 1u : 0u) + (c1 >= kCountMax ? 1u : 0u));
        s_i[11] = c0 >= kCountMax ? 2 * tid : 2 * tid + 1;
        s_u[13] = c0 >= kCountMax ? c0 : c1;
      }
      msg[kSpHdr + tid] = tagw(tag, (::min(c0, kCountMax) << 12) | ::min(c1, kCountMax));
    }
    // band [e - h, e + h]: the widest half-width h < kSpBandH whose keys fit kSpBand (monotone in h:
    // the count of passing h gives it, one barrier); band words are position (16 bits) | key - bl
    // (8 bits), so a band needs L < 2^16
    bool bok = false;
    if (tid >= kSpWin / 2 && tid < kSpWin / 2 + kSpBandH) {
      const int h = tid - kSpWin / 2;
      bok = e >= 0 && e - h >= theta && e + h + 1 <= theta + kSpFine && L < (1 << 16) &&
            cloc(e - h) - cloc(e + h + 1) <= (uint32_t)kSpBand;
    }
    const int hs = __syncthreads_count(bok) - 1;  // -1: no band
    const int bl = e - hs, bh = e + hs;
    if (tid == 0) {
      s_i[9] = hs;
      s_i[4] = bl;
      s_i[5] = bh;
      s_u[9] = hs >= 0 ? cloc(bl) - cloc(bh + 1) : 0u;
      s_u[8] = e < 0 ? 0u : cloc(lo + kSpWin);
    }
    if (hs >= 0 && !ovf) {  // the band's keys from the list (bl >= theta), in position order
      const int cp = (nc + T - 1) / T, i0 = ::min(nc, tid * cp), i1 = ::min(nc, i0 + cp);
      const uint32_t dl = (uint32_t)(bl - theta), dh = (uint32_t)(bh - theta);
      uint32_t cb = 0;
      for (int i = i0; i < i1; ++i) {
        const uint32_t d = cand[i] & 0xFFFu;
        cb += d >= dl && d <= dh;
      }
      uint32_t btot;
      uint32_t o = v2_scan<T>(cb, SPWS, &btot);
      if (tid == 0) s_u[5] = btot;
      for (int i = i0; i < i1 && cb; ++i) {
        const uint32_t x = cand[i], d = x & 0xFFFu;
        if (d >= dl && d <= dh) {
          if (o < (uint32_t)kSpBand) msg[kSpHdr + kSpWin / 2 + o] = tagw(tag, (x >> 12) | ((d - dl) << 16));
          ++o;
        }
      }
    } else if (hs >= 0) {  // the band's keys (<= kSpBand by the counts) with their positions
      const SpTh tl = sp_th(bl), th1 = sp_th(bh + 1);
      auto visit = [&](int g, const uint4& r) {
        if (g >= ng) return;
        uint32_t mk = sp_mask(r, tl) & ~sp_mask(r, th1);
        if (mk) {
          const uint4 kw = sp_keys(r);
#pragma unroll 1
          while (mk) {
            const int j = __ffs(mk) - 1;
            mk &= mk - 1;
            const uint32_t s = atomicAdd(&s_u[5], 1u);
            if (s < kSpBand) bnd[s] = make_uint2((uint32_t)(8 * g + j), sp_key_at(kw, j));
          }
        }
      };
#pragma unroll 1
      for (int g = tid; g < ng; g += T) visit(g, grp(g));
    }
    __syncthreads();
    SPDBG(15);
    const int nb = ::min((int)s_u[5], kSpBand);
    const bool bvalid = hs >= 0 && (uint32_t)nb == s_u[9] && s_u[5] <= (uint32_t)kSpBand;
    if (tid < kSpBand) {
      if (ovf && tid < nb && bvalid) {
        const uint2 me = bnd[tid];
        int rk = 0;
        for (int s = 0; s < nb; ++s) rk += bnd[s].x < me.x;
        msg[kSpHdr + kSpWin / 2 + rk] = tagw(tag, me.x | ((me.y - (uint32_t)bl) << 16));
      }
      if (tid >= (bvalid ? nb : 0)) msg[kSpHdr + kSpWin / 2 + tid] = tagw(tag, 0u);
    } else if (tid == kSpBand) {
      const bool sat1 = s_u[12] == 1 && s_u[13] < (1u << 24);  // one saturated key: exact count sent
      const uint32_t flags = (e >= 0 && (s_u[6] == 0 || sat1) ? 1u : 0u) | (bvalid ? 2u : 0u) | (ovf ? 0u : 4u);
      if (p.stats != nullptr) {  // DIAG (header thread)
        if (e < 0) atomicAdd(&p.stats[17], 1);
        if (s_u[6]) atomicAdd(&p.stats[15], 1);
        if (s_u[14]) atomicAdd(&p.stats[16], 1);
        if (s_u[12] > 1) atomicAdd(&p.stats[18], 1);
      }
      msg[0] = tagw(tag, (uint32_t)lo | (flags << 16));
      msg[1] = tagw(tag, ::min(s_u[8], 0xFFFFFFu));
      msg[2] = tagw(tag, ::min(s_u[3], 0xFFFFFFu));
      msg[3] = tagw(tag, (uint32_t)(bl & 0xFFFF) | ((uint32_t)(bh - bl) << 16));
      msg[4] = tagw(tag, bvalid ? (uint32_t)nb : 0u);
      msg[5] = tagw(tag, sat1 ? s_u[13] : 0u);
      msg[6] = tagw(tag, sat1 ? (0x100u | (uint32_t)s_i[11]) : 0u);
      msg[7] = tagw(tag, 0u);
    }
    __syncthreads();
    SPDBG(14);
    // ---- 4. round 1
    const int64_t r1row = p.off_r1 + ((int64_t)(par * p.cap + row) * W) * kV2Msg;
    auto send = [&](const uint32_t* m, int nw, int64_t off0) {
      for (int j = tid * 4; j < nw; j += T * 4) {
        const uint4 v = make_uint4(m[j], m[j + 1], m[j + 2], m[j + 3]);
        const int64_t off = off0 + j;
        if (p.mc) mc_st_v4(p.mc + off * 4, v);
        else
          for (int d = 0; d < W; ++d) st_v4(reinterpret_cast<uint32_t*>(p.peers[d]) + off, v);
      }
    };
    auto poll = [&](const uint32_t* src, uint32_t* dst, int code) {
      const long long t0 = now_ns();
      uint4 v = ld_sys_v4(src);
      while (!tagged(v, tag)) {
        if (spin_late(p.spin_ns, t0)) { spin_fail(p.err, code); break; }
        __nanosleep(128);
        v = ld_sys_v4(src);
      }
      *reinterpret_cast<uint4*>(dst) = v;
    };
    send(msg, kSpMsg, r1row + (int64_t)p.rank * kV2Msg);
    SPDBG(3);
    // in the round's shadow: the row's block-table entries
    const int nblk = bs_sh >= 0 ? (L + p.block_size - 1) >> bs_sh : kSpBt + 1;
    const bool bt_sm = btrow != nullptr && nblk <= kSpBt;
    if (bt_sm)
      for (int i = tid; i < nblk; i += T) btab[i] = btrow[i];
    for (int i = tid; i < W * (kSpMsg / 4); i += T) {
      const int r = i / (kSpMsg / 4), j = (i % (kSpMsg / 4)) * 4;
      poll(p.box + r1row + (int64_t)r * kV2Msg + j, rx + r * kSpMsg + j, 51);
    }
    __syncthreads();
    SPDBG(4);
    // ---- 5. tau, need, tie cut (every rank alike)
    auto hdr = [&](int r, int w) -> uint32_t { return rx[r * kSpMsg + w] & 0xFFFFFFu; };
    uint32_t vtot = 0;
    bool wok = true;
    int lo_max = 0, hi = 1 << 30;
    for (int r = 0; r < W; ++r) {
      const uint32_t v = hdr(r, 2);
      vtot += v;
      if (v) {
        const int lr = (int)(hdr(r, 0) & 0xFFFFu);
        wok = wok && ((hdr(r, 0) >> 16) & 1u);
        lo_max = ::max(lo_max, lr);
        hi = ::min(hi, lr + kSpWin);
      }
    }
    int tau = 1, lev0 = -1;
    uint32_t gstar = 0xFFFFFFFFu, rem = 0;
    uint32_t* const rsuf = fine;  // [W][kSpRs]: keys >= lo_r + u of rank r
    if (vtot <= (uint32_t)K) {
      SPSTAT(4);
    } else if (!wok || lo_max >= hi) {
      SPSTAT(!wok ? 7 : 8);
      lev0 = 0;
      rem = (uint32_t)K;
    } else {
      if (warp < W) {
        const int r = warp;
        uint32_t c[8], s = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint32_t w = rx[r * kSpMsg + kSpHdr + 4 * lane + j];
          c[2 * j] = (w >> 12) & 0xFFFu;
          c[2 * j + 1] = w & 0xFFFu;
          const uint32_t h6 = hdr(r, 6);  // the exact count of a saturated key
          if ((h6 & 0x100u) && (int)(h6 & 0xFFu) == 8 * lane + 2 * j) c[2 * j] = hdr(r, 5);
          if ((h6 & 0x100u) && (int)(h6 & 0xFFu) == 8 * lane + 2 * j + 1) c[2 * j + 1] = hdr(r, 5);
          s += c[2 * j] + c[2 * j + 1];
        }
        uint32_t x = s;  // suffix over lanes (lane 31 holds the top keys)
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
          const uint32_t y = __shfl_down_sync(0xffffffffu, x, o);
          if (lane + o < 32) x += y;
        }
        uint32_t run = hdr(r, 1) + x - s;
#pragma unroll
        for (int j = 7; j >= 0; --j) {
          run += c[j];
          rsuf[r * kSpRs + 8 * lane + j] = run;
        }
        if (lane == 31) rsuf[r * kSpRs + kSpWin] = hdr(r, 1);
      }
      __syncthreads();
      auto cglob = [&](int v) -> uint32_t {
        uint32_t c = 0;
        for (int r = 0; r < W; ++r)
          if (hdr(r, 2)) c += rsuf[r * kSpRs + v - (int)(hdr(r, 0) & 0xFFFFu)];
        return c;
      };
      const int v = lo_max + tid;
      const bool pr = v <= hi && cglob(v) >= (uint32_t)K;
      if (pr && v == hi) s_i[6] = 1;
      SPDBG(16);
      const int n = __syncthreads_count(pr);
      SPDBG(17);
      if (n == 0 || s_i[6]) {
        SPSTAT(n == 0 ? 9 : 10);
        lev0 = 0;
        rem = (uint32_t)K;
      } else {
        tau = lo_max + n - 1;
        const uint32_t c1 = cglob(tau + 1), c0 = cglob(tau);
        const uint32_t need = (uint32_t)K - c1;
        if (need != c0 - c1) {
          bool bok = true;
          for (int r = 0; r < W; ++r) {
            if (!hdr(r, 2)) continue;
            const int lr = (int)(hdr(r, 0) & 0xFFFFu);
            const bool ties = rsuf[r * kSpRs + tau - lr] != rsuf[r * kSpRs + tau + 1 - lr];
            const int blr = (int)(hdr(r, 3) & 0xFFFFu), bhr = blr + (int)((hdr(r, 3) >> 16) & 0xFFu);
            if (ties && !(((hdr(r, 0) >> 17) & 1u) && blr <= tau && tau <= bhr)) bok = false;
          }
          SPDBG(18);
          // a band missed ties at tau: ONE tie round of complete ascending tie lists (<= kV2Tie ids
          // per rank; ranks holding ties send them from their candidate lists), else id radix
          bool tfit = true;
          if (!bok) {
            for (int r = 0; r < W; ++r) {
              int t = 0;
              if (hdr(r, 2)) {
                const int lr = (int)(hdr(r, 0) & 0xFFFFu);
                t = (int)(rsuf[r * kSpRs + tau - lr] - rsuf[r * kSpRs + tau + 1 - lr]);
              }
              if (t > kV2Tie || (t > 0 && !((hdr(r, 0) >> 18) & 1u))) tfit = false;
              if (tid == 0) s_tn[r] = t;
            }
          }
          if (!bok && !tfit) {
            SPSTAT(11);
            lev0 = 2;
            rem = need;
          } else if (!bok) {
            SPSTAT(13);
            __syncthreads();  // s_tn; every thread is past the round-1 reads of msg
            const int tme = s_tn[p.rank];
            if (tme > 0) {  // this rank's ties at tau from its list (position = id order)
              const uint32_t dt = (uint32_t)(tau - theta);
              const int cp = (nc + T - 1) / T, i0 = ::min(nc, tid * cp), i1 = ::min(nc, i0 + cp);
              uint32_t ct = 0;
              for (int i = i0; i < i1; ++i) ct += (cand[i] & 0xFFFu) == dt;
              uint32_t ttot;
              uint32_t o = v2_scan<T>(ct, SPWS, &ttot);
              for (int i = i0; i < i1 && ct; ++i) {
                const uint32_t x = cand[i];
                if ((x & 0xFFFu) == dt) {
                  if (o < (uint32_t)kV2Tie) msg[o] = tagw(tag, gid_of((int)(x >> 12), p.rank));
                  ++o;
                }
              }
              if (tid == 0 && (int)ttot != tme) spin_fail(p.err, 54);
            }
            const int nmp = (tme + 3) & ~3;
            if (tme + tid < nmp) msg[tme + tid] = tagw(tag, 0u);
            __syncthreads();
            const int64_t trow = p.off_tie + ((int64_t)(par * p.cap + row) * W) * kV2Tie;
            send(msg, nmp, trow + (int64_t)p.rank * kV2Tie);
            for (int i = tid; i < W * (kV2Tie / 4); i += T) {
              const int r = i / (kV2Tie / 4), j = (i % (kV2Tie / 4)) * 4;
              if (j < ((s_tn[r] + 3) & ~3)) poll(p.box + trow + (int64_t)r * kV2Tie + j, tg + r * kV2Tie + j, 55);
            }
            __syncthreads();
            for (int c = tid; c < W * kV2Tie; c += T) {  // the need-th smallest id of the union
              const int r = c / kV2Tie, j = c % kV2Tie;
              if (j >= s_tn[r]) continue;
              const uint32_t x = tg[c] & 0xFFFFFFu;
              int rk = j;
              for (int r2 = 0; r2 < W; ++r2)
                if (r2 != r) rk += count_below(tg + r2 * kV2Tie, s_tn[r2], x);
              if (rk == (int)need - 1) s_u[7] = x;
            }
            for (int c = tid; c < W * kV2Tie; c += T)
              if (c % kV2Tie < ((s_tn[c / kV2Tie] + 3) & ~3)) st_u32(p.box + trow + c, kSent);
            __syncthreads();
            gstar = s_u[7];
          } else {
            for (int i = tid; i < W * kSpBand; i += T) {  // every rank's ties at tau
              const int r = i / kSpBand, j = i % kSpBand;
              if (j < (int)hdr(r, 4)) {
                const uint32_t w = hdr(r, kSpHdr + kSpWin / 2 + j);
                if ((int)(hdr(r, 3) & 0xFFFFu) + (int)((w >> 16) & 0xFFu) == tau)
                  tg[atomicAdd(&s_u[11], 1u)] = gid_of((int)(w & 0xFFFFu), r);
              }
            }
            __syncthreads();
            const int nt = (int)s_u[11];
            for (int i = tid; i < nt; i += T) {  // the need-th smallest id
              const uint32_t gv = tg[i];
              uint32_t rho = 0;
              for (int s = 0; s < nt; ++s) rho += tg[s] < gv;
              if (rho == need - 1) s_u[7] = gv;
            }
            __syncthreads();
            gstar = s_u[7];
            SPSTAT(1);
          }
        } else {
          SPSTAT(1);
        }
      }
    }
    SPDBG(5);
    // paths that read the row after the round (radix rounds, full selection) re-land it: the
    // received slots overwrote its head
    if (K31_RARE(ovf || lev0 >= 0 || vtot <= (uint32_t)K || tau < theta)) {
      __syncthreads();
      if ((tid & 31) == 0 && (tid >> 5) < kV2Cp) {
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        const int c = tid >> 5, a = nsm * c / kV2Cp, b = nsm * (c + 1) / kV2Cp;
        if (b > a) bulk_g2s(srow + a, reinterpret_cast<const uint4*>(lrow) + a, (uint32_t)(b - a) * 16u, &s_bar);
        else asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(&s_bar)) : "memory");
      }
      mbar_wait(&s_bar, bar_phase);
      bar_phase ^= 1u;
    }
    if (K31_RARE(lev0 >= 0)) {
      // radix rounds over c = key << 24 | (0xFFFFFF - gid): 8 bits per round, from the top
      SPSTAT(lev0 == 0 ? 3 : 2);
      uint64_t pre = lev0 == 0 ? 0ull : (uint64_t)tau;
      for (int lev = lev0; lev < 5; ++lev) {
        for (int i = tid; i < 256; i += T) coarse[i] = 0;
        __syncthreads();
        {
          int cur = -1;
          uint32_t run = 0;
          auto visit = [&](int g, const uint4& r) {
            if (g >= ng) return;
            const uint4 kw = sp_keys(r);
#pragma unroll 1
            for (int j = 0; j < 8; ++j) {
              const uint32_t k = sp_key_at(kw, j);
              if (!k) continue;
              int dg;
              if (lev < 2) {
                if (lev == 1 && (k >> 8) != (uint32_t)pre) continue;
                dg = lev == 0 ? (int)(k >> 8) : (int)(k & 255u);
              } else {
                if (k != (uint32_t)(pre >> (8 * (lev - 2)))) continue;
                const uint64_t c40 = ((uint64_t)k << 24) | (0xFFFFFFu - gid_of(8 * g + j, p.rank));
                if ((c40 >> (40 - 8 * lev)) != pre) continue;
                dg = (int)((c40 >> (32 - 8 * lev)) & 255u);
              }
              if (dg != cur) {
                if (run) atomicAdd(&coarse[cur], run);
                cur = dg;
                run = 0;
              }
              ++run;
            }
          };
#pragma unroll 1
          for (int g = tid; g < ng; g += T) visit(g, grp(g));
          if (run) atomicAdd(&coarse[cur], run);
        }
        __syncthreads();
        const int64_t rrow = p.off_rad + (((int64_t)(par * p.cap + row) * kV2RadR + lev) * W) * 256;
        if (tid < 64) {
          const uint4 v = make_uint4(tagw(tag, coarse[4 * tid]), tagw(tag, coarse[4 * tid + 1]),
                                     tagw(tag, coarse[4 * tid + 2]), tagw(tag, coarse[4 * tid + 3]));
          const int64_t off = rrow + (int64_t)p.rank * 256 + 4 * tid;
          if (p.mc) mc_st_v4(p.mc + off * 4, v);
          else
            for (int d = 0; d < W; ++d) st_v4(reinterpret_cast<uint32_t*>(p.peers[d]) + off, v);
        }
        for (int i = tid; i < W * 64; i += T) poll(p.box + rrow + 4 * (int64_t)i, fine + 4 * i, 52);
        __syncthreads();
        for (int i = tid; i < W * 256; i += T) st_u32(p.box + rrow + i, 0xFFFFFFFFu);  // re-arm
        uint32_t h = 0;
        if (tid < 256)
          for (int r = 0; r < W; ++r) h += fine[r * 256 + 255 - tid] & 0xFFFFFFu;
        uint32_t tot;
        const uint32_t ex = v2_scan<T>(h, SPWS, &tot);
        const uint32_t rm = lev == lev0 ? rem : s_u[10];
        if (tid < 256 && ex < rm && ex + h >= rm) {
          s_i[9] = 255 - tid;
          s_u[10] = rm - ex;
          s_i[10] = (int)h;
        }
        __syncthreads();
        pre = (pre << 8) | (uint64_t)s_i[9];
        if (lev == 1 && (uint32_t)s_i[10] == s_u[10]) {  // every tie at tau
          tau = (int)pre;
          gstar = 0xFFFFFFFFu;
          break;
        }
        if (lev == 4) {
          tau = (int)(pre >> 24);
          gstar = 0xFFFFFFu - (uint32_t)(pre & 0xFFFFFFu);
        }
        __syncthreads();
      }
      SPDBG(6);
    }
    // ---- 6. selection -> position bitmaps (chunks of CG groups) -> write; rows of more
    // than one chunk count every chunk first (the ties follow all keys above tau)
    {
      const SpTh t1 = sp_th(tau + 1), t0 = sp_th(tau);
      auto slot = [&](int pos) -> int {
        if (btrow == nullptr) return pos;
        if (bt_sm) return btab[pos >> bs_sh] * p.block_stride + (pos & (p.block_size - 1));
        return btrow[pos / p.block_size] * p.block_stride + pos % p.block_size;
      };
      const int nch = (ng + CG - 1) / CG;
      int gall = 0, cnt = 0;
      // list path: every selected key is a candidate (tau >= lo >= theta); thread t
      // owns the contiguous list range [cp t, cp t + cp) (position order)
      const bool lsel = !ovf && lev0 < 0 && vtot > (uint32_t)K && tau >= theta;
      SPSTAT(ovf ? 6 : 5);
      if (lsel) {
        SPSTAT(12);
        const int cp = (nc + T - 1) / T, i0 = ::min(nc, tid * cp), i1 = ::min(nc, i0 + cp);
        const uint32_t dt = (uint32_t)(tau - theta);
        auto pick = [&](uint32_t x) -> int {  // 1 above tau, 2 selected tie, 0 neither
          const uint32_t d = x & 0xFFFu;
          if (d > dt) return 1;
          return d == dt && (gstar == 0xFFFFFFFFu || gid_of((int)(x >> 12), p.rank) <= gstar) ? 2 : 0;
        };
        uint32_t cg = 0, ce = 0;
        for (int i = i0; i < i1; ++i) {
          const int s = pick(cand[i]);
          cg += s == 1;
          ce += s == 2;
        }
        uint32_t tot;
        const uint32_t pre = v2_scan<T>(cg | (ce << 16), SPWS, &tot);
        int gp = (int)(pre & 0xFFFFu), epi = (int)(tot & 0xFFFFu) + (int)(pre >> 16);
        for (int i = i0; i < i1; ++i) {
          const uint32_t x = cand[i];
          const int s = pick(x);
          if (s == 1) {
            if (gp < K) orow[gp] = slot((int)(x >> 12));
            ++gp;
          } else if (s == 2) {
            if (epi < K) orow[epi] = slot((int)(x >> 12));
            ++epi;
          }
        }
        cnt = (int)(tot & 0xFFFFu) + (int)(tot >> 16);
        SPDBG(7);
      }
#pragma unroll 1
      for (int sweep = lsel ? 2 : nch > 1 ? 0 : 1; sweep < 2; ++sweep) {
        int grun = 0, erun = 0;
#pragma unroll 1
        for (int c = 0; c < nch; ++c) {
          const int g0 = c * CG, g1 = ::min(ng, g0 + CG);
#pragma unroll 1
          for (int gb = g0; gb < g1; gb += T) {  // CTA-uniform trip count: quads OR their words
            const int g = gb + tid;
            uint32_t mg = 0, me = 0;
            if (g < g1) {
              const uint4 r = grp(g);
              mg = sp_mask(r, t1);
              me = sp_mask(r, t0) & ~mg;
              if (K31_RARE(me && gstar != 0xFFFFFFFFu)) {
                uint32_t m = me;
#pragma unroll 1
                while (m) {
                  const int j = __ffs(m) - 1;
                  m &= m - 1;
                  if (gid_of(8 * g + j, p.rank) > gstar) me &= ~(1u << j);
                }
              }
            }
            mg <<= 8 * (tid & 3);
            me <<= 8 * (tid & 3);
            mg |= __shfl_xor_sync(0xffffffffu, mg, 1);
            me |= __shfl_xor_sync(0xffffffffu, me, 1);
            mg |= __shfl_xor_sync(0xffffffffu, mg, 2);
            me |= __shfl_xor_sync(0xffffffffu, me, 2);
            if ((tid & 3) == 0 && g < g1 + 3) {
              bmg[(g - g0) >> 2] = mg;
              bme[(g - g0) >> 2] = me;
            }
          }
          __syncthreads();
          SPDBG(7);
          const int nw = (g1 - g0 + 3) >> 2, wpt = (nw + T - 1) / T;
          const int w0 = ::min(nw, tid * wpt), w1 = ::min(nw, w0 + wpt);
          uint32_t cg = 0, ce = 0;
          for (int w = w0; w < w1; ++w) { cg += __popc(bmg[w]); ce += __popc(bme[w]); }
          uint32_t tot;
          const uint32_t pre = v2_scan<T>(cg | (ce << 16), SPWS, &tot);
          if (sweep == 1) {
            const int gt_all = nch > 1 ? gall : (int)(tot & 0xFFFFu);
            int gp = grun + (int)(pre & 0xFFFFu), epi = gt_all + erun + (int)(pre >> 16);
            for (int w = w0; w < w1; ++w) {
              uint32_t m = bmg[w];
              while (m) {
                const int b = __ffs(m) - 1;
                m &= m - 1;
                if (gp < K) orow[gp] = slot(8 * g0 + 32 * w + b);
                ++gp;
              }
              m = bme[w];
              while (m) {
                const int b = __ffs(m) - 1;
                m &= m - 1;
                if (epi < K) orow[epi] = slot(8 * g0 + 32 * w + b);
                ++epi;
              }
            }
          }
          grun += (int)(tot & 0xFFFFu);
          erun += (int)(tot >> 16);
          __syncthreads();
        }
        gall = grun;
        cnt = grun + erun;
      }
      for (int i = cnt + tid; i < K; i += T) orow[i] = -1;
      if (tid == 0) {
        if (cnt > K) spin_fail(p.err, 53);
        p.out_count[row] = ::min(cnt, K);
        p.epoch[row] = ep + 1 == 510 ? 0 : ep + 1;
      }
    }
    SPDBG(8);
    __syncthreads();
    zero_row_smem();
    __syncthreads();
  }
  if (p.stats != nullptr && tid == 0) {  // DIAG: the last CTA copies the counters to pinned host memory
    __threadfence();
    if (atomicAdd(&p.stats[22], 1) == (int)gridDim.x - 1) {
      __threadfence();
      int* h = reinterpret_cast<int*>(*reinterpret_cast<volatile unsigned long long*>(p.stats + 20));
      if (h != nullptr)
        for (int i = 0; i < 20; ++i) h[i] = *reinterpret_cast<volatile int*>(p.stats + i);
      p.stats[22] = 0;
    }
  }
#undef SPDBG
#undef SPSTAT
#undef SPWS
}

#ifdef SP_SKEL
// Sync-only skeleton of the topk_sp hot path (SOL protocol calibration, built only with -DSP_SKEL, launched in
// place of topk_sp_kernel): the same landing, CTA barriers (19 per row), scans, message bytes and round as the R4
// one-round path, without compares, atomics or selection. Rows are written as -1 (count 0). Same stamps.
template <int T>
__global__ void __launch_bounds__(T, T >= 1024 ? 1 : 2) topk_skel_kernel(V2Params p) {
  using C = SpCfg<T>;
  constexpr int CG = C::kCapG, FB = C::kFB;
  __shared__ __align__(128) uint4 srow[CG];
  __shared__ __align__(8) uint64_t s_bar;
  __shared__ __align__(16) uint32_t fine[kSpFine + 64];
  __shared__ __align__(16) uint32_t coarse[256];
  __shared__ __align__(16) uint32_t rx[kV2MaxW * kSpMsg];
  __shared__ __align__(16) uint32_t msg[kSpMsg];
  __shared__ __align__(16) uint32_t suf[T];
  __shared__ int btab[kSpBt];
  __shared__ uint32_t warp_sums[64];
  __shared__ int s_i[16];
  __shared__ uint32_t s_u[16];
  __shared__ long long s_dbg[32];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  uint32_t ws_par = 0;
#define SKWS (warp_sums + 32 * (ws_par ^= 1u))
  for (int i = tid; i < kSpFine + 64; i += T) fine[i] = 0;
  for (int i = tid; i < 256; i += T) coarse[i] = 0;
  if (tid < 16) { s_u[tid] = 0; s_i[tid] = 0; }
  if (tid == 0) {
    mbar_init(&s_bar, kV2Cp);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  uint32_t bar_phase = 0;
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const int W = p.W, K = p.K;
  const int bs_sh = (p.block_size > 0 && (p.block_size & (p.block_size - 1)) == 0) ? __ffs(p.block_size) - 1 : -1;
#define SKDBG(i)                                                                     \
  if (p.dbg && tid == 0) {                                                           \
    if ((i) == 0) {                                                                  \
      for (int z = 1; z < 30; ++z) s_dbg[z] = 0;                                     \
      p.dbg[(int64_t)row * kV2Dbg] = now_ns();                                       \
    }                                                                                \
    s_dbg[i] = (long long)clock64();                                                 \
    if ((i) == 8) {                                                                  \
      p.dbg[(int64_t)row * kV2Dbg + 8] = now_ns();                                   \
      for (int z = 0; z < 30; ++z) p.dbg[(int64_t)row * kV2Dbg + 32 + z] = s_dbg[z]; \
    }                                                                                \
  }
  for (int row = blockIdx.x; row < p.T; row += gridDim.x) {
    SKDBG(0);
    const uint16_t* lrow = p.logits + (int64_t)row * p.lstride;
    const int L = p.local_len[row];
    const int ng = (L + 7) >> 3, nsm = ::max(1, ::min(ng, CG));
    if ((tid & 31) == 0 && (tid >> 5) < kV2Cp) {
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      const int c = tid >> 5, a = nsm * c / kV2Cp, b = nsm * (c + 1) / kV2Cp;
      if (b > a) bulk_g2s(srow + a, reinterpret_cast<const uint4*>(lrow) + a, (uint32_t)(b - a) * 16u, &s_bar);
      else asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(&s_bar)) : "memory");
    }
    int* orow = p.out + (int64_t)row * p.ostride;
    const int req = p.req_of_row != nullptr ? p.req_of_row[row] : row / p.next_n;
    const int* btrow = p.block_table != nullptr ? p.block_table + (int64_t)req * p.bt_stride : nullptr;
    const int ep = p.epoch[row];
    const uint32_t tag = (uint32_t)(ep % 255);
    const int par = ep & 1;
    mbar_wait(&s_bar, bar_phase);
    bar_phase ^= 1u;
    SKDBG(9);
    // 1. sample: max barrier, histogram barrier, warp-0 theta barrier
    uint32_t x = srow[(tid * 8) % nsm].x;
    if (lane == 0) s_u[0] = x;
    __syncthreads();
    x += s_u[0];
    __syncthreads();
    SKDBG(10);
    if (warp == 0) {
      uint32_t c8[8], s = 0;
#pragma unroll
      for (int j = 0; j < 8; ++j) { c8[j] = coarse[255 - 8 * lane - j]; s += c8[j]; }
      uint32_t y = s;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const uint32_t z = __shfl_up_sync(0xffffffffu, y, o);
        if (lane >= o) y += z;
      }
      if (lane == 0) s_i[0] = (int)(y + (x & 1u));
    }
    __syncthreads();
    SKDBG(1);
    // 2. mask pass scan, compaction barrier, histogram barrier
    uint32_t tot;
    const uint32_t c0 = v2_scan<T>((x >> 3) & 1u, SKWS, &tot);
    SKDBG(11);
    __syncthreads();
    SKDBG(12);
    __syncthreads();
    SKDBG(2);
    // 3. e: scan over the fine bins, search barrier; window / band barrier; band barrier
    {
      uint32_t v = 0;
#pragma unroll
      for (int j = 0; j < FB; ++j) v += fine[kSpFine - 1 - FB * tid - j];
      const uint32_t pre = v2_scan<T>(v, SKWS, &tot);
      suf[tid] = pre + c0;
    }
    __syncthreads();
    SKDBG(13);
    if (tid < kSpWin / 2) msg[kSpHdr + tid] = tagw(tag, fine[2 * tid] + suf[tid]);
    __syncthreads();
    __syncthreads();
    SKDBG(15);
    if (tid < kSpBand) {
      msg[kSpHdr + kSpWin / 2 + tid] = tagw(tag, 0u);
    } else if (tid < kSpBand + kSpHdr) {
      msg[tid - kSpBand] = tagw(tag, tid == kSpBand ? 0x10000u : 0u);
    }
    __syncthreads();
    SKDBG(14);
    // 4. round 1 (same message bytes, multicast, poll)
    const int64_t r1row = p.off_r1 + ((int64_t)(par * p.cap + row) * W) * kV2Msg;
    for (int j = tid * 4; j < kSpMsg; j += T * 4) {
      const uint4 v = make_uint4(msg[j], msg[j + 1], msg[j + 2], msg[j + 3]);
      const int64_t off = r1row + (int64_t)p.rank * kV2Msg + j;
      if (p.mc) mc_st_v4(p.mc + off * 4, v);
      else
        for (int d = 0; d < W; ++d) st_v4(reinterpret_cast<uint32_t*>(p.peers[d]) + off, v);
    }
    SKDBG(3);
    const int nblk = bs_sh >= 0 ? (L + p.block_size - 1) >> bs_sh : kSpBt + 1;
    if (btrow != nullptr && nblk <= kSpBt)
      for (int i = tid; i < nblk; i += T) btab[i] = btrow[i];
    for (int i = tid; i < W * (kSpMsg / 4); i += T) {
      const int r = i / (kSpMsg / 4), j = (i % (kSpMsg / 4)) * 4;
      const uint32_t* src = p.box + r1row + (int64_t)r * kV2Msg + j;
      const long long t0 = now_ns();
      uint4 v = ld_sys_v4(src);
      while (!tagged(v, tag)) {
        if (spin_late(p.spin_ns, t0)) { spin_fail(p.err, 51); break; }
        __nanosleep(128);
        v = ld_sys_v4(src);
      }
      *reinterpret_cast<uint4*>(rx + r * kSpMsg + j) = v;
    }
    __syncthreads();
    SKDBG(4);
    // 5. resolve: suffix barrier, tau count barrier, tie collection barrier, tie rank barrier
    if (warp < W) {
      uint32_t s = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) s += rx[warp * kSpMsg + kSpHdr + 4 * lane + j] & 0xFFFu;
      uint32_t y = s;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const uint32_t z = __shfl_down_sync(0xffffffffu, y, o);
        if (lane + o < 32) y += z;
      }
      fine[warp * kSpRs + lane] = y;
    }
    __syncthreads();
    SKDBG(16);
    const int n = __syncthreads_count(fine[(tid & 3) * kSpRs + (tid & 31)] > (uint32_t)K);
    SKDBG(17);
    if (tid < 16) s_u[tid] = (uint32_t)n;
    __syncthreads();
    SKDBG(18);
    __syncthreads();
    SKDBG(5);
    // 6. selection scan, writes (-1 rows), count, epoch
    const uint32_t pre = v2_scan<T>(s_u[tid & 15] & 1u, SKWS, &tot);
    SKDBG(7);
    for (int i = tid; i < K; i += T) orow[i] = (int)(pre & 0u) - 1;
    if (tid == 0) {
      p.out_count[row] = 0;
      p.epoch[row] = ep + 1 == 510 ? 0 : ep + 1;
    }
    SKDBG(8);
    __syncthreads();
    for (int i = tid; i < kSpFine + 64; i += T) fine[i] = 0;
    if (tid < 16) { s_u[tid] = 0; s_i[tid] = 0; }
    __syncthreads();
  }
#undef SKDBG
#undef SKWS
}
#endif  // SP_SKEL

}  // namespace k31dcpx
