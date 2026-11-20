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
// One CTA per row; the row stays in registers (warp w holds the 8-key groups
// [w * 32 * NRR, +32 * NRR), lane l the groups 32 i + l, plus one SMEM group per thread):
//  1. CTA max -> 1/NR sample histogram of the keys within 4096 of it -> theta (sample's
//     (q + 256)-th key, q = ceil(K * L / G)) and an estimate of the q-th key;
//  2. one pass: keys >= theta -> fine histogram (2048 keys), valid keys, band staging;
//  3. e = the q-th key; message = header + exact counts of the 256 keys around e + the
//     positions of every key in [e - d, e + d] (<= kSpBand, position order);
//  4. one round (multicast tagged words): tau, need and the tie cut by global id;
//  5. selection from registers into position bitmaps, one scan, the write.
// The only other path (decided alike on every rank from the exchanged words): radix rounds
// over the 40-bit composite (key << 24 | 0xFFFFFF - gid), 8 bits per round, from the key
// digits, or from the id digits when round 1 fixed tau. No second kernel.

namespace k31dcpx {

constexpr int kSpHdr = 8;
constexpr int kSpWin = 256;
constexpr int kSpBand = 64;
constexpr int kSpMsg = kSpHdr + kSpWin / 2 + kSpBand;  // 200 words
static_assert(kSpMsg <= kV2Msg && kSpMsg % 4 == 0, "round-1 slot");
constexpr int kSpFine = 2048;
constexpr int kSpBt = 1024;   // block-table entries cached per row
constexpr int kSpRs = 257;    // per-rank window suffix stride

template <int T>
struct SpCfg {
  static constexpr int kWarps = T / 32;
  static constexpr int kNRR = T >= 1024 ? 4 : 8;  // register groups per thread
  static constexpr int kNR = kNRR + 1;            // + one SMEM group
  static constexpr int kBmW = kNR * T / 4;        // bitmap words (slot 8g + j)
  static constexpr int kFB = kSpFine / T;
  static_assert(kFB * T == kSpFine && kV2MaxW * kSpRs <= kSpFine + 64, "SMEM aliasing");
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
  constexpr int NRR = C::kNRR, NR = C::kNR, FB = C::kFB;
  __shared__ __align__(16) uint4 sgrp[T];
  __shared__ __align__(16) uint32_t bm[2 * C::kBmW];
  __shared__ __align__(16) uint32_t fine[kSpFine + 64];  // later: window suffixes / radix rx
  __shared__ __align__(16) uint32_t coarse[256];         // later: radix histogram
  __shared__ __align__(16) uint32_t rx[kV2MaxW * kSpMsg];
  __shared__ __align__(16) uint32_t msg[kSpMsg];
  __shared__ __align__(16) uint2 bnd[kSpBand];
  __shared__ __align__(16) uint32_t suf[T];
  __shared__ __align__(16) uint32_t tg[kV2MaxW * kSpBand];
  __shared__ int btab[kSpBt];
  __shared__ uint32_t warp_sums[64];
  __shared__ int s_i[16];
  __shared__ uint32_t s_u[16];
  __shared__ long long s_dbg[32];
  uint32_t* const bmg = bm;
  uint32_t* const bme = bm + C::kBmW;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  uint32_t ws_par = 0;
#define SPWS (warp_sums + 32 * (ws_par ^= 1u))
  // s_u: 0 max, 1 nc, 2 beyond, 3 valid, 4 nst, 5 nb, 6 sat, 7 gstar, 8 A, 9 bcnt, 10 rem
  // s_i: 0 theta, 1 e_est, 2 e, 3 lo, 4 bl, 5 bh, 6 hifail, 7 tau, 8 lev0, 9 dstar, 10 hd
  auto zero_row_smem = [&]() {
    for (int i = tid; i < 2 * C::kBmW; i += T) bm[i] = 0;
    for (int i = tid; i < kSpFine + 64; i += T) fine[i] = 0;
    for (int i = tid; i < 256; i += T) coarse[i] = 0;
    if (tid < 16) { s_u[tid] = 0; s_i[tid] = 0; }
  };
  zero_row_smem();
  __syncthreads();
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
    // checks); positions >= L read as NaN
    auto ld_group = [&](int g) -> uint4 {
      uint4 r = __ldg(reinterpret_cast<const uint4*>(lrow) + g);
      const int nv = L - 8 * g;
      if (K31_RARE(nv < 8)) {
        r.x |= (nv > 0 ? 0u : 0xFFFFu) | (nv > 1 ? 0u : 0xFFFF0000u);
        r.y |= (nv > 2 ? 0u : 0xFFFFu) | (nv > 3 ? 0u : 0xFFFF0000u);
        r.z |= (nv > 4 ? 0u : 0xFFFFu) | (nv > 5 ? 0u : 0xFFFF0000u);
        r.w |= (nv > 6 ? 0u : 0xFFFFu) | (nv > 7 ? 0u : 0xFFFF0000u);
      }
      return r;
    };
    const int gw0 = warp * 32 * NRR;
    const int gsm = NRR * T + tid;  // this thread's SMEM group
    uint4 rg[NRR];
#pragma unroll
    for (int i = 0; i < NRR; ++i) {
      const int g = gw0 + 32 * i + lane;
      rg[i] = g < ng ? ld_group(g) : make_uint4(~0u, ~0u, ~0u, ~0u);
    }
    const uint4 rs0 = gsm < ng ? ld_group(gsm) : make_uint4(~0u, ~0u, ~0u, ~0u);
    int* orow = p.out + (int64_t)row * p.ostride;
    const int req = p.req_of_row != nullptr ? p.req_of_row[row] : row / p.next_n;
    const int* btrow = p.block_table != nullptr ? p.block_table + (int64_t)req * p.bt_stride : nullptr;
    if (K31_RARE(Gl <= K)) {  // the whole context (every rank sees the same global length)
      for (int i = tid; i < L; i += T)
        orow[i] = btrow == nullptr ? i : btrow[i / p.block_size] * p.block_stride + i % p.block_size;
      for (int i = L + tid; i < K; i += T) orow[i] = -1;
      if (tid == 0) p.out_count[row] = L;
      __syncthreads();
      continue;
    }
    if (tid == 0 && (ng > NR * T || L >= (1 << 18) || Gl >= (1 << 24))) spin_fail(p.err, 50);
    const int ep = p.epoch[row];
    const uint32_t tag = (uint32_t)(ep % 255);
    const int par = ep & 1;
    SPSTAT(0);
    const unsigned long long kl = (unsigned long long)K * (unsigned)L + (unsigned)(Gl - 1);
    const uint32_t q = max(1u, (uint32_t)(kl / (unsigned)Gl));
    // ---- 1. CTA max key bound, then the sample histogram near it
    {
      sgrp[tid] = rs0;
      uint32_t m = max_key_bound(rs0);
#pragma unroll
      for (int i = 0; i < NRR; ++i) m = ::max(m, max_key_bound(rg[i]));
      m = __reduce_max_sync(0xffffffffu, m);
      if (lane == 0) atomicMax(&s_u[0], m);
    }
    __syncthreads();
    SPDBG(9);
    const uint32_t mb = s_u[0];
    // sample keys within 8 octaves of the max, 4 keys per bin
    const uint32_t base = mb > 1024u ? mb - 1024u : 1u;
    {
      const int is = (lane + warp) % NR;
      uint4 r = sgrp[tid];
#pragma unroll
      for (int i = 0; i < NRR; ++i)
        if (i == is) r = rg[i];
      const int g = is < NRR ? gw0 + 32 * is + lane : gsm;
      if (g < ng) {
        const Keys8 k8(r);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const uint32_t k = k8.key(j);
          if (k >= base) atomicAdd(&coarse[(k - base) >> 2], 1u);
        }
      }
    }
    __syncthreads();
    SPDBG(10);
    if (warp == 0) {  // theta: the sample's (q + 256)-th key's bin; e_est: its q-th key's bin
      const uint32_t want = (q + 256u + NR - 1) / NR, want_e = max(1u, q / NR);
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
          const uint32_t b = 255 - 8 * lane - j;
          const bool a = run < want, ae = run < want_e;
          run += c[j];
          if (ft && a && run >= want) s_i[0] = (int)(base + 4 * b);
          if (fe && ae && run >= want_e) s_i[1] = (int)(base + 4 * b + 2);
        }
      }
      const uint32_t bt = __ballot_sync(0xffffffffu, ft), be = __ballot_sync(0xffffffffu, fe);
      if (lane == 0) {
        if (!bt) s_i[0] = (int)base;
        if (!be) s_i[1] = (int)base;
        s_i[2] = -1;
      }
    }
    __syncthreads();
    SPDBG(1);
    const int theta = s_i[0], e_est = s_i[1];
    // ---- 2. keys >= theta: fine histogram, valid keys
    {
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
#pragma unroll
      for (int i = 0; i < NRR; ++i) visit(gw0 + 32 * i + lane, rg[i]);
      visit(gsm, sgrp[tid]);
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
    const int lo = e < 0 ? 0 : ::min(::max(theta, e - kSpWin / 2), theta + kSpFine - kSpWin);
    if (tid < kSpWin / 2) {
      const uint32_t c0 = e < 0 ? 0u : fine[lo - theta + 2 * tid], c1 = e < 0 ? 0u : fine[lo - theta + 2 * tid + 1];
      if (c0 >= kCountMax || c1 >= kCountMax) s_u[6] = 1;
      msg[kSpHdr + tid] = tagw(tag, (::min(c0, kCountMax) << 12) | ::min(c1, kCountMax));
    } else if (warp == C::kWarps - 1) {
      // band [e - d, e + d]: the largest d whose keys fit kSpBand
      const int d = lane;
      bool ok = e >= 0 && e - d >= theta && e + d + 1 <= theta + kSpFine;
      uint32_t c = 0;
      if (ok) {
        c = cloc(e - d) - cloc(e + d + 1);
        ok = c <= (uint32_t)kSpBand;
      }
      const uint32_t bal = __ballot_sync(0xffffffffu, ok);
      const int ds = ~bal ? __ffs(~bal) - 2 : 31;  // the longest prefix of ok
      if (d == ::max(ds, 0)) {
        s_i[9] = ds;
        s_i[4] = e - ds;
        s_i[5] = e + ds;
        s_u[9] = c;
        s_u[8] = e < 0 ? 0u : cloc(lo + kSpWin);
      }
    }
    __syncthreads();
    const int bl = s_i[4], bh = s_i[5];
    if (s_i[9] >= 0) {  // the band's keys (<= kSpBand by the counts) with their positions
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
#pragma unroll
      for (int i = 0; i < NRR; ++i) visit(gw0 + 32 * i + lane, rg[i]);
      visit(gsm, sgrp[tid]);
    }
    __syncthreads();
    SPDBG(15);
    const int nb = ::min((int)s_u[5], kSpBand);
    const bool bvalid = s_i[9] >= 0 && (uint32_t)nb == s_u[9] && s_u[5] <= (uint32_t)kSpBand;
    if (tid < kSpBand) {
      if (tid < nb && bvalid) {
        const uint2 me = bnd[tid];
        int rk = 0;
        for (int s = 0; s < nb; ++s) rk += bnd[s].x < me.x;
        msg[kSpHdr + kSpWin / 2 + rk] = tagw(tag, me.x | ((me.y - (uint32_t)bl) << 18));
      }
      if (tid >= (bvalid ? nb : 0)) msg[kSpHdr + kSpWin / 2 + tid] = tagw(tag, 0u);
    } else if (tid == kSpBand) {
      const uint32_t flags = (e >= 0 && s_u[6] == 0 ? 1u : 0u) | (bvalid ? 2u : 0u);
      msg[0] = tagw(tag, (uint32_t)lo | (flags << 16));
      msg[1] = tagw(tag, ::min(s_u[8], 0xFFFFFFu));
      msg[2] = tagw(tag, ::min(s_u[3], 0xFFFFFFu));
      msg[3] = tagw(tag, (uint32_t)(bl & 0xFFFF) | ((uint32_t)(bh - bl) << 16));
      msg[4] = tagw(tag, bvalid ? (uint32_t)nb : 0u);
      msg[5] = tagw(tag, 0u);
      msg[6] = tagw(tag, 0u);
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
          if (!bok) {
            lev0 = 2;
            rem = need;
          } else {
            for (int i = tid; i < W * kSpBand; i += T) {  // every rank's ties at tau
              const int r = i / kSpBand, j = i % kSpBand;
              if (j < (int)hdr(r, 4)) {
                const uint32_t w = hdr(r, kSpHdr + kSpWin / 2 + j);
                if ((int)(hdr(r, 3) & 0xFFFFu) + (int)(w >> 18) == tau)
                  tg[atomicAdd(&s_u[11], 1u)] = gid_of((int)(w & 0x3FFFFu), r);
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
#pragma unroll
          for (int i = 0; i < NRR; ++i) visit(gw0 + 32 * i + lane, rg[i]);
          visit(gsm, sgrp[tid]);
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
    // ---- 6. selection -> position bitmaps -> write
    {
      const SpTh t1 = sp_th(tau + 1), t0 = sp_th(tau);
      auto visit = [&](int g, const uint4& r) {
        if (g >= ng) return;
        const uint32_t mg = sp_mask(r, t1);
        uint32_t me = sp_mask(r, t0) & ~mg;
        if (K31_RARE(me && gstar != 0xFFFFFFFFu)) {
          uint32_t m = me;
#pragma unroll 1
          while (m) {
            const int j = __ffs(m) - 1;
            m &= m - 1;
            if (gid_of(8 * g + j, p.rank) > gstar) me &= ~(1u << j);
          }
        }
        if (mg) atomicOr(&bmg[g >> 2], mg << (8 * (g & 3)));
        if (me) atomicOr(&bme[g >> 2], me << (8 * (g & 3)));
      };
#pragma unroll
      for (int i = 0; i < NRR; ++i) visit(gw0 + 32 * i + lane, rg[i]);
      visit(gsm, sgrp[tid]);
    }
    __syncthreads();
    SPDBG(7);
    {
      const int nw = (ng + 3) >> 2, wpt = (nw + T - 1) / T;
      const int w0 = ::min(nw, tid * wpt), w1 = ::min(nw, w0 + wpt);
      uint32_t cg = 0, ce = 0;
      for (int w = w0; w < w1; ++w) { cg += __popc(bmg[w]); ce += __popc(bme[w]); }
      uint32_t tot;
      const uint32_t pre = v2_scan<T>(cg | (ce << 16), SPWS, &tot);
      const int gtot = (int)(tot & 0xFFFFu), cnt = gtot + (int)(tot >> 16);
      auto slot = [&](int pos) -> int {
        if (btrow == nullptr) return pos;
        if (bt_sm) return btab[pos >> bs_sh] * p.block_stride + (pos & (p.block_size - 1));
        return btrow[pos / p.block_size] * p.block_stride + pos % p.block_size;
      };
      int gp = (int)(pre & 0xFFFFu), epi = gtot + (int)(pre >> 16);
      for (int w = w0; w < w1; ++w) {
        uint32_t m = bmg[w];
        while (m) {
          const int b = __ffs(m) - 1;
          m &= m - 1;
          if (gp < K) orow[gp] = slot(32 * w + b);
          ++gp;
        }
        m = bme[w];
        while (m) {
          const int b = __ffs(m) - 1;
          m &= m - 1;
          if (epi < K) orow[epi] = slot(32 * w + b);
          ++epi;
        }
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
#undef SPDBG
#undef SPSTAT
#undef SPWS
}

}  // namespace k31dcpx
