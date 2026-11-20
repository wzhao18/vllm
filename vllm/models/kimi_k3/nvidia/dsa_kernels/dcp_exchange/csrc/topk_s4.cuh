// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Kimi K3.1 DSA decode under DCP: exact global top-K with the row held in registers
// (included by dcp_exchange.cu after topk_sp.cuh; same parameters, mailboxes, epochs and
// output as topk_v2 / topk_sp / topk_merge: every key above the K-th key tau, then the
// `need` keys equal to tau with the smallest global ids; this rank's keys above tau in
// position order, then its selected ties in position order (KV slots), -1 padding, count).
//
// One CTA per row. Thread (warp w, lane l) loads the 8-key groups g = 32 NR w + 32 i + l
// (i < NR, coalesced) once; every pass after that runs on registers (rolled loops that
// rotate the register row, so the hot code stays small):
//  1. sample: 4 keys of group l % NR per lane -> whole-range histograms (key >> 8,
//     key >> 4) -> theta = the 16-key bin of the sample's share of q + 768 keys
//     (q = ceil(K L / G)), found by every warp alike;
//  2. keys >= theta -> exact histogram of key - theta (2048 keys; the top bin clamps);
//  3. warp 0: e = the q-th key; message = header + exact counts of the 256 keys around e
//     (136 words); one round (multicast tagged words, warps 1..W poll); tau, need;
//  4. keys above tau, then the ties -> position order (one CTA scan) -> KV slots.
// CTA barriers per row on that path: histograms zeroed, sample, pass, resolved, scan.
// Rows the round does not resolve (tau outside the windows' common range, windows that do
// not overlap, saturated counts, ties cut by global id) take exact radix rounds over the
// 40-bit composite (key << 24 | 0xFFFFFF - gid), 8 bits per round, on the row re-read
// from global memory (two key rounds, then id rounds only while the ties at tau are cut).

namespace k31dcpx {

constexpr int kS4Hdr = 8;
constexpr int kS4Band = 128;  // band entries per rank: (key - lo) << 16 | position
constexpr int kS4Msg = kS4Hdr + 128 + kS4Band;  // header, 256 window counts (12-bit pairs), band
constexpr int kS4Cand = 4096;  // candidate list (key << 16 | position)
constexpr int kS4Fine = 2048;         // exact keys above theta (the last bin clamps)
constexpr int kS4WantX = 768;         // theta admits ~q + kS4WantX keys
static_assert(kS4Msg <= kV2Msg && kS4Msg % 4 == 0, "round-1 slot");
static_assert(kV2RadR >= 5, "two key rounds + three id rounds");

template <int T>
struct S4Cfg {
  static constexpr int kWarps = T / 32;
  static constexpr int kNR = T >= 1024 ? 5 : 9;  // 8-key groups per thread
  static constexpr int kCapG = kNR * T;          // groups per row
  static constexpr int kPk = (kNR + 2) / 3;      // packed scan words (3 x 10-bit fields)
};

__device__ __forceinline__ uint32_t s4_wscan(uint32_t x, int lane) {  // inclusive, lane 0 first
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const uint32_t y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x += y;
  }
  return x;
}
__device__ __forceinline__ uint32_t s4_fld(uint32_t w, int f) { return (w >> (10 * f)) & 0x3FFu; }
// positions >= nv of a group read as NaN (key 0)
__device__ __forceinline__ uint4 s4_tail(uint4 r, int nv) {
  auto m = [&](int k) -> uint32_t { return (nv <= k ? 0xFFFFu : 0u) | (nv <= k + 1 ? 0xFFFF0000u : 0u); };
  return make_uint4(r.x | m(0), r.y | m(2), r.z | m(4), r.w | m(6));
}
__device__ __noinline__ uint32_t s4_gid_slow(int pos, int rank, int W, int IL) {
  return (uint32_t)(((pos / IL) * W + rank) * IL + pos % IL);
}
__device__ __forceinline__ uint32_t s4_gid(int pos, int rank, int W, int IL, int il_sh, int gid_sh) {
  if (gid_sh >= 0) return (uint32_t)((((pos >> il_sh) << gid_sh) | (rank << il_sh)) | (pos & (IL - 1)));
  return s4_gid_slow(pos, rank, W, IL);
}
// ties of group g (mask) kept iff gid <= gcut
__device__ __noinline__ uint32_t s4_tiecut(uint32_t m, int g, uint32_t gcut, int rank, int W, int IL, int il_sh,
                                           int gid_sh) {
  uint32_t keep = 0;
  while (m) {
    const int j = __ffs(m) - 1;
    m &= m - 1;
    if (s4_gid(8 * g + j, rank, W, IL, il_sh, gid_sh) <= gcut) keep |= 1u << j;
  }
  return keep;
}
__device__ __noinline__ int s4_slot_slow(const int* btrow, int pos, int bs, int bstride) {
  return btrow[pos / bs] * bstride + pos % bs;
}
__device__ __noinline__ void s4_put_peers(const uint64_t* peers, int W, int64_t off, uint4 v) {
  for (int d = 0; d < W; ++d) st_v4(reinterpret_cast<uint32_t*>(peers[d]) + off, v);
}
__device__ __forceinline__ void s4_put(const V2Params& p, int64_t off, uint4 v) {
  if (p.mc) mc_st_v4(p.mc + off * 4, v);
  else s4_put_peers(p.peers, p.W, off, v);
}
// waits until v (re-read from src) carries tag
__device__ __forceinline__ uint4 s4_poll(const uint32_t* src, uint4 v, uint32_t tag, long long spin_ns, int* err,
                                         long long t0) {
  while (!tagged(v, tag)) {
    if (spin_late(spin_ns, t0)) { spin_fail(err, 51); break; }
    __nanosleep(32);
    v = ld_sys_v4(src);
  }
  return v;
}

// Exact radix rounds for rows the one round does not resolve (rare): over the composite
// (key << 24 | 0xFFFFFF - gid), 8 bits per round, the row re-read from global memory.
// mode 1: tau known, the ties cut by id (need0 of them); mode 2: from scratch. Every rank
// takes the same rounds (decided alike from the exchanged words). Returns tau (0x10000:
// nothing selected) and the id cut of the ties (*gcut).
template <int T>
__device__ __noinline__ int s4_radix(V2Params p, int row, int L, uint32_t tag, int par, int mode, int tau0,
                                     uint32_t need0, uint32_t* h1, int* s_i, uint32_t* s_u, uint32_t* gcut) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int W = p.W, IL = p.interleave;
  const int il_sh = (IL & (IL - 1)) == 0 ? __ffs(IL) - 1 : -1, w_sh = (W & (W - 1)) == 0 ? __ffs(W) - 1 : -1;
  const int gid_sh = (il_sh >= 0 && w_sh >= 0) ? il_sh + w_sh : -1;
  const uint4* lr4 = reinterpret_cast<const uint4*>(p.logits + (int64_t)row * p.lstride);
  const int ng = (L + 7) >> 3;
  unsigned long long pb = mode == 1 ? (unsigned long long)tau0 : 0ull;
  int plen = mode == 1 ? 16 : 0;
  uint32_t nr = mode == 1 ? need0 : (uint32_t)p.K;
  *gcut = 0xFFFFFFFFu;
#pragma unroll 1
  for (int rd = 0;; ++rd) {
    if (tid < 64) reinterpret_cast<uint4*>(h1)[tid] = make_uint4(0u, 0u, 0u, 0u);
    __syncthreads();
    const int tk = plen >= 16 ? (int)(pb >> (plen - 16)) : 0;
    const uint32_t qp = plen > 16 ? (uint32_t)(pb & ((1ull << (plen - 16)) - 1ull)) : 0u;
#pragma unroll 1
    for (int g = tid; g < ng; g += T) {
      uint4 x = __ldg(lr4 + g);
      if (L - 8 * g < 8) x = s4_tail(x, L - 8 * g);
      const uint4 kw = sp_keys(x);
#pragma unroll 1
      for (int j = 0; j < 8; ++j) {
        const uint32_t k = sp_key_at(kw, j);
        if (k == 0u) continue;
        uint32_t d;
        if (plen < 16) {
          if (plen == 8 && (k >> 8) != (uint32_t)pb) continue;
          d = plen == 0 ? (k >> 8) : (k & 0xFFu);
        } else {
          if ((int)k != tk) continue;
          const uint32_t cg = 0xFFFFFFu - s4_gid(8 * g + j, p.rank, W, IL, il_sh, gid_sh);
          if (plen > 16 && (cg >> (40 - plen)) != qp) continue;
          d = (cg >> (32 - plen)) & 0xFFu;
        }
        atomicAdd(&h1[d], 1u);
      }
    }
    __syncthreads();
    if (warp == 0) {
      const int64_t rb = p.off_rad + ((((int64_t)par * p.cap + row) * kV2RadR + rd) * W) * 256;
      for (int j = lane * 4; j < 256; j += 128)
        s4_put(p, rb + (int64_t)p.rank * 256 + j,
               make_uint4(tagw(tag, h1[j]), tagw(tag, h1[j + 1]), tagw(tag, h1[j + 2]), tagw(tag, h1[j + 3])));
      // lane l: digits 255 - 8 l - j (j < 8), summed over the ranks
      uint32_t c[8], s = 0;
#pragma unroll
      for (int j = 0; j < 8; ++j) c[j] = 0;
      const long long t0 = now_ns();
#pragma unroll 1
      for (int rr = 0; rr < W; ++rr) {
        const uint32_t* src = p.box + rb + (int64_t)rr * 256 + 248 - 8 * lane;
        const uint4 a = s4_poll(src, ld_sys_v4(src), tag, p.spin_ns, p.err, t0);
        const uint4 b = s4_poll(src + 4, ld_sys_v4(src + 4), tag, p.spin_ns, p.err, t0);
        c[0] += b.w & 0xFFFFFFu; c[1] += b.z & 0xFFFFFFu; c[2] += b.y & 0xFFFFFFu; c[3] += b.x & 0xFFFFFFu;
        c[4] += a.w & 0xFFFFFFu; c[5] += a.z & 0xFFFFFFu; c[6] += a.y & 0xFFFFFFu; c[7] += a.x & 0xFFFFFFu;
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) s += c[j];
      const uint32_t x = s4_wscan(s, lane);
      const uint32_t tot = __shfl_sync(0xffffffffu, x, 31);
      uint32_t want = nr;
      if (rd == 0 && mode == 2 && tot < want) want = tot;  // fewer valid keys than K: all of them
      int done = 0;
      if (want == 0u) {
        done = 2;  // no valid key: nothing selected
      } else {
        const bool hit = x - s < want && x >= want;
        uint32_t dd = 0, cab = 0, cd = 0;
        if (hit) {
          uint32_t run = x - s;
          bool f = false;
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            if (!f && run + c[j] >= want) { f = true; dd = 255 - 8 * lane - j; cab = run; cd = c[j]; }
            run += c[j];
          }
        }
        const int src = __ffs(__ballot_sync(0xffffffffu, hit)) - 1;
        dd = __shfl_sync(0xffffffffu, dd, src);
        cab = __shfl_sync(0xffffffffu, cab, src);
        cd = __shfl_sync(0xffffffffu, cd, src);
        want -= cab;
        pb = (pb << 8) | dd;
        plen += 8;
        if (plen >= 16 && (cd == want || plen == 40)) done = 1;
      }
      if (lane == 0) {
        s_i[2] = done;
        s_i[3] = plen;
        s_u[4] = want;
        s_u[5] = (uint32_t)(pb >> 32);
        s_u[6] = (uint32_t)pb;
      }
    }
    __syncthreads();
    const int done = s_i[2];
    plen = s_i[3];
    nr = s_u[4];
    pb = ((unsigned long long)s_u[5] << 32) | s_u[6];
    if (done) {
      if (done == 2) return 0x10000;
      if (plen > 16) *gcut = 0xFFFFFFu - ((uint32_t)(pb & ((1ull << (plen - 16)) - 1ull)) << (40 - plen));
      return (int)(pb >> (plen - 16));
    }
  }
}

// Selection of rows longer than the register row (rare): two passes over the row in global
// memory (counts, then chunks of T groups with one CTA scan each) -> KV slots, -1 padding,
// count. ws: 64 words of scratch.
template <int T>
__device__ __noinline__ void s4_select_global(V2Params p, int row, int L, int tau, uint32_t gcut, int* orow,
                                              const int* btrow, const int* btab, bool bt_sm, uint32_t* ws) {
  const int tid = threadIdx.x;
  const int W = p.W, IL = p.interleave, K = p.K;
  const int il_sh = (IL & (IL - 1)) == 0 ? __ffs(IL) - 1 : -1, w_sh = (W & (W - 1)) == 0 ? __ffs(W) - 1 : -1;
  const int gid_sh = (il_sh >= 0 && w_sh >= 0) ? il_sh + w_sh : -1;
  const int bs_sh = (p.block_size & (p.block_size - 1)) == 0 ? __ffs(p.block_size) - 1 : -1;
  const uint4* lr4 = reinterpret_cast<const uint4*>(p.logits + (int64_t)row * p.lstride);
  const int ng = (L + 7) >> 3;
  const bool none = tau > 0xFFFF;
  const SpTh t1 = sp_th(::min(tau + 1, 0xFFFF)), t0 = sp_th(::min(tau, 0xFFFF));
  auto masks = [&](int g, uint32_t& gt, uint32_t& tie) {
    gt = tie = 0u;
    if (g >= ng || none) return;
    uint4 x = __ldg(lr4 + g);
    if (L - 8 * g < 8) x = s4_tail(x, L - 8 * g);
    const uint32_t ge = sp_mask(x, t0);
    gt = tau >= 0xFFFF ? 0u : sp_mask(x, t1);
    tie = ge & ~gt;
    if (gcut != 0xFFFFFFFFu && tie != 0u) tie = s4_tiecut(tie, g, gcut, p.rank, W, IL, il_sh, gid_sh);
  };
  uint32_t par = 0, tot = 0, ngt = 0;
  {
    uint32_t c = 0;
    for (int g = tid; g < ng; g += T) {
      uint32_t gt, tie;
      masks(g, gt, tie);
      c += (uint32_t)__popc(gt) | ((uint32_t)__popc(tie) << 16);
    }
    v2_scan<T>(c, ws + 32 * (par ^= 1u), &tot);
    ngt = tot & 0xFFFFu;
  }
  uint32_t og = 0, oe = ngt;
  for (int c0 = 0; c0 < ng; c0 += T) {
    uint32_t gt, tie;
    masks(c0 + tid, gt, tie);
    const uint32_t v = (uint32_t)__popc(gt) | ((uint32_t)__popc(tie) << 16);
    const uint32_t ex = v2_scan<T>(v, ws + 32 * (par ^= 1u), &tot);
    int o = (int)(og + (ex & 0xFFFFu)), oo = (int)(oe + (ex >> 16));
    const int pos0 = 8 * (c0 + tid);
    for (int pass = 0; pass < 2; ++pass) {
      uint32_t m = pass == 0 ? gt : tie;
      int& oc = pass == 0 ? o : oo;
      while (m) {
        const int j = __ffs(m) - 1;
        m &= m - 1;
        const int pos = pos0 + j;
        int sl = pos;
        if (btrow != nullptr)
          sl = bt_sm ? btab[pos >> bs_sh] * p.block_stride + (pos & (p.block_size - 1))
                     : s4_slot_slow(btrow, pos, p.block_size, p.block_stride);
        if (oc < K) orow[oc] = sl;
        ++oc;
      }
    }
    og += tot & 0xFFFFu;
    oe += tot >> 16;
  }
  const int cnt = (int)oe;
  for (int i = cnt + tid; i < K; i += T) orow[i] = -1;
  if (tid == 0) {
    if (cnt > K) spin_fail(p.err, 53);
    p.out_count[row] = ::min(cnt, K);
  }
}

template <int T>
__global__ void __launch_bounds__(T, T >= 1024 ? 1 : 2) topk_s4_kernel(V2Params p) {
  using C = S4Cfg<T>;
  constexpr int CG = C::kCapG, NW = C::kWarps;
  constexpr int kBmW = CG / 4;  // bitmap words of the SMEM row's positions
  __shared__ __align__(128) uint4 srow[CG];  // the row; after the pass: rx | bitmaps | tie list
  __shared__ __align__(8) uint64_t s_bar;
  __shared__ __align__(16) uint32_t h1[256];       // sample key >> 8; radix digit counts
  __shared__ __align__(16) uint32_t h2c[kS4Cand];  // sample key >> 4, then the candidate list
  __shared__ __align__(16) uint32_t fine[kS4Fine];
  __shared__ __align__(16) uint32_t band[kS4Band];
  __shared__ int btab[kSpBt];
  __shared__ uint32_t s_ws[64];
  __shared__ int s_i[8];
  __shared__ uint32_t s_u[16];
  __shared__ long long s_dbg[32];
  static_assert(4096 <= kS4Cand, "sample bins in the candidate list");
  uint32_t* const rx = reinterpret_cast<uint32_t*>(srow);  // kV2MaxW * kS4Msg
  uint32_t* const bmg = rx + kV2MaxW * kS4Msg;              // keys above tau (bit per position)
  uint32_t* const bme = bmg + kBmW;                          // selected ties
  uint2* const tl = reinterpret_cast<uint2*>(bme + kBmW);    // tie list: (gid, pos | own << 31)
  static_assert((kV2MaxW * kS4Msg + 2 * kBmW) * 4 + kV2MaxW * kS4Band * 8 <= CG * 16, "post-pass aliases");
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (tid == 0) {
    mbar_init(&s_bar, kV2Cp);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  uint32_t bar_phase = 0, ws_par = 0;
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const int W = p.W, K = p.K, IL = p.interleave;
  auto log2_or = [](int x) -> int { return (x > 0 && (x & (x - 1)) == 0) ? __ffs(x) - 1 : -1; };
  const int bs_sh = log2_or(p.block_size), il_sh = log2_or(IL), w_sh = log2_or(W);
  const int gid_sh = (il_sh >= 0 && w_sh >= 0) ? il_sh + w_sh : -1;
#define S4DBG(i)                                                                     \
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
#define S4STAT(i) \
  if (p.stats && tid == 0) atomicAdd(&p.stats[i], 1);

#pragma unroll 1
  for (int row = blockIdx.x; row < p.T; row += gridDim.x) {
    S4DBG(0);
    const uint4* lr4 = reinterpret_cast<const uint4*>(p.logits + (int64_t)row * p.lstride);
    const int L = p.local_len[row];
    const int Gl = p.global_len[row];
    const int ng = (L + 7) >> 3;
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
    // ---- landing: kV2Cp bulk copies; rows longer than the SMEM row land nothing (NaN
    // pass -> the message asks for radix rounds, which read global memory)
    const bool lr = ng > CG;
    const int nsm = lr ? 0 : ng;
    if ((tid & 31) == 0 && (tid >> 5) < kV2Cp) {
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      const int c = tid >> 5, a = nsm * c / kV2Cp, b = nsm * (c + 1) / kV2Cp;
      if (b > a) bulk_g2s(srow + a, lr4 + a, (uint32_t)(b - a) * 16u, &s_bar);
      else asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(&s_bar)) : "memory");
    }
    {
      uint4* a = reinterpret_cast<uint4*>(h2c);
#pragma unroll 1
      for (int i = tid; i < 4096 / 4; i += T) a[i] = make_uint4(0u, 0u, 0u, 0u);
      uint4* b = reinterpret_cast<uint4*>(fine);
#pragma unroll 1
      for (int i = tid; i < kS4Fine / 4; i += T) b[i] = make_uint4(0u, 0u, 0u, 0u);
      if (tid < 64) reinterpret_cast<uint4*>(h1)[tid] = make_uint4(0u, 0u, 0u, 0u);
      if (tid < 16) s_u[tid] = 0;
      if (tid < 8) s_i[tid] = 0;
    }
    if (tid == 0 && Gl >= (1 << 24)) spin_fail(p.err, 50);
    const int ep = p.epoch[row];
    const uint32_t tag = (uint32_t)(ep % 255);
    const int par = ep & 1;
    S4STAT(0);
    const uint32_t q = max(1u, L <= (1 << 20) ? ((uint32_t)K * (uint32_t)L + (uint32_t)Gl - 1u) / (uint32_t)Gl
                                              : (uint32_t)(((unsigned long long)K * L + Gl - 1) / Gl));
    const int nblk = bs_sh >= 0 ? (L + p.block_size - 1) >> bs_sh : kSpBt + 1;
    const bool bt_sm = btrow != nullptr && nblk <= kSpBt;
    __syncthreads();  // histograms zeroed
    mbar_wait(&s_bar, bar_phase);
    bar_phase ^= 1u;
    S4DBG(9);
    // ---- 1. sample: group 16 t + t % 16 per thread; lanes with the same 16-key bin add once
    {
      const int gs = tid * 16 + (tid & 15);
      uint4 x = make_uint4(~0u, ~0u, ~0u, ~0u);
      if (gs < nsm) {
        x = srow[gs];
        if (K31_RARE(L - 8 * gs < 8)) x = s4_tail(x, L - 8 * gs);
      }
      const uint4 kw = sp_keys(x);
      uint32_t ns = 0;
#pragma unroll 1
      for (int j = 0; j < 8; ++j) {
        const uint32_t b = sp_key_at(kw, j) >> 4;  // 0 <=> NaN (the smallest valid key is 0x7F)
        const uint32_t peers = __match_any_sync(0xffffffffu, b);
        if (b != 0u && lane == __ffs(peers) - 1) {
          const uint32_t c = __popc(peers);
          atomicAdd(&h2c[b], c);
          atomicAdd(&h1[b >> 4], c);
        }
        ns += b != 0u;
      }
      ns = __reduce_add_sync(0xffffffffu, ns);
      if (lane == 0 && ns) atomicAdd(&s_u[2], ns);
    }
    __syncthreads();  // sample histograms
    S4DBG(10);
    // theta (every warp alike): the 16-key bin holding the sample's want-th key
    int theta = 1;
    {
      const uint32_t ns = s_u[2];
      const uint32_t want = ((q + (uint32_t)kS4WantX) * ns + (uint32_t)L - 1u) / (uint32_t)max(L, 1);
      uint32_t c[8], s = 0;
      {
        const uint4 a = *reinterpret_cast<const uint4*>(h1 + 248 - 8 * lane);
        const uint4 b = *reinterpret_cast<const uint4*>(h1 + 252 - 8 * lane);
        c[0] = b.w; c[1] = b.z; c[2] = b.y; c[3] = b.x; c[4] = a.w; c[5] = a.z; c[6] = a.y; c[7] = a.x;
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) s += c[j];
      const uint32_t x = s4_wscan(s, lane);
      const bool hit = want > 0u && x - s < want && x >= want;
      int b1 = -1;
      uint32_t rem = 0;
      if (hit) {
        uint32_t run = x - s;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          if (b1 < 0 && run + c[j] >= want) { b1 = 255 - 8 * lane - j; rem = want - run; }
          run += c[j];
        }
      }
      const uint32_t hb = __ballot_sync(0xffffffffu, hit);
      if (hb) {
        const int src = __ffs(hb) - 1;
        b1 = __shfl_sync(0xffffffffu, b1, src);
        rem = __shfl_sync(0xffffffffu, rem, src);
        const uint32_t v = lane < 16 ? h2c[(b1 << 4) + 15 - lane] : 0u;
        const uint32_t y = s4_wscan(v, lane);
        const uint32_t h = __ballot_sync(0xffffffffu, lane < 16 && y - v < rem && y >= rem);
        theta = max(1, ((b1 << 4) + 15 - (__ffs(h) - 1)) << 4);
      }
    }
    __syncthreads();  // theta read: the candidate list reuses the sample bins
    S4DBG(1);
    // ---- 2. keys >= theta -> fine[min(key - theta, kS4Fine - 1)] + the candidate list
    // (key << 16 | position, warp-aggregated appends, any order)
    {
      const SpTh th = sp_th(theta);
#pragma unroll 1
      for (int c0 = 0; c0 < nsm; c0 += T) {
        const int g = c0 + tid;
        uint32_t m = 0;
        uint4 x = make_uint4(~0u, ~0u, ~0u, ~0u);
        if (g < nsm) {
          x = srow[g];
          if (K31_RARE(L - 8 * g < 8)) x = s4_tail(x, L - 8 * g);
          m = sp_mask(x, th);
        }
        const uint32_t cm = __popc(m);
        if (__ballot_sync(0xffffffffu, cm != 0u)) {
          const uint32_t incl = s4_wscan(cm, lane);
          uint32_t base = 0;
          if (lane == 31) base = atomicAdd(&s_u[3], incl);
          uint32_t o = __shfl_sync(0xffffffffu, base, 31) + incl - cm;
          if (m) {
            const uint4 kw = sp_keys(x);
            do {
              const int j = __ffs(m) - 1;
              m &= m - 1;
              const uint32_t k = sp_key_at(kw, j);
              atomicAdd(&fine[::min(k - (uint32_t)theta, (uint32_t)kS4Fine - 1u)], 1u);
              if (o < (uint32_t)kS4Cand) h2c[o] = (k << 16) | (uint32_t)(8 * g + j);
              ++o;
            } while (m);
          }
        }
      }
    }
    __syncthreads();  // fine histogram + candidate list
    S4DBG(2);
    const uint32_t nct = s_u[3];
    // ---- 3. e (the q-th key), window, band (every warp alike), band entries from the list
    int lo = theta, bl = 0, bh = -1;
    uint32_t flags = 0;
    if (nct < q || nct > (uint32_t)kS4Cand) {
      flags = 1;  // too few keys >= theta, or a list overflow: radix rounds
    } else {
      uint32_t s = 0;
      const uint4* f4 = reinterpret_cast<const uint4*>(fine + kS4Fine - 64 * (lane + 1));
#pragma unroll 4
      for (int j = 0; j < 16; ++j) {
        const uint4 v = f4[j];
        s += v.x + v.y + v.z + v.w;
      }
      const uint32_t x = s4_wscan(s, lane);
      const int cl = __ffs(__ballot_sync(0xffffffffu, x - s < q && x >= q)) - 1;
      const uint32_t exc = __shfl_sync(0xffffffffu, x - s, cl);
      const int top = kS4Fine - 1 - 64 * cl;
      const uint32_t a0 = fine[top - 2 * lane], a1 = fine[top - 2 * lane - 1];
      const uint32_t pre = exc + s4_wscan(a0 + a1, lane) - (a0 + a1);
      const bool c0 = pre < q && pre + a0 >= q, c1 = pre + a0 < q && pre + a0 + a1 >= q;
      const int el = __ffs(__ballot_sync(0xffffffffu, c0 || c1)) - 1;
      const int eb = __shfl_sync(0xffffffffu, c0 ? top - 2 * lane : top - 2 * lane - 1, el);
      if (eb >= kS4Fine - 1) {
        flags = 1;  // e among the clamped keys
      } else {
        lo = ::min(::max(theta + eb - 127, theta), theta + kS4Fine - 1 - 256);
        const int lb = lo - theta, eo = theta + eb - lo;
        // band: the widest [eo - d, eo + d] (inside the window) with <= kS4Band keys
        const uint32_t ce = fine[lb + eo];
        int d = 0;
        if (ce <= (uint32_t)kS4Band) {
          // lane l tests d = l + 1 (counts of the 2 (l + 1) keys around e)
          const int dl = lane + 1;
          const uint32_t up = eo + dl <= 255 ? fine[lb + eo + dl] : 0x10000u;
          const uint32_t dn = eo - dl >= 0 ? fine[lb + eo - dl] : 0x10000u;
          const uint32_t add = up + dn;
          const uint32_t cum = ce + s4_wscan(add, lane);
          const uint32_t ok = __ballot_sync(0xffffffffu, cum <= (uint32_t)kS4Band);
          d = __popc(~ok) == 0 ? 32 : __ffs(~ok) - 1;  // the first d that does not fit
          bl = eo - d;
          bh = eo + d;
        }
        if (warp == 0) {
          // above: keys >= lo + 256 = bins >= lb + 256
          const int hbn = lb + 256;
          const int ch = (kS4Fine - 1 - hbn) >> 6;
          const uint32_t exh = __shfl_sync(0xffffffffu, x - s, ch);
          const int ctop = kS4Fine - 1 - 64 * ch;
          uint32_t part = 0;
          if (ctop - 2 * lane >= hbn) part += fine[ctop - 2 * lane];
          if (ctop - 2 * lane - 1 >= hbn) part += fine[ctop - 2 * lane - 1];
          const uint32_t above = exh + __reduce_add_sync(0xffffffffu, part);
          if (lane == 0) s_u[6] = above;
        }
      }
    }
    // band entries (every warp, a slice of the list): (key - lo) << 16 | position
    if (bh >= bl) {
      const uint32_t klo = (uint32_t)(lo + bl), khi = (uint32_t)(lo + bh);
#pragma unroll 1
      for (int i = tid; i < (int)nct; i += T) {
        const uint32_t e = h2c[i], k = e >> 16;
        if (k >= klo && k <= khi) band[atomicAdd(&s_u[7], 1u)] = ((k - (uint32_t)lo) << 16) | (e & 0xFFFFu);
      }
    }
    // the bitmaps (post-pass SMEM row) are zeroed here, ordered before the marks by the poll
#pragma unroll 1
    for (int i = tid; i < 2 * kBmW; i += T) bmg[i] = 0u;
    __syncthreads();  // band entries
    S4DBG(13);
    // send: words [0, 8) header, [8, 136) window count pairs, [136, 136 + kS4Band) band
    const int64_t r1row = p.off_r1 + ((int64_t)(par * p.cap + row) * W) * kV2Msg;
    if (tid < kS4Msg / 4) {
      const int w0 = 4 * tid;
      uint4 v;
      if (w0 < kS4Hdr) {
        const uint32_t nb = s_u[7], sat = 0;
        v = w0 == 0 ? make_uint4(tagw(tag, (uint32_t)lo | (flags << 16)), tagw(tag, ::min(s_u[6], 0xFFFFFFu)),
                                 tagw(tag, sat), tagw(tag, 0u))
                    : make_uint4(tagw(tag, (uint32_t)(bl & 0xFF) | ((uint32_t)(bh & 0xFF) << 8) | (nb << 16)),
                                 tagw(tag, bh >= bl ? 1u : 0u), tagw(tag, 0u), tagw(tag, 0u));
      } else if (w0 < kS4Hdr + 128) {
        const int b0 = lo - theta + 2 * (w0 - kS4Hdr);
        uint32_t c8[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) c8[j] = flags & 1u ? 0u : fine[b0 + j];
        v = make_uint4(pack2(tag, c8[0], c8[1]), pack2(tag, c8[2], c8[3]), pack2(tag, c8[4], c8[5]),
                       pack2(tag, c8[6], c8[7]));
      } else {
        const int j0 = w0 - kS4Hdr - 128;
        const uint32_t nb = s_u[7];
        v = make_uint4(tagw(tag, j0 < (int)nb ? band[j0] : 0u), tagw(tag, j0 + 1 < (int)nb ? band[j0 + 1] : 0u),
                       tagw(tag, j0 + 2 < (int)nb ? band[j0 + 2] : 0u), tagw(tag, j0 + 3 < (int)nb ? band[j0 + 3] : 0u));
      }
      s4_put(p, r1row + (int64_t)p.rank * kV2Msg + w0, v);
    }
    S4DBG(3);
    // poll (all threads, one v4 each)
    {
      const long long t0 = now_ns();
#pragma unroll 1
      for (int i = tid; i < W * (kS4Msg / 4); i += T) {
        const int rr = i / (kS4Msg / 4), j = 4 * (i % (kS4Msg / 4));
        const uint32_t* src = p.box + r1row + (int64_t)rr * kV2Msg + j;
        *reinterpret_cast<uint4*>(rx + rr * kS4Msg + j) = s4_poll(src, ld_sys_v4(src), tag, p.spin_ns, p.err, t0);
      }
    }
    if (bt_sm)
      for (int i = tid; i < nblk; i += T) btab[i] = btrow[i];
    __syncthreads();  // polled
    S4DBG(4);
    if (warp == 0) {
      // resolve (every rank alike): common range [CL, CH] of the windows, the global count
      // above every key in it, tau = the largest key with >= K keys at or above it; ties cut
      // by id when every rank's band holds tau
      int mode = 0, tau = 0;
      uint32_t need = 0, ties = 0;
      const uint32_t hw0 = lane < W ? rx[lane * kS4Msg] & 0xFFFFFFu : 0u;
      const int lol = (int)(hw0 & 0xFFFFu);
      const bool bad = __any_sync(0xffffffffu, lane < W && ((hw0 >> 16) & 1u));
      const int CL = __reduce_max_sync(0xffffffffu, lane < W ? lol : 0);
      const int CH = __reduce_min_sync(0xffffffffu, lane < W ? lol + 255 : 0x7FFFFFFF);
      if (bad || CL > CH) {
        mode = 2;
      } else {
        uint32_t g[8], tsum = 0;
        bool sat = false;
#pragma unroll
        for (int j = 0; j < 8; ++j) g[j] = 0;
#pragma unroll 1
        for (int rr = 0; rr < W; ++rr) {
          const int lo_r = __shfl_sync(0xffffffffu, lol, rr);
          const uint32_t* win = rx + rr * kS4Msg + kS4Hdr;
          auto cnt = [&](int off) -> uint32_t {
            const uint32_t w = win[off >> 1];
            const uint32_t c = (off & 1) ? (w & 0xFFFu) : ((w >> 12) & 0xFFFu);
            sat |= c >= kCountMax;
            return c;
          };
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const int v = CH - 8 * lane - j;
            if (v >= CL) g[j] += cnt(v - lo_r);
            const int u = CH + 1 + 8 * lane + j;
            if (u <= lo_r + 255) tsum += cnt(u - lo_r);
          }
          if (lane == 0) tsum += rx[rr * kS4Msg + 1] & 0xFFFFFFu;
        }
        const uint32_t ctop = __reduce_add_sync(0xffffffffu, tsum);
        uint32_t s = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) s += g[j];
        const uint32_t x = s4_wscan(s, lane);
        const uint32_t tot = __shfl_sync(0xffffffffu, x, 31);
        if (__any_sync(0xffffffffu, sat) || ctop >= (uint32_t)K || ctop + tot < (uint32_t)K) {
          mode = 2;  // tau above / below the common range, or a saturated count
        } else {
          const uint32_t kk = (uint32_t)K - ctop;
          const bool hit = x - s < kk && x >= kk;
          int tlv = 0;
          uint32_t nd = 0, ti = 0;
          if (hit) {
            uint32_t run = x - s;
            bool f = false;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
              if (!f && run + g[j] >= kk) { f = true; tlv = CH - 8 * lane - j; nd = kk - run; ti = g[j]; }
              run += g[j];
            }
          }
          const int src = __ffs(__ballot_sync(0xffffffffu, hit)) - 1;
          tau = __shfl_sync(0xffffffffu, tlv, src);
          need = __shfl_sync(0xffffffffu, nd, src);
          ties = __shfl_sync(0xffffffffu, ti, src);
          if (need < ties) {  // the ties are cut by id: every rank with ties at tau holds them in its band
            bool miss = false;
            if (lane < W) {
              const uint32_t* hd = rx + lane * kS4Msg;
              const uint32_t bw = hd[4] & 0xFFFFFFu;
              const int to = tau - lol;
              const uint32_t w = hd[kS4Hdr + (to >> 1)];
              const uint32_t ct = (to & 1) ? (w & 0xFFFu) : ((w >> 12) & 0xFFFu);
              miss = ct != 0u && !((hd[5] & 1u) && to >= (int)(bw & 0xFFu) && to <= (int)((bw >> 8) & 0xFFu));
            }
            if (__any_sync(0xffffffffu, miss)) mode = 1;
          }
        }
      }
      if (lane == 0) {
        s_i[0] = mode;
        s_i[1] = tau;
        s_u[0] = need;
        s_u[1] = ties;
      }
    }
    __syncthreads();  // resolved
    S4DBG(5);
    const int mode = s_i[0];
    if (K31_RARE(mode != 0 || lr)) {
      // ---- radix rounds (rare) + selection from global memory
      S4STAT(mode == 1 ? 2 : 3);
      uint32_t gcut = 0xFFFFFFFFu;
      const int tau = s4_radix<T>(p, row, L, tag, par, mode == 0 ? 2 : mode, s_i[1], s_u[0], h1, s_i, s_u, &gcut);
      S4DBG(6);
      s4_select_global<T>(p, row, L, tau, gcut, orow, btrow, btab, bt_sm, s_ws);
      if (tid == 0) p.epoch[row] = ep + 1 == 510 ? 0 : ep + 1;
      S4DBG(8);
      __syncthreads();
      continue;
    }
    S4STAT(1);
    const int tau = s_i[1];
    const uint32_t need = s_u[0], ties = s_u[1];
    const bool cut = need < ties;
    if (cut) {
      // ties cut by id: every rank's band entries at tau -> tl (gid, pos | own << 31); one
      // thread per tie counts the ties with a smaller gid; the first `need` are selected
#pragma unroll 1
      for (int i = tid; i < W * kS4Band; i += T) {
        const int rr = i / kS4Band, j = i % kS4Band;
        const uint32_t* hd = rx + rr * kS4Msg;
        if (j < (int)((hd[4] >> 16) & 0xFFu)) {
          const uint32_t e = hd[kS4Hdr + 128 + j] & 0xFFFFFFu;
          if ((int)(hd[0] & 0xFFFFu) + (int)(e >> 16) == tau) {
            const int pos = (int)(e & 0xFFFFu);
            tl[atomicAdd(&s_u[8], 1u)] =
                make_uint2(s4_gid(pos, rr, W, IL, il_sh, gid_sh), (uint32_t)pos | (rr == p.rank ? 0x80000000u : 0u));
          }
        }
      }
    }
    // keys above tau (and every tie when none is cut) from the candidate list
#pragma unroll 1
    for (int i = tid; i < (int)nct; i += T) {
      const uint32_t e = h2c[i], k = e >> 16, pos = e & 0xFFFFu;
      if (k > (uint32_t)tau) atomicOr(&bmg[pos >> 5], 1u << (pos & 31));
      else if (!cut && k == (uint32_t)tau) atomicOr(&bme[pos >> 5], 1u << (pos & 31));
    }
    if (cut) {
      __syncthreads();  // tie list
      const int nt = (int)s_u[8];
#pragma unroll 1
      for (int i = tid; i < nt; i += T) {
        const uint2 me = tl[i];
        if (!(me.y >> 31)) continue;
        uint32_t rank = 0;
#pragma unroll 4
        for (int j = 0; j < nt; ++j) rank += tl[j].x < me.x;
        if (rank < need) {
          const uint32_t pos = me.y & 0xFFFFu;
          atomicOr(&bme[pos >> 5], 1u << (pos & 31));
        }
      }
    }
    __syncthreads();  // bitmaps
    S4DBG(7);
    // ---- 4. bitmaps -> position order -> KV slots: thread t owns words [nw t, nw t + nw)
    {
      const int nbw = (L + 31) >> 5, nw = (nbw + T - 1) / T, w0 = nw * tid;
      uint32_t cg = 0, ce = 0;
#pragma unroll 1
      for (int w = w0; w < ::min(w0 + nw, nbw); ++w) { cg += __popc(bmg[w]); ce += __popc(bme[w]); }
      uint32_t tot;
      const uint32_t ex = v2_scan<T>(cg | (ce << 16), s_ws + 32 * (ws_par ^= 1u), &tot);
      const uint32_t ngt = tot & 0xFFFFu;
      int og = (int)(ex & 0xFFFFu), oe = (int)(ngt + (ex >> 16));
#pragma unroll 1
      for (int w = w0; w < ::min(w0 + nw, nbw); ++w) {
#pragma unroll 1
        for (int pass = 0; pass < 2; ++pass) {
          uint32_t m = pass == 0 ? bmg[w] : bme[w];
          int& o = pass == 0 ? og : oe;
          while (m) {
            const int pos = 32 * w + __ffs(m) - 1;
            m &= m - 1;
            int sl = pos;
            if (btrow != nullptr)
              sl = bt_sm ? btab[pos >> bs_sh] * p.block_stride + (pos & (p.block_size - 1))
                         : s4_slot_slow(btrow, pos, p.block_size, p.block_stride);
            if (o < K) orow[o] = sl;
            ++o;
          }
        }
      }
      const int cnt = (int)(ngt + (tot >> 16));
      for (int i = cnt + tid; i < K; i += T) orow[i] = -1;
      if (tid == 0) {
        if (cnt > K) spin_fail(p.err, 53);
        p.out_count[row] = ::min(cnt, K);
        p.epoch[row] = ep + 1 == 510 ? 0 : ep + 1;
      }
    }
    S4DBG(8);
    __syncthreads();  // the next row lands into the row buffer
  }
#undef S4DBG
#undef S4STAT
}

#ifdef S4_SKEL
// Sync-only skeleton of the topk_s4 structure (SOL-protocol calibration; built only with
// -DS4_SKEL=<message words>, launched in place of topk_s4_kernel): the row bulk-copied to
// SMEM (kV2Cp issuers, one mbarrier), CTA barriers (sample, pass, resolved, scan), a
// message of S4_SKEL tagged words sent and polled by all threads (one v4 each), the K-entry
// row write; no compares, histograms or selection (rows written as -1, count 0).
template <int T>
__global__ void __launch_bounds__(T, T >= 1024 ? 1 : 2) topk_s4_skel(V2Params p) {
  using C = S4Cfg<T>;
  constexpr int CG = C::kCapG, MW = S4_SKEL;
  static_assert(MW % 4 == 0 && MW <= kV2Msg, "message words");
  __shared__ __align__(128) uint4 srow[CG];
  __shared__ __align__(8) uint64_t s_bar;
  __shared__ __align__(16) uint32_t rx[kV2MaxW * MW];
  __shared__ uint32_t s_x[8];
  __shared__ long long s_dbg[32];
  const int tid = threadIdx.x;
  if (tid == 0) {
    mbar_init(&s_bar, kV2Cp);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  uint32_t bar_phase = 0;
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const int W = p.W, K = p.K;
#define S4DBG(i)                                                                     \
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
#pragma unroll 1
  for (int row = blockIdx.x; row < p.T; row += gridDim.x) {
    S4DBG(0);
    const uint4* lr4 = reinterpret_cast<const uint4*>(p.logits + (int64_t)row * p.lstride);
    const int L = p.local_len[row];
    const int ng = (L + 7) >> 3, nsm = ::max(1, ::min(ng, CG));
    if ((tid & 31) == 0 && (tid >> 5) < kV2Cp) {
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      const int c = tid >> 5, a = nsm * c / kV2Cp, b = nsm * (c + 1) / kV2Cp;
      if (b > a) bulk_g2s(srow + a, lr4 + a, (uint32_t)(b - a) * 16u, &s_bar);
      else asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(&s_bar)) : "memory");
    }
    int* orow = p.out + (int64_t)row * p.ostride;
    if (tid < 8) s_x[tid] = 0;
    const int ep = p.epoch[row];
    const uint32_t tag = (uint32_t)(ep % 255);
    const int par = ep & 1;
    mbar_wait(&s_bar, bar_phase);
    bar_phase ^= 1u;
    S4DBG(9);
    if ((tid & 31) == 0) atomicXor(&s_x[0], srow[(tid * 16) % nsm].x);
    __syncthreads();  // sample
    S4DBG(10);
    S4DBG(1);
    {
      uint32_t x = 0;
      for (int g = tid; g < nsm; g += T) x ^= srow[g].x ^ srow[g].w;
      x = __reduce_xor_sync(0xffffffffu, x);
      if ((tid & 31) == 0) atomicXor(&s_x[1], x);
    }
    __syncthreads();  // pass
    S4DBG(2);
    S4DBG(13);
    const int64_t r1row = p.off_r1 + ((int64_t)(par * p.cap + row) * W) * kV2Msg;
    {
      const uint32_t v = s_x[1] & 0xFFFFu;
      for (int j = 4 * tid; j < MW; j += 4 * T) {
        const int64_t off = r1row + (int64_t)p.rank * kV2Msg + j;
        const uint4 w = make_uint4(tagw(tag, v), tagw(tag, j), tagw(tag, 0u), tagw(tag, 1u));
        if (p.mc) mc_st_v4(p.mc + off * 4, w);
        else s4_put_peers(p.peers, W, off, w);
      }
    }
    S4DBG(3);
    {
      const long long t0 = now_ns();
      for (int i = tid; i < W * (MW / 4); i += T) {
        const int rr = i / (MW / 4), j = 4 * (i % (MW / 4));
        const uint32_t* src = p.box + r1row + (int64_t)rr * kV2Msg + j;
        *reinterpret_cast<uint4*>(rx + rr * MW + j) = s4_poll(src, ld_sys_v4(src), tag, p.spin_ns, p.err, t0);
      }
    }
    __syncthreads();  // polled
    S4DBG(4);
    if (tid == 0) s_x[2] = rx[0] ^ rx[MW];
    __syncthreads();  // resolved
    S4DBG(5);
    __syncthreads();  // scan
    S4DBG(7);
    for (int i = tid; i < K; i += T) orow[i] = -1;
    if (tid == 0) {
      p.out_count[row] = 0;
      p.epoch[row] = ep + 1 == 510 ? 0 : ep + 1;
    }
    S4DBG(8);
    __syncthreads();
  }
#undef S4DBG
}
#endif  // S4_SKEL

}  // namespace k31dcpx
