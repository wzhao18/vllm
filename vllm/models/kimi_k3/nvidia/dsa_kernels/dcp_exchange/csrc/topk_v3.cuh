// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Kimi K3.1 DSA decode under DCP: the hot row body of the exact global top-K (included by
// dcp_exchange.cu after topk_v2.cuh). Same message, mailboxes, epochs, outputs and redo
// kernel as topk_v2's hot instance; a row that is not resolved in round 1 is left to
// topk_v2_redo_kernel (the complete body) exactly as there.
//
// Per row (one CTA): the sampled q-th key places the 256-key window BEFORE the candidate
// pass (lo = sampled q-th key - kV3Below), so
//  1. sample (1/8 of the row's groups) -> lo;
//  2. one pass over the SMEM row: every key >= lo with its position (position order);
//  3. the window histogram over the candidates (keys above the window counted), one
//     suffix scan of it: the exact local q-th key e, the window prefix, the band;
//  4. the round (multicast, Lamport-tagged words), the tier from the exchanged words;
//  5. WRITE from the candidates.
// No theta pass, no 2048-key fine histogram, no suffix array, no window copy: about half
// of v2's barrier-separated phases and of its executed code before the round.
// Every key >= tau lies in the candidates whenever the round resolves (tier 1 puts tau in
// every rank's window, so tau >= this rank's lo); a rank whose candidates overflow or
// whose row does not fit the SMEM path sends a saturated message (tier 3 on every rank).

namespace k31dcpx {

constexpr int kV3Below = 16;  // window keys below the sampled q-th key (~9% in value)

struct V3Sm {
  uint32_t* hist;  // sample histogram (kCoarseBins), then the exchanged words
  uint32_t* cpos;
  uint16_t* ckey;
  uint4* srow;     // the row, then the candidates' slots + the message
  uint32_t* win;   // kV2Win
  uint32_t* P;     // kV2Win + 1: window keys below u; P[kV2Win] = window total
  uint32_t* bcur;  // kV2Win band cursors, then the tie ranks
  uint64_t* bar;
  uint32_t* warp_sums;
  int* s_i;
  uint32_t* s_u;
  long long* dbg;
};

template <int T>
__device__ __forceinline__ int v3_row(const V2Params& p, const V3Sm& sm, const int row,
                                      uint32_t& bar_phase) {
  uint32_t* const hist = sm.hist;
  uint32_t* const rx = sm.hist;
  uint32_t* const cpos = sm.cpos;
  uint16_t* const ckey = sm.ckey;
  uint4* const srow = sm.srow;
  int* const cslot = reinterpret_cast<int*>(srow);
  uint32_t* const msg = reinterpret_cast<uint32_t*>(srow) + kV2Cand;
  uint32_t* const win = sm.win;
  uint32_t* const P = sm.P;
  uint32_t* const bcur = sm.bcur;
  uint64_t& s_bar = *sm.bar;
  uint32_t* const warp_sums = sm.warp_sums;
  uint32_t ws_par = 0;
#define V3WS (warp_sums + 32 * (ws_par ^= 1u))
  int* const s_i = sm.s_i;
  uint32_t* const s_u = sm.s_u;
  long long* const s_dbg = sm.dbg;
  const int tid = threadIdx.x;
  constexpr int kNR = V2Cfg<T>::kNR, kPB = V2Cfg<T>::kPB;
  const int W = p.W, K = p.K, IL = p.interleave;
  auto log2_or = [](int x) -> int { return (x > 0 && (x & (x - 1)) == 0) ? __ffs(x) - 1 : -1; };
  const int bs_sh = log2_or(p.block_size), il_sh = log2_or(IL), w_sh = log2_or(W);
  const int gid_sh = (il_sh >= 0 && w_sh >= 0) ? il_sh + w_sh : -1;
  int& s_tier = s_i[0];
  int& s_tau = s_i[1];
  int& s_need = s_i[2];
  int& s_q = s_i[3];
  int& s_cnt = s_i[5];
  int& s_e = s_i[6];
  int& s_lo = s_i[7];
  uint32_t& s_valid = s_u[0];
  uint32_t& s_above = s_u[1];
  uint32_t& s_sat = s_u[5];
#define V3DBG(i)                                                                     \
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
#define V3STAT(i) \
  if (p.stats && tid == 0) atomicAdd(&p.stats[i], 1);

  V3DBG(0);
  const uint16_t* lrow = p.logits + (int64_t)row * p.lstride;
  const uintptr_t la = reinterpret_cast<uintptr_t>(lrow);
  const int head0 = (int)(((16u - (uint32_t)(la & 15u)) & 15u) >> 1);
  const int sh0 = (8 - head0) & 7;
  {
    const int gs_lo = sh0 ? 1 : 0, gs_hi = min((p.lcols + sh0) >> 3, kNR * T);
    if ((tid & 31) == 0 && (tid >> 5) < kV2Cp) {
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      const int n = max(0, gs_hi - gs_lo), c = tid >> 5;
      const int a = gs_lo + (int)((int64_t)n * c / kV2Cp), b = gs_lo + (int)((int64_t)n * (c + 1) / kV2Cp);
      const uint4* gb = reinterpret_cast<const uint4*>(lrow - sh0);
      if (b > a) bulk_g2s(srow + a, gb + a, (uint32_t)(b - a) * 16u, &s_bar);
      else asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(&s_bar)) : "memory");
    }
  }
  const int L = p.local_len[row];
  const int Gl = p.global_len[row];
  int* orow = p.out + (int64_t)row * p.ostride;
  const int req = p.req_of_row != nullptr ? p.req_of_row[row] : row / p.next_n;
  const int* btrow = p.block_table != nullptr ? p.block_table + (int64_t)req * p.bt_stride : nullptr;
  auto slot_of = [&](int pos) -> int {
    if (btrow == nullptr) return pos;
    if (bs_sh >= 0) return btrow[pos >> bs_sh] * p.block_stride + (pos & (p.block_size - 1));
    return btrow[pos / p.block_size] * p.block_stride + pos % p.block_size;
  };
  auto gid_of = [&](int pos) -> uint32_t {
    if (gid_sh >= 0)
      return (uint32_t)((((pos >> il_sh) << gid_sh) | (p.rank << il_sh)) | (pos & (IL - 1)));
    return (uint32_t)(((pos / IL) * W + p.rank) * IL + pos % IL);
  };
  if (K31_RARE(Gl <= K)) {  // the whole context (every rank sees the same global length)
    mbar_wait(&s_bar, bar_phase);
    bar_phase ^= 1u;
    for (int i = tid; i < L; i += T) orow[i] = slot_of(i);
    for (int i = L + tid; i < K; i += T) orow[i] = -1;
    if (tid == 0) p.out_count[row] = L;
    __syncthreads();
    return 0;
  }
  if (Gl >= (1 << 24) && tid == 0) spin_fail(p.err, 13);
  const int ep = p.epoch[row];
  const uint32_t tag = (uint32_t)(ep % 255);
  const int par = ep & 1;
  V3STAT(0);
  const unsigned long long kl = (unsigned long long)K * (unsigned)L + (unsigned)(Gl - 1);
  const uint32_t q = max(1u, kl <= 0xFFFFFFFFull ? (uint32_t)kl / (uint32_t)Gl : (uint32_t)(kl / (unsigned)Gl));

  // ---- 1. sample -> lo
  for (int b = tid; b < kCoarseBins / 4; b += T) reinterpret_cast<uint4*>(hist)[b] = make_uint4(0u, 0u, 0u, 0u);
  if (tid < kV2Win) { win[tid] = 0u; bcur[tid] = 0u; }
  if (tid == 0) { s_valid = 0; s_above = 0; s_sat = 0; s_cnt = 0; s_lo = 1; s_e = -1; }
  const int head = min(L, head0);
  const int sh = (8 - head) & 7;
  const int ng = (L + sh + 7) >> 3;
  const int g_hi = (L + sh) >> 3;
  const bool regp = L >= 8 && ng <= kNR * T;
  auto load_part = [&](int g) -> uint4 {  // a partial group, key by key (NaN outside [0, L))
    uint32_t w[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int a = 8 * g - sh + 2 * j, b = a + 1;
      const uint32_t lo16 = (a >= 0 && a < L) ? lrow[a] : 0xFFFFu;
      const uint32_t hi16 = (b >= 0 && b < L) ? lrow[b] : 0xFFFFu;
      w[j] = lo16 | (hi16 << 16);
    }
    return make_uint4(w[0], w[1], w[2], w[3]);
  };
  const bool phead = regp && sh > 0, ptail = regp && g_hi < ng && g_hi > 0;
  uint4 pg = make_uint4(~0u, ~0u, ~0u, ~0u);
  if (tid == 0 && phead) pg = load_part(0);
  if (tid == 32 && ptail) pg = load_part(g_hi);
  mbar_wait(&s_bar, bar_phase);
  bar_phase ^= 1u;
  if (tid == 0 && phead) srow[0] = pg;
  if (tid == 32 && ptail) srow[g_hi] = pg;
  __syncthreads();
  V3DBG(9);
  if (K31_LIKELY(regp)) {
#pragma unroll 1
    for (int g = tid * kV2SampleStride + (tid & 7); g < ng; g += T * kV2SampleStride) {
      const Keys8 k8(srow[g]);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const uint32_t k = k8.key(j);
        if (k) atomicAdd(&hist[k >> 4], 1u);
      }
    }
  }
  __syncthreads();
  V3DBG(10);
  {
    // the coarse bin where the sample's suffix count reaches q / 8: the q-th key's estimate
    const uint32_t want = max(1u, q / kV2SampleStride);
    const int bb = T - 1 - tid;
    uint32_t hv[kPB];
    uint32_t v = 0;
#pragma unroll
    for (int i = 0; i < kPB; ++i) { hv[i] = hist[bb * kPB + i]; v += hv[i]; }
    uint32_t tot;
    const uint32_t ab = v2_scan<T>(v, V3WS, &tot);
    if (ab < want && ab + v >= want) {
      uint32_t run = ab;
      int e16 = bb * kPB;
#pragma unroll
      for (int i = kPB - 1; i >= 0; --i) {
        const uint32_t nx = run;
        run += hv[i];
        if (run >= want && nx < want) e16 = bb * kPB + i;
      }
      s_lo = min(max(1, e16 * 16 + 8 - kV3Below), 0x10000 - kV2Win);
    }
  }
  __syncthreads();
  const int lo = s_lo;
  V3DBG(1);

  // ---- 2. candidates: every key >= lo with its position, in position order
  {
    const int tg0 = tid * kNR;
    const uint32_t t2 = v2_th_bf16(lo);
    const bool kx = lo == 0x8000;
    constexpr int kMW = (kNR + 3) / 4;
    uint32_t mk[kMW];
#pragma unroll
    for (int w = 0; w < kMW; ++w) mk[w] = 0u;
    uint32_t cnt = 0, nv = 0, nacc = 0u;
    if (regp) {
#pragma unroll
      for (int i = 0; i < kNR; ++i) {
        const int g = tg0 + i;
        if (g < ng) {
          const uint4 r = srow[g];
          {
            const uint32_t m = v2_max_nan(r);
            asm("max.NaN.bf16x2 %0, %1, %2;" : "=r"(nacc) : "r"(nacc), "r"(m));
          }
          nv += 8u;
          uint32_t m8;
          if (!kx) {
            const uint32_t l4 = __byte_perm(v2_ge2(r.x, t2), v2_ge2(r.y, t2), 0x6420);
            const uint32_t h4 = __byte_perm(v2_ge2(r.z, t2), v2_ge2(r.w, t2), 0x6420);
            m8 = (((l4 & 0x08040201u) * 0x01010101u) >> 24) | ((((h4 & 0x08040201u) * 0x01010101u) >> 20) & 0xF0u);
          } else {
            const Keys8 k8(r);
            m8 = 0;
#pragma unroll 1
            for (int j = 0; j < 8; ++j) m8 |= (k8.key(j) >= (uint32_t)lo ? 1u : 0u) << j;
          }
          cnt += __popc(m8);
          mk[i / 4] |= m8 << (8 * (i % 4));
        }
      }
    }
    if (K31_RARE(v2_is_nan2(nacc))) {  // NaN keys (or the masked partial groups): exact count
      nv = 0;
#pragma unroll 1
      for (int i = 0; i < kNR; ++i)
        if (tg0 + i < ng) nv += v2_nvalid(srow[tg0 + i]);
    }
    V3DBG(24);
    uint32_t ctot;
    int c = (int)v2_scan<T>(cnt, V3WS, &ctot);
    if (tid == 0) s_cnt = (int)ctot;
    const uint16_t* srow16 = reinterpret_cast<const uint16_t*>(srow);
#pragma unroll
    for (int w = 0; w < kMW; ++w) {
      uint32_t m = mk[w];
      while (m) {
        const int b = __ffs(m) - 1;
        m &= m - 1;
        const int g = tg0 + 4 * w + (b >> 3), j = b & 7;
        if (c < kV2Cand) {
          cpos[c] = (uint32_t)(8 * g - sh + j);
          ckey[c] = (uint16_t)(keys2(srow16[8 * g + j]) & 0xFFFFu);
        }
        ++c;
      }
    }
    for (int o = 16; o > 0; o >>= 1) nv += __shfl_xor_sync(0xffffffffu, nv, o);
    if ((tid & 31) == 0 && nv) atomicAdd(&s_valid, nv);
  }
  __syncthreads();
  const int nc = s_cnt;
  const bool ok = regp && nc <= kV2Cand;
  V3DBG(2);
  if (p.dbg && tid == 0) p.dbg[(int64_t)row * kV2Dbg + 31] = nc;

  // ---- 3. window histogram over the candidates -> e, window prefix, band, message
  if (K31_LIKELY(ok)) {
    uint32_t above = 0;
    for (int c = tid; c < nc; c += T) {
      const uint32_t u = (uint32_t)ckey[c] - (uint32_t)lo;
      if (u >= (uint32_t)kV2Win) ++above;
      else atomicAdd(&win[u], 1u);
    }
    for (int o = 16; o > 0; o >>= 1) above += __shfl_xor_sync(0xffffffffu, above, o);
    if ((tid & 31) == 0 && above) atomicAdd(&s_above, above);
  } else if (tid == 0) {
    s_sat = 1u;
  }
  __syncthreads();
  V3DBG(12);
  uint32_t wtot;
  {
    const int u = kV2Win - 1 - tid;  // thread t < 256 <-> window key 255 - t
    const uint32_t c = u >= 0 ? win[u] : 0u;
    if (c > kCountMax) s_sat = 1u;
    const uint32_t ex = v2_scan<T>(c, V3WS, &wtot);  // window keys above u
    const uint32_t ab = ex + s_above;
    if (u >= 0) {
      P[u] = wtot - ex - c;
      if (ab < q && ab + c >= q) s_e = u;
    }
    if (tid == 0) P[kV2Win] = wtot;
  }
  __syncthreads();
  V3DBG(14);
  const int e = ok ? s_e : -1;
  // widest symmetric band around e with <= kV2Band keys (monotone in the half-width)
  const bool bcand = e >= 0 && win[e] <= (uint32_t)kV2Band;
  bool bok = false;
  if (bcand && tid < kV2Win) {
    const int a = max(0, e - tid), b = min(kV2Win - 1, e + tid);
    bok = P[b + 1] - P[a] <= (uint32_t)kV2Band;
  }
  const int lo_h = __syncthreads_count(bok) - 1;
  int bl = 1, bh = 0, nb = 0;
  if (bcand) {
    bl = max(0, e - lo_h);
    bh = min(kV2Win - 1, e + lo_h);
    nb = (int)(P[bh + 1] - P[bl]);
  }
  V3DBG(15);
  if (nb > 0) {
    const int k0 = lo + bl, k1 = lo + bh;
    const uint32_t p0 = P[bl];
    for (int c = tid; c < nc; c += T) {
      const int k = ckey[c];
      if (k >= k0 && k <= k1) {
        const int u = k - lo;
        const uint32_t i = P[u] - p0 + atomicAdd(&bcur[u], 1u);
        if (i < (uint32_t)kV2Band) msg[kV2Fix + i] = tagw(tag, gid_of((int)cpos[c]));
      }
    }
  }
  if (tid < kV2Win / 2) msg[kV2Hdr + tid] = pack2(tag, win[2 * tid], win[2 * tid + 1]);
  if (tid == 0) {
    const uint32_t sat = s_sat;
    msg[0] = tagw(tag, (uint32_t)(sat ? 0 : lo) | (sat ? (1u << 16) : 0u));
    msg[1] = tagw(tag, sat ? 0u : min(s_above, 0xFFFFFFu));
    msg[2] = tagw(tag, nb > 0 ? ((1u << 16) | ((uint32_t)bh << 8) | (uint32_t)bl) : 0u);
    msg[3] = tagw(tag, (uint32_t)nb);
    msg[4] = tagw(tag, min(s_valid, 0xFFFFFFu));
    msg[5] = tagw(tag, 0u);
    msg[6] = tagw(tag, 0u);
    msg[7] = tagw(tag, 0u);
  }
  for (int j = kV2Fix + nb + tid; j < kV2Msg; j += T) msg[j] = tagw(tag, 0u);
  __syncthreads();
  V3DBG(3);

  // ---- 4. round 1: send, the candidates' slots in its shadow, poll every rank's words
  const int64_t r1row = p.off_r1 + ((int64_t)(par * p.cap + row) * W) * kV2Msg;
  {
    const int64_t base = r1row + (int64_t)p.rank * kV2Msg;
    for (int j = tid * 4; j < kV2Msg; j += T * 4) {
      const uint4 v = make_uint4(msg[j], msg[j + 1], msg[j + 2], msg[j + 3]);
      const int64_t off = base + j;
      if (p.mc) mc_st_v4(p.mc + off * 4, v);
      else
        for (int d = 0; d < W; ++d) st_v4(reinterpret_cast<uint32_t*>(p.peers[d]) + off, v);
    }
  }
  if (tid == 0) { s_tau = -1; s_need = 0; s_q = 0; s_tier = 0; }
  if (ok) {
    for (int c = tid; c < nc; c += T) cslot[c] = slot_of((int)cpos[c]);
    // the whole output row -1 (the WRITE overwrites its head) and the tie-rank scratch
    for (int i = tid; i < K; i += T) orow[i] = -1;
  }
  if (tid < kV2Win) bcur[tid] = 0u;
  V3DBG(28);
  for (int i = tid; i < W * (kV2Msg / 4); i += T) {
    const int r = i / (kV2Msg / 4), j = (i % (kV2Msg / 4)) * 4;
    const uint32_t* src = p.box + r1row + (int64_t)r * kV2Msg + j;
    const long long t0 = now_ns();
    uint4 v = ld_sys_v4(src);
    while (!tagged(v, tag)) {
      if (spin_late(p.spin_ns, t0)) { spin_fail(p.err, 31); break; }
      __nanosleep(128);
      v = ld_sys_v4(src);
    }
    *reinterpret_cast<uint4*>(rx + r * kV2Msg + j) = v;
  }
  __syncthreads();
  V3DBG(4);

  // ---- resolve (as topk_v2): tier 1 resolved with quota s_q; anything else -> redo
  {
    __shared__ int s_tc[kV2MaxW], s_toff[kV2MaxW], s_tok[kV2MaxW];
    int mx = 0, mn = 1 << 30, sat = 0;
    uint32_t tv = 0;
    for (int r = 0; r < W; ++r) {
      const uint32_t* s = rx + r * kV2Msg;
      const int lr = (int)(s[0] & 0xFFFFu);
      sat |= (int)((s[0] >> 16) & 1u);
      mx = max(mx, lr);
      mn = min(mn, lr + kV2Win);
      tv += s[4] & 0xFFFFFFu;
    }
    int rt = tv <= (uint32_t)K ? 4 : (sat || mx >= mn) ? 3 : 0;
    int tau = -1, need = 0, qv = 0;
    V3DBG(19);
    if (rt == 0) {
      constexpr bool kOne = T >= 2 * kV2Win;
      const int tb = kOne ? tid - kV2Win : tid;
      uint32_t a = 0, v = 0;
      for (int r = 0; r < W; ++r) {
        const uint32_t* s = rx + r * kV2Msg;
        const int lr = (int)(s[0] & 0xFFFFu);
        if (tid < kV2Win) {
          if (lr + tid >= mn) a += v2_cnt(s, tid);
          if (tid == 0) a += s[1] & 0xFFFFFFu;
        }
        if (tb >= 0 && tb < kV2Win && mn - 1 - tb >= mx) v += v2_cnt(s, mn - 1 - tb - lr);
      }
      uint32_t tot, tota = 0, ex;
      if constexpr (kOne) {
        ex = v2_scan<T>(a + v, V3WS, &tot);
      } else {
        v2_scan<T>(a, V3WS, &tota);
        ex = v2_scan<T>(v, V3WS, &tot) + tota;
      }
      if (tb >= 0 && tb < kV2Win && ex < (uint32_t)K && ex + v >= (uint32_t)K) {
        s_tau = mn - 1 - tb;
        s_need = K - (int)ex;
      }
      __syncthreads();
      tau = s_tau;
      need = s_need;
      if (tau < 0) rt = 3;
      V3DBG(20);
    }
    if (rt == 0) {
      const int warp = tid >> 5, lane = tid & 31;
      if (warp < W) {
        const uint32_t* s = rx + warp * kV2Msg;
        const int lr = (int)(s[0] & 0xFFFFu);
        const uint32_t bw = s[2] & 0xFFFFFFu;
        const int blr = (int)(bw & 0xFFu), bhr = (int)((bw >> 8) & 0xFFu);
        const int tok = ((bw >> 16) & 1u) && tau - lr >= blr && tau - lr <= bhr;
        uint32_t c = 0;
        if (tok)
          for (int u = blr + lane; u < tau - lr; u += 32) c += v2_cnt(s, u);
        for (int o = 16; o > 0; o >>= 1) c += __shfl_xor_sync(0xffffffffu, c, o);
        if (lane == 0) {
          s_tc[warp] = (int)v2_cnt(s, tau - lr);
          s_tok[warp] = tok;
          s_toff[warp] = (int)c;
        }
      }
      __syncthreads();
      V3DBG(21);
      int tot_t = 0, nr = 0;
      bool band_ok = true;
      for (int r = 0; r < W; ++r) {
        tot_t += s_tc[r];
        nr += s_tc[r] > 0;
        if (s_tc[r] > 0 && !s_tok[r]) band_ok = false;
      }
      const int mine = s_tc[p.rank];
      if (tot_t == need) { rt = 1; qv = mine; }
      else if (nr == 1) { rt = 1; qv = mine > 0 ? need : 0; }
      else if (band_ok) rt = 5;
      else rt = 2;
      if (rt == 5) {
        // this rank's quota: its ties whose global id ranks below `need` among every
        // rank's ties (ids distinct, groups unsorted)
        const int m = mine;
        int* rk = reinterpret_cast<int*>(bcur);  // zeroed in the round's shadow
        const uint32_t* my = rx + p.rank * kV2Msg + kV2Fix + s_toff[p.rank];
        for (int it = tid; it < m * W; it += T) {
          const int i = it / W, r2 = it - i * W;
          const uint32_t x = my[i] & 0xFFFFFFu;
          const uint32_t* g = rx + r2 * kV2Msg + kV2Fix + s_toff[r2];
          const int n2 = s_tc[r2];
          int c = 0;
#pragma unroll 4
          for (int j = 0; j < n2; ++j) c += (g[j] & 0xFFFFFFu) < x;
          if (c) atomicAdd(&rk[i], c);
        }
        __syncthreads();
        rt = 1;
        qv = __syncthreads_count(tid < m && rk[tid] < need);
      }
    }
    if (rt != 1) {  // the same on every rank: the complete body takes the row
      __syncthreads();
      return 1 + ep;
    }
    V3DBG(5);
    // ---- 5. WRITE from the candidates (sorted by position, slots ready): one scan of
    // (above tau, ties at tau) over contiguous candidate chunks, then scatter
    if (tid == 0 && (!ok || tau < lo)) spin_fail(p.err, 41);
    const int qq = qv;
    const int cpt = (nc + T - 1) / T;
    const int c0 = min(nc, tid * cpt), c1 = min(nc, c0 + cpt);
    uint32_t ca = 0, ct = 0;
    for (int c = c0; c < c1; ++c) {
      const int k = ckey[c];
      ca += k > tau;
      ct += k == tau;
    }
    uint32_t tot;
    const uint32_t pre = v2_scan<T>(ca | (ct << 16), V3WS, &tot);
    const int gtt = (int)(tot & 0xFFFFu), nties = (int)(tot >> 16);
    int a = (int)(pre & 0xFFFFu), t = (int)(pre >> 16);
    for (int c = c0; c < c1; ++c) {
      const int k = ckey[c];
      if (k > tau) {
        if (a < K) orow[a] = cslot[c];
        ++a;
      } else if (k == tau) {
        if (t < qq && gtt + t < K) orow[gtt + t] = cslot[c];
        ++t;
      }
    }
    const int count = min(K, gtt + qq);
    if (tid == 0 && (gtt + qq > K || nties < qq)) spin_fail(p.err, 40);
    if (tid == 0) {
      p.out_count[row] = count;
      p.epoch[row] = ep + 1 == 510 ? 0 : ep + 1;
    }
    V3STAT(1);
    V3DBG(18);
  }
  V3DBG(8);
  __syncthreads();  // SMEM reused by the next row
  return 0;
#undef V3DBG
#undef V3WS
#undef V3STAT
}

#define V3_KERNEL_PROLOGUE(T)                                                       \
  __shared__ __align__(16) uint32_t hist[kCoarseBins];                              \
  __shared__ __align__(16) uint32_t cpos[kV2Cand];                                  \
  __shared__ __align__(16) uint16_t ckey[kV2Cand];                                  \
  __shared__ __align__(128) uint4 srow[V2Cfg<T>::kNR * (T)];                        \
  __shared__ __align__(16) uint32_t win[kV2Win];                                    \
  __shared__ __align__(16) uint32_t Pw[kV2Win + 4];                                 \
  __shared__ __align__(16) uint32_t bcur[kV2Win];                                   \
  __shared__ __align__(8) uint64_t s_bar;                                           \
  __shared__ uint32_t warp_sums[64];                                                \
  __shared__ int s_i[16];                                                           \
  __shared__ uint32_t s_u[16];                                                      \
  __shared__ long long s_dbg[32];                                                   \
  const V3Sm sm{hist, cpos, ckey, srow, win, Pw, bcur, &s_bar, warp_sums, s_i, s_u, s_dbg}; \
  if (threadIdx.x == 0) {                                                           \
    mbar_init(&s_bar, kV2Cp);                                                       \
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");             \
  }                                                                                 \
  __syncthreads();                                                                  \
  uint32_t bar_phase = 0;                                                           \
  asm volatile("griddepcontrol.wait;" ::: "memory");

// The hot kernel (PDL: the prologue overlaps the logits kernel's tail; every global read
// follows the wait); rows it leaves go to topk_v2_redo_kernel, launched right after.
template <int T>
__global__ void __launch_bounds__(T, T >= 1024 ? 1 : 2) topk_v3_kernel(V2Params p) {
  V3_KERNEL_PROLOGUE(T)
  for (int row = blockIdx.x; row < p.T; row += gridDim.x) {
    const int r = v3_row<T>(p, sm, row, bar_phase);
    if (threadIdx.x == 0) p.redo[row] = r;
  }
}

}  // namespace k31dcpx
