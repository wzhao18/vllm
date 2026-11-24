// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Kimi K3.1 DSA decode under DCP: exact global top-K, one CTA per row (included by
// dcp_exchange.cu; uses its memory / key / scan / streaming helpers).
//
// Same selection and output as topk_merge: every key above the K-th key tau plus the
// `need` keys equal to tau with the smallest GLOBAL token ids (the stock stable merge's
// rule); this rank's tokens above tau in position order, then its selected ties in
// position order (KV slots through the block table), -1 padding, count.
//
// Per row (one CTA, no cluster barriers, no pre-pass kernels):
//  1. sample: a 1/8 sample of the row's keys gives a key bound theta below this rank's
//     estimate of tau (its q-th largest key, q = ceil(K * L / G)) with margin;
//  2. one pass (16-byte loads, groups skipped by a packed bound) appends every key >= theta to
//     an SMEM candidate list (~2q + 256 keys) and counts the valid keys;
//  3. from the candidates: the exact q-th key e, a 256-key window around it (16-bit
//     resolution), the count above the window and a band of keys around e whose global
//     ids (sorted by key, then id) ride on the same message;
//  4. one NVLink round (multicast, Lamport-tagged words); every rank derives tau, `need`,
//     and, when every rank holding ties at tau sent them in its band, the global-id cut of
//     the ties and its own quota;
//  5. WRITE from the candidates (bitmaps by position, one scan; slots gathered in parallel).
// Exact tiers, decided identically on every rank from the exchanged words: (T2) tau known
// but a band missed it, every rank <= kV2Tie ties: one tie round of complete sorted lists;
// (T3) anything else (tau outside a window, saturated counts): radix select over the 40-bit
// composite (key << 24 | 0xFFFFFF - gid), 5 rounds of 256-bin histograms; rows with at
// most K valid keys globally select every valid key. Rows whose sample bound fails (too
// few / too many candidates) build the same message from passes over all keys.
// Mailboxes: one per (parity, row, rank) for round 1, the tie round and each radix round;
// per-row device epochs (tag = epoch % 255, parity = epoch & 1) as topk_merge; tie and
// radix words are re-armed after their poll (written only on rows that take the tier).
// Hot-body rare paths that keep a row out of the complete body: a candidate list that overflows
// (or falls short) is re-thresholded from exact per-key counts and the candidate pass re-run
// (v2_retheta); a tau found in round 1 with ties beyond the bands and tie lists takes the id cut
// by radix over the candidate ties (v2_tiecut).

namespace k31dcpx {

#define K31_RARE(c) __builtin_expect(!!(c), 0)
#define K31_LIKELY(c) __builtin_expect(!!(c), 1)

constexpr int kV2Win = 256;                  // window keys
constexpr int kV2Hdr = 8;                    // header words
constexpr int kV2Fix = kV2Hdr + kV2Win / 2;  // header + window count pairs
constexpr int kV2Band = 256;                 // band ids per rank
constexpr int kV2Msg = kV2Fix + kV2Band;     // round-1 slot words
constexpr int kV2MaxW = 8;
constexpr int kV2Tie = 256;     // tie-round ids per rank
constexpr int kV2RadR = 5;      // radix rounds (40-bit composite, 8 bits each)
constexpr int kV2Cand = 3840;   // candidate list (dense-bin content overshoots 3072; 2 CTAs / SM still fit)
constexpr int kV2Fine = 2048;   // fine histogram keys above theta
constexpr int kV2Chunk = 65536; // WRITE positions per bitmap chunk (2048 words)
// threads per CTA, by shape: 1024 (one CTA per SM) when every row has an SM of its own,
// else 512 (two CTAs per SM; 256 measured slower). kV2T: the two-per-SM variant.
constexpr int kV2T = 512;
constexpr int kV2RegG = 4;      // 8-key groups per thread per candidate batch (16384 keys)
constexpr int kV2Dbg = 64;      // dbg stamps per row
constexpr int kV2SampleStride = 8;  // sample: one 8-key group in 8
template <int T>
struct V2Cfg {
  static constexpr int kWarps = T / 32;
  static constexpr int kNR = T >= 1024 ? 5 : 9;  // SMEM row: <= kNR x T groups, odd 16-byte pitch
  static constexpr int kPB = kCoarseBins / T;    // coarse bins per thread in scans
  static constexpr int kFB = kV2Fine / T;        // fine bins per thread in scans
  static_assert(T >= kV2Win && kPB * T == kCoarseBins && kFB * T == kV2Fine && kFB % 2 == 0 && kWarps <= 32);
};
constexpr int kR2Rows = 512;   // rows with an R2 mailbox (others keep the complete body)
constexpr int kR2W = 2048;     // R2 slot words per (parity, row, rank)
constexpr int kR2Cap = 1022;   // R2 entries (key, position) per rank
constexpr int kCgBins = 496;   // CG round: bins of the bracket (12-bit counts, 2 per word)
constexpr int kCgMsg = kV2Hdr + kCgBins / 2;  // CG slot words = one radix-round slot
static_assert(kCgMsg == 256 && kV2RadR >= 2 && kV2Cand <= 4095);
constexpr int kV2RedoCtas = 32;  // grid of topk_v2_redo_kernel
constexpr int kV2Cp = 8;        // bulk copies (issuing warps) per row
constexpr int kV2RxW = ((kV2MaxW * kV2Msg + 31) / 32) * 32;  // rx words in the row buffer
// stats (rows): 0 exchanged, 1 resolved in round 1, 2 tie round, 3 radix, 4 select all
// valid, 5 full-pass message, 6 sample bound failed (re-thresholded from the SMEM row),
// 7 fine range rebased (the q-th key far above theta)

struct V2Params {
  const uint16_t* logits;
  int64_t lstride;
  int lcols;  // logits columns: every row's [0, lcols) is readable
  const int* local_len;
  const int* global_len;
  int* out;
  int64_t ostride;
  int* out_count;
  const int* block_table;
  int64_t bt_stride;
  int block_size, block_stride, next_n;
  const int* req_of_row;
  uint32_t* box;
  uint64_t mc;
  const uint64_t* peers;
  int W, rank, K, T, cap, interleave;
  int64_t off_r1, off_tie, off_rad, off_r2;
  int* epoch;
  int* redo;  // per row: 0, or the epoch the complete body runs the row at (+1)
  int* err;
  long long spin_ns;
  int* stats;
  long long* dbg;
};

// Pairwise NaN-propagating max of 8 bf16 values (two halves; a NaN anywhere shows up).
__device__ __forceinline__ uint32_t v2_max_nan(const uint4& r) {
  uint32_t a, b, m;
  asm("max.NaN.bf16x2 %0, %1, %2;" : "=r"(a) : "r"(r.x), "r"(r.y));
  asm("max.NaN.bf16x2 %0, %1, %2;" : "=r"(b) : "r"(r.z), "r"(r.w));
  asm("max.NaN.bf16x2 %0, %1, %2;" : "=r"(m) : "r"(a), "r"(b));
  return m;
}
__device__ __forceinline__ bool v2_is_nan2(uint32_t m) {
  return (((m & 0x7FFF7FFFu) + 0x007F007Fu) & 0x80008000u) != 0u;
}

__device__ __forceinline__ uint32_t v2_cnt(const uint32_t* slot, int u) {
  const uint32_t v = slot[kV2Hdr + (u >> 1)];
  return (u & 1) ? (v & 0xFFFu) : ((v >> 12) & 0xFFFu);
}

// Exclusive prefix over the kV2T-thread block; the total in *total. One barrier: every
// warp scans the warp totals itself. `ws` (>= kV2Warps words) must not be the buffer of
// the previous scan (callers alternate two buffers, V2WS), so no barrier protects reuse.
template <int T>
__device__ __forceinline__ uint32_t v2_scan(uint32_t v, uint32_t* ws, uint32_t* total) {
  constexpr int kV2Warps = V2Cfg<T>::kWarps;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  uint32_t x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const uint32_t y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x += y;
  }
  if (lane == 31) ws[warp] = x;
  __syncthreads();
  const uint32_t w0 = lane < kV2Warps ? ws[lane] : 0u;
  uint32_t t = w0;
#pragma unroll
  for (int o = 1; o < kV2Warps; o <<= 1) {
    const uint32_t y = __shfl_up_sync(0xffffffffu, t, o);
    if (lane >= o) t += y;
  }
  const uint32_t before = __shfl_sync(0xffffffffu, t - w0, warp);
  *total = __shfl_sync(0xffffffffu, t, kV2Warps - 1);
  return before + x - v;
}

// kV2FB consecutive (16-byte aligned) histogram words -> v; returns their sum
template <int FB>
__device__ __forceinline__ uint32_t v2_ldbins(const uint32_t* h, uint32_t (&v)[FB]) {
  uint32_t s = 0;
#pragma unroll
  for (int i = 0; i < FB; i += 2) {
    const uint2 x = *reinterpret_cast<const uint2*>(h + i);
    v[i] = x.x; v[i + 1] = x.y;
    s += x.x + x.y;
  }
  return s;
}

// Candidate-pass helpers. Against a key threshold th the test runs on the raw bf16
// values (one packed compare per 2 keys, no key conversion): key(x) >= th <=> x >= t
// as floats for every non-NaN x, with t = v2_th_bf16(th), except th = key(+0) (-0 would
// pass): that threshold takes the key path (kx). NaN fails every ordered compare (key 0).
__device__ __forceinline__ uint32_t v2_th_bf16(int th) {
  const uint32_t t = th < 0x7F ? 0xFF80u : th < 0x8000 ? ((uint32_t)th ^ 0xFFFFu) : ((uint32_t)th ^ 0x8000u);
  return t | (t << 16);
}
__device__ __forceinline__ uint32_t v2_ge2(uint32_t w, uint32_t t2) {
  __nv_bfloat162 a, b;
  memcpy(&a, &w, 4);
  memcpy(&b, &t2, 4);
  return __hge2_mask(a, b);
}
// keys >= th among 8 raw bf16 values
__device__ __forceinline__ uint32_t v2_cnt_ge(const uint4& r, uint32_t th, uint32_t t2, bool kx) {
  if (!kx)
    return (__popc(v2_ge2(r.x, t2)) + __popc(v2_ge2(r.y, t2)) + __popc(v2_ge2(r.z, t2)) +
            __popc(v2_ge2(r.w, t2))) >> 4;
  const Keys8 k8(r);
  uint32_t c = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) c += k8.key(j) >= th;
  return c;
}
__device__ __noinline__ uint32_t v2_nvalid(uint4 r) {
  const Keys8 k8(r);
  uint32_t n = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) n += k8.key(j) != 0u;
  return n;
}
// keys >= th of the group at positions [p0, p0 + 8) -> candidates cc, cc + 1, ...
__device__ __forceinline__ void v2_emit(const uint4& r, uint32_t th, uint32_t t2, bool kx, int p0,
                                        int cc, uint32_t* cpos, uint16_t* ckey) {
  const uint32_t w4[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint32_t kw = keys2(w4[j]);
    const uint32_t m = kx ? (((kw & 0xFFFFu) >= th ? 0xFFFFu : 0u) | ((kw >> 16) >= th ? 0xFFFF0000u : 0u))
                          : v2_ge2(w4[j], t2);
    if (m & 1u) {
      if (cc < kV2Cand) { cpos[cc] = (uint32_t)(p0 + 2 * j); ckey[cc] = (uint16_t)(kw & 0xFFFFu); }
      ++cc;
    }
    if (m >> 31) {
      if (cc < kV2Cand) { cpos[cc] = (uint32_t)(p0 + 2 * j + 1); ckey[cc] = (uint16_t)(kw >> 16); }
      ++cc;
    }
  }
}

// SMEM shared by the hot (kFast) and the complete instance of the row body
struct V2Sm {
  uint32_t* hist;
  uint32_t* cpos;
  uint16_t* ckey;
  uint4* srow;
  uint64_t* bar;
  uint32_t* warp_sums;
  int* s_i;
  uint32_t* s_u;
  unsigned long long* pref;
  long long* dbg;
};

// CG rounds for rank-skewed rows (round-1 windows disjoint, or tau outside their common
// range), out of line so the hot row body keeps its code layout. hi2 = the largest exact
// q-th key bounds tau from above (above it every rank has <= q_r - 1 keys, sum (q_r - 1) <
// K); lo2 = the largest theta: every rank's keys >= lo2 are all candidates. Each rank counts
// its candidates in [lo2, hi2] into kCgBins bins of one common power-of-two width (one round
// in the radix mailbox, received off the round-1 words) and every rank finds tau's bin from
// the same words; a bin wider than one key takes one more round of width 1. tau >= lo2 (>=
// every theta): tier 1 (quota) or the tie round from the candidates, as after round 1; the
// received words go to rx (round-1 slot format, base lo2) for the tie round. Returns false,
// rx untouched, when a rank is saturated or tau < lo2 (a rank holding most of the top K:
// its theta lies above tau): the row keeps R2. Same decision on every rank (same words).
template <int T>
__device__ __noinline__ bool v2_cg(const V2Params p, uint32_t* rx, uint32_t* hist, uint32_t* msg,
                                   const uint16_t* ckey, int nc, uint32_t tag, int par, int row,
                                   uint32_t* warp_sums, int* s_i) {
  __shared__ int s_cg[4];  // 0 base (-1: not eligible), 1 width shift, 2 bin, 3 need
  __shared__ uint32_t s_cgab;
  const int tid = threadIdx.x, W = p.W, K = p.K;
  uint32_t ws_par = 0;
  if (tid == 0) {
    int lo2 = 0, hi2 = -1;
    bool ok = true;
    for (int r = 0; r < W; ++r) {
      const uint32_t* s = rx + r * kV2Msg;
      const uint32_t w5 = s[5] & 0xFFFFFFu;
      ok = ok && !((s[0] >> 16) & 1u) && ((w5 >> 16) & 1u);
      hi2 = max(hi2, (int)(w5 & 0xFFFFu));
      lo2 = max(lo2, (int)(s[6] & 0xFFFFu));
    }
    int sh = 0;
    while (sh < 8 && hi2 - lo2 >= (kCgBins << sh)) ++sh;
    s_cg[0] = (ok && lo2 >= 1 && hi2 >= lo2) ? lo2 : -1;
    s_cg[1] = sh;
  }
  __syncthreads();
  if (s_cg[0] < 0) return false;
  uint32_t* const cgrx = hist + 512;  // W x kCgMsg received words
  const uint4 dead4 = make_uint4(0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu);
  for (int it = 0; it < 2; ++it) {
    const int lo2 = s_cg[0], sh = s_cg[1];
    for (int b = tid; b < kCgBins; b += T) hist[b] = 0u;
    if (tid == 0) { s_cgab = 0u; s_cg[2] = -1; }
    __syncthreads();
    uint32_t ab = 0;
    for (int c = tid; c < nc; c += T) {
      const int u = (int)ckey[c] - lo2;
      if (u >= 0) {
        if ((u >> sh) >= kCgBins) ++ab;
        else atomicAdd(&hist[u >> sh], 1u);
      }
    }
    for (int o = 16; o > 0; o >>= 1) ab += __shfl_xor_sync(0xffffffffu, ab, o);
    if ((tid & 31) == 0 && ab) atomicAdd(&s_cgab, ab);
    __syncthreads();
    if (tid < kCgBins / 2) msg[kV2Hdr + tid] = pack2(tag, hist[2 * tid], hist[2 * tid + 1]);
    if (tid < kV2Hdr) msg[tid] = tagw(tag, tid == 0 ? (uint32_t)lo2 : tid == 1 ? s_cgab : 0u);
    __syncthreads();
    const int64_t base = p.off_rad + (((int64_t)(par * p.cap + row) * kV2RadR + it) * W) * kCgMsg;
    for (int j = tid * 4; j < kCgMsg; j += T * 4) {
      const uint4 v = make_uint4(msg[j], msg[j + 1], msg[j + 2], msg[j + 3]);
      const int64_t off = base + (int64_t)p.rank * kCgMsg + j;
      if (p.mc) mc_st_v4(p.mc + off * 4, v);
      else
        for (int d = 0; d < W; ++d) st_v4(reinterpret_cast<uint32_t*>(p.peers[d]) + off, v);
    }
    for (int i = tid; i < W * (kCgMsg / 4); i += T) {
      uint32_t* const src = p.box + base + 4 * (int64_t)i;
      const long long t0 = now_ns();
      uint4 v = ld_sys_v4(src);
      while (!tagged(v, tag)) {
        if (spin_late(p.spin_ns, t0)) { spin_fail(p.err, 38); break; }
        __nanosleep(128);
        v = ld_sys_v4(src);
      }
      *reinterpret_cast<uint4*>(cgrx + 4 * i) = v;
      st_v4(src, dead4);  // re-armed: radix slots are used only on rare rows
    }
    __syncthreads();
    // tau's bin: thread t <-> bin kCgBins - 1 - t; keys above it from the scan
    const int b = kCgBins - 1 - tid;
    uint32_t g = 0, A = 0;
    for (int r = 0; r < W; ++r) {
      const uint32_t* s = cgrx + r * kCgMsg;
      A += s[1] & 0xFFFFFFu;
      if (b >= 0) g += v2_cnt(s, b);
    }
    uint32_t tot;
    const uint32_t ex = v2_scan<T>(g, warp_sums + 32 * (ws_par ^= 1u), &tot) + A;
    if (b >= 0 && ex < (uint32_t)K && ex + g >= (uint32_t)K) { s_cg[2] = b; s_cg[3] = K - (int)ex; }
    __syncthreads();
    const int bin = s_cg[2];
    if (bin < 0) return false;  // tau < lo2: R2 (rx untouched)
    if (sh > 0) {
      __syncthreads();
      if (tid == 0) { s_cg[0] = lo2 + (bin << sh); s_cg[1] = 0; }
      __syncthreads();
      continue;
    }
    if (tid == 0) {  // tau exact: the tie counts at tau from the bins
      const int need = s_cg[3];
      int tot_t = 0, nr = 0, mine = 0;
      bool fit = true;
      for (int r = 0; r < W; ++r) {
        const int t = (int)v2_cnt(cgrx + r * kCgMsg, bin);
        tot_t += t;
        nr += t > 0;
        fit = fit && t <= kV2Tie;
        if (r == p.rank) mine = t;
      }
      s_i[1] = lo2 + bin;  // s_tau
      s_i[2] = need;       // s_need
      if (tot_t == need) { s_i[0] = 1; s_i[3] = mine; }
      else if (nr == 1) { s_i[0] = 1; s_i[3] = mine > 0 ? need : 0; }
      else s_i[0] = fit ? 2 : 3;
    }
    for (int i = tid; i < W * kCgMsg; i += T) rx[(i / kCgMsg) * kV2Msg + i % kCgMsg] = cgrx[i];
    __syncthreads();
    if (p.stats && tid == 0) atomicAdd(&p.stats[3], 1);
    return true;
  }
  return false;  // unreachable: the second round has width 1
}

// Hot body, rare (out of line): the sample's theta left more than kV2Cand candidates (dense,
// clustered bins the 1/8 group sample under-counts) or fewer than q. Exact counts per key over
// the whole row (SMEM when the row fits, else global memory with kV2ReLd loads in flight per
// thread) in up to 4 ranges of kV2Fine keys give the threshold admitting about q + 256 keys (at
// least q, at most kV2Cand), as the complete body's re-theta; the position-ordered candidate pass
// then runs again over every chunk (re-copied into SMEM for long rows). The row stays in round 1.
// Returns the new theta (candidates in cpos / ckey, s_i[5] = their count) or -1 when no
// threshold fits (a tie wider than the list): candidates and counts are then untouched.
constexpr int kV2ReLd = 4;
template <int T>
__device__ __noinline__ int v2_retheta(const V2Params p, const V2Sm sm, const uint16_t* lrow, const int L,
                                       const int sh, const int ng, const int g_lo, const int g_hi, const int th,
                                       const int nc, const uint32_t q, uint32_t* bar_phase) {
  constexpr int kNR = V2Cfg<T>::kNR, kFB = V2Cfg<T>::kFB, kCap = kNR * T;
  const int tid = threadIdx.x;
  uint32_t* const hist = sm.hist;
  uint4* const srow = sm.srow;
  uint32_t* const warp_sums = sm.warp_sums;
  int* const s_i = sm.s_i;
  uint32_t* const s_u = sm.s_u;
  uint32_t ws_par = 0;
  __syncthreads();  // the caller's last scan buffer is free from here
  auto load_group = [&](int g) -> uint4 {
    if (g >= g_lo && g < g_hi) return __ldg(reinterpret_cast<const uint4*>(lrow - sh) + g);
    if (g < 0 || g >= ng) return make_uint4(~0u, ~0u, ~0u, ~0u);
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
  const uint32_t want = q + 256u;
  int base = nc > kV2Cand ? th : max(1, th - kV2Fine);
  int th2 = -1;
  for (int it = 0; it < 4 && th2 < 0 && base >= 1 && base < 65536; ++it) {
    for (int b = tid; b < kV2Fine; b += T) hist[b] = 0;
    if (tid == 0) { s_u[2] = 0; s_i[12] = -1; s_i[13] = -1; }
    __syncthreads();
    uint32_t bey = 0;
    // keys of a group (registers, fully unrolled); groups whose bound is below base skip it
    auto count8 = [&](const uint4& r) {
      if (max_key_bound(r) < (uint32_t)base) return;
      const uint32_t kw[4] = {keys2(r.x), keys2(r.y), keys2(r.z), keys2(r.w)};
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const uint32_t k = (j & 1) ? (kw[j >> 1] >> 16) : (kw[j >> 1] & 0xFFFFu);
        const int u = (int)k - base;
        if (u >= kV2Fine) ++bey;
        else if (u >= 0 && k) atomicAdd(&hist[u], 1u);
      }
    };
    if (ng <= kCap) {
#pragma unroll 1
      for (int g = tid; g < ng; g += T) count8(srow[g]);
    } else {
#pragma unroll 1
      for (int g0 = tid; g0 < ng; g0 += kV2ReLd * T) {
        uint4 r[kV2ReLd];
#pragma unroll
        for (int u = 0; u < kV2ReLd; ++u) r[u] = load_group(g0 + u * T);
#pragma unroll
        for (int u = 0; u < kV2ReLd; ++u) count8(r[u]);
      }
    }
    for (int o = 16; o > 0; o >>= 1) bey += __shfl_xor_sync(0xffffffffu, bey, o);
    if ((tid & 31) == 0 && bey) atomicAdd(&s_u[2], bey);
    __syncthreads();
    const int f0 = (T - 1 - tid) * kFB;
    uint32_t hv[kFB];
    const uint32_t hs = v2_ldbins(hist + f0, hv);
    uint32_t tot;
    const uint32_t sb = s_u[2];
    uint32_t run = v2_scan<T>(hs, warp_sums + 32 * (ws_par ^= 1u), &tot) + sb;
#pragma unroll
    for (int i = kFB - 1; i >= 0; --i) {  // run: keys >= base + f0 + i
      const uint32_t nx = run;
      run += hv[i];
      if (run >= want && nx < want) { s_i[12] = base + f0 + i; s_u[8] = run; }
      if (run >= q && nx < q) { s_i[13] = base + f0 + i; s_u[9] = run; }
    }
    __syncthreads();
    const uint32_t s_all = tot + sb;  // keys >= base
    if (s_i[12] >= 0 && s_u[8] <= (uint32_t)kV2Cand) th2 = s_i[12];
    else if (s_i[13] >= 0 && s_u[9] <= (uint32_t)kV2Cand) th2 = s_i[13];
    else if (s_i[13] < 0 && sb >= q) base += kV2Fine;                   // the q-th key lies above
    else if (s_all < q) base = base > 1 ? max(1, base - kV2Fine) : 0;  // below
    else break;                                                         // a tie wider than kV2Cand
    __syncthreads();
  }
  if (tid == 0) s_u[2] = 0;
  if (th2 < 1) { __syncthreads(); return -1; }
  // the candidate pass again (as the hot body's cand_regp, chunk by chunk, position order)
  const uint32_t t2 = v2_th_bf16(th2);
  const bool kx = th2 == 0x8000;
  const int gs_lo = sh ? 1 : 0;
  const uint16_t* srow16 = reinterpret_cast<const uint16_t*>(srow);
  int cbase = 0;
#pragma unroll 1
  for (int gb = 0; gb < ng; gb += kCap) {
    const int ge = min(ng, gb + kCap), ngc = ge - gb;
    if (ng > kCap) {  // long row: chunk gb into the SMEM row (the previous one is read)
      __syncthreads();
      if ((tid & 31) == 0 && (tid >> 5) < kV2Cp) {
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        const int g0 = gb == 0 ? gs_lo : gb, gfull = min((p.lcols + sh) >> 3, ge);
        const int n = max(0, gfull - g0), c = tid >> 5;
        const int a = g0 + (int)((int64_t)n * c / kV2Cp), b = g0 + (int)((int64_t)n * (c + 1) / kV2Cp);
        const uint4* gsrc = reinterpret_cast<const uint4*>(lrow - sh);
        if (b > a) bulk_g2s(srow + (a - gb), gsrc + a, (uint32_t)(b - a) * 16u, sm.bar);
        else asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(sm.bar)) : "memory");
      }
      const bool hd = gb == 0 && sh > 0;
      const bool tl = g_hi < ng && g_hi > 0 && g_hi >= gb && g_hi < ge;
      uint4 ph = make_uint4(~0u, ~0u, ~0u, ~0u), pt = ph;
      if (tid == 0 && hd) ph = load_group(0);
      if (tid == 32 && tl) pt = load_group(g_hi);
      mbar_wait(sm.bar, *bar_phase);
      *bar_phase ^= 1u;
      if (tid == 0 && hd) srow[0] = ph;
      if (tid == 32 && tl) srow[g_hi - gb] = pt;
      __syncthreads();
    }
    const int tg0 = tid * kNR;
    constexpr int kMW = (kNR + 3) / 4;
    uint32_t mk[kMW];
#pragma unroll
    for (int w = 0; w < kMW; ++w) mk[w] = 0u;
    uint32_t cnt = 0;
#pragma unroll
    for (int i = 0; i < kNR; ++i) {
      const int g = tg0 + i;
      if (g < ngc) {
        const uint4 r = srow[g];
        uint32_t m8;
        if (!kx) {
          const uint32_t lo = __byte_perm(v2_ge2(r.x, t2), v2_ge2(r.y, t2), 0x6420);
          const uint32_t hi = __byte_perm(v2_ge2(r.z, t2), v2_ge2(r.w, t2), 0x6420);
          m8 = (((lo & 0x08040201u) * 0x01010101u) >> 24) | ((((hi & 0x08040201u) * 0x01010101u) >> 20) & 0xF0u);
        } else {
          const Keys8 k8(r);
          m8 = 0;
#pragma unroll 1
          for (int j = 0; j < 8; ++j) m8 |= (k8.key(j) >= (uint32_t)th2 ? 1u : 0u) << j;
        }
        cnt += __popc(m8);
        mk[i / 4] |= m8 << (8 * (i % 4));
      }
    }
    uint32_t ctot;
    int c = cbase + (int)v2_scan<T>(cnt, warp_sums + 32 * (ws_par ^= 1u), &ctot);
#pragma unroll
    for (int w = 0; w < kMW; ++w) {
      uint32_t m = mk[w];
      while (m) {
        const int b = __ffs(m) - 1;
        m &= m - 1;
        const int i = 4 * w + (b >> 3), j = b & 7;
        const int g = tg0 + i;
        if (c < kV2Cand) {
          sm.cpos[c] = (uint32_t)(8 * (gb + g) - sh + j);
          sm.ckey[c] = (uint16_t)(keys2(srow16[8 * g + j]) & 0xFFFFu);
        }
        ++c;
      }
    }
    cbase += (int)ctot;
  }
  if (tid == 0) s_i[5] = cbase;  // s_cnt
  __syncthreads();
  return th2;
}

// Hot body, rare (out of line): round 1 found tau exactly, but the ties at tau straddle the
// ranks beyond the bands and the tie list (some rank holds more than kV2Tie). Every rank's ties at
// tau are among its candidates (tau >= every rank's theta: round 1 found it inside every fast
// window), so the global-id cut is a radix select over the candidate ties' ids: kTcBins-bin rounds
// of kTcBits id bits, top bits first, through the radix mailboxes (one slot per round, re-armed
// after the poll); the same words on every rank give the same cut. Returns this rank's quota: its
// ties with id <= the cut (the first ones in position order, since ids rise with position).
constexpr int kTcBits = 9, kTcBins = 1 << kTcBits;  // 512 bins: 12-bit counts, 2 per mailbox word
static_assert(kTcBins / 2 <= 256 && kV2RadR >= 3 && kV2Cand <= (int)kCountMax);
template <int T>
__device__ __noinline__ int v2_tiecut(const V2Params p, const V2Sm sm, const int nc, const int tau, const int need,
                                      const uint32_t tag, const int par, const int row, const int Gl,
                                      uint32_t* rx) {
  __shared__ int s_tc[2];  // 0 digit, 1 remaining need
  const int tid = threadIdx.x, W = p.W, IL = p.interleave;
  uint32_t* const hist = sm.hist;
  uint32_t ws_par = 0;
  auto gid_of = [&](int pos) -> uint32_t { return (uint32_t)(((pos / IL) * W + p.rank) * IL + pos % IL); };
  // id bits: every id of the row is < Gl + W * IL (the same bound on every rank)
  const uint32_t gmax = (uint32_t)Gl + (uint32_t)(W * IL) - 1u;
  const int nbits = 32 - __clz((int)max(gmax, 1u));
  const int rounds = (nbits + kTcBits - 1) / kTcBits;
  if (rounds > kV2RadR) { if (tid == 0) spin_fail(p.err, 43); return 0; }
  uint32_t prefix = 0;
  int left = need;
  const uint4 dead4 = make_uint4(0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu);
  __syncthreads();
  for (int rd = 0; rd < rounds; ++rd) {
    const int lo_bit = max(0, nbits - kTcBits * (rd + 1));
    const int hi_bit = nbits - kTcBits * rd;  // this round's digit: id bits [lo_bit, hi_bit)
    for (int b = tid; b < kTcBins; b += T) hist[b] = 0u;
    __syncthreads();
    for (int c = tid; c < nc; c += T) {
      if ((int)sm.ckey[c] != tau) continue;
      const uint32_t g = gid_of((int)sm.cpos[c]);
      if (rd > 0 && (g >> hi_bit) != prefix) continue;
      atomicAdd(&hist[(g >> lo_bit) & ((1u << (hi_bit - lo_bit)) - 1u)], 1u);
    }
    __syncthreads();
    const int64_t base = p.off_rad + (((int64_t)(par * p.cap + row) * kV2RadR + rd) * W) * 256;
    if (tid < kTcBins / 8) {  // 4 words (8 bins) per thread
      const int b = tid * 8;
      const uint4 v = make_uint4(pack2(tag, hist[b], hist[b + 1]), pack2(tag, hist[b + 2], hist[b + 3]),
                                 pack2(tag, hist[b + 4], hist[b + 5]), pack2(tag, hist[b + 6], hist[b + 7]));
      const int64_t off = base + (int64_t)p.rank * 256 + 4 * tid;
      if (p.mc) mc_st_v4(p.mc + off * 4, v);
      else
        for (int d = 0; d < W; ++d) st_v4(reinterpret_cast<uint32_t*>(p.peers[d]) + off, v);
    }
    for (int i = tid; i < W * (kTcBins / 8); i += T) {
      const int r = i / (kTcBins / 8), j = i % (kTcBins / 8);
      uint32_t* const src = p.box + base + (int64_t)r * 256 + 4 * j;
      const long long t0 = now_ns();
      uint4 v = ld_sys_v4(src);
      while (!tagged(v, tag)) {
        if (spin_late(p.spin_ns, t0)) { spin_fail(p.err, 44); break; }
        __nanosleep(128);
        v = ld_sys_v4(src);
      }
      *reinterpret_cast<uint4*>(rx + r * (kTcBins / 2) + 4 * j) = v;
      st_v4(src, dead4);
    }
    __syncthreads();
    // digit: ascending ids; thread t <-> bin t
    uint32_t cb = 0;
    if (tid < kTcBins)
      for (int r = 0; r < W; ++r) {
        const uint32_t w = rx[r * (kTcBins / 2) + (tid >> 1)];
        cb += (tid & 1) ? (w & 0xFFFu) : ((w >> 12) & 0xFFFu);
      }
    uint32_t tot;
    const uint32_t ex = v2_scan<T>(cb, sm.warp_sums + 32 * (ws_par ^= 1u), &tot);
    if (tid == 0) s_tc[0] = -1;
    __syncthreads();
    if (tid < kTcBins && ex < (uint32_t)left && ex + cb >= (uint32_t)left) { s_tc[0] = tid; s_tc[1] = left - (int)ex; }
    __syncthreads();
    if (s_tc[0] < 0) { if (tid == 0) spin_fail(p.err, 45); return 0; }
    prefix = (prefix << (hi_bit - lo_bit)) | (uint32_t)s_tc[0];
    left = s_tc[1];
    __syncthreads();
  }
  // prefix = the cut id: this rank's ties with id <= it
  int mine = 0;
  for (int c = tid; c < nc; c += T)
    mine += (int)sm.ckey[c] == tau && gid_of((int)sm.cpos[c]) <= prefix;
  if (tid == 0) s_tc[1] = 0;
  __syncthreads();
  for (int o = 16; o > 0; o >>= 1) mine += __shfl_xor_sync(0xffffffffu, mine, o);
  if ((tid & 31) == 0 && mine) atomicAdd(&s_tc[1], mine);
  __syncthreads();
  return s_tc[1];
}

// One row. kFast: the common case only, compact code (in-step every call fetches the
// executed code cold from L2; the rare paths inline made the hot path 2x slower there):
// a rank whose candidates or window fall outside the fast path sends a saturated
// message, which every rank reads as tier 3; any tier other than 1 returns 1 and the
// redo kernel runs the row again with the complete body at the next epoch, on
// every rank alike (the tier comes from the exchanged words).
template <bool kFast, int T>
// ep_in < 0: the row's epoch is read here (after the copy is issued). Returns 0, or
// 1 + the epoch used when the complete body must run the row.
__device__ __forceinline__ int v2_row(const V2Params& p, const V2Sm& sm, const int row, const int ep_in,
                                      uint32_t& bar_phase) {
  uint32_t* const hist = sm.hist;
  uint32_t* const cpos = sm.cpos;
  uint16_t* const ckey = sm.ckey;
  uint4* const srow = sm.srow;
  uint64_t& s_bar = *sm.bar;
  uint32_t* const rx = reinterpret_cast<uint32_t*>(srow);  // kV2MaxW * kV2Msg words
  int* const cslot = reinterpret_cast<int*>(srow) + kV2RxW;  // kV2Cand
  uint32_t* const msg = reinterpret_cast<uint32_t*>(srow) + kV2RxW + kV2Cand;  // kV2Msg
  uint32_t* const win = msg + kV2Msg;  // kV2Win
  uint32_t* const warp_sums = sm.warp_sums;
  uint32_t ws_par = 0;
#define V2WS (warp_sums + 32 * (ws_par ^= 1u))
  int* const s_i = sm.s_i;
  uint32_t* const s_u = sm.s_u;
  unsigned long long& s_pref = *sm.pref;
  long long* const s_dbg = sm.dbg;
  const int tid = threadIdx.x;
  constexpr int kV2T = T, kV2Warps = V2Cfg<T>::kWarps, kV2NR = V2Cfg<T>::kNR;
  constexpr int kV2PB = V2Cfg<T>::kPB, kV2FB = V2Cfg<T>::kFB;
  const int W = p.W, K = p.K, IL = p.interleave;
  auto log2_or = [](int x) -> int { return (x > 0 && (x & (x - 1)) == 0) ? __ffs(x) - 1 : -1; };
  const int bs_sh = log2_or(p.block_size), il_sh = log2_or(IL), w_sh = log2_or(W);
  // gid = ((pos / IL) * W + rank) * IL + pos % IL = (pos / IL) << (w_sh + il_sh) | rank << il_sh | pos % IL
  const int gid_sh = (il_sh >= 0 && w_sh >= 0) ? il_sh + w_sh : -1;
  // scalar slots
  int& s_tier = s_i[0];
  int& s_tau = s_i[1];
  int& s_need = s_i[2];
  int& s_q = s_i[3];
  int& s_gs = s_i[4];
  int& s_cnt = s_i[5];
  int& s_e = s_i[6];
  int& s_lo = s_i[7];
  int& s_bl = s_i[8];
  int& s_bh = s_i[9];
  int& s_n = s_i[10];
  int& s_th = s_i[11];
  uint32_t& s_valid = s_u[0];
  uint32_t& s_above = s_u[1];
  uint32_t& s_beyond = s_u[2];
  uint32_t& s_gt = s_u[4];
  uint32_t& s_sat = s_u[5];
  // dbg stamps: clock64 into SMEM (no global store on the row's chain); globaltimer only
  // at the row's start / end; flushed to global at the end
#define V2DBG(i)                                                                     \
  if (p.dbg && tid == 0) {                                                           \
    if ((i) == 0) {                                                                  \
      for (int z = 1; z < 30; ++z) s_dbg[z] = 0;                                     \
      p.dbg[(int64_t)row * kV2Dbg] = now_ns();                                       \
    }                                                                                \
    /* BAR.SYNC defers blocking to the next memory access: read SMEM and branch on */ \
    /* it first, so the stamp follows the barrier's completion */                    \
    if (*reinterpret_cast<volatile int*>(&s_i[15]) == 0x7FFFFFF1) __trap();         \
    s_dbg[i] = (long long)clock64();                                                 \
    if ((i) == 8) {                                                                  \
      p.dbg[(int64_t)row * kV2Dbg + 8] = now_ns();                                   \
      for (int z = 0; z < 30; ++z) p.dbg[(int64_t)row * kV2Dbg + 32 + z] = s_dbg[z]; \
    }                                                                                \
  }
#define V2STAT(i) \
  if (p.stats && tid == 0) atomicAdd(&p.stats[i], 1);

  {
    V2DBG(0);
    // SMEM front end: the row's groups (group g = positions [8g - sh0, +8), 16-byte
    // aligned) are bulk-copied into srow[g] before the row's lengths are known, by
    // kV2Cp warps at once; partial groups (head, tail, past lcols) are read key by key.
    const uint16_t* lrow = p.logits + (int64_t)row * p.lstride;
    const uintptr_t la = reinterpret_cast<uintptr_t>(lrow);
    const int head0 = (int)(((16u - (uint32_t)(la & 15u)) & 15u) >> 1);
    const int sh0 = (8 - head0) & 7;
    const int gs_lo = sh0 ? 1 : 0, gs_hi = min((p.lcols + sh0) >> 3, kV2NR * kV2T);
    if ((tid & 31) == 0 && (tid >> 5) < kV2Cp) {
      // the previous row's generic SMEM writes before the async-proxy copy
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      const int n = max(0, gs_hi - gs_lo), c = tid >> 5;
      const int a = gs_lo + (int)((int64_t)n * c / kV2Cp), b = gs_lo + (int)((int64_t)n * (c + 1) / kV2Cp);
      const uint4* gb = reinterpret_cast<const uint4*>(lrow - sh0);
      if (b > a) bulk_g2s(srow + a, gb + a, (uint32_t)(b - a) * 16u, &s_bar);
      else asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(&s_bar)) : "memory");
    }
    const int L = p.local_len[row];
    const int Gl = p.global_len[row];
    int* orow = p.out + (int64_t)row * p.ostride;
    const int req = p.req_of_row != nullptr ? p.req_of_row[row] : row / p.next_n;
    const int* btrow = p.block_table != nullptr ? p.block_table + (int64_t)req * p.bt_stride : nullptr;
    // power-of-two block size / interleave / world (the deployed shapes): shifts, no divides
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
      for (int i = tid; i < L; i += kV2T) orow[i] = slot_of(i);
      for (int i = L + tid; i < K; i += kV2T) orow[i] = -1;
      if (tid == 0) p.out_count[row] = L;
      __syncthreads();
      return 0;
    }
    if (Gl >= (1 << 24) && tid == 0) spin_fail(p.err, 13);
    const int ep = ep_in >= 0 ? ep_in : p.epoch[row];
    const uint32_t tag = (uint32_t)(ep % 255);
    const int par = ep & 1;
    V2STAT(0);
    const unsigned long long kl = (unsigned long long)K * (unsigned)L + (unsigned)(Gl - 1);
    const uint32_t q = max(1u, kl <= 0xFFFFFFFFull ? (uint32_t)kl / (uint32_t)Gl : (uint32_t)(kl / (unsigned)Gl));

    // ---- 1-2. keys -> sample bound theta -> candidates (every key >= theta with its
    // position), valid keys. Positions outside [0, L) read as NaN (key 0). Rows of at
    // most kV2NR x kV2T groups are read from the SMEM copy (fast path: short loops);
    // longer rows sample from global memory, then compact in batches of kV2RegG x 256 groups.
    for (int b = tid; b < kCoarseBins; b += kV2T) hist[b] = 0;
    if (tid == 0) { s_valid = 0; s_sat = 0; s_cnt = 0; s_beyond = 0; s_th = 0; s_i[14] = -1; }
    const int head = min(L, head0);
    const int sh = (8 - head) & 7;  // == sh0 when L >= 8
    const int ng = (L + sh + 7) >> 3;
    const int g_lo = sh ? 1 : 0, g_hi = (L + sh) >> 3;  // full groups [g_lo, g_hi)
    // the SMEM path (same group layout). kFast: rows longer than the SMEM row too, in chunks
    // of kV2NR x kV2T groups (the candidate pass walks them in order)
    constexpr int kCap = kV2NR * kV2T;
    const bool regp = L >= 8 && (kFast || ng <= kCap);
    auto load_group = [&](int g) -> uint4 {
      if (g >= g_lo && g < g_hi) return __ldg(reinterpret_cast<const uint4*>(lrow - sh) + g);
      if (g < 0 || g >= ng) return make_uint4(~0u, ~0u, ~0u, ~0u);
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
    // group g of the SMEM path (g < ng; the partial head / tail groups are patched in below)
    auto sgroup = [&](int g) -> uint4 { return srow[g]; };
    // visit groups [ga, gb): fn(position of the group's first key, its 8 raw bf16 values);
    // every thread must call it (it ends with a CTA barrier)
    auto for_keys = [&](int ga, int gb, auto&& fn) {
      for (int g = ga + tid; g < gb; g += kV2T) fn(8 * g - sh, load_group(g));
      __syncthreads();
    };
    // partial head / tail groups (masked, key by key), loaded while the copy lands
    const bool phead = regp && sh > 0, ptail = regp && g_hi < ng && g_hi > 0 && g_hi < kCap;
    uint4 pg = make_uint4(~0u, ~0u, ~0u, ~0u);
    if (tid == 0 && phead) pg = load_group(0);
    if (tid == 32 && ptail) pg = load_group(g_hi);
    // (sampling from global memory while the copy lands measured slower: r24 20.0 vs 18.5 us)
    mbar_wait(&s_bar, bar_phase);
    bar_phase ^= 1u;
    if (tid == 0 && phead) srow[0] = pg;
    if (tid == 32 && ptail) srow[g_hi] = pg;
    __syncthreads();  // histogram zeroed, partial groups in place
    V2DBG(9);
    if (K31_LIKELY(regp)) {
      // sample: one group in kV2SampleStride (spread over the 8 residues)
#pragma unroll 1
      for (int g = tid * kV2SampleStride + (tid & 7); g < min(ng, kCap); g += kV2T * kV2SampleStride) {
        const Keys8 k8(sgroup(g));
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const uint32_t k = k8.key(j);
          if (k) atomicAdd(&hist[k >> 4], 1u);
        }
      }
      if constexpr (kFast) {
        // a row longer than the SMEM row: its later chunks are sampled from global memory
        if (K31_RARE(ng > kCap))
#pragma unroll 1
          for (int g = kCap + tid * kV2SampleStride + (tid & 7); g < ng; g += kV2T * kV2SampleStride) {
            const Keys8 k8(load_group(g));
#pragma unroll 1
            for (int j = 0; j < 8; ++j) {
              const uint32_t k = k8.key(j);
              if (k) atomicAdd(&hist[k >> 4], 1u);
            }
          }
      }
    } else if constexpr (!kFast) {
      // sample: one 8-key group in kV2SampleStride, all loads in flight before any use
      const int step = kV2T * kV2SampleStride;
      for (int g0 = tid * kV2SampleStride; g0 < ng; g0 += 4 * step) {
        uint4 s4[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) s4[i] = load_group(g0 + i * step);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const Keys8 k8(s4[i]);
#pragma unroll 1
          for (int j = 0; j < 8; ++j) {
            const uint32_t k = k8.key(j);
            if (k) atomicAdd(&hist[k >> 4], 1u);
          }
        }
      }
    }
    __syncthreads();
    V2DBG(10);
    {
      // theta: the bottom of the coarse bin where the sample's suffix count first reaches
      // its share of q + 512 keys (8-key groups of adjacent tokens correlate: q + 256 left
      // fewer than q candidates on ~3 of 1M rows, each costing the DCP group the complete body)
      const uint32_t want = (q + 512u + kV2SampleStride - 1) / kV2SampleStride;
      const int bb = kV2T - 1 - tid;
      uint32_t v = 0;
#pragma unroll
      for (int i = 0; i < kV2PB; ++i) v += hist[bb * kV2PB + i];
      uint32_t tot;
      const uint32_t ab = v2_scan<T>(v, V2WS, &tot);
      if (ab < want && ab + v >= want) {
        uint32_t run = ab;
        for (int i = kV2PB - 1; i >= 0; --i) {
          run += hist[bb * kV2PB + i];
          if (run >= want) {
            s_th = (bb * kV2PB + i) * 16;
            s_u[13] = run;                       // sample keys >= theta's coarse bin
            s_u[14] = hist[bb * kV2PB + i];      // sample keys in that bin
            break;
          }
        }
      }
      // the sample's estimate of the q-th key (its coarse bin): the fine histogram is
      // centered there, so rows with a wide upper tail (q-th key more than kV2Fine keys
      // above theta) need no rebase pass
      const uint32_t want_e = max(1u, q / kV2SampleStride);
      if (ab < want_e && ab + v >= want_e) {
        uint32_t run = ab;
        for (int i = kV2PB - 1; i >= 0; --i) {
          run += hist[bb * kV2PB + i];
          if (run >= want_e) { s_i[14] = (bb * kV2PB + i) * 16; break; }
        }
      }
      __syncthreads();
      if (tid == 0 && s_th == 0) { s_th = 1; s_u[13] = 0u; }  // short / sparse sample: every valid key
      __syncthreads();
      // fine theta: when the sample predicts more candidates than the list holds
      // (kV2SampleStride x the sample keys >= theta), theta's coarse bin (16 keys) is
      // resolved from the same sample groups at key resolution and theta moves up to the
      // largest key whose sample suffix count still reaches `want`. theta only bounds
      // the candidate set; every later check is unchanged (exact either way).
      const int cb = s_th >> 4;
      if (s_th > 1 && s_u[13] * (uint32_t)kV2SampleStride > (uint32_t)kV2Cand) {
        // the bin's 16 key counters in warp_sums (free between CTA scans)
        if (tid < 16) warp_sums[tid] = 0u;
        __syncthreads();
        auto fine_add = [&](const uint4& r) {
          const Keys8 k8(r);
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const uint32_t k = k8.key(j);
            if (k && (int)(k >> 4) == cb) atomicAdd(&warp_sums[k & 15u], 1u);
          }
        };
        if (K31_LIKELY(regp)) {
#pragma unroll 1
          for (int g = tid * kV2SampleStride + (tid & 7); g < min(ng, kCap); g += kV2T * kV2SampleStride)
            fine_add(sgroup(g));
          if constexpr (kFast) {
            if (K31_RARE(ng > kCap))
#pragma unroll 1
              for (int g = kCap + tid * kV2SampleStride + (tid & 7); g < ng; g += kV2T * kV2SampleStride)
                fine_add(load_group(g));
          }
        } else if constexpr (!kFast) {
          const int step = kV2T * kV2SampleStride;
#pragma unroll 1
          for (int g0 = tid * kV2SampleStride; g0 < ng; g0 += 4 * step)
            for (int i = 0; i < 4; ++i) fine_add(load_group(g0 + i * step));
        }
        __syncthreads();
        if (tid == 0) {
          uint32_t run = s_u[13] - s_u[14];  // sample keys above theta's coarse bin
          for (int t = 15; t >= 0; --t) {
            run += warp_sums[t];
            if (run >= want) { s_th = max(s_th, (cb << 4) + t); break; }
          }
        }
        __syncthreads();
      }
    }
    int th = s_th;
    V2DBG(1);
    // the SMEM path's candidate pass: every key >= thv with its position, in position
    // order; returns this thread's valid-key count
    auto cand_regp = [&](int thv, int gb, int ngc, int cbase) -> uint32_t {
      // candidates in POSITION order: thread t owns the contiguous groups [kV2NR t, +kV2NR)
      // (16-byte SMEM reads at an odd 16-byte pitch: conflict free), so one CTA scan of the
      // per-thread counts places every candidate. The group tests are independent.
      const int tg0 = tid * kV2NR;
      const uint32_t t2 = v2_th_bf16(thv);
      const bool kx = thv == 0x8000;
      constexpr int kMW = (kV2NR + 3) / 4;
      uint32_t mk[kMW];  // 8-bit key masks of the groups, 4 per word
#pragma unroll
      for (int w = 0; w < kMW; ++w) mk[w] = 0u;
      uint32_t cnt = 0;
      uint32_t nv = 0;
#pragma unroll
      for (int i = 0; i < kV2NR; ++i) {
        const int g = tg0 + i;
        if (g < ngc) {
          const uint4 r = sgroup(g);
          // (one NaN accumulator per thread + a recount measured no faster: the partial
          // groups' threads recount on the scan's critical path)
          nv += v2_is_nan2(v2_max_nan(r)) ? v2_nvalid(r) : 8u;
          uint32_t m8;
          if (!kx) {
            // 0xFFFF / 0 per key -> one byte per key (prmt) -> one bit per key (mask, mul)
            const uint32_t lo = __byte_perm(v2_ge2(r.x, t2), v2_ge2(r.y, t2), 0x6420);  // keys 0-3
            const uint32_t hi = __byte_perm(v2_ge2(r.z, t2), v2_ge2(r.w, t2), 0x6420);  // keys 4-7
            m8 = (((lo & 0x08040201u) * 0x01010101u) >> 24) | ((((hi & 0x08040201u) * 0x01010101u) >> 20) & 0xF0u);
          } else {
            const Keys8 k8(r);
            m8 = 0;
#pragma unroll 1
            for (int j = 0; j < 8; ++j)
              m8 |= (k8.key(j) >= (uint32_t)thv ? 1u : 0u) << j;
          }
          cnt += __popc(m8);
          mk[i / 4] |= m8 << (8 * (i % 4));
        }
      }
      V2DBG(24);
      uint32_t ctot;
      int c = cbase + (int)v2_scan<T>(cnt, V2WS, &ctot);
      V2DBG(25);
      if (tid == 0) s_cnt = cbase + (int)ctot;
      const uint16_t* srow16 = reinterpret_cast<const uint16_t*>(srow);
      // (the fine histogram counted here instead of in its own pass measured slower:
      // same-bin atomics serialize the emit loop; r24 21.2 vs 18.3 us)
#pragma unroll
      for (int w = 0; w < kMW; ++w) {
        uint32_t m = mk[w];
        while (m) {
          const int b = __ffs(m) - 1;
          m &= m - 1;
          const int i = 4 * w + (b >> 3), j = b & 7;
          const int g = tg0 + i;
          if (c < kV2Cand) {
            const uint32_t raw = srow16[8 * g + j];
            cpos[c] = (uint32_t)(8 * (gb + g) - sh + j);
            ckey[c] = (uint16_t)(keys2(raw) & 0xFFFFu);
          }
          ++c;
        }
      }
      V2DBG(26);
      return nv;
    };
    {
      uint32_t nv = 0;
      if (regp) {
#pragma unroll 1
        for (int gb = 0;; gb += kCap) {
          int cb = 0;
          if (K31_RARE(gb > 0)) {
            // the next chunk of a long row into the SMEM row (the previous one is read)
            __syncthreads();
            cb = s_cnt;
            const int ge = min(ng, gb + kCap), gfull = min((p.lcols + sh0) >> 3, ge);
            if ((tid & 31) == 0 && (tid >> 5) < kV2Cp) {
              asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
              const int n = max(0, gfull - gb), c = tid >> 5;
              const int a = gb + (int)((int64_t)n * c / kV2Cp), b = gb + (int)((int64_t)n * (c + 1) / kV2Cp);
              const uint4* gsrc = reinterpret_cast<const uint4*>(lrow - sh0);
              if (b > a) bulk_g2s(srow + (a - gb), gsrc + a, (uint32_t)(b - a) * 16u, &s_bar);
              else asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(&s_bar)) : "memory");
            }
            const bool tl = g_hi < ng && g_hi >= gb && g_hi < ge;
            uint4 pt = make_uint4(~0u, ~0u, ~0u, ~0u);
            if (tid == 32 && tl) pt = load_group(g_hi);
            mbar_wait(&s_bar, bar_phase);
            bar_phase ^= 1u;
            if (tid == 32 && tl) srow[g_hi - gb] = pt;
            __syncthreads();
          }
          nv += cand_regp(th, gb, min(kCap, ng - gb), cb);
          if (!kFast || gb + kCap >= ng) break;
        }
      }
      const int nfull = regp ? 0 : g_hi - g_lo;
      const int npart = regp ? 0 : ng - max(0, nfull);
      for (int b0 = 0; !kFast && K31_RARE(!regp) && b0 < max(0, nfull) + (npart > 0 ? kV2RegG * kV2T : 0); b0 += kV2RegG * kV2T) {
        const bool part = b0 >= max(0, nfull);
        const int g0 = part ? 0 : g_lo + b0;
        uint4 rb[kV2RegG];
        if (!part) {
#pragma unroll
          for (int u = 0; u < kV2RegG; ++u) {
            const int g = g0 + u * kV2T + tid;
            rb[u] = g < g_hi ? __ldg(reinterpret_cast<const uint4*>(lrow - sh) + g)
                             : make_uint4(~0u, ~0u, ~0u, ~0u);
          }
        } else {
          // partial groups: group 0 (head, sh > 0) and group g_hi (tail) by threads 0 / 1
          const int g = tid == 0 ? (sh ? 0 : -1) : tid == 1 ? (g_hi < ng && g_hi > 0 ? g_hi : (g_hi < ng && !sh ? g_hi : -1)) : -1;
          rb[0] = g >= 0 ? load_group(g) : make_uint4(~0u, ~0u, ~0u, ~0u);
#pragma unroll
          for (int u = 1; u < kV2RegG; ++u) rb[u] = make_uint4(~0u, ~0u, ~0u, ~0u);
        }
        uint32_t cnt = 0, hit = 0;  // hit: groups holding a key >= theta
#pragma unroll
        for (int u = 0; u < kV2RegG; ++u) {
          // one NaN-propagating max: no NaN -> it is the group's max (its key bounds the
          // group); a NaN (rare) -> count the valid keys, bound from the full test
          const uint32_t mx = v2_max_nan(rb[u]);
          bool any;
          if (v2_is_nan2(mx)) {
            const Keys8 k8(rb[u]);
#pragma unroll 1
            for (int j = 0; j < 8; ++j) nv += k8.key(j) != 0u;
            any = (int)max_key_bound(rb[u]) >= th;
          } else {
            nv += 8;
            const uint32_t km = keys2(mx);
            // +1: max() may return -0 over +0 (equal as floats; keys 0x7FFF < 0x8000)
            any = (int)::max(km & 0xFFFFu, km >> 16) + 1 >= th;
          }
          if (any) {
            hit |= 1u << u;
            const Keys8 k8(rb[u]);
#pragma unroll 1
            for (int j = 0; j < 8; ++j) cnt += k8.key(j) >= (uint32_t)th;
          }
        }
        uint32_t incl = cnt;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
          const uint32_t y = __shfl_up_sync(0xffffffffu, incl, o);
          if ((tid & 31) >= o) incl += y;
        }
        int base = 0;
        if ((tid & 31) == 31 && incl) base = atomicAdd(&s_cnt, (int)incl);
        int c = __shfl_sync(0xffffffffu, base, 31) + (int)(incl - cnt);
        if (cnt) {
#pragma unroll
          for (int u = 0; u < kV2RegG; ++u) {
            if (!((hit >> u) & 1u)) continue;
            const Keys8 k8(rb[u]);
            const int p0 = part ? 8 * (tid == 0 ? 0 : g_hi) - sh : 8 * (g0 + u * kV2T + tid) - sh;
#pragma unroll 1
            for (int j = 0; j < 8; ++j) {
              const uint32_t k = k8.key(j);
              if (k >= (uint32_t)th) {
                if (c < kV2Cand) { cpos[c] = (uint32_t)(p0 + j); ckey[c] = (uint16_t)k; }
                ++c;
              }
            }
          }
        }
      }
      for (int o = 16; o > 0; o >>= 1) nv += __shfl_xor_sync(0xffffffffu, nv, o);
      if ((tid & 31) == 0 && nv) atomicAdd(&s_valid, nv);
    }
    __syncthreads();
    int nc = s_cnt;
    if constexpr (kFast) {
      // the sample bound failed (dense bins): re-threshold from exact counts, stay in round 1
      if (K31_RARE(regp && (nc > kV2Cand || (uint32_t)nc < q))) {
        V2STAT(6);
        const int t2 = v2_retheta<T>(p, sm, lrow, L, sh, ng, g_lo, g_hi, th, nc, q, &bar_phase);
        if (t2 >= 1) {
          th = t2;
          nc = s_cnt;
        }
      }
    }
    if (!kFast && K31_RARE(regp && (nc > kV2Cand || (uint32_t)nc < q))) {
      // the sample bound failed (tie-heavy or skewed rows): exact counts per key over
      // 2048-key ranges of the SMEM row (from theta up on overflow, down otherwise) give a
      // threshold admitting about q + 256 keys (at least q, at most kV2Cand); then a second
      // candidate pass. Ranges that cannot hold it leave the row to the full passes.
      V2STAT(6);
      const uint32_t want = q + 256u;
      int base = nc > kV2Cand ? th : max(1, th - kV2Fine);
      int th2 = -1;
      for (int it = 0; it < 4 && th2 < 0 && base >= 1 && base < 65536; ++it) {
        for (int b = tid; b < kV2Fine; b += kV2T) hist[b] = 0;
        if (tid == 0) { s_beyond = 0; s_i[12] = -1; s_i[13] = -1; }
        __syncthreads();
        uint32_t bey = 0;
        for (int g = tid; g < ng; g += kV2T) {
          const Keys8 k8(srow[g]);
#pragma unroll 1
          for (int j = 0; j < 8; ++j) {
            const int u = (int)k8.key(j) - base;
            if (u >= kV2Fine) ++bey;
            else if (u >= 0 && k8.key(j)) atomicAdd(&hist[u], 1u);
          }
        }
        for (int o = 16; o > 0; o >>= 1) bey += __shfl_xor_sync(0xffffffffu, bey, o);
        if ((tid & 31) == 0 && bey) atomicAdd(&s_beyond, bey);
        __syncthreads();
        const int f0 = (kV2T - 1 - tid) * kV2FB;
        uint32_t hv[kV2FB];
        const uint32_t hs = v2_ldbins(hist + f0, hv);
        uint32_t tot;
        const uint32_t sb = s_beyond;
        uint32_t run = v2_scan<T>(hs, V2WS, &tot) + sb;
#pragma unroll
        for (int i = kV2FB - 1; i >= 0; --i) {  // run: keys >= base + f0 + i
          const uint32_t nx = run;
          run += hv[i];
          if (run >= want && nx < want) { s_i[12] = base + f0 + i; s_u[8] = run; }
          if (run >= q && nx < q) { s_i[13] = base + f0 + i; s_u[9] = run; }
        }
        __syncthreads();
        const uint32_t s_all = tot + sb;  // keys >= base
        if (s_i[12] >= 0 && s_u[8] <= (uint32_t)kV2Cand) th2 = s_i[12];
        else if (s_i[13] >= 0 && s_u[9] <= (uint32_t)kV2Cand) th2 = s_i[13];
        else if (s_i[13] < 0 && sb >= q) base += kV2Fine;      // the q-th key lies above
        else if (s_all < q) base = base > 1 ? max(1, base - kV2Fine) : 0;  // below
        else break;                                            // a tie wider than kV2Cand
        __syncthreads();
      }
      if (tid == 0) s_beyond = 0;  // the fine histogram's count again
      if (th2 >= 1) {
        th = th2;
        cand_regp(th, 0, ng, 0);
        __syncthreads();
        nc = s_cnt;
      }
    }
    V2DBG(2);
    if (p.dbg && tid == 0) p.dbg[(int64_t)row * kV2Dbg + 31] = nc;
    // ---- 3. fine histogram above theta from the candidates -> e, window, band
    bool fast = nc <= kV2Cand && (uint32_t)nc >= q;
    for (int b = tid; b < kV2Fine; b += kV2T) hist[b] = 0;
    for (int b = tid; b < kV2Win; b += kV2T) win[b] = 0;
    if (tid == 0) s_e = -1;
    __syncthreads();
    // fast path: one suffix scan of the fine histogram gives e, the window, the count above
    // it and the window prefix P, so the band pick needs no further scan
    bool pdone = false;
    int ew = -1;
    uint32_t wtot = 0;
    // the fine histogram's base key: theta, or kV2Fine / 2 below the estimated q-th key
    int fb = s_i[14] >= 0 ? max(th, s_i[14] - kV2Fine / 2) : th;
    if (fast) {
      // thread t owns fine keys [kV2FB (kV2T - 1 - t), + kV2FB); Sx[b]: candidates with
      // fine key >= b (beyond included); the row buffer is free until the round
      uint32_t* Sx = reinterpret_cast<uint32_t*>(cslot);
      const int f0 = (kV2T - 1 - tid) * kV2FB;
      for (int pass = 0; pass < 2; ++pass) {
        uint32_t bey = 0;
        for (int c = tid; c < nc; c += kV2T) {
          const int u = (int)ckey[c] - fb;
          if (u >= kV2Fine) ++bey;
          else if (u >= 0) atomicAdd(&hist[u], 1u);
        }
        for (int o = 16; o > 0; o >>= 1) bey += __shfl_xor_sync(0xffffffffu, bey, o);
        if ((tid & 31) == 0 && bey) atomicAdd(&s_beyond, bey);
        __syncthreads();
        V2DBG(12);
        uint32_t hv[kV2FB];
        const uint32_t hs = v2_ldbins(hist + f0, hv);
        uint32_t tot;
        uint32_t run = v2_scan<T>(hs, V2WS, &tot) + s_beyond;
#pragma unroll
        for (int i = kV2FB - 1; i >= 0; --i) {
          run += hv[i];
          Sx[f0 + i] = run;
          if (run >= q && run - hv[i] < q) s_e = fb + f0 + i;
        }
        if (tid == 0) Sx[kV2Fine] = s_beyond;
        __syncthreads();
        if (K31_LIKELY(pass == 1 || s_e >= 0)) break;
        // the q-th key lies outside the fine range: above it (a wide upper tail), or below
        // it when the sample's estimate of the q-th key was high (the range starts kV2Fine / 2
        // under that estimate, above theta). Every candidate is >= theta and nc >= q, so
        // coarse counts of the candidates (32 keys per bin, the whole key range) locate it;
        // rebase there (pass 1 then finds e in [fb, fb + kV2Fine))
        V2STAT(7);
        for (int x = tid; x < kV2Fine; x += kV2T) hist[x] = 0;
        if (tid == 0) s_i[12] = -1;
        __syncthreads();
        if (tid == 0) s_beyond = 0;
        for (int c = tid; c < nc; c += kV2T) atomicAdd(&hist[ckey[c] >> 5], 1u);
        __syncthreads();
        uint32_t gv[kV2FB];
        const uint32_t gs = v2_ldbins(hist + f0, gv);
        uint32_t crun = v2_scan<T>(gs, V2WS, &tot);
#pragma unroll
        for (int i = kV2FB - 1; i >= 0; --i) {
          crun += gv[i];
          if (crun >= q && crun - gv[i] < q) s_i[12] = (f0 + i) * 32;
        }
        __syncthreads();
        fb = max(th, s_i[12] - kV2Fine / 2);  // room for the window below the q-th key
        for (int x = tid; x < kV2Fine; x += kV2T) hist[x] = 0;
        __syncthreads();
      }
      V2DBG(13);
      fast = s_e >= 0;
      if (fast) {
        const int e = s_e;
        const int lo = min(max(fb, e - kV2Win / 2), fb + kV2Fine - kV2Win);
        const int b0 = lo - fb;
        const uint32_t sa = Sx[b0 + kV2Win];
        if (tid < kV2Win) {
          win[tid] = hist[b0 + tid];
          hist[kV2Fine + tid] = Sx[b0] - Sx[b0 + tid];  // P[u]: window keys below u
        }
        if (tid == 0) { s_above = sa; s_lo = lo; }
        pdone = true;
        ew = e - lo;
        wtot = Sx[b0] - sa;
      }
    }
    if (K31_RARE(!fast)) {
      if constexpr (kFast) {
        // tier 3 on every rank, then R2. The message carries a valid lower bound only: an
        // empty window just below theta and the exact count of keys >= theta above it.
        if (tid == 0) {
          s_sat = 1u;
          s_lo = min(max(0, th - kV2Win), 65536 - kV2Win);
          s_above = s_lo + kV2Win == th ? (uint32_t)s_cnt : 0u;
        }
      } else {
      V2STAT(5);
      // the same message from passes over every key: coarse histogram, the coarse bin of
      // the q-th key, the window around it
      for (int b = tid; b < kCoarseBins; b += kV2T) hist[b] = 0;
      __syncthreads();
      for_keys(0, ng, [&](int, const uint4& r) {
          const uint32_t valid = 0xFFu;
        const Keys8 k8(r);
#pragma unroll 1
        for (int j = 0; j < 8; ++j) {
          const uint32_t k = k8.key(j);
          if (((valid >> j) & 1u) && k) atomicAdd(&hist[k >> 4], 1u);
        }
      });
      const int bb = kV2T - 1 - tid;
      uint32_t v = 0;
#pragma unroll
      for (int i = 0; i < kV2PB; ++i) v += hist[bb * kV2PB + i];
      uint32_t tot;
      const uint32_t ab = v2_scan<T>(v, V2WS, &tot);
      if (tid == 0) s_e = 0;
      __syncthreads();
      if (ab < q && ab + v >= q) {
        uint32_t run = ab;
        for (int i = kV2PB - 1; i >= 0; --i) {
          run += hist[bb * kV2PB + i];
          if (run >= q) { s_e = (bb * kV2PB + i) * 16; break; }
        }
      }
      __syncthreads();
      const int lo = min(max(0, s_e - 112), 65536 - kV2Win);
      uint32_t a = 0;
      for (int b = (lo + kV2Win) / 16 + tid; b < kCoarseBins; b += kV2T) a += hist[b];
      for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
      if (tid == 0) { s_above = 0; s_lo = lo; }
      __syncthreads();
      if ((tid & 31) == 0 && a) atomicAdd(&s_above, a);
      for_keys(0, ng, [&](int, const uint4& r) {
          const uint32_t valid = 0xFFu;
        if ((int)max_key_bound(r) < lo) return;
        const Keys8 k8(r);
#pragma unroll 1
        for (int j = 0; j < 8; ++j) {
          const uint32_t k = k8.key(j), u = k - (uint32_t)lo;
          if (((valid >> j) & 1u) && k && u < (uint32_t)kV2Win) atomicAdd(&win[u], 1u);
        }
      });
      }
    }
    __syncthreads();
    V2DBG(14);
    const int lo = s_lo;
    // band around the exact local q-th key: [bl, bh] (window-relative), <= kV2Band keys
    {
      int e;
      uint32_t tot;
      if (pdone) {  // e, P and the window total from the suffix scan
        if (tid < kV2Win && win[tid] > kCountMax) atomicOr(&s_sat, 1u);
        if (tid == 0) { s_bl = 1; s_bh = 0; s_n = 0; }
        e = ew;
        tot = wtot;
      } else {
        const int u = kV2Win - 1 - tid;  // thread t < 256 <-> window key 255 - t
        const uint32_t c = u >= 0 ? win[u] : 0u;
        if (c > kCountMax) atomicOr(&s_sat, 1u);
        const uint32_t ab = v2_scan<T>(c, V2WS, &tot) + s_above;
        if (tid == 0) { s_e = -1; s_bl = 1; s_bh = 0; s_n = 0; }
        __syncthreads();
        if (ab < q && ab + c >= q) s_e = u;
        if (u >= 0) hist[kV2Fine + u] = tot - (ab - s_above) - c;  // P[u]: window keys below u
        __syncthreads();
        e = s_e;
      }
      // widest symmetric band around e with <= kV2Band keys: the test is monotone in the
      // half-width h, so the count of passing h in [0, 256) gives it (one barrier)
      const bool bcand = e >= 0 && win[e] <= (uint32_t)kV2Band;
      const uint32_t* P = hist + kV2Fine;
      auto Pf = [&](int x) -> uint32_t { return x >= kV2Win ? tot : P[x]; };
      bool bok = false;
      if (bcand && tid < kV2Win) {
        const int a = max(0, e - tid), b = min(kV2Win - 1, e + tid);
        bok = Pf(b + 1) - Pf(a) <= (uint32_t)kV2Band;
      }
      const int lo_h = __syncthreads_count(bok) - 1;
      if (tid == 0 && bcand) {
        int a = max(0, e - lo_h), b = min(kV2Win - 1, e + lo_h);
        for (int it = 0; it < 4 && b + 1 < kV2Win && Pf(b + 2) - Pf(a) <= (uint32_t)kV2Band; ++it) ++b;
        for (int it = 0; it < 4 && a > 0 && Pf(b + 1) - Pf(a - 1) <= (uint32_t)kV2Band; ++it) --a;
        s_bl = a;
        s_bh = b;
        s_n = (int)(Pf(b + 1) - Pf(a));
      }
      __syncthreads();
    }
    V2DBG(15);
    const int bl = s_bl, bh = s_bh;
    const int nb = bl <= bh ? s_n : 0;
    // band ids grouped by key in key order (ids of one key in no particular order):
    // msg[kV2Fix + (band keys below k) + i]
    uint32_t* bcur = hist + kV2Fine + kV2Win;
    if (tid < kV2Win) bcur[tid] = 0;
    __syncthreads();
    if (nb > 0) {
      const uint32_t* P = hist + kV2Fine;
      const int k0 = lo + bl, k1 = lo + bh;
      const uint32_t p0 = P[bl];
      auto add = [&](int k, int pos) {
        const int u = k - lo;
        const uint32_t i = P[u] - p0 + atomicAdd(&bcur[u], 1u);
        if (i < (uint32_t)kV2Band) msg[kV2Fix + i] = tagw(tag, gid_of(pos));
      };
      if (fast) {
        for (int c = tid; c < nc; c += kV2T) {
          const int k = ckey[c];
          if (k >= k0 && k <= k1) add(k, (int)cpos[c]);
        }
      } else {
        for_keys(0, ng, [&](int i0, const uint4& r) {
          const uint32_t valid = 0xFFu;
          if ((int)max_key_bound(r) < k0) return;
          const Keys8 k8(r);
#pragma unroll 1
          for (int j = 0; j < 8; ++j) {
            const int k = (int)k8.key(j);
            if (((valid >> j) & 1u) && k && k >= k0 && k <= k1) add(k, i0 + j);
          }
        });
      }
    }
    __syncthreads();
    V2DBG(16);
    if (tid < kV2Win / 2) msg[kV2Hdr + tid] = pack2(tag, win[2 * tid], win[2 * tid + 1]);
    if (tid == 0) {
      msg[0] = tagw(tag, (uint32_t)lo | (s_sat ? (1u << 16) : 0u));
      msg[1] = tagw(tag, min(s_above, 0xFFFFFFu));
      msg[2] = tagw(tag, nb > 0 ? ((1u << 16) | ((uint32_t)bh << 8) | (uint32_t)bl) : 0u);
      msg[3] = tagw(tag, (uint32_t)nb);
      msg[4] = tagw(tag, min(s_valid, 0xFFFFFFu));
      // for the CG rounds: the exact q-th key and theta
      msg[5] = tagw(tag, pdone ? ((uint32_t)s_e | (1u << 16)) : 0u);
      msg[6] = tagw(tag, (uint32_t)th);
      msg[7] = tagw(tag, 0u);
    }
    // the whole slot goes out (unused band words tagged), so one poll pass reads it
    const int words = kV2Msg;
    for (int j = kV2Fix + nb + tid; j < kV2Msg; j += kV2T) msg[j] = tagw(tag, 0u);
    __syncthreads();
    V2DBG(3);
    // ---- 4. round 1: send, then poll every rank's fixed part and band
    const int64_t r1row = p.off_r1 + ((int64_t)(par * p.cap + row) * W) * kV2Msg;
    auto send = [&](const uint32_t* m, int nw, int64_t base) {
      for (int j = tid * 4; j < nw; j += kV2T * 4) {
        const uint4 v = make_uint4(m[j], m[j + 1], m[j + 2], m[j + 3]);
        const int64_t off = base + j;
        if (p.mc) mc_st_v4(p.mc + off * 4, v);
        else
          for (int d = 0; d < W; ++d) st_v4(reinterpret_cast<uint32_t*>(p.peers[d]) + off, v);
      }
    };
    auto poll_words = [&](const uint32_t* src, uint32_t* dst, int code) {
      const long long t0 = now_ns();
      uint4 v = ld_sys_v4(src);
      while (!tagged(v, tag)) {
        if (spin_late(p.spin_ns, t0)) { spin_fail(p.err, code); break; }
        __nanosleep(128);  // spinning warps steal issue slots / L2 from co-resident rows
        v = ld_sys_v4(src);
      }
      *reinterpret_cast<uint4*>(dst) = v;
    };
    send(msg, words, r1row + (int64_t)p.rank * kV2Msg);
    if (tid == 0) { s_tau = -1; s_need = 0; s_q = 0; s_gs = -1; s_tier = 0; }
    // in the round's shadow: the KV slot of every (position-sorted) candidate
    if (fast && regp)
      for (int c = tid; c < nc; c += kV2T) cslot[c] = slot_of((int)cpos[c]);
    V2DBG(28);
    for (int i = tid; i < W * (kV2Msg / 4); i += kV2T) {
      const int r = i / (kV2Msg / 4), j = (i % (kV2Msg / 4)) * 4;
      poll_words(p.box + r1row + (int64_t)r * kV2Msg + j, rx + r * kV2Msg + j, 31);
    }
    __syncthreads();
    V2DBG(4);
    if (p.dbg && tid == 0) p.dbg[(int64_t)row * kV2Dbg + 30] = nb;
    V2DBG(27);

    // ---- resolve. tier: 1 resolved (quota s_q), 2 tie round, 3 radix (id cut s_gs),
    // 4 every valid key, 5 cut from the bands. Every thread derives the tier from the
    // exchanged words; three barriers in the common case.
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
      V2DBG(19);
      if (rt == 0) {
        // thread t < 256: a = the keys above the common range [mx, mn) at window offset t
        // (thread 0 also the counts above every window), b = the common key mn - 1 - t;
        // in the order (all a, then b from the top) the thread where the prefix crosses K
        // holds tau
        // (kV2T >= 512: thread t >= 256 carries b of t - 256, one scan)
        constexpr bool kOne = kV2T >= 2 * kV2Win;
        const int tb = kOne ? tid - kV2Win : tid;  // this thread's b slot
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
          ex = v2_scan<T>(a + v, V2WS, &tot);  // a and v live on disjoint threads
        } else {
          v2_scan<T>(a, V2WS, &tota);
          ex = v2_scan<T>(v, V2WS, &tot) + tota;
        }
        if (tb >= 0 && tb < kV2Win && ex < (uint32_t)K && ex + v >= (uint32_t)K) {
          s_tau = mn - 1 - tb;
          s_need = K - (int)ex;
        }
        __syncthreads();
        tau = s_tau;
        need = s_need;
        if (tau < 0) rt = 3;  // the K-th key lies above or below the common range
        V2DBG(20);
      }
      if (rt == 0) {
        // warp r < W: rank r's tie count at tau, whether its band holds them, and their
        // offset in its band (its band keys in [bl_r, tau))
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
        V2DBG(21);
        int tot_t = 0, nr = 0;
        bool band_ok = true, fit = true;
        for (int r = 0; r < W; ++r) {
          tot_t += s_tc[r];
          nr += s_tc[r] > 0;
          if (s_tc[r] > 0 && !s_tok[r]) band_ok = false;
          fit = fit && s_tc[r] <= kV2Tie;
        }
        const int mine = s_tc[p.rank];
        if (tot_t == need) { rt = 1; qv = mine; }
        else if (nr == 1) { rt = 1; qv = mine > 0 ? need : 0; }
        else if (band_ok) rt = 5;
        else rt = fit ? 2 : 3;
        if (rt == 5) {
          // this rank's quota: its ties whose global id ranks below `need` among every
          // rank's ties (ids distinct, groups unsorted). Item (i, r2): how many of rank r2's
          // ties lie below this rank's tie i; only this rank's ties are ranked.
          const int m = s_tc[p.rank];
          int* rk = reinterpret_cast<int*>(hist + kV2Fine + 2 * kV2Win);
          if (tid < m) rk[tid] = 0;
          __syncthreads();
          const uint32_t* my = rx + p.rank * kV2Msg + kV2Fix + s_toff[p.rank];
          for (int it = tid; it < m * W; it += kV2T) {
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
      if (tid == 0) {
        s_tier = rt;
        if (rt != 3 || tau >= 0) { s_tau = tau; s_need = need; }
        if (qv >= 0) s_q = qv;
      }
    }
    __syncthreads();
    V2DBG(5);
    int tier = s_tier;
    if constexpr (kFast) {
      // ---- R2: the bounded exact fallback in this CTA (one more NVLink round, no redo
      // launch). LB = the largest key whose lower-bound global count of keys >= LB (from the
      // round-1 windows) reaches K, the same on every rank; every rank sends ALL its keys >= LB
      // with their positions (candidates, or one re-scan of its row when LB < lo); every rank
      // then derives tau, need and the tie cut by global id from the union (exact). Rows whose
      // keys >= LB overflow the message keep the complete body. Tier 2 (tau known, a band
      // missed its ties) takes the tie round below instead: tau lies in every window, so every
      // rank's ties are in its candidates.
      //
      // ---- CG first (rank-skewed rows), out of line (v2_cg); rows it cannot take keep R2
      bool cg = false;
      if (K31_RARE(tier == 3) && s_tau >= 0) {
        // tau exact, ties beyond the bands and the tie list: the id cut from the candidates
        // (kFast: every rank was fast, tau >= every theta)
        const int qv = v2_tiecut<T>(p, sm, nc, s_tau, s_need, tag, par, row, Gl, rx);
        if (tid == 0) { s_q = qv; s_tier = 1; }
        __syncthreads();
        tier = 1;
        V2STAT(2);
      }
      if (K31_RARE(tier == 3) && s_tau < 0) {
        cg = v2_cg<T>(p, rx, hist, msg, ckey, nc, tag, par, row, warp_sums, s_i);
        if (cg) tier = s_tier;
      }
      if (tier == 3 && !cg) {
        // saturated windows / failed fast paths take R2 too: every window is a lower bound
        if (row < kR2Rows) {
        __shared__ int s_r2n[kV2MaxW];
        __shared__ int s_r2[6];  // 0 LB, 1 tie count, 2 tau, 3 keys > tau, 4 quota, 5 bad
        uint32_t* const z = reinterpret_cast<uint32_t*>(srow) + 4096;
        constexpr int kAvail = kV2NR * T * 4 - 4096;
        const int capE = min(kR2Cap, (kAvail / (W + 1) - 8) / 2);
        const int RW = (4 + 2 * capE + 3) & ~3;
        uint32_t* const snd = z;
        uint32_t* const rcv = z + RW;
        uint32_t* const suf = rcv;  // W x (kV2Win + 1): above + window keys >= lr + i
        // suffix counts of every rank's window, one warp per rank (no block scan): lane l holds
        // the window's keys kV2Win - 1 - [8 l, 8 l + 8), counted from the top
        if ((tid >> 5) < W) {
          constexpr int kPer = kV2Win / 32;
          const int lane = tid & 31;
          const uint32_t* s = rx + (tid >> 5) * kV2Msg;
          const uint32_t above = s[1] & 0xFFFFFFu;
          uint32_t c[kPer], sum = 0;
#pragma unroll
          for (int j = 0; j < kPer; ++j) {
            c[j] = v2_cnt(s, kV2Win - 1 - (lane * kPer + j));
            sum += c[j];
          }
          uint32_t x = sum;
#pragma unroll
          for (int o = 1; o < 32; o <<= 1) {
            const uint32_t y = __shfl_up_sync(0xffffffffu, x, o);
            if (lane >= o) x += y;
          }
          uint32_t run = above + x - sum;  // the rank's keys above this lane's top key
          uint32_t* const sr = suf + (tid >> 5) * (kV2Win + 1);
#pragma unroll
          for (int j = 0; j < kPer; ++j) {
            run += c[j];
            sr[kV2Win - 1 - (lane * kPer + j)] = run;
          }
          if (lane == 0) sr[kV2Win] = above;
        }
        if (tid == 0) { s_r2[0] = -1; s_r2[1] = 0; s_r2[2] = -1; s_r2[4] = 0; s_r2[5] = 0; }
        __syncthreads();
        {
          int xlo = 1 << 30, xhi = 0;
          for (int r = 0; r < W; ++r) {
            const int lr = (int)(rx[r * kV2Msg] & 0xFFFFu);
            xlo = min(xlo, lr);
            xhi = max(xhi, lr + kV2Win);
          }
          xhi = min(xhi, 0x10000);
          // a rank's bound is constant below its window and above it, so g changes only at
          // window keys: the largest x with g(x) >= K is a window key or the key just below a
          // window. W x (kV2Win + 1) candidates instead of every key of [xlo, xhi), which
          // spans up to the whole key range when the windows lie far apart.
          int best = -1;  // every x below LB qualifies too: reduce per warp, then one atomic
          for (int c = tid; c < W * (kV2Win + 1); c += T) {
            const int r0 = c / (kV2Win + 1);
            const int x = (int)(rx[r0 * kV2Msg] & 0xFFFFu) + (c - r0 * (kV2Win + 1)) - 1;
            if (x < xlo || x >= xhi) continue;
            uint32_t g = 0;
            for (int r = 0; r < W; ++r) {
              const int lr = (int)(rx[r * kV2Msg] & 0xFFFFu);
              const uint32_t* sr = suf + r * (kV2Win + 1);
              g += x < lr ? sr[0] : (x < lr + kV2Win ? sr[x - lr] : 0u);
            }
            if (g >= (uint32_t)K) best = max(best, x);
          }
          best = __reduce_max_sync(0xffffffffu, best);
          if ((tid & 31) == 0 && best >= 0) atomicMax(&s_r2[0], best);
        }
        __syncthreads();
        V2DBG(6);
        const bool lbok = s_r2[0] >= 0;  // the same on every rank
        if (lbok) {
          const uint32_t LB = (uint32_t)max(s_r2[0], 1);
          const int64_t r2row = p.off_r2 + ((int64_t)(par * min(p.cap, kR2Rows) + row) * W) * kR2W;
          // this rank's keys >= LB in position order: from the candidates (SMEM, contiguous
          // chunks per thread), or one re-scan of the row (thread t owns the kV2NR consecutive
          // 8-key groups [kV2NR t, kV2NR t + kV2NR), all loads issued at once into registers)
          const bool fromc = regp && nc <= kV2Cand && (int)LB >= th;
          bool over;
          int n;
          if (fromc) {
            const int per = (nc + T - 1) / T;
            const int a0 = min(nc, tid * per), a1 = min(nc, a0 + per);
            uint32_t m = 0;
            for (int i = a0; i < a1; ++i) m += (uint32_t)ckey[i] >= LB;
            uint32_t ntot;
            uint32_t o = v2_scan<T>(m, V2WS, &ntot);
            over = ntot > (uint32_t)capE;
            n = over ? 0 : (int)ntot;
            if (!over)
              for (int i = a0; i < a1; ++i)
                if ((uint32_t)ckey[i] >= LB) {
                  snd[4 + 2 * o] = tagw(tag, (uint32_t)ckey[i]);
                  snd[5 + 2 * o] = tagw(tag, cpos[i]);
                  ++o;
                }
          } else {
            // one re-scan of the row from L2 (count, then emit; one 16-byte group live per
            // thread: the hot kernel's register budget is kept): thread t owns the kV2NR
            // consecutive 8-key groups [kV2NR t, kV2NR t + kV2NR)
            const uint4* gb = reinterpret_cast<const uint4*>(lrow - sh0);
            const int ngr = (L + sh0 + 7) >> 3;
            const int gfull = min((p.lcols + sh0) >> 3, ngr);  // whole 16-byte groups inside the row buffer
            auto grp = [&](int g) -> uint4 {
              if (g >= ngr) return make_uint4(0u, 0u, 0u, 0u);
              if (g < gfull && (g > 0 || sh0 == 0)) return __ldcg(gb + g);
              uint32_t w4[4];  // the partial head / tail groups, key by key (<= 2 per row)
  #pragma unroll
              for (int h = 0; h < 4; ++h) {
                const int a = 8 * g - sh0 + 2 * h;
                const uint32_t lo16 = (a >= 0 && a < L) ? lrow[a] : 0xFFFFu;
                const uint32_t hi16 = (a + 1 >= 0 && a + 1 < L) ? lrow[a + 1] : 0xFFFFu;
                w4[h] = lo16 | (hi16 << 16);
              }
              return make_uint4(w4[0], w4[1], w4[2], w4[3]);
            };
            // chunks of kV2NR x T groups (rows longer than the SMEM row walk several), each:
            // count, one scan, emit; position order kept
            uint32_t nb2 = 0;
            over = false;
#pragma unroll 1
            for (int gc = 0; gc < ngr && !over; gc += kV2NR * T) {
              uint32_t m = 0;
#pragma unroll 1
              for (int j = 0; j < kV2NR; ++j) {
                const int g = gc + tid * kV2NR + j;
                const uint4 v = grp(g);
                const uint32_t w4[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                for (int q2 = 0; q2 < 8; ++q2) {
                  const int pos = 8 * g - sh0 + q2;
                  const uint32_t k = bf16_key(w4[q2 >> 1] >> (16 * (q2 & 1)));
                  m += (pos >= 0 && pos < L && k >= LB);
                }
              }
              uint32_t ntot;
              uint32_t o = nb2 + v2_scan<T>(m, V2WS, &ntot);
              over = nb2 + ntot > (uint32_t)capE;
              if (!over)
#pragma unroll 1
                for (int j = 0; j < kV2NR; ++j) {
                  const int g = gc + tid * kV2NR + j;
                  const uint4 v = grp(g);
                  const uint32_t w4[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                  for (int q2 = 0; q2 < 8; ++q2) {
                    const int pos = 8 * g - sh0 + q2;
                    const uint32_t k = bf16_key(w4[q2 >> 1] >> (16 * (q2 & 1)));
                    if (pos >= 0 && pos < L && k >= LB) {
                      snd[4 + 2 * o] = tagw(tag, k);
                      snd[5 + 2 * o] = tagw(tag, (uint32_t)pos);
                      ++o;
                    }
                  }
                }
              nb2 += ntot;
            }
            n = over ? 0 : (int)nb2;
          }
          V2DBG(7);
          const int nv = (4 + 2 * n + 3) >> 2;
          if (tid < 4) snd[tid] = tagw(tag, tid == 0 ? (over ? 0x800000u : (uint32_t)n) : 0u);
          if (tid < 4 && 4 + 2 * n + tid < 4 * nv) snd[4 + 2 * n + tid] = tagw(tag, 0u);
          __syncthreads();
          {
            const int64_t base = r2row + (int64_t)p.rank * kR2W;
            for (int j = tid; j < nv; j += T) {
              const uint4 v = *reinterpret_cast<const uint4*>(snd + 4 * j);
              if (p.mc) mc_st_v4(p.mc + (base + 4 * j) * 4, v);
              else
                for (int d = 0; d < W; ++d) st_v4(reinterpret_cast<uint32_t*>(p.peers[d]) + base + 4 * j, v);
            }
          }
          const uint4 dead = make_uint4(0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu);
          auto poll = [&](int r, int j) -> uint4 {  // vector j of rank r's slot, then re-armed
            uint32_t* src = p.box + r2row + (int64_t)r * kR2W + 4 * j;
            const long long t0 = now_ns();
            uint4 v = ld_sys_v4(src);
            while (!tagged(v, tag)) {
              if (spin_late(p.spin_ns, t0)) { spin_fail(p.err, 32); break; }
              v = ld_sys_v4(src);
            }
            st_v4(src, dead);
            return v;
          };
          if (tid < W) {
            const uint4 v = poll(tid, 0);
            s_r2n[tid] = (v.x & 0x800000u) ? -1 : (int)(v.x & 0x7FFFFFu);
          }
          __syncthreads();
          for (int r = 0; r < W; ++r) {
            const int nr = max(s_r2n[r], 0);
            const int nvr = (4 + 2 * nr + 3) >> 2;
            for (int j = 1 + tid; j < nvr; j += T)
              *reinterpret_cast<uint4*>(rcv + r * RW + 4 * j) = poll(r, j);
          }
          __syncthreads();
          V2DBG(11);
          bool bad = false;
          for (int r = 0; r < W; ++r) bad |= s_r2n[r] < 0;
          if (!bad) {
            // tau: the K-th largest key of the union (two 8-bit levels)
            uint32_t* const h = hist;
            int b1 = -1, above1 = 0;
            for (int lvl = 0; lvl < 2; ++lvl) {
              for (int i = tid; i < 256; i += T) h[i] = 0u;
              __syncthreads();
              for (int r = 0; r < W; ++r) {
                const uint32_t* e = rcv + r * RW + 4;
                const int nr = s_r2n[r];
                // keys >= LB fall in a few bins: one atomic per (warp, bin), not per key
                for (int i0 = tid & ~31; i0 < nr; i0 += T) {
                  const int i = i0 + (tid & 31);
                  const uint32_t k = i < nr ? e[2 * i] & 0xFFFFu : 0u;
                  const bool in = i < nr && (lvl == 0 || (int)(k >> 8) == b1);
                  const uint32_t bin = lvl == 0 ? k >> 8 : k & 0xFFu;
                  const unsigned inm = __ballot_sync(0xFFFFFFFFu, in);
                  if (in) {
                    const unsigned peers = __match_any_sync(inm, bin);
                    if ((tid & 31) == __ffs(peers) - 1) atomicAdd(&h[bin], (uint32_t)__popc(peers));
                  }
                }
              }
              __syncthreads();
              const uint32_t c = tid < 256 ? h[255 - tid] : 0u;
              uint32_t tot;
              const uint32_t ex = v2_scan<T>(c, V2WS, &tot) + (uint32_t)above1;
              if (tid < 256 && ex < (uint32_t)K && ex + c >= (uint32_t)K) {
                s_r2[2] = lvl == 0 ? 255 - tid : (b1 << 8) | (255 - tid);
                s_r2[3] = (int)ex;
              }
              __syncthreads();
              const int got = s_r2[2], gab = s_r2[3];
              __syncthreads();
              if (got < 0) break;
              if (lvl == 0) { b1 = got; above1 = gab; if (tid == 0) s_r2[2] = -1; __syncthreads(); }
            }
            V2DBG(13);
            const int tau2 = s_r2[2], need2 = K - s_r2[3];
            if (tau2 < 0) bad = true;
            if (!bad) {
              // every rank's ties at tau by global id; this rank's quota
              auto gid_r = [&](int r, uint32_t pos) -> uint32_t {
                return (uint32_t)((((int)pos / IL) * W + r) * IL + (int)pos % IL);
              };
              for (int r = 0; r < W; ++r) {
                const uint32_t* e = rcv + r * RW + 4;
                for (int i = tid; i < s_r2n[r]; i += T)
                  if ((int)(e[2 * i] & 0xFFFFu) == tau2) {
                    const int t = atomicAdd(&s_r2[1], 1);
                    if (t < kCoarseBins) h[t] = gid_r(r, e[2 * i + 1] & 0xFFFFFFu);
                  }
              }
              __syncthreads();
              const int mt = s_r2[1];
              bad = mt > kCoarseBins || mt < need2;
              if (!bad) {
                const uint32_t* me = rcv + p.rank * RW + 4;
                const int nme = s_r2n[p.rank];
                int sel = 0;
                for (int i = tid; i < nme; i += T)
                  if ((int)(me[2 * i] & 0xFFFFu) == tau2) {
                    const uint32_t g = gid_r(p.rank, me[2 * i + 1] & 0xFFFFFFu);
                    int c = 0;
                    for (int j = 0; j < mt; ++j) c += h[j] < g;
                    sel += c < need2;
                  }
                if (sel) atomicAdd(&s_r2[4], sel);
                __syncthreads();
                V2DBG(16);
                const int qq = s_r2[4];
                // WRITE from this rank's list (position order): keys > tau, then its first qq ties
                const int pe = (nme + T - 1) / T;
                const int e0 = min(nme, tid * pe), e1 = min(nme, e0 + pe);
                uint32_t ca = 0, ct = 0;
                for (int i = e0; i < e1; ++i) {
                  const int k = (int)(me[2 * i] & 0xFFFFu);
                  ca += k > tau2;
                  ct += k == tau2;
                }
                uint32_t tot;
                const uint32_t pre = v2_scan<T>(ca | (ct << 16), V2WS, &tot);
                const int gtt = (int)(tot & 0xFFFFu);
                int a = (int)(pre & 0xFFFFu), t = (int)(pre >> 16);
                for (int i = e0; i < e1; ++i) {
                  const int k = (int)(me[2 * i] & 0xFFFFu);
                  const int pos = (int)(me[2 * i + 1] & 0xFFFFFFu);
                  if (k > tau2) {
                    if (a < K) orow[a] = slot_of(pos);
                    ++a;
                  } else if (k == tau2) {
                    if (t < qq && gtt + t < K) orow[gtt + t] = slot_of(pos);
                    ++t;
                  }
                }
                const int count = min(K, gtt + qq);
                for (int i = count + tid; i < K; i += T) orow[i] = -1;
                if (tid == 0) {
                  if (gtt + qq > K) spin_fail(p.err, 42);
                  p.out_count[row] = count;
                  p.epoch[row] = ep + 1 == 510 ? 0 : ep + 1;
                }
                V2STAT(2);
                V2DBG(17);
                V2DBG(8);
                __syncthreads();  // SMEM reused by the next row
                return 0;
              }
            }
          }
          __syncthreads();
        }
      }
    }
      if (tier != 1 && tier != 2) {  // the same on every rank: the complete body takes the row
        __syncthreads();
        return 1 + ep;
      }
    }
    if (tier == 1) V2STAT(1);
    if (tier == 4) V2STAT(4);

    // ---- T2: tie round (complete sorted tie lists at tau, <= kV2Tie each). In the hot
    // instance tier 2 implies fast and tau >= th on every rank (tau lies in every window).
    if (K31_RARE(tier == 2)) {
      V2STAT(2);
      const int tau = s_tau, need = s_need;
      __shared__ int s_tn[kV2MaxW];
      if (tid < W) {
        const uint32_t* s = rx + tid * kV2Msg;
        s_tn[tid] = (int)v2_cnt(s, tau - (int)(s[0] & 0xFFFFu));
      }
      if (tid == 0) s_cnt = 0;
      __syncthreads();
      if (kFast || (fast && tau >= th)) {
        for (int c = tid; c < nc; c += kV2T)
          if ((int)ckey[c] == tau) {
            const int i = atomicAdd(&s_cnt, 1);
            if (i < kV2Tie) hist[i] = gid_of((int)cpos[c]);
          }
      } else {
        for_keys(0, ng, [&](int i0, const uint4& r) {
          const uint32_t valid = 0xFFu;
          if ((int)max_key_bound(r) < tau) return;
          const Keys8 k8(r);
#pragma unroll 1
          for (int j = 0; j < 8; ++j)
            if (((valid >> j) & 1u) && (int)k8.key(j) == tau) {
              const int i = atomicAdd(&s_cnt, 1);
              if (i < kV2Tie) hist[i] = gid_of(i0 + j);
            }
        });
      }
      __syncthreads();
      const int nm = s_cnt;
      if (tid == 0 && (nm > kV2Tie || nm != s_tn[p.rank])) spin_fail(p.err, 34);
      if (tid < nm) {
        const uint32_t v = hist[tid];
        int r = 0;
        for (int j = 0; j < nm; ++j) r += hist[j] < v;
        msg[r] = tagw(tag, v);
      }
      const int nmp = (nm + 3) & ~3;
      if (nm + tid < nmp) msg[nm + tid] = tagw(tag, 0u);
      __syncthreads();
      const int64_t trow = p.off_tie + ((int64_t)(par * p.cap + row) * W) * kV2Tie;
      send(msg, nmp, trow + (int64_t)p.rank * kV2Tie);
      // the lists go to rx[r * kV2Tie] (the round-1 words are no longer read)
      __syncthreads();
      for (int i = tid; i < W * (kV2Tie / 4); i += kV2T) {
        const int r = i / (kV2Tie / 4), j = (i % (kV2Tie / 4)) * 4;
        if (j < ((s_tn[r] + 3) & ~3)) poll_words(p.box + trow + (int64_t)r * kV2Tie + j, rx + r * kV2Tie + j, 36);
      }
      __syncthreads();
      if (tid == 0) s_gs = -1;
      __syncthreads();
      for (int c = tid; c < W * kV2Tie; c += kV2T) {
        const int r = c / kV2Tie, j = c % kV2Tie;
        if (j >= s_tn[r]) continue;
        const uint32_t x = rx[c] & 0xFFFFFFu;
        int rk = j;
        for (int r2 = 0; r2 < W; ++r2)
          if (r2 != r) rk += count_below(rx + r2 * kV2Tie, s_tn[r2], x);
        if (rk == need - 1) s_gs = (int)x;
      }
      for (int c = tid; c < W * kV2Tie; c += kV2T)
        if (c % kV2Tie < ((s_tn[c / kV2Tie] + 3) & ~3)) st_u32(p.box + trow + c, kSent);
      __syncthreads();
      if (tid == 0) {
        if (s_gs < 0) spin_fail(p.err, 37);
        s_q = count_below(rx + p.rank * kV2Tie, nm, (uint32_t)s_gs + 1u);
      }
      __syncthreads();
      tier = 1;
      V2DBG(6);
    }

    // ---- T3: radix select of the K-th largest composite (key << 24 | 0xFFFFFF - gid)
    if (!kFast && K31_RARE(tier == 3)) {
      V2STAT(3);
      if (tid == 0) { s_pref = 0ull; s_cnt = K; s_u[12] = 0; }
      __syncthreads();
      // Id rounds from one pass: every global id of this row is < Gl + W * IL <= 2^20 (the same
      // bound on every rank), so the ties at tau fit a 4096-bin histogram by gid >> 8 (in `hist`).
      // Round 2 takes its 256 digits from sums of that histogram, round 3 from its bins, and
      // round 4 walks only the positions of the chosen 256-id block: 1 full pass instead of 3.
      // The digits are those of the composite's inverted id (0xFFFFFF - gid = ~gid: each byte
      // inverted), so the exchanged histograms are exactly the full-pass ones.
      const bool fine = (int64_t)Gl + (int64_t)W * IL <= (int64_t)(1 << 20);
      uint32_t* const rh_fine = cpos;  // round histograms in the fine mode (cpos is free in T3)
      for (int rd = 0; rd < kV2RadR; ++rd) {
        const int shb = 32 - 8 * rd;
        const unsigned long long pref = s_pref;
        const bool fr = fine && rd >= 2;
        uint32_t* const rh = fr ? rh_fine : hist;
        if (fr && rd == 2) {
          for (int b = tid; b < kCoarseBins; b += kV2T) hist[b] = 0;
        }
        if (tid < 256) rh[tid] = 0;
        __syncthreads();
        if (fr) {
          const uint32_t kt = (uint32_t)(pref >> (24 - shb - 8));
          if (rd == 2) {
            // the one pass: ties at tau by gid >> 8
            for_keys(0, ng, [&](int i0, const uint4& r) {
              const Keys8 k8(r);
#pragma unroll 1
              for (int j = 0; j < 8; ++j) {
                const uint32_t k = k8.key(j);
                if (!k || k != kt) continue;
                atomicAdd(&hist[gid_of(i0 + j) >> 8], 1u);
              }
            });
            // round-2 digit d = 255 - (gid >> 16): the sum of fine bins [g16 * 256, + 256)
            if (tid < 16) {
              uint32_t sum = 0;
              for (int b = 0; b < 256; ++b) sum += hist[tid * 256 + b];
              rh[255 - tid] = sum;
            }
            __syncthreads();
          } else if (rd == 3) {
            const int g16 = 255 - (int)(pref & 0xFFull);
            if (tid < 256) rh[255 - tid] = g16 < 16 ? hist[g16 * 256 + tid] : 0u;
            __syncthreads();
          } else {
            // rd == 4: the positions of the chosen 256-id block [g8 << 8, + 256) on this rank
            const int g8 = ((255 - (int)((pref >> 8) & 0xFFull)) << 8) | (255 - (int)(pref & 0xFFull));
            const int64_t G0 = (int64_t)g8 << 8, ilw = (int64_t)IL * W;
            const int64_t plo = (G0 / ilw) * IL, phi = ((G0 + 255) / ilw) * IL + IL - 1;
            const int64_t ga64 = (plo + sh) >> 3, gb64 = ((phi + sh) >> 3) + 1;
            const int ga = ga64 < 0 ? 0 : (int)ga64, gb = gb64 > ng ? ng : (int)gb64;
            for_keys(ga, max(ga, gb), [&](int i0, const uint4& r) {
              const Keys8 k8(r);
#pragma unroll 1
              for (int j = 0; j < 8; ++j) {
                const uint32_t k = k8.key(j);
                if (!k || k != kt) continue;
                const unsigned long long c = ((unsigned long long)k << 24) | (0xFFFFFFu - gid_of(i0 + j));
                if ((c >> (shb + 8)) != pref) continue;
                atomicAdd(&rh[(c >> shb) & 0xFFu], 1u);
              }
            });
          }
        } else {
        // id rounds 3-4 walk the positions of the keys equal to tau, collected in round 2
        // (cpos is free in the complete body's radix tier), when they all fit; else the pass
        const uint32_t ncomp = s_u[12];
        if (rd >= 3 && ncomp <= (uint32_t)kV2Cand) {
          const uint32_t kt = (uint32_t)(pref >> (24 - shb - 8));
          for (uint32_t i = tid; i < ncomp; i += kV2T) {
            const unsigned long long c =
                ((unsigned long long)kt << 24) | (0xFFFFFFu - gid_of((int)cpos[i]));
            if ((c >> (shb + 8)) != pref) continue;
            atomicAdd(&hist[(c >> shb) & 0xFFu], 1u);
          }
          __syncthreads();  // as for_keys: the histogram is complete before it is sent
        } else
        for_keys(0, ng, [&](int i0, const uint4& r) {
          const uint32_t valid = 0xFFu;
          const Keys8 k8(r);
#pragma unroll 1
          for (int j = 0; j < 8; ++j) {
            const uint32_t k = k8.key(j);
            if (!((valid >> j) & 1u) || !k) continue;
            // rounds 0-1 (shb >= 24) read key bits only; later rounds need the key equal to
            // the prefix's key part before the token id is formed (same counts, fewer ops)
            if (shb >= 24) {
              if (rd > 0 && (k >> (shb - 16)) != (uint32_t)pref) continue;
              atomicAdd(&hist[(k >> (shb - 24)) & 0xFFu], 1u);
              continue;
            }
            if (k != (uint32_t)(pref >> (24 - shb - 8))) continue;
            if (rd == 2) {  // collect the positions of the keys equal to tau (count past the cap)
              const uint32_t slot = atomicAdd(&s_u[12], 1u);
              if (slot < (uint32_t)kV2Cand) cpos[slot] = (uint32_t)(i0 + j);
            }
            const unsigned long long c =
                ((unsigned long long)k << 24) | (0xFFFFFFu - gid_of(i0 + j));
            if ((c >> (shb + 8)) != pref) continue;
            atomicAdd(&hist[(c >> shb) & 0xFFu], 1u);
          }
        });
        }
        const int64_t rrow = p.off_rad + (((int64_t)(par * p.cap + row) * kV2RadR + rd) * W) * 256;
        if (tid < 64) {
          const int j = tid * 4;
          const uint4 v = make_uint4(tagw(tag, rh[j]), tagw(tag, rh[j + 1]), tagw(tag, rh[j + 2]),
                                     tagw(tag, rh[j + 3]));
          const int64_t off = rrow + (int64_t)p.rank * 256 + j;
          if (p.mc) mc_st_v4(p.mc + off * 4, v);
          else
            for (int d = 0; d < W; ++d) st_v4(reinterpret_cast<uint32_t*>(p.peers[d]) + off, v);
        }
        for (int i = tid; i < W * 64; i += kV2T) poll_words(p.box + rrow + (int64_t)i * 4, rx + i * 4, 38);
        __syncthreads();
        uint32_t hb = 0;  // bucket 255 - tid over the ranks
        if (tid < 256)
          for (int r = 0; r < W; ++r) hb += rx[r * 256 + (255 - tid)] & 0xFFFFFFu;
        uint32_t tot;
        const uint32_t ex = v2_scan<T>(hb, V2WS, &tot);
        const uint32_t want = (uint32_t)s_cnt;
        if (tid == 0) s_n = 0;
        __syncthreads();
        if (tid < 256 && ex < want && ex + hb >= want) {
          s_pref = (pref << 8) | (unsigned long long)(255 - tid);
          s_cnt = (int)(want - ex);
          // After the last key-bit round (rd == 1) the prefix is the full 16-bit key tau.
          // When every key equal to tau over all ranks is needed (the bucket count is the
          // remaining want: no ties to break), the id rounds 2-4 would only confirm "take
          // all": skip them. Decided from the exchanged words, so the same on every rank;
          // the unused id-round slots stay armed (they are written only on rows that run them).
          s_n = (rd == 1 && want - ex == hb) ? 2 : 1;
        }
        for (int i = tid; i < W * 256; i += kV2T) st_u32(p.box + rrow + i, kSent);
        __syncthreads();
        if (tid == 0 && s_n < 1) spin_fail(p.err, 39);  // >= K distinct composites exist
        if (s_n == 2) {
          // composite cut = (tau << 24) | 0: every id of tau qualifies (gid <= 0xFFFFFF)
          if (tid == 0) s_pref = s_pref << 24;
          __syncthreads();
          break;
        }
      }
      if (tid == 0) {
        s_tau = (int)(s_pref >> 24);
        s_gs = (int)(0xFFFFFFu - (uint32_t)(s_pref & 0xFFFFFFull));
      }
      __syncthreads();
      V2DBG(7);
    }

    // ---- 5. WRITE: tokens above tau, then ties (tier 1: the first s_q in position order;
    // tier 3: those with gid <= s_gs; tier 4: tau = 0, every valid key, no ties). The
    // selected positions go to SMEM (cpos reused when the candidates are not read), then
    // all threads map them to slots.
    {
      const int tau = tier == 4 ? 0 : s_tau;
      const int qq = s_q, gs = s_gs;
      const bool from_cand = fast && tier == 1 && tau >= th;
      // kFast: tier 1 from the round puts tau in every window, above every rank's theta
      if (kFast && tid == 0 && !(from_cand && regp)) spin_fail(p.err, 41);
      if (kFast || K31_LIKELY(from_cand && regp)) {
        // candidates sorted by position, their slots ready: one scan of (above tau,
        // ties at tau) over contiguous candidate chunks, then scatter
        const int cpt = (nc + kV2T - 1) / kV2T;
        const int c0 = min(nc, tid * cpt), c1 = min(nc, c0 + cpt);
        uint32_t ca = 0, ct = 0;
        for (int c = c0; c < c1; ++c) {
          const int k = ckey[c];
          ca += k > tau;
          ct += k == tau;
        }
        uint32_t tot;
        const uint32_t pre = v2_scan<T>(ca | (ct << 16), V2WS, &tot);
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
        for (int i = count + tid; i < K; i += kV2T) orow[i] = -1;
        if (tid == 0) {
          p.out_count[row] = count;
          p.epoch[row] = ep + 1 == 510 ? 0 : ep + 1;
        }
        V2DBG(18);
      } else {
      if (tid == 0) s_gt = 0;
      __syncthreads();
      if (tier == 1) {
        uint32_t a = (tid < kV2Win && tid > tau - lo) ? win[tid] : 0u;
        if (tid == 0) a += s_above;
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if ((tid & 31) == 0 && a) atomicAdd(&s_gt, a);
      } else {
        uint32_t c = 0;
        for_keys(0, ng, [&](int, const uint4& r) {
          const uint32_t valid = 0xFFu;
          if ((int)max_key_bound(r) <= tau) return;
          const Keys8 k8(r);
#pragma unroll 1
          for (int j = 0; j < 8; ++j) c += ((valid >> j) & 1u) && (int)k8.key(j) > tau;
        });
        for (int o = 16; o > 0; o >>= 1) c += __shfl_xor_sync(0xffffffffu, c, o);
        if ((tid & 31) == 0 && c) atomicAdd(&s_gt, c);
      }
      __syncthreads();
      const int gtt = (int)s_gt;
      V2DBG(17);
      uint32_t* bmg = hist;
      uint32_t* bme = hist + kV2Chunk / 32;
      // selected positions -> sel[] (output index order)
      uint32_t* sel = rx;  // the exchanged words are no longer read
      int run_g = 0, run_e = 0;
      for (int c0 = 0; c0 < L; c0 += kV2Chunk) {
        const int n = min(kV2Chunk, L - c0), nw = (n + 31) >> 5;
        for (int i = tid; i < nw; i += kV2T) { bmg[i] = 0; bme[i] = 0; }
        __syncthreads();
        if (from_cand) {
          for (int c = tid; c < nc; c += kV2T) {
            const int i = (int)cpos[c] - c0;
            if (i < 0 || i >= n) continue;
            const int k = ckey[c];
            if (k > tau) atomicOr(&bmg[i >> 5], 1u << (i & 31));
            else if (k == tau) atomicOr(&bme[i >> 5], 1u << (i & 31));
          }
        } else {
          for_keys(max(0, (c0 + sh) >> 3), min(ng, (c0 + n + sh + 7) >> 3), [&](int pg, const uint4& r) {
            if ((int)max_key_bound(r) < tau) return;
            const int i0 = pg - c0;  // chunk-relative position of the group's first key
            const Keys8 k8(r);
            uint32_t mg = 0, me = 0;
#pragma unroll 1
            for (int j = 0; j < 8; ++j) {
              if (i0 + j < 0 || i0 + j >= n) continue;
              const int k = (int)k8.key(j);
              if (k > tau) mg |= 1u << j;
              else if (k == tau && tau != 0 && (tier != 3 || (int)gid_of(pg + j) <= gs))
                me |= 1u << j;
            }
            if (i0 >= 0) {
              if (mg) or_bits8(bmg, i0, mg);
              if (me) or_bits8(bme, i0, me);
            } else {
              for (int j = -i0; j < 8; ++j) {
                if ((mg >> j) & 1u) atomicOr(&bmg[(i0 + j) >> 5], 1u << ((i0 + j) & 31));
                if ((me >> j) & 1u) atomicOr(&bme[(i0 + j) >> 5], 1u << ((i0 + j) & 31));
              }
            }
          });
        }
        __syncthreads();
        V2DBG(23);
        const int wpt = (nw + kV2T - 1) / kV2T;
        const int w0 = min(nw, tid * wpt), w1 = min(nw, w0 + wpt);
        uint32_t cg = 0, ce = 0;
        for (int w = w0; w < w1; ++w) { cg += __popc(bmg[w]); ce += __popc(bme[w]); }
        uint32_t tot;
        const uint32_t pre = v2_scan<T>(cg | (ce << 16), V2WS, &tot);
        int gp = run_g + (int)(pre & 0xFFFFu), epi = run_e + (int)(pre >> 16);
        for (int w = w0; w < w1; ++w) {
          uint32_t m = bmg[w];
          while (m) {
            const int bit = __ffs(m) - 1;
            m &= m - 1;
            if (gp < K) sel[gp] = (uint32_t)(c0 + w * 32 + bit);
            ++gp;
          }
          m = bme[w];
          while (m) {
            const int bit = __ffs(m) - 1;
            m &= m - 1;
            if ((tier == 3 || epi < qq) && gtt + epi < K) sel[gtt + epi] = (uint32_t)(c0 + w * 32 + bit);
            ++epi;
          }
        }
        run_g += (int)(tot & 0xFFFFu);
        run_e += (int)(tot >> 16);
        __syncthreads();
      }
      V2DBG(18);
      const int nt = tier == 3 ? run_e : (tier == 4 ? 0 : qq);
      const int count = min(K, gtt + nt);
      if (tid == 0 && (run_g != gtt || gtt + nt > K || (tier == 1 && run_e < qq))) spin_fail(p.err, 40);
      for (int i = tid; i < count; i += kV2T) orow[i] = slot_of((int)sel[i]);
      for (int i = count + tid; i < K; i += kV2T) orow[i] = -1;
      if (tid == 0) {
        p.out_count[row] = count;
        p.epoch[row] = ep + 1 == 510 ? 0 : ep + 1;
      }
      }
    }
    V2DBG(8);
    __syncthreads();  // SMEM reused by the next row
  }
  return 0;
#undef V2DBG
#undef V2WS
#undef V2STAT
}


// SMEM of one row (both kernels), the mbarrier, then the PDL wait
#define V2_KERNEL_PROLOGUE(T)                                                                   __shared__ __align__(16) uint32_t hist[kCoarseBins];                                       __shared__ __align__(16) uint32_t cpos[kV2Cand];                                           __shared__ __align__(16) uint16_t ckey[kV2Cand];                                           __shared__ __align__(128) uint4 srow[V2Cfg<T>::kNR * (T)];                                        __shared__ __align__(8) uint64_t s_bar;                                                    __shared__ uint32_t warp_sums[64];                                                         __shared__ int s_i[16];                                                                    __shared__ uint32_t s_u[16];                                                               __shared__ unsigned long long s_pref;                                                      __shared__ long long s_dbg[32];                                                            const V2Sm sm{hist, cpos, ckey, srow, &s_bar, warp_sums, s_i, s_u, &s_pref, s_dbg};        if (threadIdx.x == 0) {                                                                      mbar_init(&s_bar, kV2Cp);                                                                  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");                     }                                                                                          __syncthreads();                                                                           uint32_t bar_phase = 0;                                                                    asm volatile("griddepcontrol.wait;" ::: "memory");

// The complete body of a row the hot body left (the same rows on every rank: the tier
// comes from the exchanged words), at the next epoch, in the same CTA and out of line so
// the hot row body keeps its code layout. Replaces the separate redo kernel launch (~1.0-1.5
// us per call in-step even when no row needs it).
template <int T>
__device__ __noinline__ void v2_complete(const V2Params p, const V2Sm sm, const int row, const int ep,
                                         uint32_t* bar_phase) {
  uint32_t ph = *bar_phase;
  v2_row<false, T>(p, sm, row, ep, ph);
  *bar_phase = ph;
}

// The common case: one compact row body (PDL: the prologue overlaps the logits kernel's
// tail; every global read follows the wait); a row that needs the complete body runs it
// right away (v2_complete; the inline complete body measured ~8 us in-step: instruction fetch).
template <int T>
__global__ void __launch_bounds__(T, T >= 1024 ? 1 : 2) topk_v2_kernel(V2Params p) {
  V2_KERNEL_PROLOGUE(T)
  for (int row = blockIdx.x; row < p.T; row += gridDim.x) {
    const int r = v2_row<true, T>(p, sm, row, -1, bar_phase);
    // per shape: the 1024-thread variant (r1/r8) keeps topk_v2_redo_kernel (its in-CTA call
    // measured +0.5 us/call at r8 in-step, x1k2), the 512-thread one runs it in the CTA
    if constexpr (T < 1024) {
      if (K31_RARE(r)) v2_complete<T>(p, sm, row, r == 510 ? 0 : r, &bar_phase);
    } else {
      if (threadIdx.x == 0) p.redo[row] = r;
    }
  }
}

// The rows topk_v2_kernel left (the same rows on every rank: the tier comes from the
// exchanged words), with the complete body at the next epoch; CTAs without one exit.
// A small grid (kV2RedoCtas): row r goes to CTA r % grid on every rank, so a CTA's rows
// run in the same order everywhere; the flags of a CTA's rows are read at once.
__global__ void __launch_bounds__(kV2T, 2) topk_v2_redo_kernel(V2Params p) {
  V2_KERNEL_PROLOGUE(kV2T)
  int any = 0;
  for (int row = blockIdx.x + threadIdx.x * gridDim.x; row < p.T; row += kV2T * gridDim.x) any |= p.redo[row];
  if (!__syncthreads_or(any)) return;
  for (int row = blockIdx.x; row < p.T; row += gridDim.x) {
    const int r = p.redo[row];
    if (r) v2_row<false, kV2T>(p, sm, row, r == 510 ? 0 : r, bar_phase);
  }
}

}  // namespace k31dcpx
