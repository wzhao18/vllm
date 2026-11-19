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

namespace k31dcpx {

constexpr int kV2Win = 256;                  // window keys
constexpr int kV2Hdr = 8;                    // header words
constexpr int kV2Fix = kV2Hdr + kV2Win / 2;  // header + window count pairs
constexpr int kV2Band = 256;                 // band ids per rank
constexpr int kV2Msg = kV2Fix + kV2Band;     // round-1 slot words
constexpr int kV2MaxW = 8;
constexpr int kV2Tie = 256;     // tie-round ids per rank
constexpr int kV2RadR = 5;      // radix rounds (40-bit composite, 8 bits each)
constexpr int kV2Cand = 3072;   // candidate list
constexpr int kV2Fine = 2048;   // fine histogram keys above theta
constexpr int kV2Chunk = 65536; // WRITE positions per bitmap chunk (2048 words)
constexpr int kV2T = 512;       // threads per CTA (one CTA per row, 2 CTAs per SM)
constexpr int kV2Warps = kV2T / 32;
constexpr int kV2RegG = 4;      // 8-key groups per thread per candidate batch (16384 keys)
constexpr int kV2Dbg = 64;      // dbg stamps per row
constexpr int kV2SampleStride = 8;  // sample: one 8-key group in 8
// stats (rows): 0 exchanged, 1 resolved in round 1, 2 tie round, 3 radix, 4 select all
// valid, 5 full-pass message (sample bound failed)

struct V2Params {
  const uint16_t* logits;
  int64_t lstride;
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
  int64_t off_r1, off_tie, off_rad;
  int* epoch;
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

// Exclusive prefix over the kV2T-thread block; the total in *total.
__device__ __forceinline__ uint32_t v2_scan(uint32_t v, uint32_t* warp_sums, uint32_t* total) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  uint32_t x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const uint32_t y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x += y;
  }
  __syncthreads();
  if (lane == 31) warp_sums[warp] = x;
  __syncthreads();
  if (warp == 0) {
    const uint32_t sv = lane < kV2Warps ? warp_sums[lane] : 0u;
    uint32_t t = sv;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const uint32_t y = __shfl_up_sync(0xffffffffu, t, o);
      if (lane >= o) t += y;
    }
    if (lane < kV2Warps) warp_sums[lane] = t - sv;
    if (lane == kV2Warps - 1) warp_sums[kV2Warps] = t;
  }
  __syncthreads();
  const uint32_t r = warp_sums[warp] + x - v;
  *total = warp_sums[kV2Warps];
  __syncthreads();
  return r;
}

__global__ void __launch_bounds__(kV2T, 2) topk_v2_kernel(V2Params p) {
  __shared__ __align__(16) uint32_t hist[kCoarseBins];  // histograms; later bitmaps / scratch
  __shared__ __align__(16) uint32_t cpos[kV2Cand];
  __shared__ __align__(16) uint16_t ckey[kV2Cand];
  __shared__ __align__(16) uint32_t rx[kV2MaxW * kV2Msg];
  __shared__ __align__(16) uint32_t win[kV2Win];
  __shared__ __align__(16) uint32_t msg[kV2Msg];
  __shared__ uint32_t warp_sums[kV2Warps + 1];
  __shared__ int s_i[16];
  __shared__ uint32_t s_u[16];
  __shared__ unsigned long long s_pref;
  const int tid = threadIdx.x;
  // PDL: the prologue above overlaps the logits kernel's tail; every global read follows
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const int W = p.W, K = p.K, IL = p.interleave;
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
  int& s_fast = s_i[12];
  uint32_t& s_valid = s_u[0];
  uint32_t& s_above = s_u[1];
  uint32_t& s_beyond = s_u[2];
  uint32_t& s_atop = s_u[3];
  uint32_t& s_gt = s_u[4];
  uint32_t& s_sat = s_u[5];
#define V2DBG(i)                                                    \
  if (p.dbg && tid == 0) {                                          \
    p.dbg[(int64_t)row * kV2Dbg + (i)] = now_ns();                  \
    p.dbg[(int64_t)row * kV2Dbg + 32 + (i)] = (long long)clock64(); \
  }
#define V2STAT(i) \
  if (p.stats && tid == 0) atomicAdd(&p.stats[i], 1);

  for (int row = blockIdx.x; row < p.T; row += gridDim.x) {
    V2DBG(0);
    const int L = p.local_len[row];
    const int Gl = p.global_len[row];
    int* orow = p.out + (int64_t)row * p.ostride;
    const int req = p.req_of_row != nullptr ? p.req_of_row[row] : row / p.next_n;
    const int* btrow = p.block_table != nullptr ? p.block_table + (int64_t)req * p.bt_stride : nullptr;
    auto slot_of = [&](int pos) -> int {
      if (btrow == nullptr) return pos;
      return btrow[pos / p.block_size] * p.block_stride + pos % p.block_size;
    };
    auto gid_of = [&](int pos) -> uint32_t {
      return (uint32_t)(((pos / IL) * W + p.rank) * IL + pos % IL);
    };
    if (Gl <= K) {  // the whole context (every rank sees the same global length)
      for (int i = tid; i < L; i += kV2T) orow[i] = slot_of(i);
      for (int i = L + tid; i < K; i += kV2T) orow[i] = -1;
      if (tid == 0) p.out_count[row] = L;
      continue;
    }
    if (Gl >= (1 << 24) && tid == 0) spin_fail(p.err, 13);
    const int ep = p.epoch[row];
    const uint32_t tag = (uint32_t)(ep % 255);
    const int par = ep & 1;
    V2STAT(0);
    const uint16_t* lrow = p.logits + (int64_t)row * p.lstride;
    const uint32_t q = (uint32_t)max(1LL, ((long long)K * L + Gl - 1) / Gl);

    // ---- 1-2. keys -> sample bound theta -> candidates (every key >= theta with its
    // position), valid keys. Group g holds positions [8g - sh, 8g - sh + 8), 16-byte aligned
    // in global memory; positions outside [0, L) read as NaN (key 0). Rows of at most
    // kV2RegG x 256 groups are loaded into registers at once (one memory latency); longer
    // rows sample from global memory, then compact in batches of kV2RegG x 256 groups.
    for (int b = tid; b < kCoarseBins; b += kV2T) hist[b] = 0;
    if (tid == 0) { s_valid = 0; s_sat = 0; s_cnt = 0; s_beyond = 0; s_th = 0; }
    const uintptr_t la = reinterpret_cast<uintptr_t>(lrow);
    const int head = min(L, (int)(((16u - (uint32_t)(la & 15u)) & 15u) >> 1));
    const int sh = (8 - head) & 7;
    const int ng = (L + sh + 7) >> 3;
    const int g_lo = sh ? 1 : 0, g_hi = (L + sh) >> 3;  // full groups [g_lo, g_hi)
    const uint4* gbase = reinterpret_cast<const uint4*>(lrow - sh);
    auto load_group = [&](int g) -> uint4 {
      if (g >= g_lo && g < g_hi) return __ldg(gbase + g);
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
    // visit groups [ga, gb): fn(position of the group's first key, its 8 raw bf16 values);
    // every thread must call it (it ends with a CTA barrier)
    auto for_keys = [&](int ga, int gb, auto&& fn) {
      for (int g = ga + tid; g < gb; g += kV2T) fn(8 * g - sh, load_group(g));
      __syncthreads();
    };
    __syncthreads();  // histogram zeroed
    V2DBG(9);
    {
      // sample: one 8-key group in kV2SampleStride, all loads in flight before any use
      const int step = kV2T * kV2SampleStride;
      for (int g0 = tid * kV2SampleStride; g0 < ng; g0 += 4 * step) {
        uint4 s4[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) s4[i] = load_group(g0 + i * step);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const Keys8 k8(s4[i]);
#pragma unroll
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
      // its share of q + 256 keys (the sample's standard error at q ~ 512 is ~3% of q)
      const uint32_t want = (q + 256u + kV2SampleStride - 1) / kV2SampleStride;
      const int bb = kV2T - 1 - tid;
      uint32_t v = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) v += hist[bb * 8 + i];
      uint32_t tot;
      const uint32_t ab = v2_scan(v, warp_sums, &tot);
      if (ab < want && ab + v >= want) {
        uint32_t run = ab;
        for (int i = 7; i >= 0; --i) {
          run += hist[bb * 8 + i];
          if (run >= want) { s_th = (bb * 8 + i) * 16; break; }
        }
      }
      __syncthreads();
      if (tid == 0 && s_th == 0) s_th = 1;  // short / sparse sample: every valid key
      __syncthreads();
    }
    const int th = s_th;
    V2DBG(1);
    {
      // candidates: batches of kV2RegG groups per thread (loads in flight together); per
      // thread its keys >= theta, one warp scan, one atomic per warp
      // full groups [g_lo, g_hi) in the batches (one predicated 16-byte load each), then the
      // <= 2 partial groups (head / tail) as one more batch of a single group
      uint32_t nv = 0;
      const int nfull = g_hi - g_lo;
      const int npart = ng - max(0, nfull);
      for (int b0 = 0; b0 < max(0, nfull) + (npart > 0 ? kV2RegG * kV2T : 0); b0 += kV2RegG * kV2T) {
        const bool part = b0 >= max(0, nfull);
        const int g0 = part ? 0 : g_lo + b0;
        uint4 rb[kV2RegG];
        if (!part) {
#pragma unroll
          for (int u = 0; u < kV2RegG; ++u) {
            const int g = g0 + u * kV2T + tid;
            rb[u] = g < g_hi ? __ldg(gbase + g) : make_uint4(~0u, ~0u, ~0u, ~0u);
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
#pragma unroll
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
#pragma unroll
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
#pragma unroll
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
    const int nc = s_cnt;
    V2DBG(2);
    if (p.dbg && tid == 0) p.dbg[(int64_t)row * kV2Dbg + 31] = nc;
    // ---- 3. fine histogram above theta from the candidates -> e, window, band
    bool fast = nc <= kV2Cand && (uint32_t)nc >= q;
    for (int b = tid; b < kV2Fine; b += kV2T) hist[b] = 0;
    for (int b = tid; b < kV2Win; b += kV2T) win[b] = 0;
    __syncthreads();
    if (fast) {
      uint32_t bey = 0;
      for (int c = tid; c < nc; c += kV2T) {
        const int u = (int)ckey[c] - th;
        if (u < kV2Fine) atomicAdd(&hist[u], 1u);
        else ++bey;
      }
      for (int o = 16; o > 0; o >>= 1) bey += __shfl_xor_sync(0xffffffffu, bey, o);
      if ((tid & 31) == 0 && bey) atomicAdd(&s_beyond, bey);
      __syncthreads();
      V2DBG(12);
      // thread t owns fine keys [8 (255 - t), + 8): suffix counts from the top
      const int f0 = (kV2T - 1 - tid) * 4;
      uint32_t v = 0;
#pragma unroll
      for (int i = 0; i < 4; ++i) v += hist[f0 + i];
      uint32_t tot;
      const uint32_t ab = v2_scan(v, warp_sums, &tot) + s_beyond;
      if (tid == 0) s_e = -1;
      __syncthreads();
      if (ab < q && ab + v >= q) {
        uint32_t run = ab;
        for (int i = 3; i >= 0; --i) {
          run += hist[f0 + i];
          if (run >= q) { s_e = th + f0 + i; break; }
        }
      }
      __syncthreads();
      V2DBG(13);
      fast = s_e >= 0;
      if (fast) {
        const int e = s_e;
        const int lo = min(max(th, e - kV2Win / 2), th + kV2Fine - kV2Win);
        // window counts and keys at or above lo + 256 (fine keys above it + beyond)
        if (tid < kV2Win) win[tid] = hist[lo - th + tid];
        uint32_t a = 0;
        for (int u = lo - th + kV2Win + tid; u < kV2Fine; u += kV2T) a += hist[u];
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (tid == 0) { s_above = s_beyond; s_lo = lo; }
        __syncthreads();
        if ((tid & 31) == 0 && a) atomicAdd(&s_above, a);
      }
    }
    if (!fast) {
      V2STAT(5);
      // the same message from passes over every key: coarse histogram, the coarse bin of
      // the q-th key, the window around it
      for (int b = tid; b < kCoarseBins; b += kV2T) hist[b] = 0;
      __syncthreads();
      for_keys(0, ng, [&](int, const uint4& r) {
          const uint32_t valid = 0xFFu;
        const Keys8 k8(r);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const uint32_t k = k8.key(j);
          if (((valid >> j) & 1u) && k) atomicAdd(&hist[k >> 4], 1u);
        }
      });
      const int bb = kV2T - 1 - tid;
      uint32_t v = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) v += hist[bb * 8 + i];
      uint32_t tot;
      const uint32_t ab = v2_scan(v, warp_sums, &tot);
      if (tid == 0) s_e = 0;
      __syncthreads();
      if (ab < q && ab + v >= q) {
        uint32_t run = ab;
        for (int i = 7; i >= 0; --i) {
          run += hist[bb * 8 + i];
          if (run >= q) { s_e = (bb * 8 + i) * 16; break; }
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
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const uint32_t k = k8.key(j), u = k - (uint32_t)lo;
          if (((valid >> j) & 1u) && k && u < (uint32_t)kV2Win) atomicAdd(&win[u], 1u);
        }
      });
    }
    __syncthreads();
    V2DBG(14);
    const int lo = s_lo;
    // band around the exact local q-th key: [bl, bh] (window-relative), <= kV2Band keys
    {
      const int u = kV2Win - 1 - tid;  // thread t < 256 <-> window key 255 - t
      const uint32_t c = u >= 0 ? win[u] : 0u;
      if (c > kCountMax) atomicOr(&s_sat, 1u);
      uint32_t tot;
      const uint32_t ab = v2_scan(c, warp_sums, &tot) + s_above;
      if (tid == 0) { s_e = -1; s_bl = 1; s_bh = 0; s_n = 0; }
      __syncthreads();
      if (ab < q && ab + c >= q) s_e = u;
      if (u >= 0) hist[kV2Fine + u] = tot - (ab - s_above) - c;  // P[u]: window keys below u
      __syncthreads();
      if (tid == 0 && s_e >= 0 && win[s_e] <= (uint32_t)kV2Band) {
        const uint32_t* P = hist + kV2Fine;
        auto Pf = [&](int x) -> uint32_t { return x >= kV2Win ? tot : P[x]; };
        const int e = s_e;
        int lo_h = 0, hi_h = kV2Win;
        while (hi_h - lo_h > 1) {
          const int h = (lo_h + hi_h) >> 1;
          const int a = max(0, e - h), b = min(kV2Win - 1, e + h);
          if (Pf(b + 1) - Pf(a) <= (uint32_t)kV2Band) lo_h = h; else hi_h = h;
        }
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
#pragma unroll
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
      msg[5] = tagw(tag, 0u);
      msg[6] = tagw(tag, 0u);
      msg[7] = tagw(tag, 0u);
    }
    const int words = (kV2Fix + nb + 3) & ~3;
    if (kV2Fix + nb + tid < words) msg[kV2Fix + nb + tid] = tagw(tag, 0u);
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
        v = ld_sys_v4(src);
      }
      *reinterpret_cast<uint4*>(dst) = v;
    };
    send(msg, words, r1row + (int64_t)p.rank * kV2Msg);
    for (int i = tid; i < W * (kV2Fix / 4); i += kV2T) {
      const int r = i / (kV2Fix / 4), j = (i % (kV2Fix / 4)) * 4;
      poll_words(p.box + r1row + (int64_t)r * kV2Msg + j, rx + r * kV2Msg + j, 31);
    }
    __syncthreads();
    for (int i = tid; i < W * (kV2Band / 4); i += kV2T) {
      const int r = i / (kV2Band / 4), j = kV2Fix + (i % (kV2Band / 4)) * 4;
      const int nbr = (int)(rx[r * kV2Msg + 3] & 0xFFFFFFu);
      if (j < kV2Fix + nbr) poll_words(p.box + r1row + (int64_t)r * kV2Msg + j, rx + r * kV2Msg + j, 32);
    }
    __syncthreads();
    V2DBG(4);
    if (p.dbg && tid == 0) p.dbg[(int64_t)row * kV2Dbg + 30] = nb;

    // ---- resolve. tier: 1 resolved (quota s_q), 2 tie round, 3 radix (id cut s_gs),
    // 4 every valid key, 5 cut from the bands
    if (tid == 0) { s_tier = 0; s_atop = 0; s_tau = -1; s_need = 0; s_q = 0; s_gs = -1; }
    __syncthreads();
    {
      int& s_maxlo = s_i[13];
      int& s_minhi = s_i[14];
      if (tid == 0) {
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
        s_maxlo = mx;
        s_minhi = mn;
        if (tv <= (uint32_t)K) s_tier = 4;
        else if (sat || mx >= mn) s_tier = 3;
      }
      __syncthreads();
      V2DBG(19);
      if (s_tier == 0) {
        const int mx = s_maxlo, mn = s_minhi, n = mn - mx;
        uint32_t a = 0;
        for (int r = 0; r < W; ++r) {
          const uint32_t* s = rx + r * kV2Msg;
          const int lr = (int)(s[0] & 0xFFFFu);
          if (tid < kV2Win && lr + tid >= mn) a += v2_cnt(s, tid);
          if (tid == 0) a += s[1] & 0xFFFFFFu;
        }
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if ((tid & 31) == 0 && a) atomicAdd(&s_atop, a);
        const int u = n - 1 - tid;  // thread 0: the top key of the common range
        uint32_t gu = 0;
        if (u >= 0)
          for (int r = 0; r < W; ++r) {
            const uint32_t* s = rx + r * kV2Msg;
            gu += v2_cnt(s, mx + u - (int)(s[0] & 0xFFFFu));
          }
        uint32_t tot;
        const uint32_t ex = v2_scan(gu, warp_sums, &tot);
        const uint32_t atop = s_atop;
        if (atop >= (uint32_t)K || atop + tot < (uint32_t)K) {
          if (tid == 0) s_tier = 3;
        } else if (u >= 0 && atop + ex < (uint32_t)K && atop + ex + gu >= (uint32_t)K) {
          s_tau = mx + u;
          s_need = K - (int)(atop + ex);
        }
        __syncthreads();
        V2DBG(20);
      }
      if (s_tier == 0) {
        const int tau = s_tau, need = s_need;
        // per rank: its tie count at tau, whether its band holds them, its group offset
        __shared__ int s_tc[kV2MaxW], s_toff[kV2MaxW], s_tok[kV2MaxW];
        if (tid < W) {
          const uint32_t* s = rx + tid * kV2Msg;
          const int lr = (int)(s[0] & 0xFFFFu);
          const uint32_t bw = s[2] & 0xFFFFFFu;
          const int blr = (int)(bw & 0xFFu), bhr = (int)((bw >> 8) & 0xFFu);
          s_tc[tid] = (int)v2_cnt(s, tau - lr);
          s_tok[tid] = ((bw >> 16) & 1u) && tau - lr >= blr && tau - lr <= bhr;
          s_toff[tid] = 0;
        }
        __syncthreads();
        // group offset: rank r's band keys in [bl_r, tau)
        for (int r = 0; r < W; ++r) {
          const uint32_t* s = rx + r * kV2Msg;
          const int lr = (int)(s[0] & 0xFFFFu);
          const int blr = (int)((s[2] & 0xFFFFFFu) & 0xFFu);
          uint32_t c = (tid >= blr && tid < tau - lr && s_tok[r]) ? v2_cnt(s, tid) : 0u;
          for (int o = 16; o > 0; o >>= 1) c += __shfl_xor_sync(0xffffffffu, c, o);
          if ((tid & 31) == 0 && c) atomicAdd(&s_toff[r], (int)c);
        }
        __syncthreads();
        V2DBG(21);
        if (tid == 0) {
          int tot_t = 0, nr = 0;
          bool band_ok = true, fit = true;
          for (int r = 0; r < W; ++r) {
            tot_t += s_tc[r];
            nr += s_tc[r] > 0;
            if (s_tc[r] > 0 && !s_tok[r]) band_ok = false;
            fit = fit && s_tc[r] <= kV2Tie;
          }
          const int mine = s_tc[p.rank];
          if (tot_t == need) { s_tier = 1; s_q = mine; }
          else if (nr == 1) { s_tier = 1; s_q = mine > 0 ? need : 0; }
          else if (band_ok) s_tier = 5;
          else s_tier = fit ? 2 : 3;
        }
        __syncthreads();
        V2DBG(22);
        if (s_tier == 5) {
          // the need-th smallest id over every rank's tie group (ids distinct; groups in
          // no particular order): gather them into U (hist), rank by counting
          uint32_t* U = hist;
          __shared__ int s_uo[kV2MaxW + 1];
          if (tid == 0) {
            int o = 0;
            for (int r = 0; r < W; ++r) { s_uo[r] = o; o += s_tc[r]; }
            s_uo[W] = o;
            s_gs = -1;
            s_q = 0;
          }
          __syncthreads();
          for (int r = 0; r < W; ++r)
            for (int j = tid; j < s_tc[r]; j += kV2T)
              U[s_uo[r] + j] = rx[r * kV2Msg + kV2Fix + s_toff[r] + j] & 0xFFFFFFu;
          const int nu = s_uo[W];
          for (int j = nu + tid; j < ((nu + 3) & ~3); j += kV2T) U[j] = 0xFFFFFFFFu;
          __syncthreads();
          const uint4* U4 = reinterpret_cast<const uint4*>(U);
          for (int i = tid; i < nu; i += kV2T) {
            const uint32_t x = U[i];
            int rk = 0;
            for (int j = 0; j < (nu + 3) >> 2; ++j) {
              const uint4 w4 = U4[j];
              rk += (w4.x < x) + (w4.y < x) + (w4.z < x) + (w4.w < x);
            }
            if (rk == need - 1) s_gs = (int)x;
          }
          __syncthreads();
          {
            const int o = s_uo[p.rank], n = s_tc[p.rank];
            int c = 0;
            for (int j = tid; j < n; j += kV2T) c += (int)(U[o + j] <= (uint32_t)s_gs);
            for (int o2 = 16; o2 > 0; o2 >>= 1) c += __shfl_xor_sync(0xffffffffu, c, o2);
            if ((tid & 31) == 0 && c) atomicAdd(&s_q, c);
          }
          __syncthreads();
          if (tid == 0) {
            if (s_gs < 0) spin_fail(p.err, 33);
            s_tier = 1;
          }
          __syncthreads();
        }
      }
    }
    __syncthreads();
    V2DBG(5);
    int tier = s_tier;
    if (tier == 1) V2STAT(1);
    if (tier == 4) V2STAT(4);

    // ---- T2: tie round (complete sorted tie lists at tau, <= kV2Tie each)
    if (tier == 2) {
      V2STAT(2);
      const int tau = s_tau, need = s_need;
      __shared__ int s_tn[kV2MaxW];
      if (tid < W) {
        const uint32_t* s = rx + tid * kV2Msg;
        s_tn[tid] = (int)v2_cnt(s, tau - (int)(s[0] & 0xFFFFu));
      }
      if (tid == 0) s_cnt = 0;
      __syncthreads();
      if (fast && tau >= th) {
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
#pragma unroll
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
    if (tier == 3) {
      V2STAT(3);
      if (tid == 0) { s_pref = 0ull; s_cnt = K; }
      __syncthreads();
      for (int rd = 0; rd < kV2RadR; ++rd) {
        const int shb = 32 - 8 * rd;
        const unsigned long long pref = s_pref;
        if (tid < 256) hist[tid] = 0;
        __syncthreads();
        for_keys(0, ng, [&](int i0, const uint4& r) {
          const uint32_t valid = 0xFFu;
          const Keys8 k8(r);
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const uint32_t k = k8.key(j);
            if (!((valid >> j) & 1u) || !k) continue;
            const unsigned long long c =
                ((unsigned long long)k << 24) | (0xFFFFFFu - gid_of(i0 + j));
            if (rd > 0 && (c >> (shb + 8)) != pref) continue;
            atomicAdd(&hist[(c >> shb) & 0xFFu], 1u);
          }
        });
        const int64_t rrow = p.off_rad + (((int64_t)(par * p.cap + row) * kV2RadR + rd) * W) * 256;
        if (tid < 64) {
          const int j = tid * 4;
          const uint4 v = make_uint4(tagw(tag, hist[j]), tagw(tag, hist[j + 1]), tagw(tag, hist[j + 2]),
                                     tagw(tag, hist[j + 3]));
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
        const uint32_t ex = v2_scan(hb, warp_sums, &tot);
        const uint32_t want = (uint32_t)s_cnt;
        if (tid == 0) s_n = 0;
        __syncthreads();
        if (tid < 256 && ex < want && ex + hb >= want) {
          s_pref = (pref << 8) | (unsigned long long)(255 - tid);
          s_cnt = (int)(want - ex);
          s_n = 1;
        }
        for (int i = tid; i < W * 256; i += kV2T) st_u32(p.box + rrow + i, kSent);
        __syncthreads();
        if (tid == 0 && s_n != 1) spin_fail(p.err, 39);  // >= K distinct composites exist
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
#pragma unroll
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
#pragma unroll
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
        const uint32_t pre = v2_scan(cg | (ce << 16), warp_sums, &tot);
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
    V2DBG(8);
    __syncthreads();  // SMEM reused by the next row
  }
#undef V2DBG
#undef V2STAT
}

}  // namespace k31dcpx
