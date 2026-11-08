// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Kimi K3.1 DSA decode under DCP: NCCL-free exchanges over NVLink symmetric
// memory (Lamport mailboxes, no fences on the fast path).
//
// 1. dcp_topk_merge: exact global top-K over the DCP ranks' local indexer
//    logits by histogram threshold. Each rank histograms its own bf16 logits
//    (16-bit order-preserving keys), the ranks exchange small histogram windows
//    (1 round at 16-bit resolution, or a coarse 1/8-octave round + a 16-key
//    refine round), every rank derives the same K-th key ``tau`` and the same
//    per-rank tie quotas, and each rank emits only its own selected local
//    tokens (as KV slots or local positions) plus their count. The selected
//    set has exactly K tokens: every token with key > tau, plus the ``need``
//    tokens with key == tau that have the smallest GLOBAL token ids (the stock
//    stable merge's rule, independent of the DCP size and interleave). When
//    ties at tau straddle the cut on two or more ranks, one extra round sends
//    each rank's first tied global ids (bounded radix rounds over global ids
//    if a rank has more ties than one message holds). The output is
//    deterministic: tokens above tau in position order, then the selected ties
//    in position order (block scans, no atomics).
//    A rank whose local top-K spans more than the window sends a "below"
//    count; if the boundary falls under such a rank's window the kernel runs
//    further coarse rounds anchored at an agreed top (rare; still exact).
//    Centred 1-round mode (the decode default): each rank's window is centred
//    at its local estimate of tau (its local ceil(K * L / G)-th key) and the
//    message also carries the rank's 256-key bin counts; every CTA of the row
//    resolves tau itself, and a row whose tau falls outside the windows takes
//    one refine round over tau's 256-key bin (2 rounds worst case). With a known
//    rank quota every CTA writes its own tokens and ties (no leader tail).
// 2. dcp_o_exchange: per-row partial outputs [T, H, D] + LSE are pushed to the
//    head owners (rank r owns heads [r*HPR, (r+1)*HPR)) by direct peer stores
//    into sentinel-armed mailboxes; the owner polls, merges the W partials in
//    FP32 with base-2 LSE weights, rounds once to bf16 and re-arms the slots.
//    D = 128 is the post-W_UV design (each rank applies W_UV to its own
//    partial first, 4x fewer bytes than the D = 512 latent design).
//
// Mailbox words carry an 8-bit tag (epoch % 255) in the top byte; 0xFFFFFFFF
// (tag 0xFF) is the armed state. Device epochs make CUDA graph replays carry no
// host state; consumers re-arm what they read. Mailboxes alternate between two
// parities (epoch & 1); a parity may be reused only after every peer finished
// its previous use. topk_merge keeps one epoch per row, advanced by the row's
// leader CTA after each launch that exchanges on the row (consecutive exchanges
// of a row alternate parity, and each one completes only after every peer took
// part in it); it counts modulo 510 = lcm(255, 2), so tag and parity stay
// consistent across the wrap and it never overflows. A (tag, parity) pair thus
// returns every 510 uses of a row: every mailbox word a use may poll is either
// rewritten on every use of the slot or re-armed after it is read. The O exchange
// keeps one parity bit per launch (the last CTA flips it).
// Launches must stay stream-ordered (no PDL overlap between two launches of the
// same exchange).
// A spin timeout (peer missing / desync) records the code in ``err`` and traps:
// it never continues with stale mailbox data.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <cstdlib>
#include <map>
#include <utility>

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace cg = cooperative_groups;

namespace k31dcpx {

constexpr uint32_t kSent = 0xFFFFFFFFu;
constexpr int kMaxW = 16;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kCoarseBins = 4096;  // key >> 4: sign + 8 exponent + 3 mantissa
constexpr int kCoarseWin = 32;     // coarse bins per coarse message (4 octaves)
constexpr int kCoarseWords = 20;   // 2 header + 16 count words, padded to 4
constexpr int kFineWords = 8;      // 16 keys as 8 pair words
constexpr int kG = 1024;           // resolve window (units)
constexpr uint32_t kCountMax = 4095;
// Default cross-rank spin deadline (tests / warmup): a peer that never arrives traps the
// kernel with an error code instead of hanging. Serving raises it (or disables it, 0)
// through set_spin_ns: a peer rank delayed on the host (cold JIT, GC, a P/D handshake)
// must not kill the engine where a stock collective would simply wait.
constexpr long long kSpinNsDefault = 5LL * 1000 * 1000 * 1000;
constexpr int kTieList = 64;     // tied global ids per rank in the tie round
constexpr int kTieBuckets = 64;  // global-id buckets per tie radix round
constexpr int kTieWords = kTieBuckets / 2;
constexpr int kTieStore = kG;    // tied local positions kept in SMEM per row
constexpr int kMaxK = 2048;      // top-k sizes the packed WRITE scan supports

__device__ __forceinline__ void spin_fail(int* err, int code) {
  atomicCAS(err, 0, code);
  __threadfence_system();
#ifdef K31_DCPX_DEBUG_PRINT
  // debug builds: record the code (python check() raises) and continue (no trap)
  printf("k31dcpx error %d block (%d,%d) thread %d\n", code, blockIdx.x, blockIdx.y, threadIdx.x);
#else
  __trap();
#endif
}

// ------------------------------------------------------------------ memory ops
__device__ __forceinline__ uint4 ld_sys_v4(const uint32_t* p) {
  uint4 v;
  asm volatile("ld.relaxed.sys.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}
__device__ __forceinline__ uint32_t ld_sys(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_v4(uint32_t* p, uint4 v) {
  asm volatile("st.relaxed.sys.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x),
               "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}
__device__ __forceinline__ void st_u32(uint32_t* p, uint32_t v) {
  asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void mc_st_v4(uint64_t addr, uint4 v) {
  asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(addr),
               "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}
__device__ __forceinline__ long long now_ns() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
// Whether a spin that started at t0 is past its deadline (spin_ns <= 0: never).
__device__ __forceinline__ bool spin_late(long long spin_ns, long long t0) {
  return spin_ns > 0 && now_ns() - t0 > spin_ns;
}

// bf16 bits -> order-preserving 16-bit key; NaN -> 0 (never selected).
__device__ __forceinline__ uint32_t bf16_key(uint32_t u) {
  u &= 0xFFFFu;
  if ((u & 0x7FFFu) > 0x7F80u) return 0u;
  return (u & 0x8000u) ? (~u & 0xFFFFu) : (u | 0x8000u);
}

__device__ __forceinline__ uint32_t tagw(uint32_t tag, uint32_t payload24) {
  return (tag << 24) | (payload24 & 0xFFFFFFu);
}
__device__ __forceinline__ uint32_t pack2(uint32_t tag, uint32_t a, uint32_t b) {
  a = a > kCountMax ? kCountMax : a;
  b = b > kCountMax ? kCountMax : b;
  return (tag << 24) | (a << 12) | b;
}

// ------------------------------------------------------------------ block scan
// Exclusive prefix over the block (512 threads); returns the total in *total.
__device__ __forceinline__ uint32_t block_excl_scan(uint32_t v, uint32_t* warp_sums,
                                                    uint32_t* total) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  uint32_t x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    uint32_t y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x += y;
  }
  __syncthreads();
  if (lane == 31) warp_sums[warp] = x;
  __syncthreads();
  if (warp == 0) {
    uint32_t s = lane < kWarps ? warp_sums[lane] : 0u;
    uint32_t t = s;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      uint32_t y = __shfl_up_sync(0xffffffffu, t, o);
      if (lane >= o) t += y;
    }
    if (lane < kWarps) warp_sums[lane] = t - s;
    if (lane == kWarps - 1) warp_sums[kWarps] = t;
  }
  __syncthreads();
  uint32_t r = warp_sums[warp] + x - v;
  *total = warp_sums[kWarps];
  __syncthreads();
  return r;
}

// ------------------------------------------------------------------ top-k merge
struct TopkParams {
  const uint16_t* logits;  // bf16 bits [T, lstride]
  int64_t lstride;
  const int* local_len;   // [T]
  const int* global_len;  // [T]
  int* out;               // [T, ostride] (>= K columns)
  int64_t ostride;
  int* out_count;  // [T]
  const int* block_table;  // [B, bt_stride] or null (then out = local positions)
  int64_t bt_stride;
  int block_size, next_n;
  int block_stride;        // slot rows per block-table entry (>= block_size)
  const int* req_of_row;   // [T] request of each row, or null (row / next_n)
  int tie_rank;            // 1: ties at tau to ranks in rank order (no tie round)
  uint32_t* box;          // local mailbox base (u32 words)
  uint64_t mc;            // multicast address of box (0 -> peer stores)
  const uint64_t* peers;  // [W] peer box bases
  int W, rank, K, T, cap, cap1, nw1;  // nw1: 1-round window keys (0 = 2 rounds)
  int w1c;  // 1-round window: 0 ends at the local maximum, 1 centred at the local estimate of tau
  int64_t off_w1, off_c1, off_f1, off_r8, off_r4;
  int64_t off_tl, off_tr, off_tk;  // tie list (kTieList), tie radix (kTieWords), row-0 token (4)
  int interleave;                  // DCP interleave (local position -> global id)
  int small_ctx;
  int* stats;      // optional per-path row counters (see kSt*)
  long long* dbg;  // optional phase timestamps (row 0, leader, tid 0)
  int pdl;  // launched with programmatic stream serialization: release dependents early
  int* epoch;
  int* ticket;
  int* err;
  int segmax;
  long long spin_ns;  // cross-rank spin deadline (<= 0: none)
};

enum Act : int {
  kActNone = 0, kActFine = 1, kActWrite = 3, kActTieHist = 4,
  kActTieWrite = 5, kActDone = 6, kActWriteFast = 7, kActWriteDirect = 8
};
// stats counters (rows)
enum St : int {
  kStRows = 0, kSt1R = 1, kSt2R = 2, kStFallback = 3, kStTieRound = 4, kStTieRadix = 5,
  kStTieWritePass = 6, kStSelectAll = 7
};

struct Ctl {
  int act, lo, n;          // fine histogram request: keys [lo, lo + n)
  int64_t rearm_off;       // WRITE: mailbox words [rearm_off, + rearm_n) to re-arm
  int rearm_n;
  uint32_t tau;            // selected: key > tau, plus the tie quota of key == tau
  int gt, eq;              // this CTA's counts (COUNT / WRITE)
  int gt_off, eq_off;      // this CTA's offsets among the rank's gt tokens / ties
  int tie_lo, tie_w;       // TIEHIST: global-id buckets [tie_lo + b * tie_w, +tie_w)
  int q, gtt;              // TIEWRITE: rank tie quota, rank total of gt tokens
};

// Leader-only resolve state.
struct Res {
  int gtop, glo, f, kstar, status;  // status: 1 resolved, 2 unresolved, 3 select all
  uint32_t above;  // global count strictly above the boundary unit
  uint32_t carry;  // global count at or above F (unresolved)
};


__device__ __forceinline__ void send_msg(const TopkParams& p, const uint32_t* msg, int words,
                                         int64_t rowoff, int tid) {
  for (int j = tid * 4; j < words; j += kThreads * 4) {
    uint4 v = make_uint4(msg[j], msg[j + 1], msg[j + 2], msg[j + 3]);
    int64_t off = rowoff + (int64_t)p.rank * words + j;
    if (p.mc) mc_st_v4(p.mc + off * 4, v);
    else
      for (int d = 0; d < p.W; ++d) st_v4(reinterpret_cast<uint32_t*>(p.peers[d]) + off, v);
  }
}

__device__ __forceinline__ bool tagged(const uint4& v, uint32_t tag) {
  return (v.x >> 24) == tag && (v.y >> 24) == tag && (v.z >> 24) == tag && (v.w >> 24) == tag;
}
// Waits for total (<= NP * kThreads * 4) tagged words at rb and copies them to rx. Every
// thread issues all of its loads before re-polling the ones not yet arrived, so the
// L2 round trips of one thread overlap.
template <int NP>
__device__ __forceinline__ void poll_msg(const TopkParams& p, uint32_t* rx, int64_t rowoff,
                                         int total, uint32_t tag, int tid, int code) {
  const uint32_t* rb = p.box + rowoff;
  uint4 v[NP];
#pragma unroll
  for (int i = 0; i < NP; ++i) {
    const int j = (tid + i * kThreads) * 4;
    if (j < total) v[i] = ld_sys_v4(rb + j);
  }
  const long long t0 = now_ns();
#pragma unroll
  for (int i = 0; i < NP; ++i) {
    const int j = (tid + i * kThreads) * 4;
    if (j >= total) continue;
    while (!tagged(v[i], tag)) {
      if (spin_late(p.spin_ns, t0)) { spin_fail(p.err, code); break; }
      v[i] = ld_sys_v4(rb + j);
    }
    *reinterpret_cast<uint4*>(rx + j) = v[i];
  }
}

// Waits for W ranks x n tagged words (n a multiple of 4; rank r's at rb + r * slot)
// and copies them to rx + r * dst (as poll_msg: every load issued before re-polling).
template <int NP>
__device__ __forceinline__ void poll_ranks(const TopkParams& p, uint32_t* rx, const uint32_t* rb,
                                           int slot, int n, int dst, uint32_t tag, int tid,
                                           int code) {
  uint4 v[NP];
  const int total = p.W * n;
#pragma unroll
  for (int i = 0; i < NP; ++i) {
    const int j = (tid + i * kThreads) * 4;
    if (j < total) v[i] = ld_sys_v4(rb + (j / n) * slot + j % n);
  }
  const long long t0 = now_ns();
#pragma unroll
  for (int i = 0; i < NP; ++i) {
    const int j = (tid + i * kThreads) * 4;
    if (j >= total) continue;
    while (!tagged(v[i], tag)) {
      if (spin_late(p.spin_ns, t0)) { spin_fail(p.err, code); break; }
      v[i] = ld_sys_v4(rb + (j / n) * slot + j % n);
    }
    *reinterpret_cast<uint4*>(rx + (j / n) * dst + j % n) = v[i];
  }
}

// Largest unit u in [0, n) (n even, <= 1024) with sum(G[u..n)) >= need -> res->kstar,
// res->above = sum(G[kstar+1..n)); also res->carry = sum(G[f..n)) for f = res->f - glo
// handled by callers that need it. Caller resets res->kstar = -1 before.
__device__ __forceinline__ void suffix_find(const uint32_t* G, int n, uint32_t need, Res* res,
                                            uint32_t* warp_sums, int tid) {
  // thread t owns units [hi - per, hi), hi = n - t * per (thread 0 owns the top units)
  const int per = (n + kThreads - 1) / kThreads;
  const int hi = n - tid * per, lo = max(0, hi - per);
  uint32_t gs = 0;
  for (int u = lo; u < hi; ++u) gs += G[u];
  uint32_t tot;
  const uint32_t ex = block_excl_scan(gs, warp_sums, &tot);
  uint32_t run = ex;  // count of units above u
  for (int u = hi - 1; u >= lo; --u) {
    if (run < need && run + G[u] >= need) {
      res->kstar = u;  // unique
      res->above = run;
    }
    run += G[u];
  }
  __syncthreads();
}


// Entries of a sorted tagged-id list (24-bit payloads) that are < x.
__device__ __forceinline__ int count_below(const uint32_t* list, int n, uint32_t x) {
  int lo = 0, hi = n;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if ((list[mid] & 0xFFFFFFu) < x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

__device__ __forceinline__ void load8_keys(const uint16_t* lrow, int i, int n, bool vec,
                                           uint32_t (&k8)[8]) {
  if (vec && i + 8 <= n) {
    uint4 v = *reinterpret_cast<const uint4*>(lrow + i);
    uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      k8[2 * j] = bf16_key(w[j] & 0xFFFFu);
      k8[2 * j + 1] = bf16_key(w[j] >> 16);
    }
  } else {
#pragma unroll
    for (int j = 0; j < 8; ++j) k8[j] = (i + j < n) ? bf16_key(lrow[i + j]) : 0u;
  }
}
constexpr int kUnroll = 1;  // 16-byte loads in flight per thread (4: slower, reg-capped)

// 8 bf16 logits [i, i+8) as raw words (out-of-range halves read as -NaN 0xFFFF,
// whose key is 0 = never selected).
__device__ __forceinline__ uint4 load8_raw(const uint16_t* lrow, int i, int n, bool vec) {
  if (vec && i + 8 <= n) return *reinterpret_cast<const uint4*>(lrow + i);
  uint32_t w[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    uint32_t lo = (i + 2 * j < n) ? lrow[i + 2 * j] : 0xFFFFu;
    uint32_t hi = (i + 2 * j + 1 < n) ? lrow[i + 2 * j + 1] : 0xFFFFu;
    w[j] = lo | (hi << 16);
  }
  return make_uint4(w[0], w[1], w[2], w[3]);
}
__device__ __forceinline__ uint32_t raw_key(const uint4& v, int j) {
  const uint32_t w = (j >> 1) == 0 ? v.x : (j >> 1) == 1 ? v.y : (j >> 1) == 2 ? v.z : v.w;
  return bf16_key((j & 1) ? (w >> 16) : (w & 0xFFFFu));
}

// Visits this CTA's keys (unordered): fn(i, key) for i in [0, n). The SMEM copy
// holds the raw bf16 bits.
template <typename F>
__device__ __forceinline__ void for_keys(const uint16_t* keys, bool smem, const uint16_t* lrow,
                                         int n, int tid, F fn) {
  if (smem) {
    for (int i = tid; i < n; i += kThreads) fn(i, bf16_key(keys[i]));
    return;
  }
  const bool vec = ((reinterpret_cast<uintptr_t>(lrow) & 15) == 0);
  for (int b = tid * 8; b < n; b += kThreads * 8 * kUnroll) {
    uint4 raw[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) raw[u] = load8_raw(lrow, b + u * kThreads * 8, n, vec);
#pragma unroll
    for (int u = 0; u < kUnroll; ++u)
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int i = b + u * kThreads * 8 + j;
        if (i < n) fn(i, raw_key(raw[u], j));
      }
  }
}

template <int CS_MAX>
__global__ void __launch_bounds__(kThreads, 4) topk_merge_kernel(TopkParams p) {
  cg::cluster_group cl = cg::this_cluster();
  const int crank = (int)cl.block_rank();
  const int cs = (int)cl.num_blocks();
  const int tid = threadIdx.x;
  const bool leader = crank == 0;
#define DBG(i) \
  if (p.dbg && row == 0 && crank == 0 && threadIdx.x == 0) p.dbg[i] = now_ns();
  // dbg (32 stamps, row 0, leader, thread 0): 0 row start, 1 pass A, 2 S1, 4 1R send
  // start, 16 estimate, 17 window counts, 19 1R header, 20 1R sent, 5 1R polled, 21 1R
  // accumulated, 22 1R resolved, 23 1R done, 6 quota, 7 WRITE, 12 bitmap, 24 offsets,
  // 15 enumerated, 11 ties, 8 row end
#define STAT(i) \
  if (p.stats && tid == 0) atomicAdd(&p.stats[i], 1);

  extern __shared__ uint16_t keys[];  // [segmax]
  __shared__ __align__(16) uint32_t hist[kCoarseBins];
  __shared__ uint32_t G[kG];
  __shared__ __align__(16) uint32_t fine[512];
  __shared__ uint32_t fine2[512];
  __shared__ uint32_t msg[264];
  __shared__ __align__(16) uint32_t rx[kMaxW * 260];  // received messages
  __shared__ uint32_t warp_sums[kWarps + 1];
  __shared__ int s_base[kMaxW], s_below[kMaxW], s_adj[kMaxW], s_ties[kMaxW];
  __shared__ Ctl ctl;
  __shared__ Res res;
  __shared__ int s_tag, s_par, s_enext;
  __shared__ uint32_t s_below_tot;
  __shared__ int s_tcnt;  // leader: ties appended to G[] by the fast WRITE
  // leader-only row state
  __shared__ int s_need;         // ties at tau to select over all ranks
  __shared__ int s_tmode, s_q;   // tie mode (0 quota known, 1 tie round), rank quota
  __shared__ int s_gtt, s_eqt;   // this rank's tokens above tau / ties at tau
  __shared__ int s_tlo, s_tw, s_tgb, s_trr;  // tie radix state
  __shared__ int s_tbl[kMaxW];   // per-rank ties below s_tlo
  __shared__ int s_padded;       // leader: the CTAs padded [count, K) in WRITE
  __shared__ int64_t s_rearm_off;  // leader: mailbox words to re-arm in WRITE (centred 1R)
  __shared__ int s_rearm_n;
  __shared__ uint32_t s_seq;     // act broadcasts received (non-leaders)
  __shared__ int s_ag;           // WRITE: this rank's tokens above tau
  __shared__ uint32_t s_x8[256];  // centred 1R: keys above each 256-key bin (cluster sums)
  __shared__ int s_b8c, s_top8, s_cq;
  __shared__ uint32_t s_wlo;  // centred 1R: (row + 1) << 17 | window start, from the leader
  __shared__ uint32_t s_above_c1r;  // centred 1R, leader: keys above the window
  __shared__ int s_fkey_c1r;        // centred 1R fallback: the octave rounds' key bound
  __shared__ int s_cnt2[2 * 16];    // centred 1R: every CTA's (tokens above tau, ties)
  __shared__ uint32_t s_myabove;    // centred 1R: this CTA's keys above the window
  __shared__ uint32_t s_v8[256];    // centred 1R, leader: the 256-key bin counts (sums)
  __shared__ uint32_t s_a8c;

  // PDL: the launch, CTA placement and the shared-memory carve-out above overlap
  // the logits kernel's tail; every global read (logits, lengths, epoch,
  // mailboxes) follows its completion. Dependents are released only after the
  // row loop (below): an early release at entry let the sparse core launch while
  // this cross-rank lockstep grid ran and hung the B8 step (trtllm-gen, 64 rows;
  // root cause open), so no dependent ever shares the GPU with the lockstep phase.
  // Shared-memory only (overlaps the predecessor under PDL): every CTA's coarse
  // histogram and the leader's fine sums start at zero, and no CTA adds into the
  // leader's before it is cleared. Rows are re-cleared at the end of each row.
  for (int b = tid; b < kCoarseBins; b += kThreads) hist[b] = 0;
  for (int b = tid; b < 512; b += kThreads) fine2[b] = 0;
  if (tid == 0) { s_wlo = 0; s_seq = 0; s_fkey_c1r = 0; }
  if (cs > 1) cl.sync();
  else __syncthreads();
  asm volatile("griddepcontrol.wait;" ::: "memory");

  // Persistent grid (host caps gridDim.y at the co-resident cluster count):
  // every rank walks the rows in the same order, so the cross-rank waits of a
  // row never depend on a non-resident CTA.
  bool arrived = false;  // this CTA arrived at a split cluster barrier it has not waited on
  uint32_t bseq = 0;     // act broadcasts sent (leader) / awaited (others)
  for (int row = blockIdx.y; row < p.T; row += gridDim.y) {
  DBG(0);
  const int L = p.local_len[row];
  const int Gl = p.global_len[row];
  const int K = p.K;
  int seg = (L + cs - 1) / cs;
  seg = (seg + 7) & ~7;
  const int s0 = min(L, crank * seg);
  const int s1 = min(L, s0 + seg);
  const bool smem_keys = p.segmax > 0;
  const int n = smem_keys ? min(s1 - s0, p.segmax) : (s1 - s0);
  if (tid == 0 && smem_keys && s1 - s0 > p.segmax) spin_fail(p.err, 5);
  const uint16_t* lseg = p.logits + (int64_t)row * p.lstride + s0;
  int* orow = p.out + (int64_t)row * p.ostride;
  const int req = p.req_of_row != nullptr ? p.req_of_row[row] : row / p.next_n;

  auto slot_of = [&](int pos) -> int {
    if (p.block_table == nullptr) return pos;
    int blk = p.block_table[(int64_t)req * p.bt_stride + pos / p.block_size];
    return blk * p.block_stride + pos % p.block_size;
  };

  if (Gl <= K) {
    // The whole context is selected: no exchange (every rank sees the same
    // global length), and the row's epoch stays (its mailboxes are untouched).
    for (int i = tid; i < n; i += kThreads) orow[s0 + i] = slot_of(s0 + i);
    const int fill = K - L;
    const int per = (fill + cs - 1) / cs;
    for (int i = crank * per + tid; i < min(fill, (crank + 1) * per); i += kThreads)
      orow[L + i] = -1;
    if (leader && tid == 0) p.out_count[row] = L;
    goto next_row;
  }
  if (Gl >= (1 << 24) && tid == 0) spin_fail(p.err, 13);  // global ids are 24-bit
  // Per-row epoch (tag e % 255, mailbox parity e & 1): every rank runs the same
  // rows, and a row's epoch advances once per launch that exchanges on it, so a
  // row's mailbox parity alternates between its consecutive exchanges, and a rank
  // can only write a row's parity p after every peer exchanged on that row at
  // p ^ 1, i.e. finished (and re-armed) the previous use of p. The load overlaps
  // pass A (read after the first barrier).
  if (tid == 0) {  // every CTA (centred 1R: every CTA polls the row's messages)
    const int e = p.epoch[row];
    s_tag = e % 255;
    s_par = e & 1;
    s_enext = e + 1 == 510 ? 0 : e + 1;  // epochs live in [0, 510)
  }

  {
    if (leader) STAT(kStRows);
    // ---- pass A: keys into shared memory, local coarse histogram (zeroed in the
    // prologue / at the previous row's end)
    const bool vec = ((reinterpret_cast<uintptr_t>(lseg) & 15) == 0);
    for (int b0 = tid * 8; b0 < n; b0 += kThreads * 8 * kUnroll) {
      uint4 raw[kUnroll];
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) raw[u] = load8_raw(lseg, b0 + u * kThreads * 8, n, vec);
#pragma unroll
      for (int u = 0; u < kUnroll; ++u)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int i = b0 + u * kThreads * 8 + j;
          const uint32_t k = raw_key(raw[u], j);
          if (i < n) {
            if (smem_keys) {
              const uint32_t w = (j >> 1) == 0 ? raw[u].x : (j >> 1) == 1 ? raw[u].y
                               : (j >> 1) == 2 ? raw[u].z : raw[u].w;
              keys[i] = (uint16_t)((j & 1) ? (w >> 16) : (w & 0xFFFFu));
            }
            if (k != 0u) atomicAdd(&hist[k >> 4], 1u);
          }
        }
    }
    DBG(1);
    if (!leader) {
      // add this CTA's non-zero bins into the leader's histogram (DSMEM reductions;
      // the leader's own atomics commute with them)
      __syncthreads();
      uint32_t* lh = cl.map_shared_rank(hist, 0);
      for (int b = tid; b < kCoarseBins; b += kThreads)
        if (hist[b]) atomicAdd(&lh[b], hist[b]);
    }
    cl.sync();  // S1: the leader's hist holds the cluster sums
    DBG(2);
    const uint32_t tag = (uint32_t)s_tag;
    const int par = s_par;
    // Centred 1-round window: every CTA derives the same window from the leader's
    // summed histogram (no broadcast barrier) and adds its keys in it to the
    // leader's fine2[]; the leader then starts at the 1R send (stage 11).
    const bool c1r = p.w1c && p.nw1 > 0 && row < p.cap1;
    int c1r_lo = 0;
    if (c1r) {
      // The leader derives the window from its summed histogram (thread t: the
      // 256-key bin t = coarse bins [16t, 16t + 16)) and publishes it to every CTA
      // with one release store each (bulk DSMEM reads of the leader's histogram by
      // every CTA are far slower); the others spin on their own copy.
      if (leader) {
        const uint4* lh4 = reinterpret_cast<const uint4*>(hist) + tid * 4;
        uint32_t cb16[16];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const uint4 v = lh4[i];
          cb16[4 * i] = v.x; cb16[4 * i + 1] = v.y; cb16[4 * i + 2] = v.z; cb16[4 * i + 3] = v.w;
        }
        uint32_t v8 = 0;
#pragma unroll
        for (int i = 0; i < 16; ++i) v8 += cb16[i];
        // exclusive scan from the top: thread t holds bin 255 - t
        if (tid == 0) { s_b8c = -1; s_top8 = -1; }
        s_v8[tid] = v8;
        __syncthreads();
        const int bin = 255 - tid;
        const uint32_t vb = s_v8[bin];
        uint32_t tot8;
        const uint32_t ex = block_excl_scan(vb, warp_sums, &tot8);
        s_x8[bin] = ex;
        const uint32_t q = (uint32_t)max(1LL, ((long long)K * L + Gl - 1) / Gl);
        if (ex < q && ex + vb >= q) { s_b8c = bin; s_a8c = ex; }
        if (vb > 0 && ex == 0) s_top8 = bin;
        __syncthreads();
        int b8 = s_b8c;
        uint32_t need = q, cum = s_a8c;
        if (b8 < 0) { b8 = s_top8; need = 1; cum = 0; }  // fewer than q valid keys: the top
        if (b8 >= 0 && tid == b8) {
          int cq = b8 * 16;
          for (int k = 15; k >= 0; --k) {
            if (cum + cb16[k] >= need) { cq = b8 * 16 + k; break; }
            cum += cb16[k];
          }
          s_cq = cq;
        }
        if (tid == 0 && b8 < 0) s_cq = 0;
        __syncthreads();
        const int nb = p.nw1 / 16;
        c1r_lo = min(max(0, s_cq - nb / 2), kCoarseBins - nb) * 16;
        const uint32_t word = ((uint32_t)(row + 1) << 17) | (uint32_t)c1r_lo;
        if (tid > 0 && tid < cs) {
          uint32_t* f = cl.map_shared_rank(&s_wlo, tid);
          asm volatile("st.release.cluster.u32 [%0], %1;" ::"l"(f), "r"(word) : "memory");
        }
        // the 1R header while the CTAs count: keys at or above coarse bin c are
        // s_x8[c / 16] + coarse bins [c, c | 15] (warp 0: every key, 1: >= lo,
        // 2: >= lo + nw1)
        {
          __shared__ uint32_t s_ge[3];
          const int w = tid >> 5, lane = tid & 31;
          if (w < 3) {
            const int c = w == 0 ? 0 : w == 1 ? c1r_lo / 16 : (c1r_lo + p.nw1) / 16;
            uint32_t v = 0;
            if (c < kCoarseBins && lane < 16 && lane >= (c & 15)) v = hist[(c & ~15) + lane];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0) s_ge[w] = c < kCoarseBins ? v + s_x8[c >> 4] : 0u;
          }
          __syncthreads();
          if (tid == 0) {
            const uint32_t below = s_ge[0] - s_ge[1];
            msg[0] = tagw(tag, (uint32_t)c1r_lo);
            msg[1] = tagw(tag, below > kCountMax ? kCountMax : below);
            s_above_c1r = s_ge[2];
          }
        }
      } else {
        if (tid == 0) {
          const long long t0 = now_ns();
          uint32_t v;
          while (true) {
            asm volatile("ld.acquire.cluster.u32 %0, [%1];" : "=r"(v) : "l"(&s_wlo) : "memory");
            if ((v >> 17) == (uint32_t)(row + 1)) break;
            if (spin_late(p.spin_ns, t0)) { spin_fail(p.err, 18); break; }
          }
        }
        __syncthreads();
        c1r_lo = (int)(s_wlo & 0x1FFFFu);
      }
      DBG(16);
      if (p.dbg && row == 0 && crank == 1 && tid == 0) p.dbg[30] = now_ns();
      // fine counts of [lo, lo + nw1): every CTA counts its keys locally; the
      // non-leaders store their counts into the leader's rx[] (free until the 1R
      // poll; plain DSMEM stores, which unlike reductions do not serialize at the
      // leader's SM), and the leader sums them.
      const int nn = p.nw1;
      const bool parts = cs <= kMaxW * 260 / 512;
      for (int b = tid; b < nn; b += kThreads) fine[b] = 0;
      if (tid == 0) s_myabove = 0;
      __syncthreads();
      uint32_t myab = 0;  // this CTA's keys above the window
      for_keys(keys, smem_keys, lseg, n, tid, [&](int, uint32_t k) {
        const int u = (int)k - c1r_lo;
        if (u >= 0 && u < nn) atomicAdd(&fine[u], 1u);
        myab += u >= nn;
      });
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) myab += __shfl_xor_sync(0xffffffffu, myab, o);
      if ((tid & 31) == 0 && myab) atomicAdd(&s_myabove, myab);
      DBG(28);
      if (!leader) {
        __syncthreads();
        if (parts) {
          uint4* dst = reinterpret_cast<uint4*>(cl.map_shared_rank(rx, 0) + crank * 512);
          const uint4* src4 = reinterpret_cast<const uint4*>(fine);
          for (int j = tid; j < nn / 4; j += kThreads) dst[j] = src4[j];
        } else {
          uint32_t* lf = cl.map_shared_rank(fine2, 0);
          for (int b = tid; b < nn; b += kThreads)
            if (fine[b]) atomicAdd(&lf[b], fine[b]);
        }
      }
      if (p.dbg && row == 0 && crank == 1 && tid == 0) p.dbg[29] = now_ns();
      cl.sync();  // B: every CTA's window counts are in the leader
      if (leader) {
        DBG(17);
        // the 1R message: pair words of the summed window counts, the count above
        // the window in the first pad word
        const int words = 2 + nn / 2, wpad = (words + 3) & ~3;
        for (int j = tid; j < nn / 2; j += kThreads) {
          uint32_t a = fine[2 * j], b = fine[2 * j + 1];
          if (parts) {
            for (int c = 1; c < cs; ++c) {
              a += rx[c * 512 + 2 * j];
              b += rx[c * 512 + 2 * j + 1];
            }
          } else {
            a += fine2[2 * j], b += fine2[2 * j + 1];
            fine2[2 * j] = 0, fine2[2 * j + 1] = 0;
          }
          msg[2 + j] = pack2(tag, a, b);
        }
        for (int j = words + tid; j < wpad; j += kThreads)
          msg[j] = tagw(tag, j == words ? min(s_above_c1r, 0xFFFFFFu) : 0u);
        if (tid == 0) { ctl.lo = c1r_lo; ctl.act = 0; }
        __syncthreads();
      }
    }

    // ---- centred 1R, every CTA: the leader sends the window counts; every CTA
    // polls the W messages and resolves tau itself (identical on every CTA: no
    // broadcast). With a known quota every CTA then writes its own tokens and ties
    // after one barrier that exchanges the CTAs' counts. Rows whose tau falls
    // outside a window (and global-id ties that straddle ranks) continue in the
    // leader state machine below (stage 20 / 30).
    int c1r_stage = -1;  // -1: row done here
    // this CTA's counts of the window that located tau (the 1R window, or the refine
    // bin), its first key and size; s_myabove: this CTA's keys above it
    uint32_t* c1r_cnt = fine;
    int c1r_clo = c1r_lo, c1r_cn = p.nw1;
    if (c1r) {
      // Mailbox slot per rank (stride wpad + 260): the window message (wpad words),
      // the 256-key bin counts (128 pair words) and a refine message (132 words).
      const int nwk = p.nw1, words = 2 + nwk / 2, wpad = (words + 3) & ~3, slot = wpad + 260;
      const int64_t rowoff = p.off_w1 + ((int64_t)(par * p.cap1 + row) * p.W) * slot;
      const uint32_t* rbox = p.box + rowoff;
      auto send = [&](const uint32_t* m, int n, int at, bool bins) {  // leader
        for (int j = tid * 4; j < n; j += kThreads * 4) {
          uint4 v;
          if (bins)
            v = make_uint4(pack2(tag, s_v8[2 * j], s_v8[2 * j + 1]),
                           pack2(tag, s_v8[2 * j + 2], s_v8[2 * j + 3]),
                           pack2(tag, s_v8[2 * j + 4], s_v8[2 * j + 5]),
                           pack2(tag, s_v8[2 * j + 6], s_v8[2 * j + 7]));
          else
            v = make_uint4(m[j], m[j + 1], m[j + 2], m[j + 3]);
          const int64_t off = rowoff + (int64_t)p.rank * slot + at + j;
          if (p.mc) mc_st_v4(p.mc + off * 4, v);
          else
            for (int d = 0; d < p.W; ++d) st_v4(reinterpret_cast<uint32_t*>(p.peers[d]) + off, v);
        }
      };
      if (leader) {
        DBG(19);
        send(msg, wpad, 0, false);
        send(nullptr, 128, wpad, true);
        DBG(20);
      }
      for (int b = tid; b < kG; b += kThreads) G[b] = 0;  // while the hop is in flight
      poll_ranks<(kMaxW * 260 + kThreads * 4 - 1) / (kThreads * 4)>(p, rx, rbox, slot, wpad, wpad,
                                                                    tag, tid, 1);
      __syncthreads();
      DBG(5);
      // tau from the W windows (every CTA, identically): the counts of each rank's
      // window keys, its count above the window (placed at the window top) and the
      // bounds F (max window start of the ranks with keys below their window) and
      // ub (min window top of the ranks with keys above it) between which the summed
      // counts are exact. Result in s_cst (1 resolved, 2 not, 3 select all), s_ctau,
      // s_need, s_ties.
      __shared__ int s_cst, s_ctau;
      __shared__ uint32_t s_ctop;
      __shared__ int s_cub;
      auto resolve = [&](int stride, int nw) {
        // G[u] = every rank's count of key glo + u over the top kG keys of the
        // windows (G zeroed by the caller); F / ub bound where the sums are exact
        const int wd = 2 + nw / 2;
        if (tid < p.W) {
          s_base[tid] = (int)(rx[tid * stride] & 0xFFFFu);
          s_below[tid] = (int)(rx[tid * stride + 1] & 0xFFFFFFu);
          s_adj[tid] = 0;
        }
        if (tid == 0) { res.gtop = -1; res.f = -1; res.kstar = -1; s_ctop = 0; s_cub = 1 << 30; }
        __syncthreads();
        if (tid < p.W) atomicMax(&res.gtop, s_base[tid] + nw - 1);
        __syncthreads();
        const int glo = max(0, res.gtop - (kG - 1));
        if (tid < p.W) {
          const uint32_t a = rx[tid * stride + wd] & 0xFFFFFFu;
          if (a) {
            const int pos = max(0, s_base[tid] + nw - glo);
            if (pos < kG) atomicAdd(&G[pos], a);
            else atomicAdd(&s_ctop, a);
            atomicMin(&s_cub, s_base[tid] + nw);
          }
        }
        for (int j = tid; j < p.W * (nw / 2); j += kThreads) {
          const int s2 = j / (nw / 2), w = j % (nw / 2);
          const uint32_t v = rx[s2 * stride + 2 + w];
          const uint32_t a = (v >> 12) & 0xFFFu, b = v & 0xFFFu;
          const int u = s_base[s2] + 2 * w;
          if (a) { if (u >= glo) atomicAdd(&G[u - glo], a); else atomicAdd(&s_adj[s2], (int)a); }
          if (b) { if (u + 1 >= glo) atomicAdd(&G[u + 1 - glo], b); else atomicAdd(&s_adj[s2], (int)b); }
        }
        __syncthreads();
        if (tid < p.W && s_below[tid] + s_adj[tid] > 0) atomicMax(&res.f, max(s_base[tid], glo));
        __syncthreads();
        const uint32_t top = s_ctop;
        if (top < (uint32_t)K) suffix_find(G, kG, (uint32_t)K - top, &res, warp_sums, tid);
        if (tid == 0) {
          const int F = res.f;
          int cst;
          if (top >= (uint32_t)K) cst = 2;
          else if (res.kstar >= 0 && glo + res.kstar >= F && glo + res.kstar < s_cub) cst = 1;
          else if (res.kstar < 0 && F < 0) cst = 3;
          else cst = 2;
          s_cst = cst;
          s_ctau = cst == 1 ? glo + res.kstar : 0;
          s_need = cst == 1 ? K - (int)(res.above + top) : 0;
        }
        __syncthreads();
        if (tid < p.W) {
          uint32_t c = 0;
          const int u = s_ctau - s_base[tid];
          if (s_cst == 1 && u >= 0 && u < nw) {
            const uint32_t v = rx[tid * stride + 2 + u / 2];
            c = (u & 1) ? (v & 0xFFFu) : ((v >> 12) & 0xFFFu);
          }
          s_ties[tid] = (int)c;
        }
        __syncthreads();
      };
      resolve(wpad, nwk);
      if (s_cst == 2) {
        // tau outside the windows: its 256-key bin from every rank's bin counts, then
        // one refine round over that bin (suffix counts; one window for all ranks)
        if (leader) STAT(kStFallback);
        __syncthreads();
        poll_ranks<(kMaxW * 128 + kThreads * 4 - 1) / (kThreads * 4)>(p, rx, rbox + wpad, slot,
                                                                      128, 128, tag, tid, 6);
        __syncthreads();
        uint32_t g8 = 0;  // thread t: bin 255 - t summed over the ranks
        {
          const int b = 255 - tid;
          for (int r = 0; r < p.W; ++r) {
            const uint32_t v = rx[r * 128 + b / 2];
            g8 += (b & 1) ? (v & 0xFFFu) : ((v >> 12) & 0xFFFu);
          }
        }
        uint32_t tot8;
        const uint32_t ab8 = block_excl_scan(g8, warp_sums, &tot8);
        __shared__ int s_rb8;
        if (tid == 0) s_rb8 = -1;
        __syncthreads();
        if (ab8 < (uint32_t)K && ab8 + g8 >= (uint32_t)K) s_rb8 = 255 - tid;
        __syncthreads();
        const int b8 = s_rb8;
        if (b8 < 0) {
          if (tid == 0) { s_cst = 3; s_ctau = 0; s_need = 0; }
          if (tid < p.W) s_ties[tid] = 0;
          __syncthreads();
        } else {
          // every CTA counts its keys of the bin (fine[], the window counts are no
          // longer needed) and above it; the non-leaders store their counts in the
          // leader's hist[] (no longer read in this row)
          const int rlo = b8 * 256;
          for (int b = tid; b < 256; b += kThreads) fine[b] = 0;
          for (int b = tid; b < kG; b += kThreads) G[b] = 0;  // for the refine resolve
          if (tid == 0) s_myabove = 0;
          __syncthreads();
          uint32_t myab = 0;
          for_keys(keys, smem_keys, lseg, n, tid, [&](int, uint32_t k) {
            const int u = (int)k - rlo;
            if (u >= 0 && u < 256) atomicAdd(&fine[u], 1u);
            myab += u >= 256;
          });
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) myab += __shfl_xor_sync(0xffffffffu, myab, o);
          if ((tid & 31) == 0 && myab) atomicAdd(&s_myabove, myab);
          __syncthreads();
          if (!leader) {
            uint4* dst = reinterpret_cast<uint4*>(cl.map_shared_rank(hist, 0) + crank * 256);
            const uint4* src4 = reinterpret_cast<const uint4*>(fine);
            if (tid < 64) dst[tid] = src4[tid];
          }
          if (cs > 1) cl.sync();
          else __syncthreads();
          c1r_cnt = fine, c1r_clo = rlo, c1r_cn = 256;
          if (leader) {
            // the bin's counts summed over the CTAs + this rank's keys above the bin
            const uint32_t own_above = s_x8[b8];
            if (tid < 128) {
              uint32_t a = fine[2 * tid], b = fine[2 * tid + 1];
              for (int c = 1; c < cs; ++c) {
                a += hist[c * 256 + 2 * tid];
                b += hist[c * 256 + 2 * tid + 1];
              }
              msg[2 + tid] = pack2(tag, a, b);
            }
            if (tid == 0) {
              msg[0] = tagw(tag, (uint32_t)rlo);
              msg[1] = tagw(tag, 1u);  // F = the bin's bottom (tau is in the bin)
              msg[130] = tagw(tag, min(own_above, 0xFFFFFFu));
              msg[131] = tagw(tag, 0u);
            }
            __syncthreads();
            send(msg, 132, wpad + 128, false);
          }
          poll_ranks<(kMaxW * 132 + kThreads * 4 - 1) / (kThreads * 4)>(
              p, rx, rbox + wpad + 128, slot, 132, 132, tag, tid, 3);
          // Re-arm the refine messages once every CTA of the cluster has its copy.
          // Unlike the window and bin words, a rank writes them only on the rows that
          // take this round, so a word left from this use would carry the current
          // tag again 510 uses later on the same parity and a fast poll would accept
          // it before the peer's new message. Every CTA takes this branch (s_cst and
          // b8 are identical on every CTA), so the barrier is uniform; the next
          // write to this parity follows this rank's next exchange on the row.
          if (cs > 1) cl.sync();
          else __syncthreads();
          {
            const int nre = p.W * 132, per = (nre + cs - 1) / cs;
            const int e1 = min(nre, (crank + 1) * per);
            for (int j = crank * per + tid; j < e1; j += kThreads)
              st_u32(p.box + rowoff + (int64_t)(j / 132) * slot + wpad + 128 + j % 132, kSent);
          }
          resolve(132, 256);
        }
      }
      const int st = s_cst;
      const uint32_t tau = (uint32_t)s_ctau;
      DBG(21);
      if (leader && st == 1) STAT(kSt1R);
      // rank quota (as stage 30): -1 when global-id ties straddle ranks
      __shared__ int s_cq1;
      if (tid == 0) {
        const int need = s_need;
        int tot = 0, nr = 0;
        for (int s2 = 0; s2 < p.W; ++s2) { tot += s_ties[s2]; nr += s_ties[s2] > 0; }
        int q;
        if (tau == 0u || need <= 0) q = 0;
        else if (tot <= need) q = s_ties[p.rank];
        else if (nr == 1) q = s_ties[p.rank] > 0 ? need : 0;
        else if (p.tie_rank) {
          int before = 0;
          for (int s2 = 0; s2 < p.rank; ++s2) before += s_ties[s2];
          q = max(0, min(s_ties[p.rank], need - before));
        } else q = -1;
        s_cq1 = q;
        if (leader) {
          ctl.tau = tau;
          if (st == 3) STAT(kStSelectAll);
        }
      }
      __syncthreads();
      const int q = s_cq1;
      if (st == 2) {
        // tau outside a window: the octave rounds over every key (leader)
        c1r_stage = 20;
        if (leader && tid == 0) s_fkey_c1r = 65536;
      } else if (q < 0) {
        c1r_stage = 30;  // tie round (leader state machine)
      } else {
        // this CTA's tokens above tau and ties at tau from its window counts
        const int d0 = (int)tau - c1r_clo;
        uint32_t g = 0;
        for (int u = max(d0 + 1, 0) + tid; u < c1r_cn; u += kThreads) g += c1r_cnt[u];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) g += __shfl_xor_sync(0xffffffffu, g, o);
        __shared__ uint32_t s_wg[kWarps];
        if ((tid & 31) == 0) s_wg[tid >> 5] = g;
        __syncthreads();
        if (tid < cs) {
          uint32_t gt = s_myabove;
          for (int w = 0; w < kWarps; ++w) gt += s_wg[w];
          // tau at or above the window top: no keys above it here (validity of tau)
          if (d0 >= c1r_cn) gt = 0;
          const uint32_t eq = (tau != 0u && d0 >= 0 && d0 < c1r_cn) ? c1r_cnt[d0] : 0u;
          int* dst = tid == crank ? s_cnt2 : cl.map_shared_rank(s_cnt2, tid);
          dst[2 * crank] = (int)gt;
          dst[2 * crank + 1] = (int)eq;
        }
        if (cs > 1) cl.sync();  // every CTA's counts in every CTA
        else __syncthreads();
        DBG(24);
        int go = 0, eo = 0, gtt = 0;
        for (int c = 0; c < cs; ++c) {
          if (c < crank) { go += s_cnt2[2 * c]; eo += s_cnt2[2 * c + 1]; }
          gtt += s_cnt2[2 * c];
        }
        // WRITE: bitmaps of the tokens above tau and of the ties, one packed scan,
        // tokens at go + rank, ties of rank < q at gtt + rank
        const uint16_t* src = smem_keys ? keys : lseg;
        const bool vec = ((reinterpret_cast<uintptr_t>(src) & 15) == 0);
        const int nw = (n + 31) >> 5;
        if (nw > kCoarseBins && tid == 0) spin_fail(p.err, 17);
        uint32_t* bmg = hist;
        uint32_t* bme = rx;
        for (int i = tid; i < nw; i += kThreads) { bmg[i] = 0; bme[i] = 0; }
        __syncthreads();
        for (int b0 = tid * 8; b0 < n; b0 += kThreads * 8) {
          const uint4 r = load8_raw(src, b0, n, vec);
          uint32_t mg = 0, me = 0;
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const uint32_t k = raw_key(r, j);
            mg |= (uint32_t)(k > tau) << j;
            me |= (uint32_t)((k == tau) && (tau != 0u)) << j;
          }
          if (mg) atomicOr(&bmg[b0 >> 5], mg << (b0 & 31));
          if (me) atomicOr(&bme[b0 >> 5], me << (b0 & 31));
        }
        __syncthreads();
        DBG(12);
        const int wpt = (nw + kThreads - 1) / kThreads;
        const int w0 = min(nw, tid * wpt), w1 = min(nw, w0 + wpt);
        uint32_t cgt = 0, ce = 0;
        for (int w = w0; w < w1; ++w) { cgt += __popc(bmg[w]); ce += __popc(bme[w]); }
        uint32_t tot;
        const uint32_t pre = block_excl_scan(cgt | (min(ce, (uint32_t)kMaxK) << 12), warp_sums, &tot);
        if (tid == 0) {
          const int eq = s_cnt2[2 * crank + 1];
          if ((int)(tot & 0xFFFu) != s_cnt2[2 * crank] || (eq < kMaxK && (int)(tot >> 12) != eq))
            spin_fail(p.err, 20);
        }
        int gp = go + (int)(pre & 0xFFFu), ep = eo + (int)(pre >> 12);
        for (int w = w0; w < w1; ++w) {
          uint32_t m = bmg[w];
          while (m) {
            const int bit = __ffs(m) - 1;
            m &= m - 1;
            orow[gp++] = slot_of(s0 + w * 32 + bit);
          }
          m = bme[w];
          while (m && ep < q) {
            const int bit = __ffs(m) - 1;
            m &= m - 1;
            orow[gtt + ep] = slot_of(s0 + w * 32 + bit);
            ++ep;
          }
        }
        const int count = gtt + q, per = (K - count + cs - 1) / cs;
        for (int i = count + crank * per + tid; i < min(K, count + (crank + 1) * per); i += kThreads)
          orow[i] = -1;
        if (leader && tid == 0) p.out_count[row] = count;
        // (no re-arm of the window and bin words: every use of a row's slot rewrites
        // them with the use's tag, and the poll accepts only that tag; the refine
        // words, written only on refine rows, are re-armed after their poll)
        DBG(15);
      }
    }

    // Leader state machine. Every CTA runs the same sequence of cl.sync().
    // stage: 0 init, 1 after 1R fine, 2 after refine fine, 3 after count.
    int stage = c1r ? c1r_stage : 0;
    if (tid == 0) s_rearm_n = 0;
    // After WRITE the non-leader CTAs exit; the leader runs the rare tie paths
    // (TIEHIST / TIEWRITE) alone over the whole row from global memory.
    bool solo = false;
    uint32_t coarse_above = 0;
    int cb = -1;
    // Leader: send this rank's first min(ties, kTieList) tied global ids (G[]
    // holds the tied local positions in position order).
    // words of a tie message holding n ids
    auto tie_words = [](int n) -> int { return (n + 3) & ~3; };
    auto send_ties = [&]() {
      DBG(9);
      STAT(kStTieRound);
      const int Wn = p.W, IL = p.interleave;
      const int64_t rowoff = p.off_tl + ((int64_t)(par * p.cap + row) * Wn) * kTieList;
      const int nm = min(s_ties[p.rank], kTieList);
      const int nmp = tie_words(nm);
      for (int j = tid; j < nmp; j += kThreads) {
        uint32_t gid = 0;
        if (j < nm) {
          const int l = (int)G[j];
          gid = (uint32_t)(((l / IL) * Wn + p.rank) * IL + l % IL);
        }
        msg[j] = tagw(tag, gid);
      }
      __syncthreads();
      for (int j = tid * 4; j < nmp; j += kThreads * 4) {
        const uint4 v = make_uint4(msg[j], msg[j + 1], msg[j + 2], msg[j + 3]);
        const int64_t off = rowoff + (int64_t)p.rank * kTieList + j;
        if (p.mc) mc_st_v4(p.mc + off * 4, v);
        else
          for (int d = 0; d < Wn; ++d) st_v4(reinterpret_cast<uint32_t*>(p.peers[d]) + off, v);
      }
    };
    while (!(c1r && c1r_stage < 0)) {
      if (leader) {
        const int ncs = solo ? 1 : cs;
        // -------------------------------------------------- leader decides
        __shared__ int s_m;
        __shared__ int s_mode;  // 0 coarse rounds, 1 1R window
        __shared__ int s_fkey, s_b8, s_kt;
        if (stage == 20 && c1r && tid == 0 && s_fkey_c1r) { s_fkey = s_fkey_c1r; s_fkey_c1r = 0; }
        if (stage == 0) {
          // the local maximum's coarse bin (one shared atomic per warp)
          if (tid == 0) s_m = -1;
          __syncthreads();
          int m = -1;
          for (int b = tid; b < kCoarseBins; b += kThreads)
            if (hist[b]) m = b;
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) m = max(m, __shfl_xor_sync(0xffffffffu, m, o));
          if ((tid & 31) == 0) atomicMax(&s_m, m);
          __syncthreads();
          if (p.nw1 > 0 && row < p.cap1) {
            // 1-round mode: fine counts for an nw1-key window, either ending at the
            // local maximum's coarse bin or centred at the coarse bin of this rank's
            // local q-th key, q = ceil(K * L / Gl) (its share of the global top-K:
            // an estimate of tau; the exchange stays exact wherever tau falls).
            const int nb = p.nw1 / 16;
            int mb;
            if (p.w1c) {
              const uint32_t q = (uint32_t)max(1LL, ((long long)K * L + Gl - 1) / Gl);
              if (tid == 0) res.kstar = -1;
              __syncthreads();
              suffix_find(hist, kCoarseBins, q, &res, warp_sums, tid);
              const int cq = res.kstar >= 0 ? res.kstar : max(s_m, 0);
              mb = min(max(0, cq - nb / 2), kCoarseBins - nb);
            } else {
              mb = max(0, max(s_m, 0) - (nb - 1));
            }
            if (tid == 0) { s_mode = 1; }
            DBG(16);
            for (int c = tid; c < cs; c += kThreads) {
              Ctl* pc = cl.map_shared_rank(&ctl, c);
              pc->act = kActFine;
              pc->lo = mb * 16;
              pc->n = p.nw1;
            }
            stage = 1;
          } else if (Gl < p.small_ctx) {
            if (tid == 0) s_fkey = 65536;
            coarse_above = 0;
            stage = 20;
          } else {
            if (tid == 0) { s_mode = 0; }
            stage = 10;  // coarse round 1 (no fine request needed)
          }
        }
        __syncthreads();

        // ---------------------------------------- 1R: send window, resolve
        if (stage == 11) {
          DBG(4);
          // fine[] holds the cluster-summed nw1 window counts (summed below)
          const int nwk = p.nw1;
          const int words = 2 + nwk / 2;
          const int wpad = (words + 3) & ~3;
          int lo = ctl.lo;
          __shared__ uint32_t s_above_tot;
          if (!c1r) {
            // (centred mode: header and message built while the CTAs counted)
            // keys below the window and keys above it (the first pad word)
            uint32_t pb = 0, pa = 0;
            for (int b = tid; b < kCoarseBins; b += kThreads) {
              if (b < lo / 16) pb += hist[b];
              else if (b >= (lo + nwk) / 16) pa += hist[b];
            }
            uint32_t tot;
            block_excl_scan(pb, warp_sums, &tot);
            if (tid == 0) s_below_tot = tot;
            block_excl_scan(pa, warp_sums, &tot);
            if (tid == 0) s_above_tot = tot;
            __syncthreads();
          }
          DBG(19);
          if (!c1r) {
            if (tid == 0) {
              uint32_t below = s_below_tot;
              msg[0] = tagw(tag, (uint32_t)lo);
              msg[1] = tagw(tag, below > kCountMax ? kCountMax : below);
            }
            for (int j = tid; j < nwk / 2; j += kThreads)
              msg[2 + j] = pack2(tag, fine[2 * j], fine[2 * j + 1]);
            for (int j = words + tid; j < wpad; j += kThreads)
              msg[j] = tagw(tag, j == words ? min(s_above_tot, 0xFFFFFFu) : 0u);
            __syncthreads();
          }
          // rows are spaced as in the centred layout (W slots of wpad + 260 words): a
          // row's region never overlaps another row's (their epochs are independent)
          const int64_t rowoff = p.off_w1 + ((int64_t)(par * p.cap1 + row) * p.W) * (wpad + 260);
          // send
          for (int j = tid * 4; j < wpad; j += kThreads * 4) {
            uint4 v = make_uint4(msg[j], msg[j + 1], msg[j + 2], msg[j + 3]);
            int64_t off = rowoff + (int64_t)p.rank * wpad + j;
            if (p.mc) mc_st_v4(p.mc + off * 4, v);
            else
              for (int d = 0; d < p.W; ++d)
                st_v4(reinterpret_cast<uint32_t*>(p.peers[d]) + off, v);
          }
          DBG(20);
          poll_msg<(kMaxW * 260 + kThreads * 4 - 1) / (kThreads * 4)>(p, rx, rowoff, p.W * wpad,
                                                                      tag, tid, 1);
          __syncthreads();
          DBG(5);
          // resolve at key level
          if (tid < p.W) {
            uint32_t h0 = rx[tid * wpad], h1 = rx[tid * wpad + 1];
            s_base[tid] = (int)(h0 & 0xFFFFu);
            s_below[tid] = (int)(h1 & 0xFFFFFFu);
            s_adj[tid] = 0;
          }
          // s_top: keys above every G unit; s_ub: tau must be below every rank's
          // first key above its window that the rank has keys at or above
          __shared__ uint32_t s_top;
          __shared__ int s_ub;
          if (tid == 0) { res.gtop = -1; res.f = -1; res.kstar = -1; s_top = 0; s_ub = 1 << 30; }
          for (int b = tid; b < kG; b += kThreads) G[b] = 0;
          __syncthreads();
          DBG(25);
          if (tid < p.W) atomicMax(&res.gtop, s_base[tid] + nwk - 1);
          __syncthreads();
          DBG(26);
          const int glo = max(0, res.gtop - (kG - 1));
          if (tid < p.W) {
            const uint32_t a = rx[tid * wpad + words] & 0xFFFFFFu;
            if (a) {
              // a window entirely below glo puts s_ub <= glo: never resolved
              const int pos = max(0, s_base[tid] + nwk - glo);
              if (pos < kG) atomicAdd(&G[pos], a);
              else atomicAdd(&s_top, a);
              atomicMin(&s_ub, s_base[tid] + nwk);
            }
          }
          for (int j = tid; j < p.W * (nwk / 2); j += kThreads) {
            int s = j / (nwk / 2), w = j % (nwk / 2);
            uint32_t v = rx[s * wpad + 2 + w];
            uint32_t a = (v >> 12) & 0xFFFu, b = v & 0xFFFu;
            int u = s_base[s] + 2 * w;
            if (a) { if (u >= glo) atomicAdd(&G[u - glo], a); else atomicAdd(&s_adj[s], (int)a); }
            if (b) { if (u + 1 >= glo) atomicAdd(&G[u + 1 - glo], b); else atomicAdd(&s_adj[s], (int)b); }
          }
          __syncthreads();
          DBG(27);
          if (tid < p.W) {
            int bot = max(s_base[tid], glo);
            if (s_below[tid] + s_adj[tid] > 0) atomicMax(&res.f, bot);
          }
          __syncthreads();
          const uint32_t top = s_top;
          DBG(21);
          if (top < (uint32_t)K) suffix_find(G, kG, (uint32_t)K - top, &res, warp_sums, tid);
          DBG(22);
          if (res.f >= 0 && !c1r) {
            uint32_t part = 0;
            for (int u = res.f - glo + tid; u < kG; u += kThreads) part += G[u];
            uint32_t tot;
            block_excl_scan(part, warp_sums, &tot);
            if (tid == 0) res.carry = tot + top;
            __syncthreads();
          }
          if (tid == 0) {
            int F = res.f;
            if (top >= (uint32_t)K) res.status = 2;
            else if (res.kstar >= 0 && glo + res.kstar >= F && glo + res.kstar < s_ub) res.status = 1;
            else if (res.kstar < 0 && F < 0) res.status = 3;
            else res.status = 2;
            res.above += top;
          }
          __syncthreads();
          if (res.status == 1) {
            uint32_t tau = (uint32_t)(glo + res.kstar);
            if (tid < p.W) {
              int u = (int)tau - s_base[tid];
              uint32_t c = 0;
              if (u >= 0 && u < nwk) {
                uint32_t v = rx[tid * wpad + 2 + u / 2];
                c = (u & 1) ? (v & 0xFFFu) : ((v >> 12) & 0xFFFu);
              }
              s_ties[tid] = (int)c;
            }
            __syncthreads();
            if (tid == 0) {
              s_need = K - (int)res.above;
              ctl.tau = tau;
            }
            STAT(kSt1R);
            stage = 30;
          } else if (res.status == 3) {
            if (tid == 0) { ctl.tau = 0; s_need = 0; }
            stage = 30;
          } else if (s_ub < (1 << 30) || top > 0) {
            // tau may lie above a window: the octave rounds over every key
            if (tid == 0) s_fkey = 65536;
            __syncthreads();
            coarse_above = 0;
            stage = 20;
          } else {
            // bounded fallback over the keys below F (aligned to 16 keys)
            if (tid == 0) s_fkey = res.f;
            __syncthreads();
            coarse_above = res.carry;
            stage = 20;
          }
          __syncthreads();
          // re-arm the consumed 1R slots (centred mode: spread over the CTAs in WRITE,
          // off the critical path)
          if (c1r) {
            if (tid == 0) { s_rearm_off = rowoff; s_rearm_n = p.W * wpad; }
          } else {
            for (int j = tid; j < p.W * wpad; j += kThreads) st_u32(p.box + rowoff + j, kSent);
          }
          __syncthreads();
          DBG(23);
        }

        // --------------------------------------------- coarse round (2R mode)
        if (stage == 10) {
          int top = max(s_m, 0);
          const int base = max(0, top - (kCoarseWin - 1));
          top = base + kCoarseWin - 1;
          {
            uint32_t part = 0;
            for (int b = tid; b < base; b += kThreads) part += hist[b];
            uint32_t tot;
            block_excl_scan(part, warp_sums, &tot);
            if (tid == 0) s_below_tot = tot;
            __syncthreads();
          }
          if (tid == 0) {
            uint32_t below = s_below_tot;
            msg[0] = tagw(tag, (uint32_t)base);
            msg[1] = tagw(tag, below > kCountMax ? kCountMax : below);
            msg[18] = tagw(tag, 0);
            msg[19] = tagw(tag, 0);
          }
          for (int j = tid; j < 16; j += kThreads)
            msg[2 + j] = pack2(tag, hist[base + 2 * j], hist[base + 2 * j + 1]);
          __syncthreads();
          const int64_t rowoff = p.off_c1 + ((int64_t)(par * p.cap + row) * p.W) * kCoarseWords;
          send_msg(p, msg, kCoarseWords, rowoff, tid);
          poll_msg<1>(p, rx, rowoff, p.W * kCoarseWords, tag, tid, 2);
          __syncthreads();
          if (tid < p.W) {
            s_base[tid] = (int)(rx[tid * kCoarseWords] & 0xFFFFu);
            s_below[tid] = (int)(rx[tid * kCoarseWords + 1] & 0xFFFFFFu);
            s_adj[tid] = 0;
          }
          if (tid == 0) { res.gtop = -1; res.f = -1; res.kstar = -1; }
          for (int b = tid; b < kG; b += kThreads) G[b] = 0;
          __syncthreads();
          if (tid < p.W) atomicMax(&res.gtop, s_base[tid] + kCoarseWin - 1);
          __syncthreads();
          const int glo = max(0, res.gtop - (kG - 1));
          for (int j = tid; j < p.W * 16; j += kThreads) {
            int s = j / 16, w = j % 16;
            uint32_t v = rx[s * kCoarseWords + 2 + w];
            uint32_t a = (v >> 12) & 0xFFFu, b = v & 0xFFFu;
            int u = s_base[s] + 2 * w;
            if (a) { if (u >= glo) atomicAdd(&G[u - glo], a); else atomicAdd(&s_adj[s], (int)a); }
            if (b) { if (u + 1 >= glo) atomicAdd(&G[u + 1 - glo], b); else atomicAdd(&s_adj[s], (int)b); }
          }
          __syncthreads();
          if (tid < p.W) {
            int bot = max(s_base[tid], glo);
            if (s_below[tid] + s_adj[tid] > 0) atomicMax(&res.f, bot);
          }
          __syncthreads();
          suffix_find(G, kG, (uint32_t)K, &res, warp_sums, tid);
          if (tid == 0) {
            if (res.kstar >= 0 && glo + res.kstar >= res.f) res.status = 1;
            else if (res.f < 0) res.status = 3;
            else res.status = 2;
          }
          __syncthreads();
          if (res.status == 2) {
            uint32_t part = 0;
            for (int u = res.f - glo + tid; u < kG; u += kThreads) part += G[u];
            uint32_t tot;
            block_excl_scan(part, warp_sums, &tot);
            if (tid == 0) res.carry = tot;
            __syncthreads();
          }
          for (int j = tid; j < p.W * kCoarseWords; j += kThreads) st_u32(p.box + rowoff + j, kSent);
          if (res.status == 1) {
            cb = glo + res.kstar;
            coarse_above = res.above;
            for (int c = tid; c < cs; c += kThreads) {
              Ctl* pc = cl.map_shared_rank(&ctl, c);
              pc->act = kActFine;
              pc->lo = cb * 16;
              pc->n = 16;
            }
            stage = 2;
          } else if (res.status == 3) {
            if (tid == 0) { ctl.tau = 0; s_need = 0; }
            stage = 30;
          } else {
            if (tid == 0) s_fkey = res.f * 16;
            coarse_above = res.carry;
            stage = 20;
          }
          __syncthreads();
        }

        // ------------------- R8: 256 bins (key >> 8) of the keys below s_fkey
        // Bounded fallback (and the path of short contexts): covers every key, so
        // R8 -> R4 -> refine always resolves (3 rounds).
        if (stage == 20) {
          STAT(kStFallback);
          const int fk = s_fkey;
          if (tid < 256) {
            uint32_t a = 0;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
              const int c = 16 * tid + i;
              if (c * 16 < fk) a += hist[c];
            }
            G[tid] = a;  // local, reused below
          }
          __syncthreads();
          if (tid < 128) msg[tid] = pack2(tag, G[2 * tid], G[2 * tid + 1]);
          __syncthreads();
          const int64_t rowoff = p.off_r8 + ((int64_t)(par * p.cap + row) * p.W) * 128;
          send_msg(p, msg, 128, rowoff, tid);
          poll_msg<2>(p, rx, rowoff, p.W * 128, tag, tid, 6);
          __syncthreads();
          if (tid < 256) {
            uint32_t a = 0;
            for (int s2 = 0; s2 < p.W; ++s2) {
              uint32_t v = rx[s2 * 128 + tid / 2];
              a += (tid & 1) ? (v & 0xFFFu) : ((v >> 12) & 0xFFFu);
            }
            G[tid] = a;
          }
          if (tid == 0) { res.kstar = -1; }
          __syncthreads();
          suffix_find(G, 256, (uint32_t)K - coarse_above, &res, warp_sums, tid);
          for (int j = tid; j < p.W * 128; j += kThreads) st_u32(p.box + rowoff + j, kSent);
          if (res.kstar < 0) {
            if (tid == 0) { ctl.tau = 0; s_need = 0; }  // fewer than K valid keys
            stage = 30;
          } else {
            if (tid == 0) s_b8 = res.kstar;
            coarse_above += res.above;
            stage = 22;
          }
          __syncthreads();
        }

        // --------------------------------- R4: the 16 coarse bins inside b8
        if (stage == 22) {
          const int fk = s_fkey, b8 = s_b8;
          if (tid < 8) {
            uint32_t c0 = 16 * b8 + 2 * tid, c1 = c0 + 1;
            msg[tid] = pack2(tag, ((int)c0 * 16 < fk) ? hist[c0] : 0u,
                             ((int)c1 * 16 < fk) ? hist[c1] : 0u);
          }
          __syncthreads();
          const int64_t rowoff = p.off_r4 + ((int64_t)(par * p.cap + row) * p.W) * 8;
          send_msg(p, msg, 8, rowoff, tid);
          poll_msg<1>(p, rx, rowoff, p.W * 8, tag, tid, 7);
          __syncthreads();
          if (tid == 0) {
            uint32_t g[16];
            for (int k = 0; k < 16; ++k) g[k] = 0;
            for (int s2 = 0; s2 < p.W; ++s2)
              for (int w = 0; w < 8; ++w) {
                uint32_t v = rx[s2 * 8 + w];
                g[2 * w] += (v >> 12) & 0xFFFu;
                g[2 * w + 1] += v & 0xFFFu;
              }
            uint32_t cum = coarse_above;
            int kt = -1;
            for (int k = 15; k >= 0; --k) {
              if (cum + g[k] >= (uint32_t)K) { kt = k; break; }
              cum += g[k];
            }
            if (kt < 0) { kt = 0; spin_fail(p.err, 8); }
            s_kt = kt;
            res.above = cum;
          }
          __syncthreads();
          for (int j = tid; j < p.W * 8; j += kThreads) st_u32(p.box + rowoff + j, kSent);
          cb = 16 * b8 + s_kt;
          coarse_above = res.above;
          for (int c = tid; c < cs; c += kThreads) {
            Ctl* pc = cl.map_shared_rank(&ctl, c);
            pc->act = kActFine;
            pc->lo = cb * 16;
            pc->n = 16;
          }
          stage = 2;
          __syncthreads();
        }

        // ---------------------------------------------------- refine round
        if (stage == 21) {
          STAT(kSt2R);
          if (tid < 4) {
            // fine[0..15] summed; 8 pair words + pad none
            msg[2 * tid] = pack2(tag, fine[4 * tid], fine[4 * tid + 1]);
            msg[2 * tid + 1] = pack2(tag, fine[4 * tid + 2], fine[4 * tid + 3]);
          }
          __syncthreads();
          const int64_t rowoff = p.off_f1 + ((int64_t)(par * p.cap + row) * p.W) * kFineWords;
          if (tid < 2) {
            int j = tid * 4;
            uint4 v = make_uint4(msg[j], msg[j + 1], msg[j + 2], msg[j + 3]);
            int64_t off = rowoff + (int64_t)p.rank * kFineWords + j;
            if (p.mc) mc_st_v4(p.mc + off * 4, v);
            else
              for (int d = 0; d < p.W; ++d)
                st_v4(reinterpret_cast<uint32_t*>(p.peers[d]) + off, v);
          }
          poll_msg<1>(p, rx, rowoff, p.W * kFineWords, tag, tid, 3);
          __syncthreads();
          if (tid == 0) {
            uint32_t g[16];
            for (int k = 0; k < 16; ++k) g[k] = 0;
            for (int s = 0; s < p.W; ++s)
              for (int w = 0; w < 8; ++w) {
                uint32_t v = rx[s * kFineWords + w];
                g[2 * w] += (v >> 12) & 0xFFFu;
                g[2 * w + 1] += v & 0xFFFu;
              }
            uint32_t cum = coarse_above;
            int kt = -1;
            for (int k = 15; k >= 0; --k) {
              if (cum + g[k] >= (uint32_t)K) { kt = k; break; }
              cum += g[k];
            }
            if (kt < 0) { kt = 0; spin_fail(p.err, 4); }
            s_need = K - (int)cum;
            for (int s = 0; s < p.W; ++s) {
              uint32_t v = rx[s * kFineWords + kt / 2];
              s_ties[s] = (int)((kt & 1) ? (v & 0xFFFu) : ((v >> 12) & 0xFFFu));
            }
            ctl.tau = (uint32_t)(cb * 16 + kt);
          }
          __syncthreads();
          for (int j = tid; j < p.W * kFineWords; j += kThreads) st_u32(p.box + rowoff + j, kSent);
          stage = 30;
          __syncthreads();
        }

        if (stage == 30) {
          DBG(6);
          // tau, need and every rank's tie count at tau are identical on all
          // ranks. Ties are selected in global-id order; the rank quota is known
          // without another round unless ties straddle the cut on >= 2 ranks.
          if (tid == 0) {
            const uint32_t tau = ctl.tau;
            const int need = s_need;
            int tot = 0, nr = 0;
            for (int s = 0; s < p.W; ++s) { tot += s_ties[s]; nr += s_ties[s] > 0; }
            int mode = 0, q = 0;
            if (tau == 0u || need <= 0) q = 0;
            else if (tot <= need) q = s_ties[p.rank];
            else if (nr == 1) q = s_ties[p.rank] > 0 ? need : 0;
            else if (p.tie_rank) {
              // ties by (rank, position): the quota is known without a round
              // (tie_order="rank")
              int before = 0;
              for (int s = 0; s < p.rank; ++s) before += s_ties[s];
              q = max(0, min(s_ties[p.rank], need - before));
            } else { mode = 1; q = -1; }
            s_tmode = mode;
            s_q = q;
            s_gtt = 0;
            s_eqt = 0;
            if (tau == 0u) STAT(kStSelectAll);
          }
          __syncthreads();
          const uint32_t tau = ctl.tau;
          // fast path: two-pass write, ties collected unordered (<= kTieList,
          // sorted by the leader) and the tie message sent before pass 2
          const bool fast = tau == 0u || s_ties[p.rank] <= kTieList;
          if (tid == 0) s_tcnt = 0;
          for (int c = tid; c < cs; c += kThreads) {
            Ctl* pc = cl.map_shared_rank(&ctl, c);
            pc->tau = tau;
            pc->gt_off = 0;
            pc->eq_off = 0;
            pc->q = s_tmode == 0 ? s_q : -1;  // known quota: every CTA pads its share
            pc->rearm_off = s_rearm_off;
            pc->rearm_n = s_rearm_n;
            // known quota: every CTA writes its own ties (tie bitmaps + one scan)
            pc->act = s_tmode == 0 ? kActWriteDirect : fast ? kActWriteFast : kActWrite;
          }
          stage = fast ? 34 : 33;
        } else if (stage == 34) {
          stage = s_tmode == 1 ? 43 : 50;
        } else if (stage == 33) {
          // WRITE done: tokens above tau are at [0, s_gtt) in position order and
          // G[] holds the first kTieStore tied local positions in position order
          // s_gtt / s_eqt were set by the leader in WRITE
          stage = s_tmode == 1 ? 40 : 50;
        }
        __syncthreads();

        // ---------------- tie round: every rank's first tied global ids
        if (stage == 40) {
          send_ties();
          stage = 43;
        }
        if (stage == 43) {
          const int Wn = p.W;
          const int64_t rowoff = p.off_tl + ((int64_t)(par * p.cap + row) * Wn) * kTieList;
          const int nm = min(s_ties[p.rank], kTieList);
          {
            const long long t0 = now_ns();
            for (int j = tid * 4; j < Wn * kTieList; j += kThreads * 4) {
              const int s = j / kTieList, jj = j % kTieList;
              if (jj >= tie_words(min(s_ties[s], kTieList))) continue;
              while (true) {
                const uint4 v = ld_sys_v4(p.box + rowoff + j);
                if ((v.x >> 24) == tag && (v.y >> 24) == tag && (v.z >> 24) == tag &&
                    (v.w >> 24) == tag) {
                  *reinterpret_cast<uint4*>(rx + j) = v;
                  break;
                }
                if (spin_late(p.spin_ns, t0)) spin_fail(p.err, 14);
              }
              if (p.dbg && row == 0 && crank == 0)  // last poll completion (profiling)
                atomicMax(reinterpret_cast<unsigned long long*>(p.dbg + 13),
                          (unsigned long long)now_ns());
            }
          }
          if (tid == 0) res.kstar = -1;  // the need-th smallest tied global id
          __syncthreads();
          DBG(10);
          // compact every rank's ids into hist[] (free after WRITE; rank order), then the
          // need-th smallest id g* by a linear rank count (ids are distinct)
          const int need = s_need;
          if (tid < 32) {
            int n_ = tid < Wn ? min(s_ties[tid], kTieList) : 0, x_ = n_;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
              const int y = __shfl_up_sync(0xffffffffu, x_, o);
              if (tid >= o) x_ += y;
            }
            if (tid < Wn) s_base[tid] = x_ - n_;  // offset of rank tid's ids
            if (tid == 31) s_adj[0] = x_;         // total
          }
          __syncthreads();
          const int ncand = s_adj[0];
          for (int c = tid; c < Wn * kTieList; c += kThreads) {
            const int s = c / kTieList, j = c % kTieList;
            if (j < min(s_ties[s], kTieList)) hist[s_base[s] + j] = rx[c] & 0xFFFFFFu;
          }
          __syncthreads();
          for (int i = tid; i < ncand; i += kThreads) {
            const uint32_t x = hist[i];
            int less = 0;
            for (int k = 0; k < ncand; ++k) less += hist[k] < x;
            if (less == need - 1) res.kstar = (int)x;
          }
          __syncthreads();
          DBG(12);
          if (tid == 0) {
            const int gs = res.kstar;
            bool ok = gs >= 0;
            // a rank that sent only its first kTieList ties must not hide one <= g*
            for (int s = 0; s < Wn && ok; ++s)
              if (s_ties[s] > kTieList && (int)(rx[s * kTieList + kTieList - 1] & 0xFFFFFFu) < gs)
                ok = false;
            int q = 0;
            if (ok)
              for (int j = 0; j < nm; ++j) q += (int)(rx[p.rank * kTieList + j] & 0xFFFFFFu) <= gs;
            s_q = ok ? q : -1;
          }
          for (int j = tid; j < Wn * kTieList; j += kThreads) {
            const int s = j / kTieList, jj = j % kTieList;
            if (jj < tie_words(min(s_ties[s], kTieList))) st_u32(p.box + rowoff + j, kSent);
          }
          DBG(14);
          __syncthreads();
          if (s_q >= 0) {
            stage = 50;
          } else {
            // bounded radix rounds over global ids (kTieBuckets per round); the
            // tie-list region is reused by odd rounds: our re-arm must land first
            __threadfence_system();
            STAT(kStTieRadix);
            if (tid == 0) {
              s_tlo = 0;
              s_tw = (Gl + kTieBuckets - 1) / kTieBuckets;
              s_tgb = 0;
              s_trr = 0;
            }
            if (tid < Wn) s_tbl[tid] = 0;
            __syncthreads();
            for (int c = tid; c < ncs; c += kThreads) {
              Ctl* pc = cl.map_shared_rank(&ctl, c);
              pc->tie_lo = s_tlo;
              pc->tie_w = s_tw;
              pc->act = kActTieHist;
            }
            stage = 41;  // wait for TIEHIST
          }
          __syncthreads();
        }

        // ---------------- tie radix round (fine[] = this rank's bucket counts)
        if (stage == 42) {
          const int Wn = p.W, rr = s_trr;
          const int stride = (rr & 1) ? kTieList : kTieWords;  // alternate regions
          const int64_t rowoff =
              ((rr & 1) ? p.off_tl : p.off_tr) + ((int64_t)(par * p.cap + row) * Wn) * stride;
          if (tid < kTieWords) msg[tid] = pack2(tag, fine[2 * tid], fine[2 * tid + 1]);
          __syncthreads();
          if (tid < kTieWords / 4) {
            const int j = tid * 4;
            const uint4 v = make_uint4(msg[j], msg[j + 1], msg[j + 2], msg[j + 3]);
            const int64_t off = rowoff + (int64_t)p.rank * stride + j;
            if (p.mc) mc_st_v4(p.mc + off * 4, v);
            else
              for (int d = 0; d < Wn; ++d) st_v4(reinterpret_cast<uint32_t*>(p.peers[d]) + off, v);
          }
          {
            const long long t0 = now_ns();
            for (int j = tid * 4; j < Wn * kTieWords; j += kThreads * 4) {
              const int s = j / kTieWords, jj = j % kTieWords;
              const uint32_t* q = p.box + rowoff + (int64_t)s * stride + jj;
              while (true) {
                const uint4 v = ld_sys_v4(q);
                if ((v.x >> 24) == tag && (v.y >> 24) == tag && (v.z >> 24) == tag &&
                    (v.w >> 24) == tag) {
                  *reinterpret_cast<uint4*>(rx + j) = v;
                  break;
                }
                if (spin_late(p.spin_ns, t0)) spin_fail(p.err, 15);
              }
            }
          }
          __syncthreads();
          auto cnt_at = [&](int s, int b) -> int {
            const uint32_t v = rx[s * kTieWords + b / 2];
            return (int)((b & 1) ? (v & 0xFFFu) : ((v >> 12) & 0xFFFu));
          };
          if (tid == 0) {
            const int need = s_need;
            int cum = s_tgb, bstar = -1;
            for (int b = 0; b < kTieBuckets; ++b) {
              int cb_ = 0;
              for (int s = 0; s < Wn; ++s) cb_ += cnt_at(s, b);
              if (cum + cb_ >= need) { bstar = b; break; }
              cum += cb_;
            }
            if (bstar < 0) spin_fail(p.err, 16);
            // buckets below bstar sum to < need <= K < kCountMax: exact counts
            for (int s = 0; s < Wn; ++s)
              for (int b = 0; b < bstar; ++b) s_tbl[s] += cnt_at(s, b);
            s_tgb = cum;
            s_tlo += bstar * s_tw;
            if (s_tw == 1) {
              s_q = s_tbl[p.rank] + cnt_at(p.rank, bstar);
            } else {
              s_tw = (s_tw + kTieBuckets - 1) / kTieBuckets;
              s_trr = rr + 1;
            }
          }
          for (int j = tid; j < Wn * kTieWords; j += kThreads) {
            const int s = j / kTieWords, jj = j % kTieWords;
            st_u32(p.box + rowoff + (int64_t)s * stride + jj, kSent);
          }
          __threadfence_system();
          __syncthreads();
          if (s_q >= 0) {
            stage = 50;
          } else {
            for (int c = tid; c < ncs; c += kThreads) {
              Ctl* pc = cl.map_shared_rank(&ctl, c);
              pc->tie_lo = s_tlo;
              pc->tie_w = s_tw;
              pc->act = kActTieHist;
            }
            stage = 41;
          }
          __syncthreads();
        }

        // ---------------- emit this rank's first s_q ties after its s_gtt tokens
        if (stage == 50) {
          DBG(11);
          const int q = s_q, gtt = s_gtt;
          if (q <= min(s_eqt, kTieStore)) {
            for (int e = tid; e < q; e += kThreads) orow[gtt + e] = slot_of((int)G[e]);
            stage = 51;
          } else {
            STAT(kStTieWritePass);
            for (int c = tid; c < ncs; c += kThreads) {
              Ctl* pc = cl.map_shared_rank(&ctl, c);
              pc->q = q;
              pc->gtt = gtt;
              pc->act = kActTieWrite;
            }
            stage = 52;  // wait for TIEWRITE
          }
        }
        if (stage == 51) {
          const int count = s_gtt + s_q;
          if (tid == 0) p.out_count[row] = count;
          if (!s_padded)
            for (int i = count + tid; i < K; i += kThreads) orow[i] = -1;
          for (int c = tid; c < ncs; c += kThreads) cl.map_shared_rank(&ctl, c)->act = kActDone;
        }
        __syncthreads();
      }  // leader decides

      // A: the leader's ctl writes reach every CTA. Every act ends at a cluster
      // barrier, so a release-store of the broadcast count per CTA (and a spin on
      // it) replaces the cluster barrier here.
      if (solo || cs == 1) {
        __syncthreads();
      } else if (leader) {
        __syncthreads();
        ++bseq;
        if (tid > 0 && tid < cs) {
          asm volatile("fence.acq_rel.cluster;" ::: "memory");
          uint32_t* f = cl.map_shared_rank(&s_seq, tid);
          asm volatile("st.relaxed.cluster.u32 [%0], %1;" ::"l"(f), "r"(bseq) : "memory");
        }
      } else {
        ++bseq;
        if (tid == 0) {
          const long long t0 = now_ns();
          uint32_t v;
          while (true) {
            asm volatile("ld.acquire.cluster.u32 %0, [%1];" : "=r"(v) : "l"(&s_seq) : "memory");
            if (v == bseq) break;
            if (spin_late(p.spin_ns, t0)) { spin_fail(p.err, 19); break; }
          }
        }
        __syncthreads();
      }
      const int act = ctl.act;
      if (act == kActDone) break;
      if (act == kActFine) {
        // every CTA adds its keys in [lo, lo + nn) into the leader's fine2[] (zero
        // between uses; DSMEM reductions), which the leader moves to fine[]
        const int lo = ctl.lo, nn = ctl.n;
        uint32_t* cnt = leader ? fine2 : fine;  // non-leaders count locally, then push
        if (!leader) {
          for (int b = tid; b < nn; b += kThreads) fine[b] = 0;
          __syncthreads();
        }
        for_keys(keys, smem_keys, lseg, n, tid, [&](int, uint32_t k) {
          int u = (int)k - lo;
          if (u >= 0 && u < nn) atomicAdd(&cnt[u], 1u);
        });
        if (!leader) {
          __syncthreads();
          uint32_t* lf = cl.map_shared_rank(fine2, 0);
          for (int b = tid; b < nn; b += kThreads)
            if (fine[b]) atomicAdd(&lf[b], fine[b]);
        }
        cl.sync();  // B: sums complete
        if (leader) {
          DBG(17);
          for (int b = tid; b < nn; b += kThreads) {
            fine[b] = fine2[b];
            fine2[b] = 0;
          }
          if (tid == 0) ctl.act = 0;
          stage = (stage == 1) ? 11 : 21;
          __syncthreads();
        }
        continue;
      }

      // ---- ordered passes over this CTA's keys (16 consecutive keys per
      // thread per step; block scans keep position order, no atomics)
      if (act == kActWrite || act == kActWriteFast || act == kActWriteDirect) {
        // WRITE: one coalesced pass over this CTA's keys sets SMEM bitmaps of the
        // tokens above tau (and of the ties, except on the fast path, which
        // collects its <= kTieList ties unordered on the leader). Popcounts of
        // contiguous word ranges + one block scan + the earlier CTAs' totals give
        // every thread its output offset; enumerating the bitmap words writes the
        // tokens in position order (deterministic, no second read of the logits).
        // The fast path sends the tie message before the enumeration, so the
        // NVLink hop overlaps it.
        DBG(7);
        const bool fastw = act == kActWriteFast;
        // direct: the rank quota q is known; every CTA writes its tokens above tau and
        // its ties of rank < q (tie ranks from the tie bitmaps and the CTA offsets),
        // the leader the count: no tie list and no leader tail
        const bool direct = act == kActWriteDirect;
        const uint32_t tau = ctl.tau;
        const uint16_t* src = smem_keys ? keys : lseg;
        const bool vec = ((reinterpret_cast<uintptr_t>(src) & 15) == 0);
        const int nw = (n + 31) >> 5;
        if (nw > kCoarseBins && tid == 0) spin_fail(p.err, 17);  // host keeps n <= 131072
        uint32_t* bmg = hist;  // tokens above tau
        uint32_t* bme = rx;    // ties (ordered path)
        for (int i = tid; i < nw; i += kThreads) {
          bmg[i] = 0;
          if (!fastw) bme[i] = 0;
        }
        __syncthreads();
        int* tc = leader ? &s_tcnt : cl.map_shared_rank(&s_tcnt, 0);
        uint32_t* tl = leader ? G : cl.map_shared_rank(G, 0);
        uint32_t neq_fast = 0;
        for (int b0 = tid * 8; b0 < n; b0 += kThreads * 8) {
          const uint4 r = load8_raw(src, b0, n, vec);
          uint32_t mg = 0, me = 0;
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const uint32_t k = raw_key(r, j);
            mg |= (uint32_t)(k > tau) << j;
            me |= (uint32_t)((k == tau) && (tau != 0u)) << j;
          }
          if (mg) atomicOr(&bmg[b0 >> 5], mg << (b0 & 31));
          if (me) {
            if (fastw) {
              neq_fast += __popc(me);
              for (int j = 0; j < 8; ++j)
                if ((me >> j) & 1u) {
                  const int idx = atomicAdd(tc, 1);
                  if (idx < kTieList) tl[idx] = (uint32_t)(s0 + b0 + j);
                }
            } else {
              atomicOr(&bme[b0 >> 5], me << (b0 & 31));
            }
          }
        }
        __syncthreads();
        DBG(12);
        const int wpt = (nw + kThreads - 1) / kThreads;
        const int w0 = min(nw, tid * wpt), w1 = min(nw, w0 + wpt);
        uint32_t cg = 0, ce = fastw ? neq_fast : 0u;
        for (int w = w0; w < w1; ++w) {
          cg += __popc(bmg[w]);
          if (!fastw) ce += __popc(bme[w]);
        }
        // one packed scan: gt in the low 12 bits (a CTA holds < K <= 2048 tokens
        // above tau), ties in the high 20 bits, clamped per thread at kMaxK (tie
        // ranks >= K are never written, so every rank that is used stays exact)
        uint32_t tot;
        const uint32_t pre =
            block_excl_scan(cg | (min(ce, (uint32_t)kMaxK) << 12), warp_sums, &tot);
        if (tid == 0) { ctl.gt = (int)(tot & 0xFFFu); ctl.eq = (int)(tot >> 12); }
        __shared__ int s_go, s_eo;
        if (cs > 1) cl.sync();  // every CTA's totals (and the leader's tie list) visible
        else __syncthreads();
        DBG(24);
        if (tid < 32) {
          int go = 0, eo = 0, ag = 0, ae = 0;
          if (tid < cs) {
            const Ctl* pc = leader && tid == 0 ? &ctl : cl.map_shared_rank(&ctl, tid);
            ag = pc->gt;
            ae = pc->eq;
            if (tid < crank) { go = ag; eo = ae; }
          }
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) {
            go += __shfl_xor_sync(0xffffffffu, go, o);
            eo += __shfl_xor_sync(0xffffffffu, eo, o);
            ag += __shfl_xor_sync(0xffffffffu, ag, o);
            ae += __shfl_xor_sync(0xffffffffu, ae, o);
          }
          if (tid == 0) {
            s_go = go;
            s_eo = eo;
            s_ag = ag;
            ctl.gt_off = go;
            ctl.eq_off = eo;
            if (leader) {
              s_gtt = ag;
              // fast path: G[] holds min(ties, kTieList) ties, sorted below
              s_eqt = fastw ? min(s_tcnt, kTieList) : ae;
            }
          }
        }
        if (fastw && leader) {
          const int nt = min(s_tcnt, kTieList);
          uint32_t mine = 0;
          int rk = 0;
          if (tid < nt) {
            mine = G[tid];
            for (int j = 0; j < nt; ++j) rk += G[j] < mine;
          }
          __syncthreads();
          if (tid < nt) fine2[rk] = mine;
          __syncthreads();
          if (tid < nt) {
            G[tid] = fine2[tid];
            fine2[tid] = 0;
          }
          __syncthreads();
          if (s_tmode == 1) send_ties();
        }
        __syncthreads();
        {
          int gp = s_go + (int)(pre & 0xFFFu), ep = s_eo + (int)(pre >> 12);
          for (int w = w0; w < w1; ++w) {
            uint32_t m = bmg[w];
            while (m) {
              const int bit = __ffs(m) - 1;
              m &= m - 1;
              orow[gp++] = slot_of(s0 + w * 32 + bit);
            }
            if (direct) {
              m = bme[w];
              const int q = ctl.q, gtt = s_ag;
              while (m && ep < q) {
                const int bit = __ffs(m) - 1;
                m &= m - 1;
                orow[gtt + ep] = slot_of(s0 + w * 32 + bit);
                ++ep;
              }
            } else if (!fastw) {
              m = bme[w];
              while (m) {
                const int bit = __ffs(m) - 1;
                m &= m - 1;
                if (ep < kTieStore) tl[ep] = (uint32_t)(s0 + w * 32 + bit);
                ++ep;
              }
            }
          }
        }
        // Known quota (fast path): every CTA pads its share of [count, K) with -1.
        const bool pad = (fastw || direct) && ctl.q >= 0;
        if (pad) {
          const int count = s_ag + ctl.q, per = (K - count + cs - 1) / cs;
          const int e1 = min(K, count + (crank + 1) * per);
          for (int i = count + crank * per + tid; i < e1; i += kThreads) orow[i] = -1;
          if (direct && leader && tid == 0) p.out_count[row] = count;
        }
        if (leader && tid == 0) s_padded = pad;
        if (ctl.rearm_n > 0) {
          const int per = (ctl.rearm_n + cs - 1) / cs;
          const int e1 = min(ctl.rearm_n, (crank + 1) * per);
          for (int j = crank * per + tid; j < e1; j += kThreads)
            st_u32(p.box + ctl.rearm_off + j, kSent);
        }
        DBG(15);
        if ((fastw || direct) && cs > 1) {
          // The tie list was complete at the offsets barrier and no peer SMEM is read
          // after the offsets: arrive now, wait before this CTA's SMEM is reused or
          // released (next row / exit).
          asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
          arrived = true;
        } else if (cs > 1) {
          cl.sync();  // tokens and tie list complete: no peer SMEM access after this
        } else {
          __syncthreads();
        }
        if (!leader || direct) break;
        solo = true;
        stage = fastw ? 34 : 33;
        continue;
      }
      {
        // TIEHIST / TIEWRITE (leader alone, whole row from global memory):
        // ordered passes (per-step block scans), rare paths
        const uint32_t tau = ctl.tau;
        const uint16_t* src = p.logits + (int64_t)row * p.lstride;
        const bool vec = ((reinterpret_cast<uintptr_t>(src) & 15) == 0);
        const int n = L, s0 = 0;
        const int gto = 0, eqo = 0;
        const int tq = ctl.q, tgtt = ctl.gtt, tlo = ctl.tie_lo, tw = ctl.tie_w;
        const int Wn = p.W, IL = p.interleave;
        if (act == kActTieHist) {
          for (int b = tid; b < kTieBuckets; b += kThreads) fine[b] = 0;
          __syncthreads();
        }
        int run_gt = 0, run_eq = 0;
        for (int b = 0; b < n; b += kThreads * 16) {
          const int i0 = b + tid * 16;
          const uint4 r0 = load8_raw(src, i0, n, vec);
          const uint4 r1 = load8_raw(src, i0 + 8, n, vec);
          int ngt = 0, neq = 0;
#pragma unroll
          for (int j = 0; j < 16; ++j) {
            const uint32_t k = raw_key(j < 8 ? r0 : r1, j & 7);
            ngt += k > tau;
            neq += (k == tau) && (tau != 0u);
          }
          if (act == kActTieHist) {
            if (neq) {
              for (int j = 0; j < 16; ++j) {
                const uint32_t k = raw_key(j < 8 ? r0 : r1, j & 7);
                if (k != tau) continue;
                const int l = s0 + i0 + j;
                const long long u = (long long)(((l / IL) * Wn + p.rank) * IL + l % IL) - tlo;
                if (u >= 0 && u < (long long)kTieBuckets * tw) atomicAdd(&fine[u / tw], 1u);
              }
            }
            continue;
          }
          if (!__syncthreads_or(ngt | neq)) continue;
          uint32_t tot;
          const uint32_t pre =
              block_excl_scan((uint32_t)ngt | ((uint32_t)neq << 16), warp_sums, &tot);
          int gp = gto + run_gt + (int)(pre & 0xFFFFu);
          int ep = eqo + run_eq + (int)(pre >> 16);
          if (ngt | neq) {
            for (int j = 0; j < 16; ++j) {
              const uint32_t k = raw_key(j < 8 ? r0 : r1, j & 7);
              const int l = s0 + i0 + j;
              if (k > tau) {
                ++gp;
              } else if (k == tau && tau != 0u) {
                if (ep < tq) orow[tgtt + ep] = slot_of(l);
                ++ep;
              }
            }
          }
          run_gt += (int)(tot & 0xFFFFu);
          run_eq += (int)(tot >> 16);
        }
        __syncthreads();  // fine[] (TIEHIST) complete
        stage = act == kActTieHist ? 42 : 51;
        continue;
      }
    }  // while
  }

  // the leader's last mailbox use of this row is behind it
  if (leader && tid == 0) p.epoch[row] = s_enext;
next_row:
  DBG(8);
  if (arrived) {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    arrived = false;
  }
  if (row + (int)gridDim.y < p.T) {
    // the cluster's SMEM is reused by its next row: histograms start at zero again
    // (after every thread of this CTA is done with this row's bitmaps in hist[])
    __syncthreads();
    for (int b = tid; b < kCoarseBins; b += kThreads) hist[b] = 0;
    cl.sync();
  }
  }  // rows

  if (p.pdl) {
    // Every row of this CTA is done: make its outputs visible, then let the
    // dependents (the sparse core) launch while the grid retires. The fence is
    // required: trtllm-gen's sparse MLA kernels read their KV lengths (our
    // counts) BEFORE griddepcontrol.wait (SASS: LDG ahead of ACQBULK in two warp
    // roles of the persistent 2-CTA variant), so the counts must be final and
    // visible when the dependent launches. A release at kernel entry let those
    // roles read stale counts of the previous layer and disagree on the tile
    // count: the B8 hang of the b1-pdl-fusion track.
    // This is NOT a programming-model guarantee: a load issued before
    // griddepcontrol.wait has no happens-before edge to our stores. It holds in
    // practice because the dependent grid launches only after every CTA of this
    // grid triggered (i.e. after the fence below); a consumer cubin or launch
    // behaviour change can break it, so the decode default keeps PDL at <= 16
    // rows (K31_DCPX_PDL). The robust fix is a consumer that reads its lengths
    // after its wait (or lengths produced by an earlier kernel).
    // A CTA counts as triggered as soon as one of its threads issues the release,
    // so every thread's outputs (the leader's count is written by one thread just
    // before this point on the centred 1R path) must be fenced before any thread
    // releases: barrier, fence, barrier.
    __syncthreads();
    __threadfence();
    __syncthreads();
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  }
}

// ------------------------------------------------------------------ O exchange
struct OxParams {
  const __nv_bfloat16* y;  // [T, H, D] strided, D contiguous
  int64_t ys_row, ys_head;
  const float* lse;  // [T, H] strided
  int64_t ls_row, ls_head;
  int lse_base_e;
  __nv_bfloat16* out;  // [T, HPR, D]
  int64_t os_row, os_head;
  float* out_lse;  // [T, HPR] base-e, or null
  uint32_t* box;
  const uint64_t* peers;
  int W, rank, T, HPR, D, cap;
  int64_t off_data, off_lse;  // u32 words
  int* epoch;
  int* ticket;
  int* err;
  long long spin_ns;  // cross-rank spin deadline (<= 0: none)
};

__device__ __forceinline__ uint32_t canon(uint32_t w) { return w == kSent ? 0x7FFF7FFFu : w; }

template <int D>
__global__ void __launch_bounds__(256) o_exchange_kernel(OxParams p) {
  const int lane = threadIdx.x & 31;
  const int gw = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
  const int nw = gridDim.x * (blockDim.x / 32);
  __shared__ int s_par;
  if (threadIdx.x == 0) s_par = (*((volatile int*)p.epoch)) & 1;
  __syncthreads();
  const int par = s_par;
  const int HPR = p.HPR, W = p.W;
  constexpr int CPH = D / 8;  // uint4 chunks per head
  const int item_words = HPR * D / 2;
  const float kLog2e = 1.4426950408889634f;

  // ---- phase 1: push this rank's partial rows to the head owners
  for (int it = gw; it < p.T * W; it += nw) {
    const int row = it / W, dst = it % W;
    uint32_t* peer = reinterpret_cast<uint32_t*>(p.peers[dst]);
    uint32_t* dd = peer + p.off_data + ((int64_t)(par * p.cap + row) * W + p.rank) * item_words;
    for (int c = lane; c < HPR * CPH; c += 32) {
      const int h = c / CPH, part = c % CPH;
      const __nv_bfloat16* src = p.y + (int64_t)row * p.ys_row + (int64_t)(dst * HPR + h) * p.ys_head + part * 8;
      uint4 v = *reinterpret_cast<const uint4*>(src);
      v.x = canon(v.x); v.y = canon(v.y); v.z = canon(v.z); v.w = canon(v.w);
      st_v4(dd + h * (D / 2) + part * 4, v);
    }
    // HPR can exceed a warp (DCP2: 48 heads per rank)
    for (int h = lane; h < HPR; h += 32) {
      float l = p.lse[(int64_t)row * p.ls_row + (int64_t)(dst * HPR + h) * p.ls_head];
      if (p.lse_base_e) l *= kLog2e;
      uint32_t b = __float_as_uint(l);
      if (b == kSent) b = 0x7FC00000u;
      uint32_t* dl = peer + p.off_lse + ((int64_t)(par * p.cap + row) * W + p.rank) * HPR;
      st_u32(dl + h, b);
    }
  }

  // ---- phase 2: merge the W partials of this rank's heads
  for (int it = gw; it < p.T * HPR; it += nw) {
    const int row = it / HPR, h = it % HPR;
    uint32_t* lb = p.box + p.off_lse + ((int64_t)(par * p.cap + row) * W) * HPR;
    float l = -INFINITY;
    long long t0 = now_ns();
    if (lane < W) {
      uint32_t b;
      while ((b = ld_sys(lb + lane * HPR + h)) == kSent) {
        if (spin_late(p.spin_ns, t0)) { spin_fail(p.err, 10); break; }
      }
      l = __uint_as_float(b);
    }
    float m = l;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float w = (lane < W && m != -INFINITY && l != -INFINITY) ? exp2f(l - m) : 0.f;
    float sum = w;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    const float inv = sum > 0.f ? 1.f / sum : 0.f;
    const float wn = w * inv;

    const uint32_t* db = p.box + p.off_data + ((int64_t)(par * p.cap + row) * W) * item_words + h * (D / 2);
    constexpr int LPC = CPH >= 32 ? 1 : 32 / CPH;  // lanes per chunk
    constexpr int CPL = CPH >= 32 ? CPH / 32 : 1;  // chunks per lane
    constexpr int NS = (kMaxW + LPC - 1) / LPC;    // sources per lane
    const int sub = lane / (CPH >= 32 ? 32 : CPH);
    float ws[NS];
#pragma unroll
    for (int i = 0; i < NS; ++i) ws[i] = __shfl_sync(0xffffffffu, wn, (sub + i * LPC) & 31);
    float acc[CPL][8];
#pragma unroll
    for (int a = 0; a < CPL; ++a)
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[a][j] = 0.f;
#pragma unroll
    for (int a = 0; a < CPL; ++a) {
      const int c = (CPH >= 32) ? (lane + 32 * a) : (lane % CPH);
      uint4 v[NS];
      // issue every source's load first, then re-poll only the ones still armed
#pragma unroll
      for (int i = 0; i < NS; ++i) {
        const int s = sub + i * LPC;
        if (s < W) v[i] = ld_sys_v4(db + (int64_t)s * item_words + c * 4);
      }
#pragma unroll
      for (int i = 0; i < NS; ++i) {
        const int s = sub + i * LPC;
        if (s >= W) continue;
        const uint32_t* q = db + (int64_t)s * item_words + c * 4;
        while (v[i].x == kSent || v[i].y == kSent || v[i].z == kSent || v[i].w == kSent) {
          if (spin_late(p.spin_ns, t0)) { spin_fail(p.err, 11); break; }
          v[i] = ld_sys_v4(q);
        }
        st_v4(const_cast<uint32_t*>(q), make_uint4(kSent, kSent, kSent, kSent));
        const float w_ = ws[i];
        if (w_ == 0.f) continue;  // empty partial (LSE -inf): its rows may hold NaN
        uint32_t ww[4] = {v[i].x, v[i].y, v[i].z, v[i].w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          acc[a][2 * j] += w_ * __uint_as_float(ww[j] << 16);
          acc[a][2 * j + 1] += w_ * __uint_as_float(ww[j] & 0xFFFF0000u);
        }
      }
    }
    if (LPC > 1) {
#pragma unroll
      for (int o = CPH; o < 32; o <<= 1)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[0][j] += __shfl_xor_sync(0xffffffffu, acc[0][j], o);
    }
    if (lane < (CPH >= 32 ? 32 : CPH)) {
#pragma unroll
      for (int a = 0; a < CPL; ++a) {
        const int c = (CPH >= 32) ? (lane + 32 * a) : lane;
        __nv_bfloat16* o = p.out + (int64_t)row * p.os_row + (int64_t)h * p.os_head + c * 8;
        uint4 r;
        uint32_t rr[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          __nv_bfloat162 t = __floats2bfloat162_rn(acc[a][2 * j], acc[a][2 * j + 1]);
          rr[j] = *reinterpret_cast<uint32_t*>(&t);
        }
        r = make_uint4(rr[0], rr[1], rr[2], rr[3]);
        *reinterpret_cast<uint4*>(o) = r;
      }
    }
    if (p.out_lse && lane == 0)
      p.out_lse[(int64_t)row * HPR + h] = (m == -INFINITY) ? -INFINITY : (m + log2f(sum)) / kLog2e;
    __syncwarp();
    if (lane < W) st_u32(lb + lane * HPR + h, kSent);
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(p.ticket, 1) == (int)gridDim.x - 1) {
      *p.ticket = 0;
      __threadfence();
      atomicXor(p.epoch, 1);  // only the parity is used: flip it (never overflows)
    }
  }
}

}  // namespace k31dcpx

// ------------------------------------------------------------------ host side
using namespace k31dcpx;

static int g_num_sms = 0;
// Captured by value into each launch (CUDA graphs keep the value at capture time).
static long long g_spin_ns = kSpinNsDefault;

void set_spin_ns(int64_t ns) { g_spin_ns = (long long)ns; }
int64_t get_spin_ns() { return (int64_t)g_spin_ns; }

// Co-resident top-k clusters of ``cluster`` CTAs with ``segmax`` staged keys each
// (cudaOccupancyMaxActiveClusters; cached per shape). A launch walks rows beyond it in
// turns, so the decode policy (``DcpxDecode.launch_params``) keeps a step's rows within
// it: cluster packing per GPC makes it smaller than SMs x CTAs per SM / cluster.
int64_t topk_resident_clusters(int64_t cluster, int64_t segmax) {
  TORCH_CHECK(cluster >= 1 && cluster <= 16 && segmax >= 0);
  static std::map<std::pair<int64_t, int64_t>, int> resident;
  auto key = std::make_pair(cluster, segmax);
  auto itr = resident.find(key);
  if (itr != resident.end()) return itr->second;
  auto kern = topk_merge_kernel<16>;
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
    cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    configured = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)cluster, 1, 1);
  cfg.blockDim = dim3(kThreads, 1, 1);
  cfg.dynamicSmemBytes = (size_t)segmax * 2;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = (unsigned)cluster;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  int nres = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&nres, (void*)kern, &cfg));
  TORCH_CHECK(nres >= 1, "topk_merge: no co-resident cluster");
  resident[key] = nres;
  return nres;
}

void topk_merge(torch::Tensor logits, torch::Tensor local_len, torch::Tensor global_len,
                torch::Tensor out, torch::Tensor out_count,
                c10::optional<torch::Tensor> block_table, int64_t block_size, int64_t next_n,
                int64_t block_stride, c10::optional<torch::Tensor> req_of_row, int64_t tie_rank,
                int64_t box, int64_t mc, torch::Tensor peers, int64_t W, int64_t rank,
                int64_t K, int64_t cap, int64_t cap1, int64_t nw1, int64_t w1c, int64_t off_w1,
                int64_t off_c1, int64_t off_f1, int64_t off_r8, int64_t off_r4,
                int64_t off_tl, int64_t off_tr, int64_t off_tk, int64_t interleave,
                int64_t small_ctx, torch::Tensor epoch,
                torch::Tensor ticket, torch::Tensor err, int64_t cluster, int64_t segmax,
                c10::optional<torch::Tensor> dbg, c10::optional<torch::Tensor> stats) {
  TORCH_CHECK(logits.dtype() == torch::kBFloat16 && logits.stride(1) == 1);
  TORCH_CHECK(W <= kMaxW && cluster >= 1 && cluster <= 16);
  TORCH_CHECK(K >= 1 && K <= kMaxK, "topk_merge: K must be in [1, 2048]");
  TORCH_CHECK(nw1 == 0 || nw1 == 256 || nw1 == 512);
  const int T = (int)logits.size(0);
  if (T == 0) return;
  const c10::cuda::CUDAGuard guard(logits.device());
  TopkParams p;
  p.logits = reinterpret_cast<const uint16_t*>(logits.data_ptr());
  p.lstride = logits.stride(0);
  p.local_len = local_len.data_ptr<int>();
  p.global_len = global_len.data_ptr<int>();
  p.out = out.data_ptr<int>();
  p.ostride = out.stride(0);
  p.out_count = out_count.data_ptr<int>();
  p.block_table = block_table.has_value() ? block_table->data_ptr<int>() : nullptr;
  p.bt_stride = block_table.has_value() ? block_table->stride(0) : 0;
  p.block_size = (int)block_size;
  p.next_n = (int)next_n;
  p.block_stride = block_stride > 0 ? (int)block_stride : (int)block_size;
  p.req_of_row = req_of_row.has_value() ? req_of_row->data_ptr<int>() : nullptr;
  p.tie_rank = (int)tie_rank;
  p.box = reinterpret_cast<uint32_t*>(box);
  p.mc = (uint64_t)mc;
  p.peers = reinterpret_cast<const uint64_t*>(peers.data_ptr<int64_t>());
  p.W = (int)W; p.rank = (int)rank; p.K = (int)K; p.T = T;
  p.cap = (int)cap; p.cap1 = (int)cap1; p.nw1 = (int)nw1; p.w1c = (int)w1c;
  p.off_w1 = off_w1; p.off_c1 = off_c1; p.off_f1 = off_f1; p.off_r8 = off_r8;
  p.off_r4 = off_r4; p.small_ctx = (int)small_ctx;
  p.off_tl = off_tl; p.off_tr = off_tr; p.off_tk = off_tk;
  p.interleave = (int)interleave;
  TORCH_CHECK(interleave >= 1);
  p.stats = stats.has_value() ? stats->data_ptr<int>() : nullptr;
  TORCH_CHECK(!dbg.has_value() || dbg->numel() >= 32, "topk_merge: dbg needs 32 int64");
  p.dbg = dbg.has_value() ? reinterpret_cast<long long*>(dbg->data_ptr<int64_t>()) : nullptr;
  p.epoch = epoch.data_ptr<int>();
  p.ticket = ticket.data_ptr<int>();
  p.err = err.data_ptr<int>();
  p.segmax = (int)segmax;
  p.spin_ns = g_spin_ns;
  TORCH_CHECK(T <= cap);
  TORCH_CHECK(epoch.numel() >= T, "topk_merge: one epoch per row");

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)cluster, 1, 1);
  cfg.blockDim = dim3(kThreads, 1, 1);
  cfg.dynamicSmemBytes = (size_t)segmax * 2;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[2];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = (unsigned)cluster;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  // Programmatic dependent launch: never (stream-serialized launches). The decode's
  // DSA sparse core polls its q tiles before its PDL wait, so the kernel before it
  // must not release it early; a PDL top-k measured neutral (16-rank B1 step, PDL at
  // <= 16 rows vs never: 2 us per 9 layers). With the release at kernel entry (since
  // removed) the 16-rank step hung at 64 rows (B8 q8, trtllm-gen).
  constexpr int s_pdl_rows = 0;
  const bool s_pdl = T <= s_pdl_rows;
  attr[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[1].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  auto kern = topk_merge_kernel<16>;
  // Persistent rows: at most the co-resident cluster count (identical on every
  // rank, so all ranks walk the rows in the same order).
  const int nres = (int)topk_resident_clusters(cluster, segmax);
  cfg.gridDim = dim3((unsigned)cluster, (unsigned)std::min(T, nres), 1);
  cfg.numAttrs = s_pdl ? 2 : 1;
  p.pdl = s_pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, p));
}

void o_exchange(torch::Tensor y, torch::Tensor lse, bool lse_base_e, torch::Tensor out,
                c10::optional<torch::Tensor> out_lse, int64_t box, torch::Tensor peers,
                int64_t W, int64_t rank, int64_t HPR, int64_t cap, int64_t off_data,
                int64_t off_lse, torch::Tensor epoch, torch::Tensor ticket, torch::Tensor err,
                int64_t grid) {
  TORCH_CHECK(y.dtype() == torch::kBFloat16 && y.stride(2) == 1);
  const int T = (int)y.size(0);
  if (T == 0) return;
  const int D = (int)y.size(2);
  TORCH_CHECK(D == 128 || D == 512);
  TORCH_CHECK(T <= cap);
  const c10::cuda::CUDAGuard guard(y.device());
  OxParams p;
  p.y = reinterpret_cast<const __nv_bfloat16*>(y.data_ptr());
  p.ys_row = y.stride(0); p.ys_head = y.stride(1);
  p.lse = lse.data_ptr<float>();
  p.ls_row = lse.stride(0); p.ls_head = lse.stride(1);
  p.lse_base_e = lse_base_e ? 1 : 0;
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.os_row = out.stride(0); p.os_head = out.stride(1);
  p.out_lse = out_lse.has_value() ? out_lse->data_ptr<float>() : nullptr;
  p.box = reinterpret_cast<uint32_t*>(box);
  p.peers = reinterpret_cast<const uint64_t*>(peers.data_ptr<int64_t>());
  p.W = (int)W; p.rank = (int)rank; p.T = T; p.HPR = (int)HPR; p.D = D; p.cap = (int)cap;
  p.off_data = off_data; p.off_lse = off_lse;
  p.epoch = epoch.data_ptr<int>(); p.ticket = ticket.data_ptr<int>(); p.err = err.data_ptr<int>();
  p.spin_ns = g_spin_ns;
  auto stream = c10::cuda::getCurrentCUDAStream();
  const int items = max(T * (int)W, T * (int)HPR);
  int g = (int)std::min<int64_t>(grid, (items + 7) / 8);
  g = std::max(g, 1);
  if (D == 128) o_exchange_kernel<128><<<g, 256, 0, stream>>>(p);
  else o_exchange_kernel<512><<<g, 256, 0, stream>>>(p);
  C10_CUDA_CHECK(cudaGetLastError());
}

int64_t max_coresident_ox(int64_t D) {
  int dev;
  cudaGetDevice(&dev);
  int sms;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  int occ = 0;
  if (D == 128)
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, o_exchange_kernel<128>, 256, 0);
  else
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, o_exchange_kernel<512>, 256, 0);
  return (int64_t)sms * occ;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("topk_merge", &topk_merge);
  m.def("o_exchange", &o_exchange);
  m.def("max_coresident_ox", &max_coresident_ox);
  m.def("topk_resident_clusters", &topk_resident_clusters);
  m.def("set_spin_ns", &set_spin_ns);
  m.def("get_spin_ns", &get_spin_ns);
}
