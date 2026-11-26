// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Kimi K3.1 DSA sparse core under DCP (= TP), small M (<= 64 rows): one kernel from
// the gathered q mailbox and this rank's DcpExchange top-k slots to the head
// owners' DcpBox O mailboxes. It replaces FlashInfer's sparse decode (trtllm-gen +
// its reduction / LSE glue) and DcpValue's push; DcpValue(pushed=True) merges.
//
// CTA (split s, head group g, row t), 512 threads, 32 query heads (2 MMA tiles of 16):
//
//  * before the PDL wait: polls its [32, 576] FP8 query tile out of the q mailbox
//    (MlaQuery multicasts it; 0xFFFFFFFF words are the armed state, a published FP8
//    word never is), so MlaQuery needs no gather poll of its own. The mailbox of this
//    parity was re-armed by the DcpValue two DSA layers back, which completed before
//    this grid could start (every kernel between them waits on its predecessor
//    before triggering its dependents).
//  * after the wait (the top-k kernel's slots and counts, the current token's KV
//    insert): the row's local count n and the slots of this split's 128-key tiles
//    (tiles s, s + S, .., two 64-row chunks each), KV rows gathered with cp.async (2
//    chunks deep), FP8 QK (mma.sync e4m3, FP32 accumulation) and the stock FP8 MLA
//    decode's precision (trtllm-gen's multi-CTA KV split; probed bitwise on VR200 at
//    rows 1-64 and local counts 100-2048, probe_split3): the batch's trtllm-gen KV CTA
//    count C (``kv_ctas``, its host heuristic for this many rows) splits a row's
//    T = ceil(n / 128) tiles into pieces of k = ceil(T / C) consecutive tiles (one
//    trtllm-gen CTA each). Per piece, base-2 softmax in FP32 with one running max
//    step m per 128-key tile, P = E4M3(448 x 2^(s - m)), the piece's FP32 P V and row
//    sum (from the unquantized P) rescaled by 2^(m_old - m) at each step, P x V with
//    both operands as exactly converted FP16 (E4M3 values; FP32 accumulation: the
//    products and accumulator precision of an E4M3 x E4M3 MMA), the piece's partial
//    O = P V x vscale / 448 rounded to BF16 unnormalized (C > 1; C == 1: no split,
//    no partial rounding); the pieces merged in FP32 (weights 2^(m_p - M), sum
//    L = sum l_p 2^(m_p - M)), O = merged / L rounded to BF16.
//  * S == 1: the normalized partial O (bf16, no negative zero: 0x8000 halves are the
//    mailbox sentinel's) and its base-2 LSE go straight to the head owners'
//    DcpBox.partial / DcpBox.lse (slot ((parity * W + rank) * cap + t) * H + h).
//    Rows without local entries push zeros and the EMPTY LSE (DcpValue's rule).
//  * S > 1 (2, 4, 8): the S splits of (t, g) are one cluster (pieces s, s + S, ..).
//    Each leaves its FP32 O (its pieces' BF16 partials merged at its running max) and (max, sum) in its shared
//    memory; after one cluster barrier split s merges latent dims [512 s / S, +512 /
//    S) of all splits through distributed shared memory (FP32 weights 2^(m_s - M) /
//    L, one BF16 rounding) and pushes them (split 0: the LSE).
//
// Every row's whole query tile is polled by every CTA of the row whatever its local
// count, so when this grid completes every rank's publish of rows < T has landed
// here, and the DcpValue after it may re-arm the mailbox (after its own PDL wait).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <algorithm>
#include <type_traits>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

namespace k31dsac {

constexpr int kThreads = 512;
constexpr uint32_t kSent = 0xFFFFFFFFu;  // q mailbox armed word
constexpr int kMaxK = 2048;
constexpr int kChunk = 64;  // KV rows per chunk (two per 128-key softmax tile)
constexpr int kTile = 2 * kChunk;  // keys per running-max step (trtllm-gen's tile)
constexpr float kPScale = 448.f;   // P = E4M3(kPScale x 2^(s - m)) (E4M3's largest)
constexpr int kRowB = 576;  // FP8 KV / Q row bytes
constexpr int kRowS = 592;  // padded shared row stride (conflict-free ldmatrix)
constexpr int kSStr = kTile + 4;   // S tile stride (floats)
constexpr int kPStr = kTile + 8;   // P tile stride (halves)
constexpr int kOStr = 520;  // O staging stride (bf16)
constexpr float kEmptyLse = -3.0e38f;

constexpr int kVStr = 520;  // FP16 V row stride (halves; conflict-free ldmatrix.trans)

// Shared memory plan of a CTA of HG query heads (16 or 32), bytes.
template <int HG>
struct Plan {
  static constexpr int kQ = 0;                              // [HG][kRowS] FP8
  static constexpr int kKV = kQ + HG * kRowS;               // 2 x [64][kRowS] FP8 rows
  static constexpr int kVh = kKV + 2 * kChunk * kRowS;      // [64][kVStr] FP16 V
  static constexpr int kS = kVh + kChunk * kVStr * 2;       // 2 x [16][kSStr] / [32][kSStr]
                                                            // (one 128-key tile)
  static constexpr int kP = kS + 32 * kSStr * 4;            // [HG][kPStr] FP16
  static constexpr int kIdx = kP + HG * kPStr * 2;          // [kMaxK] slots
  static constexpr int kStat = kIdx + kMaxK * 4;            // m, l, alpha, beta, piece m, l, alpha [HG]
  static constexpr int kSmem = kStat + 7 * HG * 4;
  static constexpr int kQChunks = HG * kRowB / 16;          // 16 B chunks per Q tile
  static constexpr int kQPer = (kQChunks + kThreads - 1) / kThreads;
  static_assert(kSmem + 2048 <= 227 * 1024, "dsa_core shared memory");
  static_assert(HG * 512 * 4 <= 2 * kChunk * kRowS, "FP32 O staging over the KV buffers");
};

struct Params {
  const uint8_t* q;       // q mailbox of this parity [rows, q_heads, 576] FP8 bytes
  int64_t q_row_stride;   // bytes per row
  const int* sel;         // [T, sel_stride] this rank's KV slots (flat rows)
  int64_t sel_stride;
  int width;              // slots per row (<= kMaxK)
  const int* cnt;         // [T] local counts
  const uint8_t* kv;      // flat KV rows (576 B)
  float s2scale;          // softmax scale x q scale x k scale x log2(e)
  float vscale;           // k scale (V dequant)
  const uint64_t* part_peers;  // [W] DcpBox.partial bases
  const uint64_t* lse_peers;   // [W] DcpBox.lse bases
  int lse_wide;  // DcpBox.lse as [2, CAP, H, W] (DcpValue's wide / spread merge)
  int split_min;  // S > 1: rows of at most this many 64-key chunks run on split 0 alone
  int kv_ctas;    // trtllm-gen's KV CTAs per row at this batch (pieces of ceil(T / kv_ctas) tiles)
  int box_cap, hpr, parity, rank, W, T, splits;
  int* err;
  long long spin_ns;
  long long* dbg;         // optional [T * G * S, 16] timestamps (ns): start, q polled,
                          // slots, loop done, pushed, after the PDL wait
  // ---- the #63 merge in the core (``mic``; see "merge in core" below). 0: the
  // slots / counts come from ``sel`` / ``cnt`` (the merge's outputs).
  int mic;
  const uint32_t* box;    // this rank's round-1 mailbox [2, max_rows, W, box_words] (u32)
  int* ordinals;          // [max_rows] merges per row so far (the mailbox's)
  int* readers;           // [2, rcap] zeroed: CTAs of a row done with round 1 / 2
  int rcap, max_rows, nloc, topk, box_words, gid_bits;
  const int* req_id;      // [T] request of a row
  const int* block_table; // [reqs, bt_width] MLA cache blocks
  int64_t bt_stride;
  int bt_width, block_size, stride_rows;
  // exact merge's second round (DCPTopkRescueWorkspace + the indexer's logits)
  const uint16_t* logits;  // [T, logits_stride] BF16 bits
  int64_t logits_stride;
  int logits_width;
  const int* row_ends;     // [T]
  const uint64_t* peer_recv;   // [W] recv bases [max_rows, slots, K + K / 32]
  const uint64_t* peer_flags;  // [W] flag bases [W, max_rows]
  int* seq2;               // [max_rows]
  int num_slots;
  int* rows_out;           // optional [T, rows_stride]: this rank's rows (CTA (0, 0) of a row)
  int64_t rows_stride;
  int* valid_out;          // optional [T]
};

// ------------------------------------------------------------------ primitives
__device__ __forceinline__ long long now_ns() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
static __device__ unsigned int g_fail_printed;  // one timeout line per process and GPU
__device__ __noinline__ void fail(int* err, int code) {
  atomicCAS(err, 0, code);
  __threadfence_system();
  // the trap ends the context before the host can read err: name it first
  if (atomicCAS(&g_fail_printed, 0u, 1u) == 0u) {
    unsigned sm;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    printf("K31_SPIN_TIMEOUT kernel=k31_dsa_core site=%d smid=%u block=%d,%d thread=%d\n", code, sm,
           (int)blockIdx.x, (int)blockIdx.y, (int)threadIdx.x);
  }
  __trap();
}
__device__ __forceinline__ uint4 ld_sys_v4(const void* p) {
  uint4 v;
  asm volatile("ld.relaxed.sys.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
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
__device__ __forceinline__ int ld_cg(const int* p) {
  int v;
  asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_launch() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
__device__ __forceinline__ void ldsm_x4(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d,
                                        const void* p) {
  uint32_t s = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
               : "r"(s));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d,
                                          const void* p) {
  uint32_t s = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
               : "r"(s));
}
__device__ __forceinline__ void ldsm_x2(uint32_t& a, uint32_t& b, const void* p) {
  uint32_t s = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
               : "=r"(a), "=r"(b)
               : "r"(s));
}
__device__ __forceinline__ void mma_fp8(float* c, uint32_t a0, uint32_t a1, uint32_t a2,
                                        uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, "
      "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_f16(float* c, uint32_t a0, uint32_t a1, uint32_t a2,
                                        uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, "
      "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// two E4M3 bytes (lo, hi) -> f16x2 (exact)
__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint32_t two) {
  uint32_t out;
  uint16_t in = (uint16_t)two;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(out) : "h"(in));
  return out;
}
__device__ __forceinline__ bool armed(uint4 v) {
  return v.x == kSent || v.y == kSent || v.z == kSent || v.w == kSent;
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
// a float / float4 at the same shared address of cluster CTA ``rank``
__device__ __forceinline__ float ld_dsmem(const float* p, int rank) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(p), r;
  asm("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
  float v;
  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(r));
  return v;
}
__device__ __forceinline__ float4 ld_dsmem_v4(const float* p, int rank) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(p), r;
  asm("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
  float4 v;
  asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
               : "r"(r));
  return v;
}
__device__ __forceinline__ uint32_t bits(__nv_bfloat162 v) {
  return *reinterpret_cast<uint32_t*>(&v);
}
// index of (row t, gathered head gh) from this rank in the owner's DcpBox.lse
__device__ __forceinline__ int64_t lse_slot(const Params& p, int t, int gh) {
  const int H = p.hpr;
  if (p.lse_wide) return (((int64_t)p.parity * p.box_cap + t) * H + gh % H) * p.W + p.rank;
  return (((int64_t)p.parity * p.W + p.rank) * p.box_cap + t) * H + gh % H;
}
// a bf16 pair without negative zeros (0x8000 halves: DcpBox.partial's sentinel)
__device__ __forceinline__ uint32_t no_neg_zero(uint32_t w) {
  if ((w & 0xFFFFu) == 0x8000u) w &= 0xFFFF0000u;
  if ((w >> 16) == 0x8000u) w &= 0xFFFFu;
  return w;
}

// ------------------------------------------------------------------ merge in core
// ``mic``: the #63 exact merge (DCPTopkSelectKernel, mailbox + fold) runs inside the
// core, so the step's chain is pack (publishes) -> core instead of pack -> select ->
// core. Every CTA of a row polls the row's W round-1 mailbox slots (before its PDL
// wait, as the select), and the last of the row's CTAs to stage them re-arms the slots
// and bumps the row's ordinal (the select's re-arm / bump, once per row). Every CTA
// then finds the cut of the global top-K under the select's total order (bf16 score
// desc, global id asc; keys (ord16 << gid_bits) | ~gid, the same radix passes) and
// takes this rank's selected keys in the select's order (slot order j of its own
// round-1 row) as its slots: the same slots and count as the select's rows_out /
// valid_out, so the core's arithmetic is unchanged (bitwise the select + core).
// Saturated rows (the select's sat mask: a rank whose N words are all above tau16) run
// the exact merge's second round here as the folded select does: each CTA of a
// saturated rank pushes its local top-K of the logits to every rank's rescue slot
// (identical bytes from every CTA of the row; the flag value is the row's seq2 + 1),
// every CTA waits for the mask's flags, re-selects over the other ranks' round-1 words
// at or above tau16 plus the pushed lists, and the row's last CTA bumps seq2.
// Every CTA pushes before it waits, and a row's CTAs are launched in the same order on
// every rank (as the folded select's "every CTA resident" argument).

__device__ __forceinline__ uint32_t ld_cg_u32(const void* p) {
  uint32_t v;
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ uint16_t ld_cg_u16(const void* p) {
  uint16_t v;
  asm volatile("ld.global.cg.u16 %0, [%1];" : "=h"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ int ld_acquire_sys(const int* p) {
  int v;
  asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_sys(int* p, int v) {
  asm volatile("st.release.sys.global.s32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t ord16(uint32_t b) {
  return (b & 0x8000u) ? (~b & 0xFFFFu) : (b | 0x8000u);
}
// inclusive block scan of 512 threads; ``total``: the block's sum
__device__ __forceinline__ int block_scan(int x, int* wt, int lane, int warp, int* total) {
  int v = x;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(~0u, v, o);
    if (lane >= o) v += y;
  }
  if (lane == 31) wt[warp] = v;
  __syncthreads();
  if (warp == 0) {
    int w = lane < kThreads / 32 ? wt[lane] : 0;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int y = __shfl_up_sync(~0u, w, o);
      if (lane >= o) w += y;
    }
    if (lane < kThreads / 32) wt[lane] = w;
  }
  __syncthreads();
  const int r = v + (warp > 0 ? wt[warp - 1] : 0);
  if (total) *total = wt[kThreads / 32 - 1];
  __syncthreads();
  return r;
}
// The cut of the top-``K`` of ``n`` keys (``key(i)``; 0 = empty; keys unique) under the
// select's radix schedule (9 + 7 score bits, then the gid bits 9 at a time while the
// K-th key's score is tied): key i is selected iff key(i) != 0 && key(i) >= the result.
// ``*full``: K keys selected (else every key: fewer than K).
template <class F>
__device__ uint64_t radix_cut(F key, int n, int K, int gb, int* hist, int* wt, int* sc, int tid,
                              int lane, int warp, bool* full) {
  uint64_t prefix = 0;
  int need = K, shift = gb + 7, bits = 9;
  for (int pass = 0;; ++pass) {
    hist[tid] = 0;  // (512 bins = 512 threads)
    __syncthreads();
    const int top = shift + bits;
    for (int i = tid; i < n; i += kThreads) {
      const uint64_t k = key(i);
      if (k != 0 && (k >> top) == prefix)
        atomicAdd(&hist[(int)((k >> shift) & ((1ull << bits) - 1))], 1);
    }
    __syncthreads();
    const int count = hist[kThreads - 1 - tid];
    const int incl = block_scan(count, wt, lane, warp, nullptr);
    if (count > 0 && incl >= need && incl - count < need) {
      sc[0] = kThreads - 1 - tid;
      sc[1] = incl - count;
      sc[2] = incl == need;
      sc[3] = 1;
    }
    if (tid == kThreads - 1 && incl < need) {  // fewer keys than K (pass 0): all of them
      sc[0] = 0;
      sc[1] = 0;
      sc[2] = 1;
      sc[3] = 0;
    }
    __syncthreads();
    const int thr = sc[0], above = sc[1], done = sc[2], kfull = sc[3];
    __syncthreads();  // (sc, hist reused)
    const uint64_t np = (prefix << bits) | (uint64_t)thr;
    if (done || shift == 0) {
      *full = kfull != 0;
      return np << shift;
    }
    need -= above;
    prefix = np;
    if (pass == 0) {
      shift = gb;
      bits = 7;
    } else {
      bits = min(9, shift);
      shift -= bits;
    }
  }
}

constexpr int kTab = 1024;  // block-table entries of a row kept in shared memory (as the select)

// The saturated rank's push (DCPTopkSelectKernel.push_local_topk): its local top-min(K,
// valid) of ``logits[t]`` in the B' layout (keys above the cut v in ascending local
// index, then keys equal to v in ascending local index up to K) -> every rank's
// recv[t, slot]; then flag (rank, t) = ``expected`` on every rank.
__device__ void mic_push(const Params& p, int t, int slot, int expected, uint32_t* buf, int* hist,
                         int* wt, int* sc, int tid, int lane, int warp) {
  const int K = p.topk, RW = K + K / 32;
  int end = min(p.logits_width, 1 << 17);
  end = min(end, ld_cg(p.row_ends + t));
  const uint16_t* lrow = p.logits + (int64_t)t * p.logits_stride;
  auto lo = [&](int l) -> uint32_t {
    const uint32_t b = ld_cg_u16(lrow + l);
    return (b & 0x7FFFu) <= 0x7F80u ? ord16(b) : 0u;  // NaN: not a candidate
  };
  hist[tid] = 0;
  if (tid == 0) {
    sc[4] = -1;
    sc[5] = 0;
  }
  __syncthreads();
  for (int l = tid; l < end; l += kThreads) {
    const uint32_t o = lo(l);
    if (o) atomicAdd(&hist[o >> 7], 1);
  }
  __syncthreads();
  {
    const int count = hist[kThreads - 1 - tid];
    const int incl = block_scan(count, wt, lane, warp, nullptr);
    if (count > 0 && incl >= K && incl - count < K) {
      sc[4] = kThreads - 1 - tid;
      sc[5] = incl - count;
    }
  }
  __syncthreads();
  const int b1 = sc[4], above = sc[5];
  uint32_t v = 0;
  int cgt = 0;
  if (b1 >= 0) {
    hist[tid] = 0;
    __syncthreads();
    for (int l = tid; l < end; l += kThreads) {
      const uint32_t o = lo(l);
      if (o && (int)(o >> 7) == b1) atomicAdd(&hist[o & 127], 1);
    }
    __syncthreads();
    const int c2 = hist[kThreads - 1 - tid];
    const int i2 = block_scan(c2, wt, lane, warp, nullptr);
    const int need2 = K - above;
    if (c2 > 0 && i2 >= need2 && i2 - c2 < need2) {
      sc[6] = kThreads - 1 - tid;
      sc[7] = above + i2 - c2;
    }
    __syncthreads();
    v = ((uint32_t)b1 << 7) | (uint32_t)sc[6];
    cgt = sc[7];
  }
  for (int i = tid; i < RW; i += kThreads) buf[i] = 0;
  __syncthreads();
  int eq_run = 0, gt_run = 0;
  for (int base = 0; base < end; base += kThreads) {
    const int l = base + tid;
    const uint32_t o = l < end ? lo(l) : 0u;
    const bool is_eq = o == v && o != 0;
    int flag = is_eq ? 1 : 0;
    if (o > v) flag = 1 << 16;
    int tot;
    const int incl = block_scan(flag, wt, lane, warp, &tot);
    int pos = -1;
    if (o > v) pos = gt_run + (incl >> 16) - 1;
    if (is_eq) {
      const int idx = eq_run + (incl & 0xFFFF) - 1;
      if (idx < K - cgt) pos = cgt + idx;
    }
    if (pos >= 0) {
      buf[pos] = (o << 16) | (0xFFFFu - ((uint32_t)l & 0xFFFFu));
      if ((uint32_t)l >> 16) atomicOr(&buf[K + (pos >> 5)], 1u << (pos & 31));
    }
    eq_run += tot & 0xFFFF;
    gt_run += tot >> 16;
  }
  __syncthreads();
  const int64_t off = ((int64_t)t * p.num_slots + slot) * RW;
  for (int pr = 0; pr < p.W; ++pr) {
    uint32_t* dst = reinterpret_cast<uint32_t*>(p.peer_recv[pr]) + off;
    for (int i = tid; i < RW / 4; i += kThreads)
      st_v4(dst + 4 * i, reinterpret_cast<const uint4*>(buf)[i]);
  }
  __syncthreads();
  if (tid < p.W) {
    asm volatile("fence.acq_rel.sys;" ::: "memory");
    int* f = reinterpret_cast<int*>(p.peer_flags[tid]) + ((int64_t)p.rank * p.max_rows + t);
    st_release_sys(f, expected);
  }
}

// The merge in the core (see above): this rank's selected slots of row t -> sIdx [0, n)
// (-1 after), returns n. Waits on the PDL predecessor (and triggers the dependents) itself.
__device__ int mic_front(const Params& p, int t, int tid, int lane, int warp, uint8_t* kvbuf,
                         uint8_t* scratch, int* sIdx, long long* dbg) {
  __shared__ int wt[kThreads / 32];
  __shared__ int sc[8];
  __shared__ int notsat[16], owner[16];
  __shared__ int s_last, s_tau;
  const int W = p.W, N = p.nloc, K = p.topk, BW = p.box_words, gb = p.gid_bits;
  const uint64_t gmask = (1ull << gb) - 1;
  uint32_t* stage = reinterpret_cast<uint32_t*>(kvbuf);
  uint32_t* pbuf = stage + W * BW;  // the push's B' row (W * BW: a multiple of 4)
  int* hist = reinterpret_cast<int*>(scratch);  // [512]
  int* tab = hist + kThreads;                   // [kTab]
  const int nreaders = gridDim.x * gridDim.y;
  const int ordinal = ld_cg(p.ordinals + t);
  const int seq = ld_cg(p.seq2 + t);
  const int req = ld_cg(p.req_id + t);
  const int* trow = p.block_table + (int64_t)req * p.bt_stride;
  const int twidth = min(p.bt_width, kTab);
  for (int i = tid; i < twidth; i += kThreads) tab[i] = ld_cg(trow + i);
  // ---- round 1: the row's W slots (Lamport: no word is the armed 0xFFFFFFFF)
  const uint32_t* base = p.box + ((int64_t)(ordinal & 1) * p.max_rows + t) * (int64_t)W * BW;
  const int chunks = W * BW / 4;
  for (int c = tid; c < chunks; c += kThreads) {
    uint4 v = ld_sys_v4(base + 4 * c);
    if (armed(v)) {
      const long long ts = now_ns();
      do {
        __nanosleep(32);
        if (now_ns() - ts > p.spin_ns) fail(p.err, 3);
        v = ld_sys_v4(base + 4 * c);
      } while (armed(v));
    }
    reinterpret_cast<uint4*>(stage)[c] = v;
  }
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    s_last = atomicAdd(p.readers + t, 1) == nreaders - 1;
  }
  __syncthreads();
  const bool last = s_last;
  if (last) {  // every CTA of the row has staged it: re-arm (the select's per-chunk re-arm)
    uint32_t* wb = const_cast<uint32_t*>(base);
    for (int c = tid; c < chunks; c += kThreads) st_v4(wb + 4 * c, make_uint4(kSent, kSent, kSent, kSent));
    if (tid == 0) p.readers[t] = 0;
  }
  if (dbg && tid == 0) dbg[19] = now_ns();
  pdl_wait();
  // DcpValue(pushed) only stages W_UV before its own wait: let it become resident
  pdl_launch();
  if (dbg && tid == 0) dbg[5] = now_ns();
  if (last && tid == 0) p.ordinals[t] = ordinal + 1;
  // ---- round-1 keys (ord16 << gid_bits) | ~gid of slot j of rank r; 0: empty
  auto r1 = [&](int i) -> uint64_t {
    const int r = i / N, j = i - r * N;
    const uint32_t* s = stage + r * BW;
    const uint32_t w = s[j];
    if (w == 0) return 0;
    const uint32_t* pl = s + N + 2 * (j >> 5);
    const uint32_t hi = (pl[0] & 0xFFFFu) | (pl[1] << 16);
    const uint32_t local = (0xFFFFu - (w & 0xFFFFu)) | (((hi >> (j & 31)) & 1u) << 16);
    const uint64_t gid = (uint64_t)local * W + r;
    return ((uint64_t)(w >> 16) << gb) | (gmask - gid);
  };
  const int n1 = W * N;
  bool full;
  const uint64_t cut = radix_cut(r1, n1, K, gb, hist, wt, sc, tid, lane, warp, &full);
  // tau16: the smallest selected score (0 when fewer than K were selected)
  {
    int mn = 0x10000;
    if (full) {
      for (int i = tid; i < n1; i += kThreads) {
        const uint64_t k = r1(i);
        if (k != 0 && k >= cut) mn = min(mn, (int)(k >> gb));
      }
    }
    mn = __reduce_min_sync(~0u, mn);
    if (lane == 0) wt[warp] = mn;
    if (tid < 16) notsat[tid] = 0;
    __syncthreads();
    if (tid == 0) {
      int m = 0x10000;
      for (int w = 0; w < kThreads / 32; ++w) m = min(m, wt[w]);
      s_tau = full ? m : 0;
    }
    __syncthreads();
  }
  const int tau = s_tau;
  // rank r is saturated iff all its N slots hold a key with ord16 > tau16
  for (int i = tid; i < n1; i += kThreads) {
    const uint64_t k = r1(i);
    if (k == 0 || (int)(k >> gb) <= tau) notsat[i / N] = 1;
  }
  __syncthreads();
  int mask = 0;
  for (int r = 0; r < W; ++r)
    if (!notsat[r]) mask |= 1 << r;
  // distinct keys bound the saturated ranks by the slots; else round 1 stands (select)
  if (__popc(mask) > p.num_slots) mask = 0;
  if (dbg && tid == 0) dbg[20] = now_ns();
  const int RW = K + K / 32;
  const uint32_t* recv_row = nullptr;
  const int expected = seq + 1;
  const bool me_sat = (mask >> p.rank) & 1;
  int sme = 0;
  // ---- the second round of a saturated row (DCPTopkRescueKernel / the folded select)
  auto r2 = [&](int i) -> uint64_t {
    if (i < n1) {
      if ((mask >> (i / N)) & 1) return 0;
      const uint64_t k = r1(i);
      return (k != 0 && (int)(k >> gb) < tau) ? 0 : k;
    }
    const int s = (i - n1) / K, j = i - n1 - s * K;
    const uint32_t* b = recv_row + (int64_t)s * RW;
    const uint32_t w = ld_cg_u32(b + j);
    if (w == 0) return 0;
    const uint32_t hi = ld_cg_u32(b + K + (j >> 5));
    const uint32_t local = (0xFFFFu - (w & 0xFFFFu)) | (((hi >> (j & 31)) & 1u) << 16);
    const uint64_t gid = (uint64_t)local * W + owner[s];
    return ((uint64_t)(w >> 16) << gb) | (gmask - gid);
  };
  uint64_t cut2 = cut;
  if (mask) {
    recv_row = reinterpret_cast<const uint32_t*>(p.peer_recv[p.rank]) + (int64_t)t * p.num_slots * RW;
    if (me_sat) {
      sme = __popc(mask & ((1 << p.rank) - 1));
      mic_push(p, t, sme, expected, pbuf, hist, wt, sc, tid, lane, warp);
    }
    if (tid == 0) {
      int k = 0;
      for (int r = 0; r < W; ++r)
        if ((mask >> r) & 1) owner[k++] = r;
    }
    if (warp == 0 && lane < W && ((mask >> lane) & 1)) {
      const int* f = reinterpret_cast<const int*>(p.peer_flags[p.rank]) + ((int64_t)lane * p.max_rows + t);
      const long long ts = now_ns();
      while (ld_acquire_sys(f) != expected) {
        __nanosleep(64);
        if (now_ns() - ts > p.spin_ns) fail(p.err, 4);
      }
    }
    __syncthreads();
    bool full2;
    cut2 = radix_cut(r2, n1 + __popc(mask) * K, K, gb, hist, wt, sc, tid, lane, warp, &full2);
  }
  if (dbg && tid == 0) dbg[21] = now_ns();
  // ---- this rank's selected keys in the select's order -> flat KV rows
  const int obase = me_sat ? n1 + sme * K : p.rank * N;
  const int ocount = me_sat ? K : N;
  const uint64_t c = mask ? cut2 : cut;
  int total = 0;
  for (int b0 = 0; b0 < ocount; b0 += kThreads) {
    const int j = b0 + tid;
    int row = -1, f = 0;
    if (j < ocount) {
      const uint64_t k = mask ? r2(obase + j) : r1(obase + j);
      if (k != 0 && k >= c) {
        const uint32_t gid = (uint32_t)(gmask - (k & gmask));
        const int local = (int)(gid / (uint32_t)W);
        const int blk = local / p.block_size;
        if (blk < p.bt_width) {
          const int e = blk < kTab ? tab[blk] : ld_cg(trow + blk);
          row = e * p.stride_rows + (local - blk * p.block_size);
          f = 1;
        }
      }
    }
    int tot;
    const int incl = block_scan(f, wt, lane, warp, &tot);
    if (f && total + incl - 1 < kMaxK) sIdx[total + incl - 1] = row;
    total += tot;
  }
  const int n = min(total, kMaxK);
  for (int k = n + tid; k < kMaxK; k += kThreads) sIdx[k] = -1;
  __syncthreads();
  if (mask && tid == 0) {  // the row's second round is over on this CTA
    __threadfence();
    if (atomicAdd(p.readers + p.rcap + t, 1) == nreaders - 1) {
      p.seq2[t] = expected;
      p.readers[p.rcap + t] = 0;
    }
  }
  if (p.rows_out && blockIdx.x == 0 && blockIdx.y == 0) {
    for (int k = tid; k < n; k += kThreads) p.rows_out[(int64_t)t * p.rows_stride + k] = sIdx[k];
    if (tid == 0) p.valid_out[t] = n;
  }
  if (dbg && tid == 0) dbg[22] = now_ns();
  return n;
}

// ------------------------------------------------------------------ kernel
// HG = 32: warp (m-tile hg, n-tile wl) of QK, 2 heads of softmax, latent dims
// [32 w, +32) of both m-tiles in PV. HG = 16: warp (n-tile w & 7, k-half w >> 3) of
// QK (the halves summed by the softmax), 1 head of softmax, dims [32 w, +32) in PV.
template <int HG>
__global__ void __launch_bounds__(kThreads, 1) dsac_kernel(const __grid_constant__ Params p) {
  using PL = Plan<HG>;
  constexpr int MT = HG / 16;  // MMA m-tiles (16 heads each)
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g8 = lane >> 2, q4 = lane & 3;
  const int S = p.splits, sp = blockIdx.x, grp = blockIdx.y, t = blockIdx.z;
  const int G = gridDim.y;
  const int h0 = grp * HG;  // first gathered head
  uint8_t* sQ = smem + PL::kQ;
  uint8_t* sKV = smem + PL::kKV;
  __half* sVh = reinterpret_cast<__half*>(smem + PL::kVh);
  float* sSall = reinterpret_cast<float*>(smem + PL::kS);
  __half* sP = reinterpret_cast<__half*>(smem + PL::kP);
  int* sIdx = reinterpret_cast<int*>(smem + PL::kIdx);
  float* sM = reinterpret_cast<float*>(smem + PL::kStat);
  float* sL = sM + HG;
  float* sA = sL + HG;
  float* sB = sA + HG;
  float* sMp = sB + HG;  // the current piece's running max, sum and rescale
  float* sLp = sMp + HG;
  float* sAp = sLp + HG;
  __shared__ int s_n;
  long long* dbg = p.dbg ? p.dbg + ((int64_t)(t * G + grp) * S + sp) * 24 : nullptr;
  if (dbg && tid == 0) dbg[0] = now_ns();

  // ---- query tile [HG, 576] FP8 from the q mailbox (polled: no PDL dependency). Split
  // 0 loads it before its PDL wait; the other splits only once their row needs them.
  auto load_q = [&]() {
    const uint8_t* qrow = p.q + (int64_t)t * p.q_row_stride + (int64_t)h0 * kRowB;
    uint4 v[PL::kQPer];
#pragma unroll
    for (int j = 0; j < PL::kQPer; ++j) {
      const int i = tid + j * kThreads;
      if (i < PL::kQChunks) v[j] = ld_sys_v4(qrow + 16 * i);
    }
#pragma unroll
    for (int j = 0; j < PL::kQPer; ++j) {
      const int i = tid + j * kThreads;
      if (i < PL::kQChunks) {
        if (armed(v[j])) {
          const long long ts = now_ns();
          do {
            __nanosleep(32);
            if (now_ns() - ts > p.spin_ns) fail(p.err, 1);
            v[j] = ld_sys_v4(qrow + 16 * i);
          } while (armed(v[j]));
        }
        const int h = i / (kRowB / 16), c = i % (kRowB / 16);
        *reinterpret_cast<uint4*>(sQ + h * kRowS + 16 * c) = v[j];
      }
    }
  };
  load_q();  // (every split: S = 2 rows all need it, solo ones split the latent dims)
  if (tid < HG) {
    sM[tid] = -INFINITY;
    sL[tid] = 0.f;
  }
  if (dbg && tid == 0) dbg[1] = now_ns();
  int n;
  if (p.mic) {
    // the merge in the core: round 1 polled before the PDL wait, this rank's slots
    n = mic_front(p, t, tid, lane, warp, sKV, reinterpret_cast<uint8_t*>(sVh), sIdx, dbg);
  } else {
    // ---- the top-k slots / counts and the KV insert of this step are ours from here
    pdl_wait();
    // DcpValue(pushed) only stages W_UV before its own wait: let it become resident
    pdl_launch();
    if (dbg && tid == 0) dbg[5] = now_ns();
    // the count and the row's whole slot list in one round trip (past the count: -1)
    const int* selrow = p.sel + (int64_t)t * p.sel_stride;
    int v[kMaxK / kThreads];
#pragma unroll
    for (int b = 0; b < kMaxK / kThreads; ++b) {
      const int k = b * kThreads + tid;
      v[b] = k < p.width ? ld_cg(selrow + k) : -1;
    }
    if (tid == 0) s_n = ld_cg(p.cnt + t);
    __syncthreads();
    n = min(s_n, p.width);
#pragma unroll
    for (int b = 0; b < kMaxK / kThreads; ++b) {
      const int k = b * kThreads + tid;
      sIdx[k] = k < n ? v[b] : -1;
    }
  }
  const int nall = (n + kChunk - 1) / kChunk;
  // rows of at most split_min chunks run on split 0 alone (no merge across the cluster;
  // the other splits leave), longer ones on every split: chunks sp, sp + S, ..
  const bool solo = S > 1 && nall <= p.split_min;
  // S = 2 solo rows: both CTAs walk every key of the row (QK and softmax computed
  // alike, so the same P), each converts and multiplies only its half of the latent
  // dims [256 sp, +256) in P V and pushes them (split 0: the LSE). Per dim the same
  // arithmetic: bitwise the same partial, half the V conversion per CTA.
  const bool dsplit = solo && S == 2;
  if (solo && !dsplit && sp > 0) return;
  const int Se = solo ? 1 : S;
  const int kspl = solo ? 0 : sp;  // this CTA's key split
  // The row's pieces (trtllm-gen's KV CTAs): kp consecutive 128-key tiles each. This
  // split's pieces: kspl, kspl + Se, ..; its local tile i is tile i % kp of its piece
  // i / kp, each tile two 64-key chunks; local chunk c is chunk c & 1 of local tile
  // c >> 1 (a row's last tile may hold one chunk)
  const int ntiles = (n + kTile - 1) / kTile;
  // (kp == 1, the common case, takes the per-tile mapping without the divisions)
  const int kp = ntiles > p.kv_ctas ? (ntiles + p.kv_ctas - 1) / p.kv_ctas : 1;
  // global 64-key chunk of local chunk c when kp > 1 (a table: no division in the loop)
  __shared__ int s_gch[kMaxK / kChunk];
  int mytiles, last;  // this split's tiles, the global tile of its last one
  if (kp == 1) {
    mytiles = ntiles > kspl ? (ntiles - kspl + Se - 1) / Se : 0;
    last = (mytiles - 1) * Se + kspl;
  } else {
    const int npieces = (ntiles + kp - 1) / kp;
    const int mypieces = npieces > kspl ? (npieces - kspl + Se - 1) / Se : 0;
    mytiles =
        mypieces > 0 ? (mypieces - 1) * kp + min(kp, ntiles - ((mypieces - 1) * Se + kspl) * kp) : 0;
    if (tid < 2 * mytiles) {
      const int i = tid >> 1;
      s_gch[tid] = 2 * (((i / kp) * Se + kspl) * kp + i % kp) + (tid & 1);
    }
    const int i = mytiles - 1;
    last = ((i / kp) * Se + kspl) * kp + i % kp;
  }
  int nchunks = 2 * mytiles;
  // (the row's last tile may hold one chunk: only a split's last local one)
  if (nchunks > 0 && 2 * last + 1 >= nall) --nchunks;
  __syncthreads();
  if (dbg && tid == 0) {
    dbg[2] = now_ns();
    dbg[18] = clock64();
  }
  // acc: the row's O so far (FP32, x vscale / 448, relative to the running max sM);
  // tacc: the current piece's P V (relative to the piece's running max sMp)
  float acc[4 * MT][4], tacc[4 * MT][4];
#pragma unroll
  for (int j = 0; j < 4 * MT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  // a finished piece (running max mp, sum lp) joins the row's running max sM / sum sL
  // (head h; alpha = 2^(M_old - M) -> A[h], beta = 2^(m_p - M) -> B[h])
  auto piece_fold = [&](int h, float mp, float lp, float* A, float* B) {
    const float mold = sM[h], mnew = fmaxf(mold, mp);
    const float alpha = mold == -INFINITY ? 0.f : exp2f(mold - mnew);
    const float beta = mp == -INFINITY ? 0.f : exp2f(mp - mnew);
    A[h] = alpha;
    B[h] = beta;
    sL[h] = sL[h] * alpha + lp * beta;
    sM[h] = mnew;
  };
  // the piece's partial O (unnormalized, rounded to BF16: trtllm-gen's multi-CTA
  // partial; kv_ctas == 1: no split, no rounding) joins the row's FP32 O
  auto acc_fold = [&](const float* A, const float* B) {
    const float vs = p.vscale / kPScale;
    const bool split = p.kv_ctas > 1;
    auto bf16r = [split](float x) {
      return split ? __bfloat162float(__float2bfloat16_rn(x)) : x;
    };
#pragma unroll
    for (int m2 = 0; m2 < MT; ++m2) {
      const float a0 = A[16 * m2 + g8], a1 = A[16 * m2 + g8 + 8];
      const float b0 = B[16 * m2 + g8], b1 = B[16 * m2 + g8 + 8];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        acc[4 * m2 + j][0] = acc[4 * m2 + j][0] * a0 + bf16r(tacc[4 * m2 + j][0] * vs) * b0;
        acc[4 * m2 + j][1] = acc[4 * m2 + j][1] * a0 + bf16r(tacc[4 * m2 + j][1] * vs) * b0;
        acc[4 * m2 + j][2] = acc[4 * m2 + j][2] * a1 + bf16r(tacc[4 * m2 + j][2] * vs) * b1;
        acc[4 * m2 + j][3] = acc[4 * m2 + j][3] * a1 + bf16r(tacc[4 * m2 + j][3] * vs) * b1;
      }
    }
  };
  // The main loop, compiled twice: P (pieces of more than one tile: the row has more
  // tiles than trtllm-gen's KV CTAs) keeps the piece's running state across its
  // tiles; !P (every piece one tile, the common case) is the per-tile form.
  auto mainloop = [&](auto pieces) {
    constexpr bool P = decltype(pieces)::value;
    auto gchunk = [&](int c) { return P ? s_gch[c] : 2 * ((c >> 1) * Se + kspl) + (c & 1); };
    // slot of row r of this CTA's chunk c
    auto slot_of = [&](int c, int r) { return sIdx[gchunk(c) * kChunk + r]; };

    // chunk c -> KV buffer c & 1 (cp.async, the two chunks of a tile); rows past the
    // count are zeroed (P = 0 must not meet a NaN code)
    auto issue = [&](int c) {
      uint8_t* buf = sKV + (c & 1) * kChunk * kRowS;
      for (int i = tid; i < kChunk * (kRowB / 16); i += kThreads) {
        const int r = i / (kRowB / 16), pc = i % (kRowB / 16);
        const int slot = slot_of(c, r);
        if (slot >= 0) {
          cp_async16(buf + r * kRowS + 16 * pc, p.kv + (int64_t)slot * kRowB + 16 * pc);
        } else {
          *reinterpret_cast<uint4*>(buf + r * kRowS + 16 * pc) = make_uint4(0, 0, 0, 0);
        }
      }
      cp_commit();
    };
    // keys of chunk c below the count (a partial last chunk skips its empty key tiles in
    // QK, the V cast and PV; the softmax masks them)
    auto valid_of = [&](int c) { return min(kChunk, n - gchunk(c) * kChunk); };
    // ---- V of chunk c -> FP16 (exact) for the PV fragments: 64 rows x 512 dims
    auto cast_v = [&](int c) {
      const uint8_t* buf = sKV + (c & 1) * kChunk * kRowS;
      constexpr int kItems = kChunk * 32 / kThreads;  // 4
      const int vrows = (valid_of(c) + 15) & ~15;  // (PV's 16-key steps)
      uint4 w[kItems];
#pragma unroll
      for (int j = 0; j < kItems; ++j) {
        const int i = tid + j * kThreads;  // (dsplit: this CTA's 16 of the 32 16-B columns)
        const int r = dsplit ? i >> 4 : i >> 5, c16 = dsplit ? 16 * sp + (i & 15) : i & 31;
        if (r < vrows) w[j] = *reinterpret_cast<const uint4*>(buf + r * kRowS + 16 * c16);
      }
#pragma unroll
      for (int j = 0; j < kItems; ++j) {
        const int i = tid + j * kThreads;
        const int r = dsplit ? i >> 4 : i >> 5, c16 = dsplit ? 16 * sp + (i & 15) : i & 31;
        if (r >= vrows) continue;
        const uint32_t ws[4] = {w[j].x, w[j].y, w[j].z, w[j].w};
        uint32_t h[8];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          h[2 * k] = e4m3x2_to_f16x2(ws[k] & 0xFFFFu);
          h[2 * k + 1] = e4m3x2_to_f16x2(ws[k] >> 16);
        }
        uint4* dst = reinterpret_cast<uint4*>(sVh + r * kVStr + 16 * c16);
        dst[0] = make_uint4(h[0], h[1], h[2], h[3]);
        dst[1] = make_uint4(h[4], h[5], h[6], h[7]);
      }
    };
    // ---- tile O += P V of chunk c (P columns [64 (c & 1), +64) of the tile; FP16
    // operands, FP32 accumulation): warp w owns latent dims [32 w, +32)
    auto pv = [&](int c) {
      if (dsplit && (warp >> 3) != sp) return;  // (the other CTA's latent dims)
      const int valid = valid_of(c);
      // ldmatrix.trans rows: lane l -> key 16 ks + 8 ((l >> 3) & 1) + (l & 7), dims
      // 32 w + 8 (l >> 4) (+ 16 for the second x4)
      const __half* vb = sVh + (((lane >> 3) & 1) * 8 + (lane & 7)) * kVStr + 32 * warp +
                         (lane >> 4) * 8;
      const __half* pb = sP + (c & 1) * kChunk;
#pragma unroll
      for (int ks = 0; ks < kChunk / 16; ++ks) {
        if (16 * ks >= valid) break;  // (empty key steps of a partial chunk)
        uint32_t b[8];
        ldsm_x4_t(b[0], b[1], b[2], b[3], vb + 16 * ks * kVStr);
        ldsm_x4_t(b[4], b[5], b[6], b[7], vb + 16 * ks * kVStr + 16);
#pragma unroll
        for (int m2 = 0; m2 < MT; ++m2) {
          uint32_t a0, a1, a2, a3;
          ldsm_x4(a0, a1, a2, a3, pb + (16 * m2 + (lane & 15)) * kPStr + 16 * ks + (lane >> 4) * 8);
          // x4.trans matrices: (k 0-7, n 0-7), (k 8-15, n 0-7), (k 0-7, n 8-15), (k 8-15, n 8-15)
          mma_f16(tacc[4 * m2 + 0], a0, a1, a2, a3, b[0], b[1]);
          mma_f16(tacc[4 * m2 + 1], a0, a1, a2, a3, b[2], b[3]);
          mma_f16(tacc[4 * m2 + 2], a0, a1, a2, a3, b[4], b[5]);
          mma_f16(tacc[4 * m2 + 3], a0, a1, a2, a3, b[6], b[7]);
        }
      }
    };
    if (nchunks > 0) issue(0);
    if (nchunks > 1) issue(1);
    for (int c = 0; c < nchunks; c += 2) {
      const int nh = min(2, nchunks - c);  // chunks of this tile
      cp_wait<0>();
      __syncthreads();
      if (dbg && tid == 0 && c == 0) dbg[10] = clock64();
      // ---- S = Q K^T of the tile's chunks (FP8, FP32 accumulation, 3 chains; fragments
      // loaded 2 steps ahead) into S columns [64 h, +64); HG = 32: warp (m-tile w >> 3,
      // n-tile w & 7); HG = 16: (n-tile w & 7, k-half w >> 3)
      for (int h = 0; h < nh; ++h) {
        const uint8_t* buf = sKV + h * kChunk * kRowS;
        const int valid = valid_of(c + h);
        const int nt = warp & 7;
        const int mt = HG == 32 ? (warp >> 3) : 0;
        const int kh = HG == 32 ? 0 : (warp >> 3);
        constexpr int KS = HG == 32 ? kRowB / 32 : kRowB / 64;  // m16n8k32 steps
        if (8 * nt < valid) {
        float sacc[3][4] = {};
        const uint8_t* qa = sQ + (mt * 16 + (lane & 15)) * kRowS + (lane >> 4) * 16 + kh * KS * 32;
        const uint8_t* kb =
            buf + (8 * nt + (lane & 7)) * kRowS + ((lane >> 3) & 1) * 16 + kh * KS * 32;
        uint32_t fa[3][4], fb[3][2];
        ldsm_x4(fa[0][0], fa[0][1], fa[0][2], fa[0][3], qa);
        ldsm_x2(fb[0][0], fb[0][1], kb);
        ldsm_x4(fa[1][0], fa[1][1], fa[1][2], fa[1][3], qa + 32);
        ldsm_x2(fb[1][0], fb[1][1], kb + 32);
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
          if (ks + 2 < KS) {
            ldsm_x4(fa[(ks + 2) % 3][0], fa[(ks + 2) % 3][1], fa[(ks + 2) % 3][2],
                    fa[(ks + 2) % 3][3], qa + 32 * (ks + 2));
            ldsm_x2(fb[(ks + 2) % 3][0], fb[(ks + 2) % 3][1], kb + 32 * (ks + 2));
          }
          mma_fp8(sacc[ks % 3], fa[ks % 3][0], fa[ks % 3][1], fa[ks % 3][2], fa[ks % 3][3],
                  fb[ks % 3][0], fb[ks % 3][1]);
        }
        float* sS = sSall + (mt + kh) * 16 * kSStr + h * kChunk;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int r = 8 * nt + 2 * q4 + (i & 1);
          sS[(g8 + 8 * (i >> 1)) * kSStr + r] = (sacc[0][i] + sacc[1][i]) + sacc[2][i];
        }
        }  // (8 nt < valid)
      }
      cast_v(c);  // (chunk c's V while the softmax waits for every warp's S)
      __syncthreads();
      if (dbg && tid == 0 && c == 0) dbg[11] = clock64();
      // ---- softmax of the tile's 128 keys (base 2), one running max step of the piece
      // (trtllm-gen's KV CTA): m = max(m_piece, tile max), P = E4M3(448 x 2^(s - m)),
      // the piece's sum from the unquantized P (l_p = l_p 2^(m_old - m) + sum, P V
      // rescaled alike, sAp); at the piece's last tile the row's running max M and sum
      // L take the piece in (alpha = 2^(M_old - M), beta = 2^(m_p - M)). Warp w owns
      // heads w * HG / 16 .., lane l keys l, 32 + l, 64 + l, 96 + l
      const int jt = P ? (c >> 1) % kp : 0;  // tile of the piece
      const bool piece_end = !P || jt == kp - 1 || c + 2 >= nchunks;
#pragma unroll
      for (int hh = 0; hh < HG / 16; ++hh) {
        const int h = warp * (HG / 16) + hh;
        const int r0 = (h >> 4) * 16 + (h & 15);  // row of the [2][16] / [32] S tiles
        float x[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          x[k] = sSall[r0 * kSStr + 32 * k + lane];
          if (HG == 16) x[k] += sSall[(16 + h) * kSStr + 32 * k + lane];  // the second k-half
          x[k] *= p.s2scale;
          if ((k >> 1) >= nh || slot_of(c + (k >> 1), 32 * (k & 1) + lane) < 0) x[k] = -INFINITY;
        }
        float mx = fmaxf(fmaxf(x[0], x[1]), fmaxf(x[2], x[3]));
#pragma unroll
        for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        const float mpold = (!P || jt == 0) ? -INFINITY : sMp[h];
        const float mp = fmaxf(mpold, mx);
        const float base = mp == -INFINITY ? 0.f : mp;
        float sum = 0.f;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const float e = exp2f(x[k] - base);
          sum += e;  // (l from the unquantized P, as trtllm-gen)
          // P = E4M3(448 e), then exactly to FP16 for the FP16 MMA
          const __nv_fp8_storage_t q8 = __nv_cvt_float_to_fp8(kPScale * e, __NV_SATFINITE, __NV_E4M3);
          sP[h * kPStr + 32 * k + lane] = __half(__nv_cvt_fp8_to_halfraw(q8, __NV_E4M3));
        }
#pragma unroll
        for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
        __syncwarp();
        if (lane == 0) {
          const float ap = mpold == -INFINITY ? 0.f : exp2f(mpold - mp);
          const float lp = ((!P || jt == 0) ? 0.f : sLp[h] * ap) + sum;
          if (P) {
            sAp[h] = ap;
            sMp[h] = mp;
            sLp[h] = lp;
          }
          if (piece_end) piece_fold(h, mp, lp, sA, sB);
        }
      }
      __syncthreads();
      if (dbg && tid == 0 && c == 0) dbg[12] = clock64();
      // the piece's P V: zero at its first tile, else rescaled to the new running max
      if (!P || jt == 0) {
#pragma unroll
        for (int j = 0; j < 4 * MT; ++j) tacc[j][0] = tacc[j][1] = tacc[j][2] = tacc[j][3] = 0.f;
      } else {
#pragma unroll
        for (int m2 = 0; m2 < MT; ++m2) {
          const float r0 = sAp[16 * m2 + g8], r1 = sAp[16 * m2 + g8 + 8];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            tacc[4 * m2 + j][0] *= r0;
            tacc[4 * m2 + j][1] *= r0;
            tacc[4 * m2 + j][2] *= r1;
            tacc[4 * m2 + j][3] *= r1;
          }
        }
      }
      pv(c);
      __syncthreads();
      if (c + 2 < nchunks) issue(c + 2);
      if (nh > 1) {
        cast_v(c + 1);
        __syncthreads();
        pv(c + 1);
        __syncthreads();
        if (c + 3 < nchunks) issue(c + 3);
      }
      // ---- at the piece's last tile: its partial O, unnormalized and rounded to BF16
      // (trtllm-gen's multi-CTA partial; kv_ctas == 1: no split, no rounding), joins the
      // row's FP32 O at its running max
      if (piece_end) acc_fold(sA, sB);
      if (dbg && tid == 0 && c == 0) dbg[13] = clock64();
    }
  };
  // ---- the one-pass form of S = 2 solo rows (dsplit, kp == 1, <= 3 chunks); HG = 16
  auto one_pass = [&]() {
    constexpr int kS2Str = kChunk + 4;  // chunk 2's S [32][kS2Str] floats
    constexpr int kP1Str = kChunk + 8;  // tile 1's P [16][kP1Str] halves
    uint8_t* const xb = reinterpret_cast<uint8_t*>(sVh);  // chunk 2's KV, S, tile 1's P
    float* const sS2 = reinterpret_cast<float*>(xb + kChunk * kRowS);
    __half* const sP1 = reinterpret_cast<__half*>(xb + kChunk * kRowS + 32 * kS2Str * 4);
    static_assert(kChunk * kRowS + 32 * kS2Str * 4 + 16 * kP1Str * 2 <= kChunk * kVStr * 2,
                  "one-pass chunk 2 in the V buffer's space");
    auto bufc = [&](int c) { return c < 2 ? sKV + c * kChunk * kRowS : xb; };
    auto validc = [&](int c) { return min(kChunk, n - c * kChunk); };
    for (int c = 0; c < nchunks; ++c) {
      uint8_t* buf = bufc(c);
      for (int i = tid; i < kChunk * (kRowB / 16); i += kThreads) {
        const int r = i / (kRowB / 16), pc = i % (kRowB / 16);
        const int slot = sIdx[c * kChunk + r];
        if (slot >= 0) {
          cp_async16(buf + r * kRowS + 16 * pc, p.kv + (int64_t)slot * kRowB + 16 * pc);
        } else {
          *reinterpret_cast<uint4*>(buf + r * kRowS + 16 * pc) = make_uint4(0, 0, 0, 0);
        }
      }
    }
    cp_commit();
    cp_wait<0>();
    __syncthreads();
    if (dbg && tid == 0) dbg[10] = clock64();
    // ---- S = Q K^T of every chunk: warp (n-tile w & 7, k-half w >> 3), as the loop
    {
      const int nt = warp & 7, kh = warp >> 3;
      constexpr int KS = kRowB / 64;
      for (int h = 0; h < nchunks; ++h) {
        if (8 * nt >= validc(h)) continue;
        const uint8_t* buf = bufc(h);
        float sacc[3][4] = {};
        const uint8_t* qa = sQ + (lane & 15) * kRowS + (lane >> 4) * 16 + kh * KS * 32;
        const uint8_t* kb = buf + (8 * nt + (lane & 7)) * kRowS + ((lane >> 3) & 1) * 16 + kh * KS * 32;
        uint32_t fa[3][4], fb[3][2];
        ldsm_x4(fa[0][0], fa[0][1], fa[0][2], fa[0][3], qa);
        ldsm_x2(fb[0][0], fb[0][1], kb);
        ldsm_x4(fa[1][0], fa[1][1], fa[1][2], fa[1][3], qa + 32);
        ldsm_x2(fb[1][0], fb[1][1], kb + 32);
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
          if (ks + 2 < KS) {
            ldsm_x4(fa[(ks + 2) % 3][0], fa[(ks + 2) % 3][1], fa[(ks + 2) % 3][2],
                    fa[(ks + 2) % 3][3], qa + 32 * (ks + 2));
            ldsm_x2(fb[(ks + 2) % 3][0], fb[(ks + 2) % 3][1], kb + 32 * (ks + 2));
          }
          mma_fp8(sacc[ks % 3], fa[ks % 3][0], fa[ks % 3][1], fa[ks % 3][2], fa[ks % 3][3],
                  fb[ks % 3][0], fb[ks % 3][1]);
        }
        float* sS = h < 2 ? sSall + kh * 16 * kSStr + h * kChunk : sS2 + kh * 16 * kS2Str;
        const int str = h < 2 ? kSStr : kS2Str;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int r = 8 * nt + 2 * q4 + (i & 1);
          sS[(g8 + 8 * (i >> 1)) * str + r] = (sacc[0][i] + sacc[1][i]) + sacc[2][i];
        }
      }
    }
    __syncthreads();
    if (dbg && tid == 0) dbg[11] = clock64();
    // ---- softmax of tile 0 (chunks 0, 1) and tile 1 (chunk 2), each its own piece:
    // warp w head w, lane l keys l + 32 k (the loop's per-tile arithmetic)
    {
      const int h = warp;
      const int nh0 = min(2, nchunks);
      float x[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        x[k] = sSall[h * kSStr + 32 * k + lane];
        x[k] += sSall[(16 + h) * kSStr + 32 * k + lane];
        x[k] *= p.s2scale;
        if ((k >> 1) >= nh0 || sIdx[(k >> 1) * kChunk + 32 * (k & 1) + lane] < 0) x[k] = -INFINITY;
      }
      float mx = fmaxf(fmaxf(x[0], x[1]), fmaxf(x[2], x[3]));
#pragma unroll
      for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
      const float mp = fmaxf(-INFINITY, mx);
      const float base = mp == -INFINITY ? 0.f : mp;
      float sum = 0.f;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float e = exp2f(x[k] - base);
        sum += e;
        const __nv_fp8_storage_t q8 = __nv_cvt_float_to_fp8(kPScale * e, __NV_SATFINITE, __NV_E4M3);
        sP[h * kPStr + 32 * k + lane] = __half(__nv_cvt_fp8_to_halfraw(q8, __NV_E4M3));
      }
#pragma unroll
      for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
      float mp1 = -INFINITY, sum1 = 0.f;
      if (nchunks == 3) {
        float y[2];
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          y[k] = sS2[h * kS2Str + 32 * k + lane];
          y[k] += sS2[(16 + h) * kS2Str + 32 * k + lane];
          y[k] *= p.s2scale;
          if (sIdx[2 * kChunk + 32 * k + lane] < 0) y[k] = -INFINITY;
        }
        float my = fmaxf(y[0], y[1]);
#pragma unroll
        for (int o = 16; o; o >>= 1) my = fmaxf(my, __shfl_xor_sync(0xffffffffu, my, o));
        // kp == 2: tile 1 continues tile 0's piece (running max max(m_0, tile max))
        mp1 = fmaxf(kp == 2 ? mp : -INFINITY, my);
        const float base1 = mp1 == -INFINITY ? 0.f : mp1;
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          const float e = exp2f(y[k] - base1);
          sum1 += e;
          const __nv_fp8_storage_t q8 = __nv_cvt_float_to_fp8(kPScale * e, __NV_SATFINITE, __NV_E4M3);
          sP1[h * kP1Str + 32 * k + lane] = __half(__nv_cvt_fp8_to_halfraw(q8, __NV_E4M3));
        }
#pragma unroll
        for (int o = 16; o; o >>= 1) sum1 += __shfl_xor_sync(0xffffffffu, sum1, o);
      }
      __syncwarp();
      if (lane == 0) {
        if (kp == 1) {
          piece_fold(h, mp, 0.f + sum, sA, sB);
          if (nchunks == 3) piece_fold(h, mp1, 0.f + sum1, sMp, sLp);
        } else {  // one piece of two tiles: the loop's piece state at its second tile
          const float lp0 = 0.f + sum;
          if (nchunks == 3) {
            const float ap = mp == -INFINITY ? 0.f : exp2f(mp - mp1);
            sAp[h] = ap;
            // (the loop's l_p = select(0, l_p x a_p) + sum: product and sum rounded apart)
            piece_fold(h, mp1, __fadd_rn(__fmul_rn(lp0, ap), sum1), sA, sB);
          } else {
            piece_fold(h, mp, lp0, sA, sB);
          }
        }
      }
    }
    // ---- V (this CTA's latent dims [256 sp, +256)) -> FP16 in place of each KV row's
    // first 512 bytes: a half-warp per row, every lane's 16 bytes read before any store
    for (int i = 0; i < 3 * kChunk / 32; ++i) {
      const int rr = 2 * warp + (lane >> 4) + 32 * i, c = rr / kChunk, r = rr % kChunk;
      const int j = lane & 15;
      const bool on = c < nchunks && r < ((validc(c) + 15) & ~15);
      uint8_t* row = on ? bufc(c) + r * kRowS : nullptr;
      uint4 w = make_uint4(0, 0, 0, 0);
      if (on) w = *reinterpret_cast<const uint4*>(row + 256 * sp + 16 * j);
      __syncwarp();
      if (on) {
        const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
        uint32_t hh[8];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          hh[2 * k] = e4m3x2_to_f16x2(ws[k] & 0xFFFFu);
          hh[2 * k + 1] = e4m3x2_to_f16x2(ws[k] >> 16);
        }
        uint4* dst = reinterpret_cast<uint4*>(row + 32 * j);
        dst[0] = make_uint4(hh[0], hh[1], hh[2], hh[3]);
        dst[1] = make_uint4(hh[4], hh[5], hh[6], hh[7]);
      }
    }
    __syncthreads();
    if (dbg && tid == 0) dbg[12] = clock64();
    // ---- P V: tile 0 then tile 1, each folded into the row as the loop does
    auto pv1 = [&](int c, const __half* pb, int pstr) {
      if ((warp >> 3) != sp) return;  // (the other CTA's latent dims)
      const int valid = validc(c);
      const __half* vb = reinterpret_cast<const __half*>(bufc(c)) +
                         (((lane >> 3) & 1) * 8 + (lane & 7)) * (kRowS / 2) + 32 * (warp & 7) +
                         (lane >> 4) * 8;
#pragma unroll
      for (int ks = 0; ks < kChunk / 16; ++ks) {
        if (16 * ks >= valid) break;
        uint32_t b[8];
        ldsm_x4_t(b[0], b[1], b[2], b[3], vb + 16 * ks * (kRowS / 2));
        ldsm_x4_t(b[4], b[5], b[6], b[7], vb + 16 * ks * (kRowS / 2) + 16);
        uint32_t a0, a1, a2, a3;
        ldsm_x4(a0, a1, a2, a3, pb + (lane & 15) * pstr + 16 * ks + (lane >> 4) * 8);
        mma_f16(tacc[0], a0, a1, a2, a3, b[0], b[1]);
        mma_f16(tacc[1], a0, a1, a2, a3, b[2], b[3]);
        mma_f16(tacc[2], a0, a1, a2, a3, b[4], b[5]);
        mma_f16(tacc[3], a0, a1, a2, a3, b[6], b[7]);
      }
    };
#pragma unroll
    for (int j = 0; j < 4; ++j) tacc[j][0] = tacc[j][1] = tacc[j][2] = tacc[j][3] = 0.f;
    pv1(0, sP, kPStr);
    if (nchunks > 1) pv1(1, sP + kChunk, kPStr);
    if (kp == 2 && nchunks == 3) {  // tile 0's P V rescaled to the piece's max, tile 1 added
      const float r0 = sAp[g8], r1 = sAp[g8 + 8];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        tacc[j][0] *= r0;
        tacc[j][1] *= r0;
        tacc[j][2] *= r1;
        tacc[j][3] *= r1;
      }
      pv1(2, sP1, kP1Str);
      acc_fold(sA, sB);
    } else {
      acc_fold(sA, sB);
    }
    if (kp == 1 && nchunks == 3) {
#pragma unroll
      for (int j = 0; j < 4; ++j) tacc[j][0] = tacc[j][1] = tacc[j][2] = tacc[j][3] = 0.f;
      pv1(2, sP1, kP1Str);
      acc_fold(sMp, sLp);
    }
    if (dbg && tid == 0) dbg[13] = clock64();
  };
  // S = 2 solo rows of at most 3 chunks (2 tiles; each its own piece, kp == 1, or one
  // piece, kp == 2: trtllm-gen's single KV CTA): one pass instead of the per-tile loop.
  // The chunks are gathered at once (chunk 2 into the V buffer's space), QK of all of them, the two tiles' softmax
  // (each piece's own running max and sum) in one step, V cast to FP16 in place of the
  // KV rows (this CTA's 256 dims; QK is done with them), then P V of tile 0 and of tile
  // 1 back to back and the two pieces folded into the row in the per-tile loop's order
  // with its arithmetic: bitwise the per-tile loop's partial and LSE.
  if constexpr (HG == 16) {
    if (dsplit && kp <= 2 && nchunks >= 1 && nchunks <= 3) {
      one_pass();
    } else if (kp > 1) {
      mainloop(std::true_type{});
    } else {
      mainloop(std::false_type{});
    }
  } else {
    if (kp > 1) {
      mainloop(std::true_type{});
    } else {
      mainloop(std::false_type{});
    }
  }
  if (dbg && tid == 0) dbg[3] = now_ns();
  constexpr int kGroupHeads = HG;
  const int H = p.hpr;
  if (S > 1 && !solo) {
    // ---- the S splits of (t, grp) are one cluster: each leaves its FP32 O (the sum
    // of its tiles' BF16 partials at its running max; shared [32, 512], the KV buffers'
    // space) and (max, sum); after one cluster barrier split s merges latent dims
    // [512 s / S, +512 / S) of all of them through distributed shared memory (FP32
    // weights 2^(m_s - M) / L, one BF16 rounding) and pushes them
    float* sOf = reinterpret_cast<float*>(sKV);
    __shared__ float s_w[8][kGroupHeads], s_lse[kGroupHeads];
#pragma unroll
    for (int m2 = 0; m2 < MT; ++m2) {
      const int h = 16 * m2 + g8;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int col = 32 * warp + 8 * j + 2 * q4;
        *reinterpret_cast<float2*>(sOf + h * 512 + col) =
            make_float2(acc[4 * m2 + j][0], acc[4 * m2 + j][1]);
        *reinterpret_cast<float2*>(sOf + (h + 8) * 512 + col) =
            make_float2(acc[4 * m2 + j][2], acc[4 * m2 + j][3]);
      }
    }
    if (nchunks == 0 && tid < kGroupHeads) sM[tid] = -INFINITY;  // (an empty split)
    if (dbg && tid == 0) dbg[6] = now_ns();
    cluster_sync();
    if (dbg && tid == 0) dbg[7] = now_ns();
    if (tid < kGroupHeads) {
      float m[8], l[8];
#pragma unroll
      for (int s2 = 0; s2 < 8; ++s2) {
        m[s2] = s2 < S ? ld_dsmem(sM + tid, s2) : -INFINITY;
        l[s2] = s2 < S ? ld_dsmem(sL + tid, s2) : 0.f;
      }
      float M = -INFINITY;
#pragma unroll
      for (int s2 = 0; s2 < 8; ++s2) M = fmaxf(M, m[s2]);
      float L = 0.f;
#pragma unroll
      for (int s2 = 0; s2 < 8; ++s2) {
        m[s2] = m[s2] == -INFINITY ? 0.f : exp2f(m[s2] - M);
        L += l[s2] * m[s2];
      }
      const bool live = n > 0 && L > 0.f && M != -INFINITY;
      const float r = live ? 1.f / L : 0.f;
#pragma unroll
      for (int s2 = 0; s2 < 8; ++s2) s_w[s2][tid] = m[s2] * r;
      float lse = kEmptyLse;
      if (live) {  // DcpValue's rule
        lse = M + log2f(L);
        if (!(lse > kEmptyLse) || !(lse < -kEmptyLse)) lse = kEmptyLse;  // NaN / inf
      }
      s_lse[tid] = lse;
    }
    __syncthreads();
    if (dbg && tid == 0) dbg[8] = now_ns();
    const int span = 512 / S, per = span / 8;  // 8-float groups per head
    for (int i = tid; i < kGroupHeads * per; i += kThreads) {
      const int h = i / per, d = sp * span + 8 * (i % per);
      float4 x[8][2];
#pragma unroll
      for (int s2 = 0; s2 < 8; ++s2) {
        if (s2 < S) {
          x[s2][0] = ld_dsmem_v4(sOf + h * 512 + d, s2);
          x[s2][1] = ld_dsmem_v4(sOf + h * 512 + d + 4, s2);
        }
      }
      float a[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int s2 = 0; s2 < 8; ++s2) {
        if (s2 < S) {
          const float w = s_w[s2][h];
          a[0] = fmaf(x[s2][0].x, w, a[0]);
          a[1] = fmaf(x[s2][0].y, w, a[1]);
          a[2] = fmaf(x[s2][0].z, w, a[2]);
          a[3] = fmaf(x[s2][0].w, w, a[3]);
          a[4] = fmaf(x[s2][1].x, w, a[4]);
          a[5] = fmaf(x[s2][1].y, w, a[5]);
          a[6] = fmaf(x[s2][1].z, w, a[6]);
          a[7] = fmaf(x[s2][1].w, w, a[7]);
        }
      }
      const int gh = h0 + h;
      const int64_t slot = (((int64_t)p.parity * p.W + p.rank) * p.box_cap + t) * H + gh % H;
      uint32_t* dst = reinterpret_cast<uint32_t*>(p.part_peers[gh / H]) + slot * 256 + d / 2;
      st_v4(dst, make_uint4(no_neg_zero(bits(__floats2bfloat162_rn(a[0], a[1]))),
                            no_neg_zero(bits(__floats2bfloat162_rn(a[2], a[3]))),
                            no_neg_zero(bits(__floats2bfloat162_rn(a[4], a[5]))),
                            no_neg_zero(bits(__floats2bfloat162_rn(a[6], a[7])))));
    }
    if (dbg && tid == 0) dbg[9] = now_ns();
    if (sp == 0 && tid < kGroupHeads) {
      const int gh = h0 + tid;
      const int64_t slot = (((int64_t)p.parity * p.W + p.rank) * p.box_cap + t) * H + gh % H;
      st_u32(reinterpret_cast<uint32_t*>(p.lse_peers[gh / H]) + lse_slot(p, t, gh), __float_as_uint(s_lse[tid]));
    }
    cluster_sync();  // (no CTA exits while a peer may still read its shared memory)
    if (dbg && tid == 0) dbg[4] = now_ns();
    return;
  }

  // ---- epilogue (one split): normalized partial (bf16) + base-2 LSE -> head owners
  __syncthreads();
  __nv_bfloat16* sO = reinterpret_cast<__nv_bfloat16*>(sKV);  // [32, kOStr]
#pragma unroll
  for (int m2 = 0; m2 < MT; ++m2) {
    const float l0 = sL[16 * m2 + g8], l1 = sL[16 * m2 + g8 + 8];
    const float r0 = (n > 0 && l0 > 0.f) ? 1.f / l0 : 0.f;  // (acc: x vscale / 448)
    const float r1 = (n > 0 && l1 > 0.f) ? 1.f / l1 : 0.f;
    __nv_bfloat16* o = sO + 16 * m2 * kOStr;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int col = 32 * warp + 8 * j + 2 * q4;
      *reinterpret_cast<__nv_bfloat162*>(o + g8 * kOStr + col) =
          __floats2bfloat162_rn(acc[4 * m2 + j][0] * r0, acc[4 * m2 + j][1] * r0);
      *reinterpret_cast<__nv_bfloat162*>(o + (g8 + 8) * kOStr + col) =
          __floats2bfloat162_rn(acc[4 * m2 + j][2] * r1, acc[4 * m2 + j][3] * r1);
    }
  }
  __syncthreads();
  for (int i = tid; i < kGroupHeads * (dsplit ? 32 : 64); i += kThreads) {
    const int h = dsplit ? i >> 5 : i >> 6, ch = dsplit ? 32 * sp + (i & 31) : i & 63;
    const int gh = h0 + h;
    const uint4 v = *reinterpret_cast<const uint4*>(sO + h * kOStr + 8 * ch);
    const int64_t slot = (((int64_t)p.parity * p.W + p.rank) * p.box_cap + t) * H + gh % H;
    uint32_t* dst = reinterpret_cast<uint32_t*>(p.part_peers[gh / H]) + slot * 256 + 4 * ch;
    st_v4(dst, make_uint4(no_neg_zero(v.x), no_neg_zero(v.y), no_neg_zero(v.z),
                          no_neg_zero(v.w)));
  }
  if (tid < kGroupHeads && sp == 0) {  // (S = 2 solo rows: split 0's)
    const int gh = h0 + tid;
    const float l = sL[tid], m = sM[tid];
    float lse = kEmptyLse;
    if (n > 0 && l > 0.f && m != -INFINITY) {  // DcpValue's rule
      lse = m + log2f(l);
      if (!(lse > kEmptyLse) || !(lse < -kEmptyLse)) lse = kEmptyLse;  // NaN / inf
    }
    const int64_t slot = (((int64_t)p.parity * p.W + p.rank) * p.box_cap + t) * H + gh % H;
    st_u32(reinterpret_cast<uint32_t*>(p.lse_peers[gh / H]) + lse_slot(p, t, gh), __float_as_uint(lse));
  }
  if (dbg && tid == 0) dbg[4] = now_ns();
}

// ------------------------------------------------------------------ host

void launch(const Params& p, int heads, int64_t heads_per_cta, int64_t pdl);

void run(torch::Tensor q, torch::Tensor sel, torch::Tensor cnt, torch::Tensor kv,
         double s2scale, double vscale, torch::Tensor part_peers, torch::Tensor lse_peers,
         int64_t box_cap, int64_t hpr, int64_t parity, int64_t rank, int64_t T, int64_t lse_wide,
         int64_t splits, int64_t split_min, int64_t kv_ctas, int64_t heads_per_cta, torch::Tensor err,
         int64_t spin_ns, int64_t pdl,
         c10::optional<torch::Tensor> dbg) {
  if (T == 0) return;
  const int W = (int)part_peers.numel();
  const int heads = (int)(W * hpr);
  TORCH_CHECK(heads_per_cta == 16 || heads_per_cta == 32, "dsa_core: 16 or 32 heads per CTA");
  TORCH_CHECK(heads % heads_per_cta == 0, "dsa_core: gathered heads not a multiple of the CTA's");
  TORCH_CHECK(q.dim() == 3 && q.size(-1) == kRowB && q.element_size() == 1 && q.size(1) >= heads,
              "dsa_core: q mailbox [rows, q_heads >= W * hpr, 576] bytes");
  TORCH_CHECK(T <= q.size(0) && T <= box_cap && T <= sel.size(0) && T <= cnt.numel());
  TORCH_CHECK(sel.dtype() == torch::kInt32 && sel.stride(1) == 1 && cnt.dtype() == torch::kInt32);
  TORCH_CHECK(kv.element_size() == 1 && kv.size(-1) == kRowB);
  const c10::cuda::CUDAGuard guard(q.device());
  Params p{};
  p.q = reinterpret_cast<const uint8_t*>(q.data_ptr());
  p.q_row_stride = q.stride(0);
  p.sel = sel.data_ptr<int>();
  p.sel_stride = sel.stride(0);
  p.width = (int)std::min<int64_t>(sel.size(1), kMaxK);
  p.cnt = cnt.data_ptr<int>();
  p.kv = reinterpret_cast<const uint8_t*>(kv.data_ptr());
  p.s2scale = (float)s2scale;
  p.vscale = (float)vscale;
  p.part_peers = reinterpret_cast<const uint64_t*>(part_peers.data_ptr<int64_t>());
  p.lse_peers = reinterpret_cast<const uint64_t*>(lse_peers.data_ptr<int64_t>());
  p.box_cap = (int)box_cap;
  p.lse_wide = (int)lse_wide;
  p.split_min = (int)split_min;
  TORCH_CHECK(kv_ctas >= 1 && kv_ctas <= kMaxK / kTile, "dsa_core: kv_ctas in [1, 16]");
  p.kv_ctas = (int)kv_ctas;
  p.hpr = (int)hpr;
  p.parity = (int)parity;
  p.rank = (int)rank;
  p.W = W;
  p.T = (int)T;
  p.splits = (int)splits;
  TORCH_CHECK(splits == 1 || splits == 2 || splits == 4 || splits == 8,
              "dsa_core: splits 1, 2, 4 or 8 (one cluster per (row, group))");
  p.err = err.data_ptr<int>();
  p.spin_ns = spin_ns;
  p.dbg = dbg.has_value() ? reinterpret_cast<long long*>(dbg->data_ptr<int64_t>()) : nullptr;
  launch(p, heads, heads_per_cta, pdl);
}

// The core with the #63 merge in it (``mic``): ``sel`` / ``cnt`` are optional outputs
// (this rank's slots and count, as the select's rows_out / valid_out; ``sel`` also sets
// the slot width trtllm_kv_ctas saw); the rest are the merge's mailbox and rescue state.
void run_mic(torch::Tensor q, torch::Tensor sel, torch::Tensor cnt, torch::Tensor kv,
             double s2scale, double vscale, torch::Tensor part_peers, torch::Tensor lse_peers,
             int64_t box_cap, int64_t hpr, int64_t parity, int64_t rank, int64_t T,
             int64_t lse_wide, int64_t splits, int64_t split_min, int64_t kv_ctas,
             int64_t heads_per_cta, torch::Tensor err, int64_t spin_ns, int64_t pdl,
             c10::optional<torch::Tensor> dbg, torch::Tensor box, torch::Tensor ordinals,
             torch::Tensor readers, int64_t topk, int64_t nloc, torch::Tensor req_id,
             torch::Tensor block_table, int64_t block_size, int64_t stride_rows,
             torch::Tensor logits, torch::Tensor row_ends, torch::Tensor peer_recv,
             torch::Tensor peer_flags, torch::Tensor seq2, int64_t write_rows) {
  if (T == 0) return;
  const int W = (int)part_peers.numel();
  const int heads = (int)(W * hpr);
  TORCH_CHECK(heads_per_cta == 16 || heads_per_cta == 32, "dsa_core: 16 or 32 heads per CTA");
  TORCH_CHECK(heads % heads_per_cta == 0, "dsa_core: gathered heads not a multiple of the CTA's");
  TORCH_CHECK(q.dim() == 3 && q.size(-1) == kRowB && q.element_size() == 1 && q.size(1) >= heads,
              "dsa_core: q mailbox [rows, q_heads >= W * hpr, 576] bytes");
  TORCH_CHECK(T <= q.size(0) && T <= box_cap);
  TORCH_CHECK(kv.element_size() == 1 && kv.size(-1) == kRowB);
  TORCH_CHECK(W <= 16 && nloc % 32 == 0 && nloc > 0 && topk % 32 == 0 && topk <= kMaxK,
              "dsa_core mic: W <= 16, N and K multiples of 32, K <= 2048");
  const int box_words = (int)(((nloc + 2 * (nloc / 32)) + 3) / 4 * 4);
  const int max_rows = (int)ordinals.numel();
  TORCH_CHECK(box.dtype() == torch::kInt32 && box.numel() == 2LL * max_rows * W * box_words,
              "dsa_core mic: mailbox [2, max_rows, W, box_words] int32");
  TORCH_CHECK(T <= max_rows && 2 * T <= readers.numel() && seq2.numel() >= max_rows);
  // stage (W x box_words) + the push's B' row in the KV buffers, the block-table cache
  // and the radix bins in the FP16 V buffer
  TORCH_CHECK((int64_t)W * box_words + topk + topk / 32 <= 2 * kChunk * kRowS / 4,
              "dsa_core mic: round-1 rows do not fit the KV buffers");
  TORCH_CHECK(block_table.dtype() == torch::kInt32 && block_table.stride(1) == 1);
  TORCH_CHECK(req_id.dtype() == torch::kInt32 && req_id.numel() >= T);
  TORCH_CHECK(logits.dtype() == torch::kBFloat16 && logits.stride(1) == 1 && logits.size(0) >= T);
  TORCH_CHECK(row_ends.dtype() == torch::kInt32 && row_ends.numel() >= T);
  TORCH_CHECK(peer_recv.numel() == W && peer_flags.numel() == W);
  const int num_slots = (int)((topk - 1) / nloc);
  const c10::cuda::CUDAGuard guard(q.device());
  Params p{};
  p.q = reinterpret_cast<const uint8_t*>(q.data_ptr());
  p.q_row_stride = q.stride(0);
  p.sel = nullptr;
  p.width = kMaxK;
  p.cnt = nullptr;
  p.kv = reinterpret_cast<const uint8_t*>(kv.data_ptr());
  p.s2scale = (float)s2scale;
  p.vscale = (float)vscale;
  p.part_peers = reinterpret_cast<const uint64_t*>(part_peers.data_ptr<int64_t>());
  p.lse_peers = reinterpret_cast<const uint64_t*>(lse_peers.data_ptr<int64_t>());
  p.box_cap = (int)box_cap;
  p.lse_wide = (int)lse_wide;
  p.split_min = (int)split_min;
  TORCH_CHECK(kv_ctas >= 1 && kv_ctas <= kMaxK / kTile, "dsa_core: kv_ctas in [1, 16]");
  p.kv_ctas = (int)kv_ctas;
  p.hpr = (int)hpr;
  p.parity = (int)parity;
  p.rank = (int)rank;
  p.W = W;
  p.T = (int)T;
  p.splits = (int)splits;
  TORCH_CHECK(splits == 1 || splits == 2 || splits == 4 || splits == 8,
              "dsa_core: splits 1, 2, 4 or 8 (one cluster per (row, group))");
  p.err = err.data_ptr<int>();
  p.spin_ns = spin_ns;
  p.dbg = dbg.has_value() ? reinterpret_cast<long long*>(dbg->data_ptr<int64_t>()) : nullptr;
  p.mic = 1;
  p.box = reinterpret_cast<const uint32_t*>(box.data_ptr<int>());
  p.ordinals = ordinals.data_ptr<int>();
  p.readers = readers.data_ptr<int>();
  p.rcap = (int)(readers.numel() / 2);
  p.max_rows = max_rows;
  p.nloc = (int)nloc;
  p.topk = (int)topk;
  p.box_words = box_words;
  int gb = 0;
  while ((1LL << gb) <= (int64_t)W * (1LL << 17) - 1) ++gb;  // _gid_bits
  p.gid_bits = gb;
  p.req_id = req_id.data_ptr<int>();
  p.block_table = block_table.data_ptr<int>();
  p.bt_stride = block_table.stride(0);
  p.bt_width = (int)block_table.size(1);
  p.block_size = (int)block_size;
  p.stride_rows = (int)stride_rows;
  p.logits = reinterpret_cast<const uint16_t*>(logits.data_ptr());
  p.logits_stride = logits.stride(0);
  p.logits_width = (int)logits.size(1);
  p.row_ends = row_ends.data_ptr<int>();
  p.peer_recv = reinterpret_cast<const uint64_t*>(peer_recv.data_ptr<int64_t>());
  p.peer_flags = reinterpret_cast<const uint64_t*>(peer_flags.data_ptr<int64_t>());
  p.seq2 = seq2.data_ptr<int>();
  p.num_slots = num_slots;
  if (write_rows) {
    TORCH_CHECK(sel.dtype() == torch::kInt32 && sel.stride(1) == 1 && cnt.dtype() == torch::kInt32);
    p.rows_out = sel.data_ptr<int>();
    p.rows_stride = sel.stride(0);
    p.valid_out = cnt.data_ptr<int>();
  }
  launch(p, heads, heads_per_cta, pdl);
}

void launch(const Params& p, int heads, int64_t heads_per_cta, int64_t pdl) {
  const int64_t splits = p.splits;
  const int HG = (int)heads_per_cta;
  auto kernel = HG == 16 ? dsac_kernel<16> : dsac_kernel<32>;
  const int smem_bytes = HG == 16 ? Plan<16>::kSmem : Plan<32>::kSmem;
  static bool attr16 = false, attr32 = false;
  bool& attr = HG == 16 ? attr16 : attr32;
  if (!attr) {
    C10_CUDA_CHECK(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)splits, (unsigned)(heads / HG), (unsigned)p.T);
  cfg.blockDim = dim3(kThreads, 1, 1);
  cfg.dynamicSmemBytes = smem_bytes;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attrs[2];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  attrs[1].id = cudaLaunchAttributeClusterDimension;
  attrs[1].val.clusterDim.x = (unsigned)splits;
  attrs[1].val.clusterDim.y = 1;
  attrs[1].val.clusterDim.z = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = 2;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, p));
}

// Shared-memory opt-ins of every variant, outside any CUDA graph capture.
void prepare() {
  C10_CUDA_CHECK(cudaFuncSetAttribute(dsac_kernel<16>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      Plan<16>::kSmem));
  C10_CUDA_CHECK(cudaFuncSetAttribute(dsac_kernel<32>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      Plan<32>::kSmem));
}

int64_t smem_bytes(int64_t heads_per_cta) {
  return heads_per_cta == 16 ? Plan<16>::kSmem : Plan<32>::kSmem;
}

}  // namespace k31dsac

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &k31dsac::run);
  m.def("run_mic", &k31dsac::run_mic);
  m.def("smem_bytes", &k31dsac::smem_bytes);
  m.def("prepare", [] { k31dsac::prepare(); });
}
