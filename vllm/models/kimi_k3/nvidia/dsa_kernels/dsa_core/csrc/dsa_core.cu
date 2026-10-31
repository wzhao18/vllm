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
//    insert): the row's local count n and the slots of this split's 64-row chunks
//    (chunks s, s + S, ..), KV rows gathered with cp.async (2 deep), FP8 QK
//    (mma.sync e4m3, FP32 accumulation), online base-2 softmax in FP32, FP16 P x
//    FP16 (exactly converted FP8) V, FP32 accumulation.
//  * S == 1: the normalized partial O (bf16, no negative zero: 0x8000 halves are the
//    mailbox sentinel's) and its base-2 LSE go straight to the head owners'
//    DcpBox.partial / DcpBox.lse (slot ((parity * W + rank) * cap + t) * H + h).
//    Rows without local entries push zeros and the EMPTY LSE (DcpValue's rule).
//  * S > 1 (2, 4, 8): the S splits of (t, g) are one cluster. Each leaves its FP32 O
//    and (max, sum) in its shared memory; after one cluster barrier split s merges
//    latent dims [512 s / S, +512 / S) of all splits through distributed shared
//    memory (weights 2^(m_s - M) * vscale / L) and pushes them (split 0: the LSE).
//
// Every row's whole query tile is polled by every CTA of the row whatever its local
// count, so when this grid completes every rank's publish of rows < T has landed
// here, and the DcpValue after it may re-arm the mailbox (after its own PDL wait).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <algorithm>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

namespace k31dsac {

constexpr int kThreads = 512;
constexpr uint32_t kSent = 0xFFFFFFFFu;  // q mailbox armed word
constexpr int kMaxK = 2048;
constexpr int kChunk = 64;  // KV rows per chunk
constexpr int kRowB = 576;  // FP8 KV / Q row bytes
constexpr int kRowS = 592;  // padded shared row stride (conflict-free ldmatrix)
constexpr int kSStr = 68;   // S tile stride (floats)
constexpr int kPStr = 72;   // P tile stride (halves)
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
  static constexpr int kP = kS + 32 * kSStr * 4;            // [HG][kPStr] FP16
  static constexpr int kIdx = kP + HG * kPStr * 2;          // [kMaxK] slots
  static constexpr int kStat = kIdx + kMaxK * 4;            // m, l, alpha [HG]
  static constexpr int kSmem = kStat + 3 * HG * 4;
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
  int box_cap, hpr, parity, rank, W, T, splits;
  int* err;
  long long spin_ns;
  long long* dbg;         // optional [T * G * S, 16] timestamps (ns): start, q polled,
                          // slots, loop done, pushed, after the PDL wait
};

// ------------------------------------------------------------------ primitives
__device__ __forceinline__ long long now_ns() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __noinline__ void fail(int* err, int code) {
  atomicCAS(err, 0, code);
  __threadfence_system();
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
  __shared__ int s_n;
  long long* dbg = p.dbg ? p.dbg + ((int64_t)(t * G + grp) * S + sp) * 24 : nullptr;
  if (dbg && tid == 0) dbg[0] = now_ns();

  // ---- query tile [HG, 576] FP8 from the q mailbox (polled: no PDL dependency)
  {
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
  }
  if (tid < HG) {
    sM[tid] = -INFINITY;
    sL[tid] = 0.f;
  }
  if (dbg && tid == 0) dbg[1] = now_ns();
  // ---- the top-k slots / counts and the KV insert of this step are ours from here
  pdl_wait();
  // DcpValue(pushed) only stages W_UV before its own wait: let it become resident
  pdl_launch();
  if (dbg && tid == 0) dbg[5] = now_ns();
  // the count and this split's slots in one round trip (slots past the count masked)
  const int* selrow = p.sel + (int64_t)t * p.sel_stride;
  int v[kMaxK / kThreads];
#pragma unroll
  for (int b = 0; b < kMaxK / kThreads; ++b) {
    const int k = b * kThreads + tid;  // local index k -> chunk k / 64, row k % 64
    const int gk = ((k / kChunk) * S + sp) * kChunk + k % kChunk;
    v[b] = gk < p.width ? ld_cg(selrow + gk) : -1;
  }
  if (tid == 0) s_n = ld_cg(p.cnt + t);
  __syncthreads();
  const int n = min(s_n, p.width);
  const int nall = (n + kChunk - 1) / kChunk;
  const int nchunks = nall > sp ? (nall - sp + S - 1) / S : 0;  // chunks sp, sp + S, ..
  {
    const int nmine = nchunks * kChunk;
#pragma unroll
    for (int b = 0; b < kMaxK / kThreads; ++b) {
      const int k = b * kThreads + tid;
      const int gk = ((k / kChunk) * S + sp) * kChunk + k % kChunk;
      if (k < nmine) sIdx[k] = gk < n ? v[b] : -1;
    }
  }
  __syncthreads();
  if (dbg && tid == 0) {
    dbg[2] = now_ns();
    dbg[18] = clock64();
  }
  float acc[4 * MT][4];
#pragma unroll
  for (int j = 0; j < 4 * MT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;

  // chunk c -> KV buffer c & 1 (cp.async, 2 deep); rows past the count are zeroed
  // (P = 0 must not meet a NaN code)
  auto issue = [&](int c) {
    uint8_t* buf = sKV + (c & 1) * kChunk * kRowS;
    for (int i = tid; i < kChunk * (kRowB / 16); i += kThreads) {
      const int r = i / (kRowB / 16), pc = i % (kRowB / 16);
      const int slot = sIdx[c * kChunk + r];
      if (slot >= 0) {
        cp_async16(buf + r * kRowS + 16 * pc, p.kv + (int64_t)slot * kRowB + 16 * pc);
      } else {
        *reinterpret_cast<uint4*>(buf + r * kRowS + 16 * pc) = make_uint4(0, 0, 0, 0);
      }
    }
    cp_commit();
  };
  if (nchunks > 0) issue(0);
  if (nchunks > 1) issue(1);
  for (int c = 0; c < nchunks; ++c) {
    if (c + 1 < nchunks) {
      cp_wait<1>();
    } else {
      cp_wait<0>();
    }
    __syncthreads();
    if (dbg && tid == 0 && c < 2) dbg[10 + 4 * c] = clock64();
    const uint8_t* buf = sKV + (c & 1) * kChunk * kRowS;
    // ---- S = Q K^T (FP8, FP32 accumulation, 3 chains; fragments loaded 2 steps ahead);
    // HG = 32: warp (m-tile w >> 3, n-tile w & 7); HG = 16: (n-tile w & 7, k-half w >> 3)
    {
      const int nt = warp & 7;
      const int mt = HG == 32 ? (warp >> 3) : 0;
      const int kh = HG == 32 ? 0 : (warp >> 3);
      constexpr int KS = HG == 32 ? kRowB / 32 : kRowB / 64;  // m16n8k32 steps
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
      float* sS = sSall + (mt + kh) * 16 * kSStr;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int r = 8 * nt + 2 * q4 + (i & 1);
        sS[(g8 + 8 * (i >> 1)) * kSStr + r] = (sacc[0][i] + sacc[1][i]) + sacc[2][i];
      }
    }
    if (p.dbg) {  // (timing diagnostics only: QK apart from the V cast)
      __syncthreads();
      if (tid == 0 && c < 2) p.dbg[((int64_t)(t * G + grp) * S + sp) * 24 + 19 + c] = clock64();
    }
    // ---- V -> FP16 (exact) for the PV fragments: 64 rows x 512 dims, 16 B per item
    {
      constexpr int kItems = kChunk * 32 / kThreads;  // 4
      uint4 w[kItems];
#pragma unroll
      for (int j = 0; j < kItems; ++j) {
        const int i = tid + j * kThreads, r = i >> 5, c16 = i & 31;
        w[j] = *reinterpret_cast<const uint4*>(buf + r * kRowS + 16 * c16);
      }
#pragma unroll
      for (int j = 0; j < kItems; ++j) {
        const int i = tid + j * kThreads, r = i >> 5, c16 = i & 31;
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
    }
    __syncthreads();
    if (dbg && tid == 0 && c < 2) dbg[11 + 4 * c] = clock64();
    // ---- online softmax (base 2): warp w owns heads w * HG / 16 ..
#pragma unroll
    for (int hh = 0; hh < HG / 16; ++hh) {
      const int h = warp * (HG / 16) + hh;
      const int r0 = (h >> 4) * 16 + (h & 15);  // row of the [2][16] / [32] S tiles
      float v0 = sSall[r0 * kSStr + lane], v1 = sSall[r0 * kSStr + 32 + lane];
      if (HG == 16) {  // the second k-half
        v0 += sSall[(16 + h) * kSStr + lane];
        v1 += sSall[(16 + h) * kSStr + 32 + lane];
      }
      v0 *= p.s2scale;
      v1 *= p.s2scale;
      if (sIdx[c * kChunk + lane] < 0) v0 = -INFINITY;
      if (sIdx[c * kChunk + 32 + lane] < 0) v1 = -INFINITY;
      float mx = fmaxf(v0, v1);
#pragma unroll
      for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
      const float mold = sM[h];
      const float mnew = fmaxf(mold, mx);
      const float base = mnew == -INFINITY ? 0.f : mnew;
      const float p0 = exp2f(v0 - base), p1 = exp2f(v1 - base);
      float sum = p0 + p1;
#pragma unroll
      for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
      sP[h * kPStr + lane] = __float2half_rn(p0);
      sP[h * kPStr + 32 + lane] = __float2half_rn(p1);
      __syncwarp();
      if (lane == 0) {
        const float alpha = mold == -INFINITY ? 0.f : exp2f(mold - base);
        sA[h] = alpha;
        sL[h] = sL[h] * alpha + sum;
        sM[h] = mnew;
      }
    }
    __syncthreads();
    if (dbg && tid == 0 && c < 2) dbg[12 + 4 * c] = clock64();
    // ---- O += P V (FP16, FP32 accumulation): warp w owns latent dims [32 w, +32)
    {
#pragma unroll
      for (int m2 = 0; m2 < MT; ++m2) {
        const float al0 = sA[16 * m2 + g8], al1 = sA[16 * m2 + g8 + 8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          acc[4 * m2 + j][0] *= al0;
          acc[4 * m2 + j][1] *= al0;
          acc[4 * m2 + j][2] *= al1;
          acc[4 * m2 + j][3] *= al1;
        }
      }
      // ldmatrix.trans rows: lane l -> key 16 ks + 8 ((l >> 3) & 1) + (l & 7), dims
      // 32 w + 8 (l >> 4) (+ 16 for the second x4)
      const __half* vb = sVh + (((lane >> 3) & 1) * 8 + (lane & 7)) * kVStr + 32 * warp +
                         (lane >> 4) * 8;
#pragma unroll
      for (int ks = 0; ks < kChunk / 16; ++ks) {
        uint32_t b[8];
        ldsm_x4_t(b[0], b[1], b[2], b[3], vb + 16 * ks * kVStr);
        ldsm_x4_t(b[4], b[5], b[6], b[7], vb + 16 * ks * kVStr + 16);
#pragma unroll
        for (int m2 = 0; m2 < MT; ++m2) {
          uint32_t a0, a1, a2, a3;
          ldsm_x4(a0, a1, a2, a3, sP + (16 * m2 + (lane & 15)) * kPStr + 16 * ks + (lane >> 4) * 8);
          // x4.trans matrices: (k 0-7, n 0-7), (k 8-15, n 0-7), (k 0-7, n 8-15), (k 8-15, n 8-15)
          mma_f16(acc[4 * m2 + 0], a0, a1, a2, a3, b[0], b[1]);
          mma_f16(acc[4 * m2 + 1], a0, a1, a2, a3, b[2], b[3]);
          mma_f16(acc[4 * m2 + 2], a0, a1, a2, a3, b[4], b[5]);
          mma_f16(acc[4 * m2 + 3], a0, a1, a2, a3, b[6], b[7]);
        }
      }
    }
    __syncthreads();
    if (dbg && tid == 0 && c < 2) dbg[13 + 4 * c] = clock64();
    if (c + 2 < nchunks) issue(c + 2);
  }
  if (dbg && tid == 0) dbg[3] = now_ns();
  constexpr int kGroupHeads = HG;
  const int H = p.hpr;
  if (S > 1) {
    // ---- the S splits of (t, grp) are one cluster: each leaves its unnormalized FP32
    // O (shared [32, 512], the KV buffers' space) and (max, sum); after one cluster
    // barrier split s merges latent dims [512 s / S, +512 / S) of all of them through
    // distributed shared memory (weights 2^(m_s - M) * vscale / L) and pushes them
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
      const float r = live ? p.vscale / L : 0.f;
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
    const float r0 = (n > 0 && l0 > 0.f) ? p.vscale / l0 : 0.f;
    const float r1 = (n > 0 && l1 > 0.f) ? p.vscale / l1 : 0.f;
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
  for (int i = tid; i < kGroupHeads * 64; i += kThreads) {
    const int h = i >> 6, ch = i & 63;
    const int gh = h0 + h;
    const uint4 v = *reinterpret_cast<const uint4*>(sO + h * kOStr + 8 * ch);
    const int64_t slot = (((int64_t)p.parity * p.W + p.rank) * p.box_cap + t) * H + gh % H;
    uint32_t* dst = reinterpret_cast<uint32_t*>(p.part_peers[gh / H]) + slot * 256 + 4 * ch;
    st_v4(dst, make_uint4(no_neg_zero(v.x), no_neg_zero(v.y), no_neg_zero(v.z),
                          no_neg_zero(v.w)));
  }
  if (tid < kGroupHeads) {
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

void run(torch::Tensor q, torch::Tensor sel, torch::Tensor cnt, torch::Tensor kv,
         double s2scale, double vscale, torch::Tensor part_peers, torch::Tensor lse_peers,
         int64_t box_cap, int64_t hpr, int64_t parity, int64_t rank, int64_t T, int64_t lse_wide,
         int64_t splits, int64_t heads_per_cta, torch::Tensor err, int64_t spin_ns, int64_t pdl,
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
  cfg.gridDim = dim3((unsigned)splits, (unsigned)(heads / HG), (unsigned)T);
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

int64_t smem_bytes(int64_t heads_per_cta) {
  return heads_per_cta == 16 ? Plan<16>::kSmem : Plan<32>::kSmem;
}

}  // namespace k31dsac

namespace k31dsau {
void run(torch::Tensor q, torch::Tensor sel, torch::Tensor cnt, torch::Tensor kv, double s2scale,
         double vscale, torch::Tensor part_peers, torch::Tensor lse_peers, int64_t box_cap,
         int64_t hpr, int64_t parity, int64_t rank, int64_t T, int64_t lse_wide, torch::Tensor err, int64_t spin_ns,
         int64_t pdl, c10::optional<torch::Tensor> dbg);
}  // namespace k31dsau

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &k31dsac::run);
  m.def("run_umma", &k31dsau::run);
  m.def("smem_bytes", &k31dsac::smem_bytes);
}
