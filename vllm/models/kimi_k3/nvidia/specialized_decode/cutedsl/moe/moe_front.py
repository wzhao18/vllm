# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SP decode MoE front as ONE kernel: router | latent down GEMM, stock router
top-k and the MegaMoE input staging (``moe_front``).

Replaces router|latent GEMM -> (cuBLASLt split-K reduce) -> ``route_stage``: the
chain from the AttnRes output ``h [n, 7168]`` to a staged MegaMoE session is one
launch, so the per-boundary grid drain + memory flush and the separate one-row
top-k grid leave the critical path.

Kernel (Rubin, tcgen05, one CTA of 256 threads per SM slot):

* GEMM ``y = h @ W^T`` with ``W = [router (E) | latent down (L)] [E + L, K]`` BF16
  (``sp_mlp.pack_router_latent``), FP32 accumulation. Swap-AB: a 128-row weight
  tile is the MMA M operand, the n <= 256 tokens the N operand. One cluster of C
  CTAs per weight tile; CTA rank r streams K blocks [r kb / C, (r + 1) kb / C).
* PDL: the kernel launches with programmatic stream serialization; the weight ring
  is issued before ``griddepcontrol.wait`` (weights never depend on the
  predecessor), the tokens after it. The predecessor (AttnRes reduce) triggers its
  dependents at entry and uses ~2 KB of shared memory per CTA, so this grid's CTAs
  can sit beside it and stream weights while it runs.
* Split-K reduce: every CTA leaves its FP32 partial in its own shared memory; after
  a cluster barrier rank r sums the token rows [r n / C, (r + 1) n / C) over ranks
  0..C-1 in rank order (deterministic) from DSMEM.
  - router tiles store the FP32 logits rows (``y``) and then count themselves done
    (release, one global counter);
  - latent tiles round to BF16 (as the stock BF16 latent down output) and run the
    MegaMoE stager's quantization (NVFP4 block16 / MXFP8 block32, ``route_stage``'s
    ``quant_*``: bitwise ``megamoe_rubin.staging._stage_kernel``) straight into the
    session's ``x`` / ``x_sf``.
* Top-k: CTA b < n waits (acquire) until every router CTA is done and runs stock's
  sigmoid + bias single-group top-k on row b with ``route_stage``'s ``route_block``
  (bitwise ``_moe_C`` ``single_group_topk``: same keys, same exact selection, same
  ``write_outputs`` math, the stager's route rules). Router tiles are claimed first,
  by clusters that are already running, so the wait cannot deadlock.
* Dead rows ``[n, capacity)`` get ``topk_idx = -1`` as ``route_stage``.

Numerics vs stock: router logits FP32 with a different summation order than
cuBLAS (same inputs, FP32 accumulation); the latent the BF16 round of an FP32 sum
(as stock); top-k / weights / quantization bitwise functions of those values.
The device code of top-k and quantization IS ``route_stage``'s (same translation
unit). Built without fast math, as ``route_stage``.
"""

import hashlib
import os
import threading

import torch

from ...shapes import Unsupported
from .route_stage import _CUDA_SRC as _ROUTE_STAGE_SRC
from .route_stage import RouteStage, route_stage_cause

_FRONT_SRC = r"""
namespace k3front {

constexpr int kThreads = kWarps * 32;  // 256: w0 TMA, w1 MMA, w2 TMEM fill, w4-7 TMEM -> partial
constexpr int kSlot = 128 * 128;       // one 128 x 64 BF16 weight K block (or <= 128 token rows)
constexpr int kMaxNS = 16;             // ring slots
constexpr int kMaxT = 16;              // TMEM-resident weight K blocks
constexpr int kMaxR = 8;               // shared-memory-resident weight K blocks
constexpr int kMaxX = 32;              // token ring slots (n x 128 B each, in the ring)
constexpr int kLine = 32;              // one counter per 128 B line
constexpr int kTileCnt = 3 * kLine;    // cnt[kTileCnt + kLine t]: partials stored for tile t
constexpr int kStampStride = 64;  // [0, 8) phases (globaltimer), [8, 32) clock after MMA issue, [32, 56) clock operands ready, per K block
constexpr int kSmemBudget = 214 * 1024;  // dynamic + static: co-resident with the AttnRes reduce

struct Params {
  Args a;            // route_stage's arguments (y = router logits rows, ldy)
  float* y;          // [n, ldy] FP32 router logits
  float* part;       // [grid][n][128] FP32 split-K partials
  unsigned* cnt;     // [2 kLine] CTAs finished, [kTileCnt + kLine t] partials stored of tile t
  unsigned long long* stamps;  // [grid][kStampStride] globaltimer or null
  int kb, rtiles, ks, ns, xs, xg, na, nt_max, nr_max, yfull;
  int dcol, acol;  // TMEM columns of D [n] and of the first weight K block  // xs: token op slots of xg blocks
  int debug;  // microbench only (0 in production): bit 0 skips the MMAs
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, int bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar),
               "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{ .reg .pred p; WAIT_%=: "
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; "
      "@!p bra WAIT_%=; }" ::"r"(bar),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void mbar_wait_sleep(uint32_t bar, uint32_t parity) {
  uint32_t done = 0;
  while (true) {
    asm volatile(
        "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; "
        "selp.u32 %0, 1, 0, p; }"
        : "=r"(done)
        : "r"(bar), "r"(parity)
        : "memory");
    if (done) break;
    __nanosleep(64);
  }
}
__device__ __forceinline__ void tma_2d(uint32_t dst, const CUtensorMap* map, int c0, int c1,
                                       uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(bar)
      : "memory");
}
__device__ __forceinline__ void tma_3d(uint32_t dst, const CUtensorMap* map, int row, int kb,
                                       uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(0), "r"(row), "r"(kb), "r"(bar)
      : "memory");
}
// K-major SWIZZLE_128B operand descriptor (8-row core groups 1024 B apart).
__device__ __forceinline__ uint64_t smem_desc(uint32_t addr) {
  return static_cast<uint64_t>((addr & 0x3FFFF) >> 4) | (1ull << 16) | (64ull << 32) |
         (1ull << 46) | (2ull << 61);
}
// Warp-uniform issue (hwdb hw28c): the whole warp runs the asm, elect.sync picks the
// issuing lane, so descriptors stay in uniform registers.
__device__ __forceinline__ void mma_ts(uint32_t d, uint32_t a_tmem, uint64_t b, uint32_t idesc,
                                       uint32_t acc) {
  asm volatile(
      "{ .reg .pred P, Q; elect.sync _|P, 0xffffffff; setp.ne.b32 Q, %4, 0; "
      "@P tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, Q; }" ::"r"(d),
      "r"(a_tmem), "l"(b), "r"(idesc), "r"(acc));
}
__device__ __forceinline__ void mma_ss(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc,
                                       uint32_t acc) {
  asm volatile(
      "{ .reg .pred P, Q; elect.sync _|P, 0xffffffff; setp.ne.b32 Q, %4, 0; "
      "@P tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, Q; }" ::"r"(d),
      "l"(a), "l"(b), "r"(idesc), "r"(acc));
}
__device__ __forceinline__ void cp_el(uint32_t taddr, uint64_t src) {
  asm volatile(
      "{ .reg .pred P; elect.sync _|P, 0xffffffff; "
      "@P tcgen05.cp.cta_group::1.128x256b [%0], %1; }" ::"r"(taddr),
      "l"(src)
      : "memory");
}
__device__ __forceinline__ void commit_el(uint32_t bar) {
  asm volatile(
      "{ .reg .pred P; elect.sync _|P, 0xffffffff; "
      "@P tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0]; }" ::"r"(
          bar)
      : "memory");
}
__device__ __forceinline__ void tmem_ld16(uint32_t taddr, float (&v)[16]) {
  uint32_t r[16];
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,"
      "%12,%13,%14,%15}, [%16];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]),
        "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]),
        "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
      : "r"(taddr));
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
  for (int i = 0; i < 16; ++i) v[i] = __uint_as_float(r[i]);
}
__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ unsigned long long globaltimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
#define STAMP(i) \
  do { \
    if (p.stamps) p.stamps[blockIdx.x * kStampStride + (i)] = globaltimer(); \
  } while (0)

// Sum of the tile's ks (<= kMaxKs) partials at float offset `off` of a [n][128]
// partial, in rank order: every load is issued before the first add.
constexpr int kMaxKs = 8;
__device__ __forceinline__ float4 tile_sum(const float* pt, int64_t stride, int ks, int64_t off) {
  float4 v[kMaxKs];
#pragma unroll
  for (int q = 0; q < kMaxKs; ++q)
    if (q < ks) v[q] = __ldcg(reinterpret_cast<const float4*>(pt + q * stride + off));
  float4 acc = v[0];
#pragma unroll
  for (int q = 1; q < kMaxKs; ++q) {
    if (q < ks) {
      acc.x += v[q].x;
      acc.y += v[q].y;
      acc.z += v[q].z;
      acc.w += v[q].w;
    }
  }
  return acc;
}

// One CTA per (128-row weight tile, K split): blockIdx = tile * ks + split. 1 CTA/SM,
// the whole grid co-resident (host check). Weight K blocks of the CTA: the first nt
// live in TMEM (A operand of the TS MMA, filled by tcgen05.cp through the ring before
// the dependency), the last nr in shared memory (A operand of the SS MMA). Ring items:
// nt weight K blocks, then nk token K blocks (after griddepcontrol.wait).
template <bool NVFP4>
__global__ void __maxnreg__(96)  // launched with kThreads
    moe_front_kernel(const __grid_constant__ CUtensorMap tw,
                     const __grid_constant__ CUtensorMap tx,
                     const __grid_constant__ Params p) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  __shared__ float s_bias[kWarps * 32 * kPer];
  __shared__ float s_unb[kWarps * 32 * kPer];
  __shared__ uint64_t s_cand[kWarps * 32];
  __shared__ __align__(8) uint64_t s_bar[2 * kMaxNS + kMaxT + 2 + 2 * kMaxX];
  __shared__ uint32_t s_tmem;
  const Args& a = p.a;
  uint8_t* base = reinterpret_cast<uint8_t*>(
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const uint32_t res_s = smem_u32(base);
  const uint32_t ring_s = res_s + p.nr_max * kSlot;
  const uint32_t full0 = smem_u32(s_bar), empty0 = full0 + 8 * kMaxNS,
                 trdy0 = empty0 + 8 * kMaxNS, res_bar = trdy0 + 8 * kMaxT,
                 acc_bar = res_bar + 8, xfull0 = acc_bar + 8, xempty0 = xfull0 + 8 * kMaxX;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int tile = static_cast<int>(blockIdx.x) / p.ks, split = static_cast<int>(blockIdx.x) % p.ks;
  const int kb0 = split * p.kb / p.ks, nk = (split + 1) * p.kb / p.ks - kb0;
  const int nt = min(nk, p.nt_max), nr = nk - nt;
  const int n = a.n, wrow = tile * 128;

  if (threadIdx.x == 0) {
    asm volatile("griddepcontrol.launch_dependents;");
    STAMP(0);
    for (int s = 0; s < p.ns; ++s) {
      mbar_init(full0 + 8 * s, 1);
      mbar_init(empty0 + 8 * s, 1);
    }
    for (int j = 0; j < kMaxT; ++j) mbar_init(trdy0 + 8 * j, 1);
    for (int s = 0; s < p.xs; ++s) {
      mbar_init(xfull0 + 8 * s, 1);
      mbar_init(xempty0 + 8 * s, 1);
    }
    mbar_init(res_bar, 1);
    mbar_init(acc_bar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if (nr > 0) mbar_expect_tx(res_bar, nr * kSlot);
  }
  if (threadIdx.x == 32) {
    asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&tw)) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&tx)) : "memory");
  }
  // Router correction bias (a weight: before the dependency).
  for (int e = threadIdx.x; e < a.num_logits; e += kThreads) s_bias[e] = a.bias[e];
  if (warp == 1) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" ::"r"(
            smem_u32(&s_tmem)));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  asm volatile("tcgen05.fence::before_thread_sync;");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;");
  const uint32_t tmem = s_tmem;  // D: columns [0, n); weight K block j < nt: na + 32 j

  // TMA issue: lane 0 of warps 0, 3 and 4-7 (the epilogue warps idle until the
  // accumulator is done). Bulk ops from one issuing warp complete about one at a
  // time (hwdb hw51), so each ring slot is owned by one issuer and the issuers run
  // in parallel. Weights never depend on the predecessor: the shared-memory-resident
  // blocks and the TMEM fill stream start at entry, beside the AttnRes reduce.
  if ((warp == 0 || warp >= 3) && lane == 0) {
    constexpr int kIssuers = 6;
    const int u = warp == 0 ? 0 : warp - 2;
    // res_bar's expect_tx is set before the CTA barrier above (thread 0).
    for (int r = u; r < nr; r += kIssuers)
      tma_2d(res_s + r * kSlot, &tw, (kb0 + nt + r) * 64, wrow, res_bar);
    for (int i = 0; i < nt; ++i) {
      const int slot = i % p.ns;
      if (slot % kIssuers != u) continue;
      if (i >= p.ns) mbar_wait(empty0 + 8 * slot, ((i / p.ns) & 1) ^ 1);
      mbar_expect_tx(full0 + 8 * slot, kSlot);
      tma_2d(ring_s + slot * kSlot, &tw, (kb0 + i) * 64, wrow, full0 + 8 * slot);
    }
    // The token ring reuses the weight ring's memory in n x 128 B slots once the
    // TMEM fill has read every weight slot (its commits complete in order).
    if (u < p.xs) {
      mbar_wait(trdy0 + 8 * (nt - 1), 0);
      asm volatile("griddepcontrol.wait;" ::: "memory");
      if (u == 0) STAMP(1);
      // One op = xg token K blocks (3-D map, [xg][n][128 B] SWIZZLE_128B tiles); the
      // last op of the split may run past it (unused blocks, zero fill past K).
      const int ng = (nk + p.xg - 1) / p.xg;
      for (int g = 0; g < ng; ++g) {
        const int gs = g % p.xs;
        if (gs % kIssuers != u) continue;
        if (g >= p.xs) mbar_wait(xempty0 + 8 * gs, ((g / p.xs) & 1) ^ 1);
        mbar_expect_tx(xfull0 + 8 * gs, p.xg * n * 128);
        tma_3d(ring_s + gs * p.xg * n * 128, &tx, 0, kb0 + g * p.xg, xfull0 + 8 * gs);
      }
    }
  }
  __syncwarp();  // the issuing lane rejoins its warp (tcgen05.ld below is .aligned)
  if (warp == 2) {
    // TMEM fill: ring weight block i -> TMEM columns [na + 32 i, na + 32 i + 32).
    for (int i = 0; i < nt; ++i) {
      const int slot = i % p.ns;
      mbar_wait(full0 + 8 * slot, (i / p.ns) & 1);
      const uint64_t src = smem_desc(ring_s + slot * kSlot);
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) cp_el(tmem + p.acol + 32 * i + 8 * kk, src + 2 * kk);
      // tcgen05.commit stalls the issuing thread's tensor pipe (measured ~0.25 us per
      // K block with a commit each): commit only where a consumer needs it.
      if (i + p.ns < nt) commit_el(empty0 + 8 * slot);
      if (i == nt - 1) commit_el(trdy0 + 8 * i);
    }
  } else if (warp == 1) {
    // kind::f16: BF16 A/B, FP32 D, K-major, M = 128 weight rows, N = n tokens.
    const uint32_t idesc = (1u << 4) | (1u << 7) | (1u << 10) |
                           (static_cast<uint32_t>(n >> 3) << 17) | (8u << 24);
    for (int j = 0; j < nk; ++j) {
      const int g = j / p.xg, gs = g % p.xs, xs = gs * p.xg + j % p.xg;
      if (j % p.xg == 0) mbar_wait(xfull0 + 8 * gs, (g / p.xs) & 1);
      if (j == 0) {
        mbar_wait(trdy0 + 8 * (nt - 1), 0);  // the whole TMEM fill (one commit)
      } else if (j == nt) {
        mbar_wait(res_bar, 0);
      }
      asm volatile("tcgen05.fence::after_thread_sync;");
      if (j == 0 && lane == 0) STAMP(6);
      if (lane == 0 && j < 24 && p.stamps) p.stamps[blockIdx.x * kStampStride + 32 + j] = clock64();
      const uint64_t b0 = smem_desc(ring_s + xs * n * 128);
      if (p.debug & 1) {
      } else if (j < nt) {
#pragma unroll
        for (int kk = 0; kk < 4; ++kk)
          mma_ts(tmem + p.dcol, tmem + p.acol + 32 * j + 8 * kk, b0 + 2 * kk, idesc, (j | kk) != 0);
      } else {
        const uint64_t a0 = smem_desc(res_s + (j - nt) * kSlot);
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) mma_ss(tmem + p.dcol, a0 + 2 * kk, b0 + 2 * kk, idesc, (j | kk) != 0);
      }
      if ((j % p.xg == p.xg - 1 || j == nk - 1) && (g + p.xs) * p.xg < nk)
        commit_el(xempty0 + 8 * gs);
      if (lane == 0 && j < 24 && p.stamps) p.stamps[blockIdx.x * kStampStride + 8 + j] = clock64();
    }
    commit_el(acc_bar);
  } else if (warp >= 4) {
    // D (TMEM lane = weight row, column = token) -> this CTA's FP32 partial.
    asm volatile("griddepcontrol.wait;" ::: "memory");
    mbar_wait_sleep(acc_bar, 0);
    if (threadIdx.x == 128) STAMP(7);
    asm volatile("tcgen05.fence::after_thread_sync;");
    const int q = warp & 3, row = q * 32 + lane;
    const uint32_t ta = tmem + p.dcol + (static_cast<uint32_t>(q * 32) << 16);
    float* dst = p.part + int64_t{blockIdx.x} * n * 128 + row;
    for (int c = 0; c < n; c += 16) {
      float v[16];
      tmem_ld16(ta + c, v);
#pragma unroll
      for (int j = 0; j < 16; ++j) __stcg(dst + (c + j) * 128, v[j]);
    }
  }
  // Every thread past the dependency before any global write below.
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("tcgen05.fence::before_thread_sync;");
  __syncthreads();
  if (warp == 1) {
    asm volatile("tcgen05.fence::after_thread_sync;");
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" ::"r"(tmem));
  }
  if (threadIdx.x == 0) {
    STAMP(2);
    __threadfence();
    atomicAdd(p.cnt + kTileCnt + kLine * tile, 1u);
    while (ld_acquire(p.cnt + kTileCnt + kLine * tile) < static_cast<unsigned>(p.ks)) __nanosleep(32);
    STAMP(3);
  }
  __syncthreads();
  {
    // Split-K reduce of token rows [t0, t1) of this tile, partials in rank order.
    const int t0 = split * n / p.ks, t1 = (split + 1) * n / p.ks;
    const float* pt = p.part + int64_t{tile} * p.ks * n * 128;
    const int64_t stride = int64_t{n} * 128;
    // Router tiles: the top-k CTA of each row sums its logits itself (below).
    if (tile >= p.rtiles) {
      // Latent: BF16 round + the stager's quantization of each block of this tile.
      constexpr int vec = NVFP4 ? 16 : 32;
      constexpr int per = 128 / vec;
      const int blk0 = (wrow - a.num_logits) / vec;
      for (int f = threadIdx.x; f < (t1 - t0) * per; f += kThreads) {
        const int t = t0 + f / per, b = f % per;
        float4 in[8];
#pragma unroll
        for (int q = 0; q < vec / 4; ++q)
          in[q] = tile_sum(pt, stride, p.ks, int64_t{t} * 128 + b * vec + 4 * q);
        if (p.yfull) {
          float4* dst = reinterpret_cast<float4*>(p.y + int64_t{t} * a.ldy + wrow + b * vec);
#pragma unroll
          for (int q = 0; q < vec / 4; ++q) dst[q] = in[q];
        }
        if constexpr (NVFP4) {
          quant_nvfp4(a, t, blk0 + b, in);
        } else {
          quant_mxfp8(a, t, blk0 + b, in);
        }
      }
    }
  }
  // Dead rows [n, capacity): routes -1 only (spread over the grid).
  {
    const int64_t lo = int64_t{n} * a.topk, hi = int64_t{a.capacity} * a.topk;
    for (int64_t i = lo + int64_t{blockIdx.x} * kThreads + threadIdx.x; i < hi;
         i += int64_t{gridDim.x} * kThreads)
      a.idx[i] = -1;
  }
  const int row = blockIdx.x;
  if (row < n) {
    // Every router tile's partials stored: sum this row's logits in rank order
    // (bitwise the tile sum) from the partials, one counter round trip fewer.
    if (threadIdx.x < p.rtiles) {
      const unsigned* c = p.cnt + kTileCnt + kLine * threadIdx.x;
      while (ld_acquire(c) < static_cast<unsigned>(p.ks)) __nanosleep(32);
    }
    __syncthreads();
    if (threadIdx.x == 0) STAMP(4);
    const int64_t stride = int64_t{n} * 128;
    float lg[kPer];
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
      const int e = j * kThreads + threadIdx.x;
      float v[kMaxKs];
      if (e < a.num_logits) {
        const float* pe = p.part + (int64_t{e / 128} * p.ks * n + row) * 128 + (e % 128);
#pragma unroll
        for (int q = 0; q < kMaxKs; ++q)
          if (q < p.ks) v[q] = __ldcg(pe + q * stride);
      }
      float acc = 0.0F;
      if (e < a.num_logits) {
        acc = v[0];
#pragma unroll
        for (int q = 1; q < kMaxKs; ++q)
          if (q < p.ks) acc += v[q];
        p.y[int64_t{row} * a.ldy + e] = acc;
      }
      lg[j] = acc;
    }
    route_block(a, row, lg, s_bias, s_unb, s_cand);  // warps 1.. return inside
  }
  if (threadIdx.x == 0) {
    STAMP(5);
    // The last CTA of the grid (every CTA is past its waits) resets the counters.
    if (atomicAdd(p.cnt + 2 * kLine, 1u) == gridDim.x - 1) {
      p.cnt[2 * kLine] = 0;
      for (int t = 0; t < static_cast<int>(gridDim.x) / p.ks; ++t) p.cnt[kTileCnt + kLine * t] = 0;
    }
  }
}

// {64 columns, rows, K blocks}: K block j of row r at r * k + 64 j; one op = box_kb
// consecutive K blocks of the rows, stored as consecutive [rows][64] SWIZZLE_128B tiles.
static void make_map3(CUtensorMap* m, const void* ptr, int rows, int k, int box_rows,
                      int box_kb) {
  cuuint64_t dims[3] = {64, static_cast<cuuint64_t>(rows), static_cast<cuuint64_t>(k / 64)};
  cuuint64_t strides[2] = {static_cast<cuuint64_t>(k) * 2, 128};
  cuuint32_t box[3] = {64, static_cast<cuuint32_t>(box_rows), static_cast<cuuint32_t>(box_kb)};
  cuuint32_t es[3] = {1, 1, 1};
  CUresult rc = cuTensorMapEncodeTiled(
      m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(ptr), dims, strides, box,
      es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(rc == CUDA_SUCCESS, "moe_front: cuTensorMapEncodeTiled ", static_cast<int>(rc));
}

static void make_map(CUtensorMap* m, const void* ptr, int rows, int k, int box_rows) {
  cuuint64_t dims[2] = {static_cast<cuuint64_t>(k), static_cast<cuuint64_t>(rows)};
  cuuint64_t strides[1] = {static_cast<cuuint64_t>(k) * 2};
  cuuint32_t box[2] = {64, static_cast<cuuint32_t>(box_rows)};
  cuuint32_t es[2] = {1, 1};
  CUresult rc = cuTensorMapEncodeTiled(
      m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides, box,
      es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(rc == CUDA_SUCCESS, "moe_front: cuTensorMapEncodeTiled ", static_cast<int>(rc));
}

struct Plan {
  int na, nt_max, nr_max, ns, smem, xs, xg;
};

static int static_smem() {
  cudaFuncAttributes fa;
  TORCH_CHECK(cudaFuncGetAttributes(&fa, moe_front_kernel<true>) == cudaSuccess);
  return static_cast<int>(fa.sharedSizeBytes);
}

// TMEM: D in columns [0, na), then as many weight K blocks (32 columns each) as fit;
// the rest of the CTA's K blocks in shared memory; the ring takes what is left.
static Plan plan_for(int n, int kb, int ks, int ns) {
  Plan pl;
  pl.na = (n + 31) / 32 * 32;
  pl.nt_max = std::min(kMaxT, (512 - pl.na) / 32);
  const int nk_max = (kb + ks - 1) / ks;
  pl.nr_max = std::max(0, nk_max - pl.nt_max);
  const int fixed = 1024 + pl.nr_max * kSlot;
  const int room = (kSmemBudget - static_smem() - fixed) / kSlot;
  pl.ns = ns > 0 ? ns : std::min(kMaxNS, room);
  pl.smem = fixed + pl.ns * kSlot;
  // Token ops of xg K blocks (>= 16 KiB per op: one op completes at a time per issuer).
  pl.xg = std::max(1, std::min(4, (kSlot + n * 128 - 1) / (n * 128)));
  pl.xs = std::min({kMaxX, (nk_max + pl.xg - 1) / pl.xg, pl.ns * kSlot / (pl.xg * n * 128)});
  return pl;
}

static void set_attrs(int smem) {
  for (const void* f : {(const void*)moe_front_kernel<true>, (const void*)moe_front_kernel<false>})
    TORCH_CHECK(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) ==
                    cudaSuccess,
                "moe_front: smem attribute ", smem);
}

}  // namespace k3front

// ---- host ----
void moe_front(const torch::Tensor& w, const torch::Tensor& h, torch::Tensor& y,
               torch::Tensor& cnt, torch::Tensor& part, const torch::Tensor& bias,
               torch::Tensor& x, torch::Tensor& sf, torch::Tensor& idx, torch::Tensor& sw,
               torch::Tensor& err, const c10::optional<torch::Tensor>& pad,
               int64_t num_logits, int64_t num_experts, bool renormalize,
               double routed_scaling_factor, bool nvfp4, double norm_const, int64_t ks,
               int64_t ns, const torch::Tensor& stamps) {
  using namespace k3front;
  TORCH_CHECK(w.scalar_type() == at::kBFloat16 && h.scalar_type() == at::kBFloat16 &&
                  w.is_contiguous() && h.is_contiguous(),
              "moe_front: BF16 contiguous weight and tokens");
  const int64_t n = h.size(0), k = h.size(1);
  const int64_t rows_w = w.size(0);
  const int64_t hidden = rows_w - num_logits;
  TORCH_CHECK(w.size(1) == k && k % 64 == 0 && rows_w % 128 == 0 &&
                  num_logits % 128 == 0 && num_logits <= kWarps * 32 * kPer &&
                  hidden > 0 && n >= 16 && n <= 128 && n % 16 == 0,
              "moe_front: shapes");
  TORCH_CHECK(y.scalar_type() == at::kFloat && y.dim() == 2 && y.stride(1) == 1 &&
                  y.size(0) >= n && y.size(1) >= num_logits && y.stride(0) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(y.data_ptr()) % 16 == 0,
              "moe_front: logits buffer");
  const int64_t tiles = rows_w / 128, kb = k / 64, grid = tiles * ks;
  TORCH_CHECK(cnt.scalar_type() == at::kInt && cnt.is_contiguous() &&
                  cnt.numel() >= kTileCnt + kLine * tiles,
              "moe_front: counters");
  TORCH_CHECK(part.scalar_type() == at::kFloat && part.is_contiguous() &&
                  part.numel() >= grid * n * 128,
              "moe_front: partials");
  TORCH_CHECK(bias.scalar_type() == at::kFloat && bias.is_contiguous() &&
                  bias.numel() == num_logits,
              "moe_front: bias");
  TORCH_CHECK(x.scalar_type() == at::kByte && sf.scalar_type() == at::kByte &&
                  x.stride(1) == 1 && sf.stride(1) == 1 &&
                  idx.scalar_type() == at::kInt && idx.is_contiguous() &&
                  sw.scalar_type() == at::kFloat && sw.is_contiguous() &&
                  err.scalar_type() == at::kLong && err.numel() == 2,
              "moe_front: session buffers");
  const int64_t capacity = idx.size(0), topk = idx.size(1);
  const int64_t vec = nvfp4 ? 16 : 32;
  TORCH_CHECK(topk <= 32 && topk <= num_logits && hidden % vec == 0 && n <= capacity &&
                  sw.size(0) == capacity && sw.size(1) == topk && x.size(0) >= capacity &&
                  x.size(1) * (nvfp4 ? 2 : 1) >= hidden && sf.size(0) >= capacity &&
                  sf.size(1) >= hidden / vec && x.stride(0) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "moe_front: session shapes");
  const Plan pl = plan_for((int)n, (int)kb, (int)ks, (int)(ns & 255));
  int dev = h.get_device(), sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  TORCH_CHECK(ks >= 1 && ks <= kMaxKs && ks <= kb && grid >= n && grid <= sms && pl.nr_max <= kMaxR &&
                  pl.ns >= 2 && pl.ns <= kMaxNS && pl.xs >= 1,
              "moe_front: launch configuration (ks ", ks, ", grid ", grid, " of ", sms,
              " SMs, ring ", pl.ns, ", smem blocks ", pl.nr_max, ")");
  const bool* pad_ptr = nullptr;
  if (pad.has_value()) {
    TORCH_CHECK(pad->scalar_type() == at::kBool && pad->dim() == 1 &&
                    pad->size(0) == n && pad->stride(0) == 1,
                "moe_front: is_padding");
    pad_ptr = pad->data_ptr<bool>();
  }
  Params p;
  Args& a = p.a;
  a.y = y.data_ptr<float>();
  a.ldy = y.stride(0);
  a.num_logits = (int32_t)num_logits;
  a.bias = bias.data_ptr<float>();
  a.x = x.data_ptr<uint8_t>();
  a.x_stride = x.stride(0);
  a.sf = sf.data_ptr<uint8_t>();
  a.sf_stride = sf.stride(0);
  a.sf_cols = (int32_t)sf.size(1);
  a.idx = idx.data_ptr<int32_t>();
  a.sw = sw.data_ptr<float>();
  a.err = err.data_ptr<int64_t>();
  a.pad = pad_ptr;
  a.n = (int32_t)n;
  a.capacity = (int32_t)capacity;
  a.num_experts = (int32_t)num_experts;
  a.topk = (int32_t)topk;
  a.renormalize = renormalize;
  a.routed_scaling_factor = (float)routed_scaling_factor;
  a.hidden = (int32_t)hidden;
  a.scale_mult = nvfp4 ? (float)(norm_const / 6.0) : 0.0F;
  a.norm_const = (float)norm_const;
  TORCH_CHECK(!nvfp4 || a.scale_mult > 0.0F, "moe_front: norm_const");
  p.y = y.data_ptr<float>();
  p.part = part.data_ptr<float>();
  p.cnt = reinterpret_cast<unsigned*>(cnt.data_ptr<int32_t>());
  p.debug = static_cast<int>(stamps.numel() > grid * kStampStride ? stamps.numel() - grid * kStampStride : 0);
  p.stamps = stamps.numel() >= grid * kStampStride
                 ? reinterpret_cast<unsigned long long*>(stamps.data_ptr())
                 : nullptr;
  p.kb = (int)kb;
  p.rtiles = (int)(num_logits / 128);
  p.ks = (int)ks;
  p.ns = pl.ns;
  p.na = pl.na;
  // ns bits 8+: TMEM layout (0: D in the first columns, 1: D in the last columns).
  const bool dtop = (ns >> 8) & 1;
  p.dcol = dtop ? 512 - pl.na : 0;
  p.acol = dtop ? 0 : pl.na;
  p.nt_max = pl.nt_max;
  p.nr_max = pl.nr_max;
  p.xs = pl.xs;
  p.xg = pl.xg;
  p.yfull = y.size(1) >= rows_w ? 1 : 0;
  CUtensorMap tw, tx;
  make_map(&tw, w.data_ptr(), (int)rows_w, (int)k, 128);
  make_map3(&tx, h.data_ptr(), (int)n, (int)k, (int)n, pl.xg);
  static int configured = 0;
  if (pl.smem > configured) {
    set_attrs(pl.smem);
    configured = pl.smem;
  }
  const c10::cuda::CUDAGuard guard(h.device());
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(static_cast<unsigned>(grid));
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = pl.smem;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  const cudaError_t rc = nvfp4 ? cudaLaunchKernelEx(&cfg, moe_front_kernel<true>, tw, tx, p)
                              : cudaLaunchKernelEx(&cfg, moe_front_kernel<false>, tw, tx, p);
  TORCH_CHECK(rc == cudaSuccess, "moe_front launch: ", cudaGetErrorString(rc));
}

// (ring slots, shared-memory weight K blocks, dynamic smem bytes) at this shape.
std::vector<int64_t> moe_front_plan(int64_t n, int64_t k, int64_t ks, int64_t ns) {
  const k3front::Plan pl = k3front::plan_for((int)n, (int)(k / 64), (int)ks, (int)(ns & 255));
  return {pl.ns, pl.nr_max, pl.smem, pl.xs};
}

std::vector<int64_t> moe_front_attrs() {
  cudaFuncAttributes fa;
  TORCH_CHECK(cudaFuncGetAttributes(&fa, k3front::moe_front_kernel<true>) == cudaSuccess);
  return {fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)fa.sharedSizeBytes};
}
"""

_CPP_SRC = """
void moe_front(const torch::Tensor& w, const torch::Tensor& h, torch::Tensor& y,
               torch::Tensor& cnt, torch::Tensor& part, const torch::Tensor& bias,
               torch::Tensor& x, torch::Tensor& sf, torch::Tensor& idx, torch::Tensor& sw,
               torch::Tensor& err, const c10::optional<torch::Tensor>& pad,
               int64_t num_logits, int64_t num_experts, bool renormalize,
               double routed_scaling_factor, bool nvfp4, double norm_const, int64_t ks,
               int64_t ns, const torch::Tensor& stamps);
std::vector<int64_t> moe_front_plan(int64_t n, int64_t k, int64_t ks, int64_t ns);
std::vector<int64_t> moe_front_attrs();
"""


def _cuda_src() -> str:
    # route_stage's translation unit (top-k, quantization, Args) + the front kernel,
    # which calls its device functions: one copy of that code, bitwise the same.
    # Its anonymous namespace is closed before the host function; reopen it.
    head, sep, tail = _ROUTE_STAGE_SRC.partition("}  // namespace\n")
    device, mark, host = _FRONT_SRC.partition("// ---- host ----\n")
    if not sep or not mark:
        raise RuntimeError("moe_front: route_stage source layout changed")
    return "#include <cuda.h>\n" + head + device + sep + tail + host


_lock = threading.Lock()
_mod = None


def _module():
    global _mod
    if _mod is not None:
        return _mod
    with _lock:
        if _mod is None:
            from torch.utils.cpp_extension import load_inline

            import vllm.envs as envs

            src = _cuda_src()
            h = hashlib.sha256((src + _CPP_SRC).encode())
            h.update(f"{torch.__version__} {torch.version.cuda}".encode())
            tag = h.hexdigest()[:16]
            build = os.path.join(envs.VLLM_CACHE_ROOT, "k3_moe_front", tag)
            os.makedirs(build, exist_ok=True)
            # No --use_fast_math: precise tanhf, IEEE division (stock numerics).
            _mod = load_inline(
                name=f"k3_moe_front_{tag}",
                cpp_sources=_CPP_SRC,
                cuda_sources=src,
                functions=["moe_front", "moe_front_plan", "moe_front_attrs"],
                extra_cuda_cflags=[
                    "-O3",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    # tcgen05 / TMA / griddepcontrol: the sm_100 family target.
                    "-gencode",
                    "arch=compute_100f,code=sm_100f",
                ],
                extra_ldflags=["-lcuda"],
                build_directory=build,
                verbose=False,
            )
    return _mod


# Weight tiles of 128 rows (router | latent: 35 at K3.1) x K splits: one CTA per
# (tile, split), the whole grid resident at 1 CTA/SM. 6 splits = 210 CTAs on VR200.
MAX_ROWS = 96
MAX_SPLITS = 6


def config_for(n: int, device=None):
    """(K splits, ring slots (0: as many as shared memory holds)) at ``n`` shard rows,
    or None without a launch shape."""
    if n > MAX_ROWS or n % 16:
        return None
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    ks = min(MAX_SPLITS, sms // 35)
    return (ks, 0) if ks >= 2 else None


class MoeFront:
    """The SP MoE front of one MegaMoE layer (module docstring): ``stage(h)`` runs
    GEMM + top-k + staging into the session of ``h``'s bucket and returns the
    handle ``RouteStage.launch`` takes. ``packed``: the [router | latent down]
    BF16 weight (``sp_mlp.pack_router_latent``)."""

    def __init__(self, mlp, rows, packed: torch.Tensor):
        cause = route_stage_cause(mlp)
        if cause is not None:
            raise Unsupported("moe", None, f"moe_front: {cause}")
        self.route = RouteStage(mlp, rows)
        self.mlp, self.packed = mlp, packed
        self.num_logits = mlp.gate.weight.shape[0]
        if self.num_logits % 128 or packed.shape[0] % 128 or packed.shape[1] % 64:
            raise Unsupported("moe", None, f"moe_front: weight {tuple(packed.shape)}")
        for n in rows:
            if config_for(n, packed.device) is None:
                raise Unsupported("moe", n, "moe_front: no launch configuration")
        dev = packed.device
        n_max = max(rows)
        self.y = torch.empty((n_max, self.num_logits), dtype=torch.float32, device=dev)
        tiles = packed.shape[0] // 128
        self.cnt = torch.zeros(32 * (3 + tiles), dtype=torch.int32, device=dev)
        ks = max(config_for(n, dev)[0] for n in rows)
        self.part = torch.empty(tiles * ks * n_max * 128, dtype=torch.float32, device=dev)
        self.no_stamps = torch.empty(0, dtype=torch.int64, device=dev)
        _module()

    def ready(self, h: torch.Tensor) -> bool:
        return self.route.ready(h.shape[0]) and h.is_contiguous()

    def stage(self, h: torch.Tensor, stamps: torch.Tensor | None = None):
        import vllm.envs as envs
        from vllm.forward_context import get_forward_context, is_forward_context_available

        experts = self.mlp.experts
        n = h.shape[0]
        bucket = experts.select_bucket(n)
        rt = experts.get_runtime()
        session = rt.sessions[bucket]
        p = rt.problem
        if (
            self.packed.shape[0] - self.num_logits != p.hidden
            or session.topk_idx.shape[1] != experts.top_k
        ):
            raise Unsupported("moe", n, "moe_front: MegaMoE problem shape")
        pad = None
        if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
            pad = get_forward_context().is_padding
            if pad is not None:
                pad = pad[:n]
        cfg = config_for(n, h.device)
        _module().moe_front(
            self.packed, h, self.y, self.cnt, self.part,
            self.mlp.gate.e_score_correction_bias.data,
            session.x.view(torch.uint8), session.x_sf.view(torch.uint8),
            session.topk_idx, session.topk_weights, rt.route_error, pad,
            self.num_logits, p.num_experts, self.route.renormalize, self.route.scale,
            p.quant_kind == "nvfp4", float(p.input_norm_const),
            *cfg,
            self.no_stamps if stamps is None else stamps,
        )  # fmt: skip
        return rt, session, n

    def launch(self, staged, output_rmsnorm=None, shared=None):
        return self.route.launch(staged, output_rmsnorm, shared)
