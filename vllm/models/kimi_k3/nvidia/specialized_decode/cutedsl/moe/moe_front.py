# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SP decode MoE front as ONE kernel: router | latent down GEMM, stock router
top-k and the MegaMoE input staging (``moe_front``).

Replaces router|latent GEMM -> (split-K reduce) -> ``route_stage``: the chain from
the AttnRes output ``h [n, 7168]`` to a staged MegaMoE session is one launch.

Kernel (Rubin, tcgen05, one CTA of 256 threads per SM, the whole grid co-resident):

* GEMM ``y = h @ W^T`` with ``W = [router (E) | latent down (L)] [E + L, K]`` BF16
  (``sp_mlp.pack_router_latent``), FP32 accumulation. Swap-AB: a 128-row weight
  tile is the MMA M operand, the n <= 96 tokens the N operand. Work item = (tile,
  K split): 35 tiles x 6 splits on 210 SMs; the host plan gives every SM the items
  whose weights its die homes (``front_image``: die-local weight image).
* Weights before the dependency: the item's first K blocks go to TMEM (tcgen05.cp
  through a shared-memory ring, A operand of the TS MMA), the rest stay in shared
  memory (SS MMA). The predecessor (AttnRes reduce) uses little shared memory, so
  this grid's CTAs enter early and the weights are on chip by griddepcontrol.wait
  (r8/r16; at r24 the fill ends shortly after it).
* Tokens after the dependency: 3-D TMA ops of up to 16 KiB per ring slot (several
  issuing warps), recycled through the ring when they do not all fit.
* MMA issue: one warp (warp 1), 4 MMAs per K block, a 4-block token group under
  one elect at n32. Its per-iteration issue latency, not the tensor pipe, bounds
  the MMA phase (drain bench: ~425 clk per iteration with no waits).
* Split-K exchange: every CTA stores its FP32 partial [n][128] and signals its tile
  (one release add). Latent CTAs wait for their tile, sum its partials in rank
  order (deterministic), round to BF16 (the stock latent down output) and run the
  MegaMoE stager's quantization (NVFP4 block16 / MXFP8 block32, ``route_stage``'s
  ``quant_*``) straight into the session's ``x`` / ``x_sf``.
* Top-k: the SM of row b (router items first, then latent items) waits for every
  router tile, sums the row's logits from the partials in rank order and runs
  stock's sigmoid + bias single-group top-k with ``route_stage``'s ``route_block``
  (bitwise ``_moe_C`` ``single_group_topk`` keys, exact selection, write_outputs
  math, the stager's route rules). Dead rows ``[n, capacity)`` get ``topk_idx =
  -1`` as ``route_stage``.
* Cold code: the tail is long straight-line code that the step's other kernels
  evict from L2 between layers; the spare SMs (no item) run it dry before the
  dependency so the real pass finds it in L2.

Numerics vs stock: router logits FP32 with a different summation order than
cuBLAS (same inputs, FP32 accumulation); the latent the BF16 round of an FP32 sum
(as stock); top-k / weights / quantization bitwise functions of those values.
The device code of top-k and quantization IS ``route_stage``'s (same translation
unit). Built without fast math, as ``route_stage``.
"""

import functools
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
constexpr int kMaxX = kMaxNS;          // token slots (<= 16 KiB each, one per W ring slot)
constexpr int kLine = 32;              // one counter per 128 B line
constexpr int kTileCnt = 3 * kLine;    // cnt[kTileCnt + kLine t]: partials stored for tile t
constexpr int kMaxSmid = 512;
constexpr int kStampStride = 64;  // [0, 8) phases (globaltimer), [8, 27) clock after MMA issue, [32, 51) clock operands ready, per K block, 27 top-k clock, 31 item + 1, 56 W in TMEM, 57 latent staged, 58-63 route_block clocks
constexpr int kSmemBudget = 214 * 1024;  // dynamic + static: co-resident with the AttnRes reduce

struct Params {
  Args a;            // route_stage's arguments (y = router logits rows, ldy)
  float* y;          // [n, ldy] FP32 router logits
  float* part;       // [grid][n][128] FP32 split-K partials
  unsigned* cnt;     // [2 kLine] CTAs finished, [kTileCnt + kLine t] partials stored of tile t
  unsigned long long* stamps;  // [grid][kStampStride] globaltimer or null
  int kb, rtiles, ks, ns, xs, xg, na, nt_max, nr_max, yfull;
  int dcol, acol;  // TMEM columns of D [n] and of the first weight K block  // xs: token op slots of xg blocks
  int debug;  // microbench only (0 in production): bit 0 skips the MMAs, bit 1 one
              // accumulator poller, bit 2 no per-K-block stamps, bit 3 no tail warm-up
  const uint8_t* wimg;  // die-local weight tile image (front_image)
  int tiles, item_split, item_pad;  // items >= item_split start item_pad blocks later
  short smid_item[kMaxSmid];        // SM -> work item (tile * ks + split), -1: spare
  // Token slot g lives in W ring slot xslot[g] once that slot's last weight block is in
  // TMEM (its wfree barrier; xfree[g] = 0: the slot never holds weights).
  signed char xslot[kMaxNS], xfree[kMaxNS];
  short smid_row[kMaxSmid];  // SM -> top-k row (router tiles first: no latent work)
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
__device__ __forceinline__ void tma_3d(uint32_t dst, const CUtensorMap* map, int row, int kb,
                                       uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(0), "r"(row), "r"(kb), "r"(bar)
      : "memory");
}
// Weights are read once per step: evict-first in L2, so the 64 MB stream does not push
// out what the step reuses (the tail's instructions, warmed before the dependency).
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes,
                                         uint32_t bar) {
  asm volatile(
      "{ .reg .b64 pol; createpolicy.fractional.L2::evict_first.b64 pol, 1.0; "
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
      "[%0], [%1], %2, [%3], pol; }"
      ::"r"(dst), "l"(reinterpret_cast<uint64_t>(src)), "r"(bytes), "r"(bar)
      : "memory");
}
// K-major SWIZZLE_128B operand descriptor (8-row core groups 1024 B apart).
__device__ __forceinline__ uint64_t smem_desc(uint32_t addr) {
  return static_cast<uint64_t>((addr & 0x3FFFF) >> 4) | (1ull << 16) | (64ull << 32) |
         (1ull << 46) | (2ull << 61);
}
// Warp-uniform issue (hwdb hw28c): the whole warp runs the asm, elect.sync picks the
// issuing lane, so descriptors stay in uniform registers. One weight K block (64) =
// 4 MMAs of K16: A +8 TMEM columns (TS) or +32 B (SS), B +32 B per MMA.
__device__ __forceinline__ void mma4_ts(uint32_t d, uint32_t a_tmem, uint64_t b, uint32_t idesc,
                                        uint32_t acc) {
  asm volatile(
      "{ .reg .pred P, Q, T; .reg .b32 a1, a2, a3; .reg .b64 b1, b2, b3;\n"
      "elect.sync _|P, 0xffffffff; setp.ne.b32 Q, %4, 0; setp.eq.u32 T, %4, %4;\n"
      "add.u32 a1, %1, 8; add.u32 a2, %1, 16; add.u32 a3, %1, 24;\n"
      "add.u64 b1, %2, 2; add.u64 b2, %2, 4; add.u64 b3, %2, 6;\n"
      "@P tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, Q;\n"
      "@P tcgen05.mma.cta_group::1.kind::f16 [%0], [a1], b1, %3, T;\n"
      "@P tcgen05.mma.cta_group::1.kind::f16 [%0], [a2], b2, %3, T;\n"
      "@P tcgen05.mma.cta_group::1.kind::f16 [%0], [a3], b3, %3, T; }" ::"r"(d),
      "r"(a_tmem), "l"(b), "r"(idesc), "r"(acc));
}
__device__ __forceinline__ void mma4_ss(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc,
                                        uint32_t acc) {
  asm volatile(
      "{ .reg .pred P, Q, T; .reg .b64 a1, a2, a3, b1, b2, b3;\n"
      "elect.sync _|P, 0xffffffff; setp.ne.b32 Q, %4, 0; setp.eq.u32 T, %4, %4;\n"
      "add.u64 a1, %1, 2; add.u64 a2, %1, 4; add.u64 a3, %1, 6;\n"
      "add.u64 b1, %2, 2; add.u64 b2, %2, 4; add.u64 b3, %2, 6;\n"
      "@P tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, Q;\n"
      "@P tcgen05.mma.cta_group::1.kind::f16 [%0], a1, b1, %3, T;\n"
      "@P tcgen05.mma.cta_group::1.kind::f16 [%0], a2, b2, %3, T;\n"
      "@P tcgen05.mma.cta_group::1.kind::f16 [%0], a3, b3, %3, T; }" ::"r"(d),
      "l"(a), "l"(b), "r"(idesc), "r"(acc));
}
// Four consecutive weight blocks (TS: TMEM columns a + 32 q; SS: smem descriptor
// a + 1024 q) against their token blocks b0..b3: 16 MMAs under one elect, so the
// operands convert to uniform registers once per batch instead of per MMA.
__device__ __forceinline__ void mma16_ts(uint32_t d, uint32_t a, uint64_t b0, uint64_t b1,
                                         uint64_t b2, uint64_t b3, uint32_t idesc,
                                         uint32_t acc) {
  asm volatile(
      "{ .reg .pred P, Q, T; .reg .b32 ta; .reg .b64 tb;\n"
      "elect.sync _|P, 0xffffffff; setp.ne.b32 Q, %7, 0; setp.eq.u32 T, %7, %7;\n"
      "add.u32 ta, %1, 0; add.u64 tb, %2, 0; @P tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], tb, %6, Q;\n"
      "add.u32 ta, %1, 8; add.u64 tb, %2, 2; @P tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], tb, %6, T;\n"
      "add.u32 ta, %1, 16; add.u64 tb, %2, 4; @P tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], tb, %6, T;\n"
      "add.u32 ta, %1, 24; add.u64 tb, %2, 6; @P tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], tb, %6, T;\n"
      "add.u32 ta, %1, 32; add.u64 tb, %3, 0; @P tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], tb, %6, T;\n"
      "add.u32 ta, %1, 40; add.u64 tb, %3, 2; @P tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], tb, %6, T;\n"
      "add.u32 ta, %1, 48; add.u64 tb, %3, 4; @P tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], tb, %6, T;\n"
      "add.u32 ta, %1, 56; add.u64 tb, %3, 6; @P tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], tb, %6, T;\n"
      "add.u32 ta, %1, 64; add.u64 tb, %4, 0; @P tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], tb, %6, T;\n"
      "add.u32 ta, %1, 72; add.u64 tb, %4, 2; @P tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], tb, %6, T;\n"
      "add.u32 ta, %1, 80; add.u64 tb, %4, 4; @P tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], tb, %6, T;\n"
      "add.u32 ta, %1, 88; add.u64 tb, %4, 6; @P tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], tb, %6, T;\n"
      "add.u32 ta, %1, 96; add.u64 tb, %5, 0; @P tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], tb, %6, T;\n"
      "add.u32 ta, %1, 104; add.u64 tb, %5, 2; @P tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], tb, %6, T;\n"
      "add.u32 ta, %1, 112; add.u64 tb, %5, 4; @P tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], tb, %6, T;\n"
      "add.u32 ta, %1, 120; add.u64 tb, %5, 6; @P tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], tb, %6, T; }"
      ::"r"(d), "r"(a), "l"(b0), "l"(b1), "l"(b2), "l"(b3), "r"(idesc), "r"(acc));
}
__device__ __forceinline__ void mma16_ss(uint32_t d, uint64_t a, uint64_t b0, uint64_t b1,
                                         uint64_t b2, uint64_t b3, uint32_t idesc,
                                         uint32_t acc) {
  asm volatile(
      "{ .reg .pred P, Q, T; .reg .b64 ta, tb;\n"
      "elect.sync _|P, 0xffffffff; setp.ne.b32 Q, %7, 0; setp.eq.u32 T, %7, %7;\n"
      "add.u64 ta, %1, 0; add.u64 tb, %2, 0; @P tcgen05.mma.cta_group::1.kind::f16 [%0], ta, tb, %6, Q;\n"
      "add.u64 ta, %1, 2; add.u64 tb, %2, 2; @P tcgen05.mma.cta_group::1.kind::f16 [%0], ta, tb, %6, T;\n"
      "add.u64 ta, %1, 4; add.u64 tb, %2, 4; @P tcgen05.mma.cta_group::1.kind::f16 [%0], ta, tb, %6, T;\n"
      "add.u64 ta, %1, 6; add.u64 tb, %2, 6; @P tcgen05.mma.cta_group::1.kind::f16 [%0], ta, tb, %6, T;\n"
      "add.u64 ta, %1, 1024; add.u64 tb, %3, 0; @P tcgen05.mma.cta_group::1.kind::f16 [%0], ta, tb, %6, T;\n"
      "add.u64 ta, %1, 1026; add.u64 tb, %3, 2; @P tcgen05.mma.cta_group::1.kind::f16 [%0], ta, tb, %6, T;\n"
      "add.u64 ta, %1, 1028; add.u64 tb, %3, 4; @P tcgen05.mma.cta_group::1.kind::f16 [%0], ta, tb, %6, T;\n"
      "add.u64 ta, %1, 1030; add.u64 tb, %3, 6; @P tcgen05.mma.cta_group::1.kind::f16 [%0], ta, tb, %6, T;\n"
      "add.u64 ta, %1, 2048; add.u64 tb, %4, 0; @P tcgen05.mma.cta_group::1.kind::f16 [%0], ta, tb, %6, T;\n"
      "add.u64 ta, %1, 2050; add.u64 tb, %4, 2; @P tcgen05.mma.cta_group::1.kind::f16 [%0], ta, tb, %6, T;\n"
      "add.u64 ta, %1, 2052; add.u64 tb, %4, 4; @P tcgen05.mma.cta_group::1.kind::f16 [%0], ta, tb, %6, T;\n"
      "add.u64 ta, %1, 2054; add.u64 tb, %4, 6; @P tcgen05.mma.cta_group::1.kind::f16 [%0], ta, tb, %6, T;\n"
      "add.u64 ta, %1, 3072; add.u64 tb, %5, 0; @P tcgen05.mma.cta_group::1.kind::f16 [%0], ta, tb, %6, T;\n"
      "add.u64 ta, %1, 3074; add.u64 tb, %5, 2; @P tcgen05.mma.cta_group::1.kind::f16 [%0], ta, tb, %6, T;\n"
      "add.u64 ta, %1, 3076; add.u64 tb, %5, 4; @P tcgen05.mma.cta_group::1.kind::f16 [%0], ta, tb, %6, T;\n"
      "add.u64 ta, %1, 3078; add.u64 tb, %5, 6; @P tcgen05.mma.cta_group::1.kind::f16 [%0], ta, tb, %6, T; }"
      ::"r"(d), "l"(a), "l"(b0), "l"(b1), "l"(b2), "l"(b3), "r"(idesc), "r"(acc));
}
// One 128 x 64 BF16 K block (SWIZZLE_128B smem) -> 32 TMEM columns: 4 x 128x256b.
__device__ __forceinline__ void cp4_el(uint32_t taddr, uint64_t src) {
  asm volatile(
      "{ .reg .pred P; .reg .b32 t1, t2, t3; .reg .b64 s1, s2, s3;\n"
      "elect.sync _|P, 0xffffffff;\n"
      "add.u32 t1, %0, 8; add.u32 t2, %0, 16; add.u32 t3, %0, 24;\n"
      "add.u64 s1, %1, 2; add.u64 s2, %1, 4; add.u64 s3, %1, 6;\n"
      "@P tcgen05.cp.cta_group::1.128x256b [%0], %1;\n"
      "@P tcgen05.cp.cta_group::1.128x256b [t1], s1;\n"
      "@P tcgen05.cp.cta_group::1.128x256b [t2], s2;\n"
      "@P tcgen05.cp.cta_group::1.128x256b [t3], s3; }" ::"r"(taddr),
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

// MMA warp: K block j of the CTA multiplies weight block j (TMEM block j < nt, then
// shared-memory block j - nt) by token block j of the token ring (xs slots of xg
// blocks). Every counter and descriptor advances incrementally: the loop body is the
// barrier wait at a token group start, the four MMAs and the slot release.
struct MmaLoop {
  uint32_t d_tm, a_tm;
  uint64_t a_ss, x0;
  uint32_t xstep, idesc;
  int nk, nt;
  uint32_t xfull0, xempty0, res_bar, trdy0;
};
template <bool kStamp, bool kGroup4>
__device__ __forceinline__ void mma_loop(const Params& p, const MmaLoop& m, bool mma, int lane) {
  const int xg = p.xg, xs = p.xs;
  // TMEM readiness is confirmed 4 weight blocks at a time (one wait + one fence: the
  // fill commits each block, in order), so the MMAs run behind the fill. A whole
  // 4-block token group inside the confirmed blocks issues its 16 MMAs under one
  // elect (kGroup4: 4-block token groups, n32); otherwise one block (4 MMAs) at a
  // time, as soon as its tokens land.
  uint32_t a_tm = m.a_tm, xph = 0;
  uint64_t a_ss = m.a_ss, bcur = m.x0 + p.xslot[0] * (kSlot >> 4);
  int jx = 0, gs = 0, g = 0, ready = 0;
  for (int j = 0; j < m.nk;) {
    const bool tm = j < m.nt;
    if (tm && j >= ready) {
      ready = min(j + 4, m.nt);
      mbar_wait(m.trdy0 + 8 * (ready - 1), 0);
      asm volatile("tcgen05.fence::after_thread_sync;");
      if (kStamp && ready == m.nt && lane == 0) {
        p.stamps[blockIdx.x * kStampStride + 56] = globaltimer();
      }
    } else if (j == m.nt) {
      mbar_wait(m.res_bar, 0);
    }
    if (jx == 0) mbar_wait(m.xfull0 + 8 * gs, xph);
    if (kStamp && lane == 0 && j < 19) p.stamps[blockIdx.x * kStampStride + 32 + j] = clock64();
    const int nb = kGroup4 && jx == 0 && j + 4 <= (tm ? ready : m.nk) ? 4 : 1;
    if (mma) {
      if (kGroup4 && nb == 4) {
        if (tm) {
          mma16_ts(m.d_tm, a_tm, bcur, bcur + m.xstep, bcur + 2 * m.xstep, bcur + 3 * m.xstep,
                   m.idesc, j);
        } else {
          mma16_ss(m.d_tm, a_ss, bcur, bcur + m.xstep, bcur + 2 * m.xstep, bcur + 3 * m.xstep,
                   m.idesc, j);
        }
      } else if (tm) {
        mma4_ts(m.d_tm, a_tm, bcur, m.idesc, j);
      } else {
        mma4_ss(m.d_tm, a_ss, bcur, m.idesc, j);
      }
    }
    if (tm) {
      a_tm += 32 * nb;
    } else {
      a_ss += (kSlot >> 4) * nb;
    }
    j += nb;
    jx += nb;
    bcur += static_cast<uint64_t>(nb) * m.xstep;
    if (jx == xg || j == m.nk) {
      if ((g + xs) * xg < m.nk) commit_el(m.xempty0 + 8 * gs);
      jx = 0;
      ++g;
      if (++gs == xs) {
        gs = 0;
        xph ^= 1;
      }
      bcur = m.x0 + p.xslot[gs] * (kSlot >> 4);
    }
    if (kStamp && lane == 0 && j - nb < 19) {
      p.stamps[blockIdx.x * kStampStride + 8 + j - nb] = clock64();
    }
  }
}

// One CTA per (128-row weight tile, K split): blockIdx = tile * ks + split. 1 CTA/SM,
// the whole grid co-resident (host check). Weight K blocks of the CTA: the first nt
// live in TMEM (A operand of the TS MMA, filled by tcgen05.cp through the ring before
// the dependency), the last nr in shared memory (A operand of the SS MMA). Ring items:
// nt weight K blocks, then nk token K blocks (after griddepcontrol.wait).
template <bool NVFP4>
__global__ void __maxnreg__(96)  // launched with kThreads
    moe_front_kernel(const __grid_constant__ CUtensorMap tx,
                     const __grid_constant__ Params p) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  __shared__ float s_bias[kWarps * 32 * kPer];
  __shared__ float s_unb[kWarps * 32 * kPer];
  __shared__ uint64_t s_cand[kWarps * 32];
  __shared__ __align__(8) uint64_t s_bar[3 * kMaxNS + kMaxT + 2 + 2 * kMaxX];
  __shared__ uint32_t s_tmem;
  const Args& a = p.a;
  uint8_t* base = reinterpret_cast<uint8_t*>(
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const uint32_t res_s = smem_u32(base);
  const uint32_t ring_s = res_s + p.nr_max * kSlot;
  const uint32_t full0 = smem_u32(s_bar), empty0 = full0 + 8 * kMaxNS,
                 trdy0 = empty0 + 8 * kMaxNS, res_bar = trdy0 + 8 * kMaxT,
                 acc_bar = res_bar + 8, xfull0 = acc_bar + 8, xempty0 = xfull0 + 8 * kMaxX,
                 wfree0 = xempty0 + 8 * kMaxX;
  // Warp-uniform (CUTLASS canonical_warp_idx_sync): role branches and the values the
  // tcgen05 / TMA loops derive from it stay in uniform registers.
  const int warp = __shfl_sync(0xffffffffu, static_cast<int>(threadIdx.x >> 5), 0);
  const int lane = threadIdx.x & 31;
  // This SM's work item (weight tile, K split) from the host plan, which gives the SMs
  // of each die the items whose weights that die homes; -1: a spare SM (dead rows,
  // top-k rows and the grid counter only). One CTA per SM (grid = SMs, 1 CTA/SM).
  // The plan entry (a parameter-bank miss) is loaded first and used only after the
  // barrier init / TMEM allocation / CTA barrier below, which hide its latency.
  uint32_t smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  const int plan_item = smid < kMaxSmid ? p.smid_item[smid] : -1;
  const int n = a.n;

  if (threadIdx.x == 0) {
    asm volatile("griddepcontrol.launch_dependents;");
    STAMP(0);
    for (int s = 0; s < p.ns; ++s) {
      mbar_init(full0 + 8 * s, 1);
      mbar_init(empty0 + 8 * s, 1);
      mbar_init(wfree0 + 8 * s, 1);
    }
    for (int j = 0; j < kMaxT; ++j) mbar_init(trdy0 + 8 * j, 1);
    for (int s = 0; s < p.xs; ++s) {
      mbar_init(xfull0 + 8 * s, 1);
      mbar_init(xempty0 + 8 * s, 1);
    }
    mbar_init(res_bar, 1);
    mbar_init(acc_bar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (threadIdx.x == 32) {
    asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&tx)) : "memory");
  }
  if (warp == 1) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" ::"r"(
            smem_u32(&s_tmem)));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  asm volatile("tcgen05.fence::before_thread_sync;");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;");
  // D: columns [dcol, dcol + n); weight K block j < nt: acol + 32 j. Warp-uniform.
  const uint32_t tmem = __shfl_sync(0xffffffffu, s_tmem, 0);
  const int item = __shfl_sync(0xffffffffu, plan_item, 0);
  const bool work = item >= 0;
  const int tile = work ? item / p.ks : 0, split = work ? item % p.ks : 0;
  const int kb0 = split * p.kb / p.ks, nk = (split + 1) * p.kb / p.ks - kb0;
  const int nt = min(nk, p.nt_max), nr = nk - nt;
  const int wrow = tile * 128;
  // The item's K blocks in the image: 16 KiB [128][64] SWIZZLE_128B tiles, items in
  // order, die 1's items after a gap that starts them on their own 2 MiB chunk.
  const uint8_t* wblk =
      p.wimg + (int64_t{tile} * p.kb + kb0 + (item >= p.item_split ? p.item_pad : 0)) * kSlot;
  if (threadIdx.x == 0) {
    if (p.stamps) p.stamps[blockIdx.x * kStampStride + 31] = item + 1;
    // A bulk copy may complete before this expectation (the phase needs the arrive).
    if (work && nr > 0) mbar_expect_tx(res_bar, nr * kSlot);
  }

  // TMA issue: lane 0 of warps 0, 3 and 4-7 (the epilogue warps idle until the
  // accumulator is done). Bulk ops from one issuing warp complete about one at a
  // time (hwdb hw51), so each ring slot is owned by one issuer and the issuers run
  // in parallel. Weights never depend on the predecessor: the TMEM fill stream starts
  // at entry, beside the AttnRes reduce, and the shared-memory-resident blocks follow.
  // Ring items run in rounds of ns slots (tokens: xs slots); issuer u owns the slots
  // s = u (mod kIssuers) and visits them in item order, counters advance without a
  // division.
  if (work && (warp == 0 || warp >= 3) && lane == 0) {
    constexpr int kIssuers = 6;
    const int u = warp == 0 ? 0 : warp - 2;
    const int ns = p.ns;
    uint32_t ph = 0;
    for (int i0 = 0; i0 < nt; i0 += ns, ph ^= 1) {
      for (int slot = u; slot < ns && i0 + slot < nt; slot += kIssuers) {
        if (i0 > 0) mbar_wait(empty0 + 8 * slot, ph ^ 1);
        mbar_expect_tx(full0 + 8 * slot, kSlot);
        bulk_g2s(ring_s + slot * kSlot, wblk + (i0 + slot) * kSlot, kSlot, full0 + 8 * slot);
      }
    }
    // The shared-memory-resident blocks (the MMAs' last K blocks) follow the TMEM
    // fill stream instead of opening it: issued at entry they were a 13-20 MB burst
    // on top of the ring's first round beside the AttnRes reduce (in-step: +1.1 us
    // on the reduce). res_bar's expect_tx is set before the CTA barrier (thread 0).
    for (int r = u; r < nr; r += kIssuers)
      bulk_g2s(res_s + r * kSlot, wblk + (nt + r) * kSlot, kSlot, res_bar);
    // Token slot gs takes over W ring slot xslot[gs] as soon as that slot's last
    // weight block is in TMEM (its empty barrier), so the tokens follow the fill and
    // the MMAs run behind it, not after the whole fill.
    if (u < p.xs) {
      asm volatile("griddepcontrol.wait;" ::: "memory");
      if (u == 0) STAMP(1);
      // One op = xg token K blocks (3-D map, [xg][n][128 B] SWIZZLE_128B tiles); the
      // last op of the split may run past it (unused blocks, zero fill past K).
      const int xg = p.xg, xs = p.xs, ng = (nk + xg - 1) / xg;
      const int gbytes = xg * n * 128;
      uint32_t xph = 0;
      for (int g0 = 0; g0 < ng; g0 += xs, xph ^= 1) {
        for (int gs = u; gs < xs && g0 + gs < ng; gs += kIssuers) {
          const int ws = p.xslot[gs];
          if (g0 > 0) {
            mbar_wait(xempty0 + 8 * gs, xph ^ 1);
          } else if (p.xfree[gs]) {
            mbar_wait(wfree0 + 8 * ws, 0);
          }
          mbar_expect_tx(xfull0 + 8 * gs, gbytes);
          tma_3d(ring_s + ws * kSlot, &tx, 0, kb0 + (g0 + gs) * xg, xfull0 + 8 * gs);
        }
      }
    }
  }
  __syncwarp();  // the issuing lane rejoins its warp (tcgen05.ld below is .aligned)
  if (!work) {
    // Spare SM: the router bias for a top-k row (read after a CTA barrier).
    for (int e = threadIdx.x; e < a.num_logits; e += kThreads) s_bias[e] = a.bias[e];
  } else if (warp == 2) {
    // TMEM fill: ring weight block i -> TMEM columns [acol + 32 i, acol + 32 i + 32).
    // Commit only where a consumer waits (a slot reused by the ring, the last block).
    const int ns = p.ns;
    uint32_t ta = tmem + p.acol, ph = 0;
    int slot = 0;
    for (int i = 0; i < nt; ++i) {
      mbar_wait(full0 + 8 * slot, ph);
      if (i == nt - 1 && lane == 0) STAMP(28);  // the last TMEM weight block landed
      cp4_el(ta, smem_desc(ring_s + slot * kSlot));
      commit_el(trdy0 + 8 * i);  // block i in TMEM: its MMAs may run
      // The slot is free: for the next weight block of the ring, or after its last
      // weight block for tokens (one commit on a barrier of its own: no parity alias).
      commit_el((i + ns < nt ? empty0 : wfree0) + 8 * slot);
      ta += 32;
      if (++slot == ns) {
        slot = 0;
        ph ^= 1;
      }
    }
  } else if (warp == 1) {
    // kind::f16: BF16 A/B, FP32 D, K-major, M = 128 weight rows, N = n tokens.
    const uint32_t idesc = (1u << 4) | (1u << 7) | (1u << 10) |
                           (static_cast<uint32_t>(n >> 3) << 17) | (8u << 24);
    mbar_wait(xfull0, 0);
    if (lane == 0) STAMP(6);
    __syncwarp();
    const MmaLoop m{tmem + p.dcol,
                    tmem + p.acol,
                    smem_desc(res_s),
                    smem_desc(ring_s),
                    static_cast<uint32_t>(n) * 8,  // n x 128 B per token K block, 16 B units
                    idesc,
                    nk,
                    nt,
                    xfull0,
                    xempty0,
                    res_bar,
                    trdy0};
    if (p.stamps && !(p.debug & 4)) {
      if (p.xg == 4) {
        mma_loop<true, true>(p, m, !(p.debug & 1), lane);
      } else {
        mma_loop<true, false>(p, m, !(p.debug & 1), lane);
      }
    } else if (p.xg == 4) {
      mma_loop<false, true>(p, m, !(p.debug & 1), lane);
    } else {
      mma_loop<false, false>(p, m, !(p.debug & 1), lane);
    }
    commit_el(acc_bar);
  } else if (warp >= 4) {
    // Router correction bias (a weight, read by the top-k after a CTA barrier): the
    // epilogue warps load it while the accumulator runs, off the W stream's start.
    for (int e = threadIdx.x - 128; e < a.num_logits; e += 128) s_bias[e] = a.bias[e];
    // D (TMEM lane = weight row, column = token) -> this CTA's FP32 partial.
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (p.debug & 2) {
      if (threadIdx.x == 128) mbar_wait_sleep(acc_bar, 0);
      asm volatile("bar.sync 1, 128;" ::: "memory");
      mbar_wait(acc_bar, 0);  // completed: acquire for every thread
    } else {
      mbar_wait_sleep(acc_bar, 0);
    }
    if (threadIdx.x == 128) STAMP(7);
    asm volatile("tcgen05.fence::after_thread_sync;");
    const int q = warp & 3, row = q * 32 + lane;
    const uint32_t ta = tmem + p.dcol + (static_cast<uint32_t>(q * 32) << 16);
    float* dst = p.part + int64_t{item} * n * 128 + row;
    for (int c = 0; c < n; c += 16) {
      float v[16];
      tmem_ld16(ta + c, v);
#pragma unroll
      for (int j = 0; j < 16; ++j) __stcg(dst + (c + j) * 128, v[j]);
    }
  }
  const int row = smid < kMaxSmid ? p.smid_row[smid] : kMaxSmid;
  // The tail below is long straight-line code run once per layer. In a decode step
  // the other kernels evict it from L2 between two layers, and fetching it cold from
  // HBM made the tail 3-4x slower in-step than warm (route_block 21.6K vs 5.9K clk;
  // bench with an L2 + instruction-cache thrash: 24.6K). A spare SM (no weight item,
  // no top-k row) runs this same code first, before the dependency, as a latent item's
  // sum + quantization and row 0's top-k with every global write and wait off ("dry"),
  // so the other CTAs' tails fetch it from L2.
  const bool dry = !work && row >= n && !(p.debug & 8);
  // With two or more spare SMs (rows items, items + 1, ...) the first warms the latent
  // sum + quantization, the row's logits and its scoring + sort, the second the rest.
  const int items = p.tiles * p.ks;
  const int warm = !dry                                    ? 0
                   : static_cast<int>(gridDim.x) - items < 2 ? 3
                   : ((row - items) & 1)                   ? 2
                                                           : 1;
  const int ttile = dry ? p.rtiles : tile, tsplit = dry ? 0 : split;
  const int trow = dry ? 0 : row;
  if (!dry) {
    // Every thread past the dependency before any global write below.
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("tcgen05.fence::before_thread_sync;");
    __syncthreads();
    // A top-k CTA holds its TMEM to its end; the others free it now.
    if (warp == 1 && row >= n) {
      asm volatile("tcgen05.fence::after_thread_sync;");
      asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" ::"r"(tmem));
    }
    if (work && threadIdx.x == 0) {
      STAMP(2);
      // Release (cumulative: the partial stores of the whole CTA precede it through the
      // CTA barrier above), then only the latent CTAs wait for their tile: they stage it.
      unsigned* const c = p.cnt + kTileCnt + kLine * tile;
      asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(c) : "memory");
      if (tile >= p.rtiles) {
        while (ld_acquire(c) < static_cast<unsigned>(p.ks)) __nanosleep(32);
      }
      STAMP(3);
    }
  }
  __syncthreads();
  {
    // Split-K reduce of token rows [t0, t1) of this tile, partials in rank order.
    const int t0 = tsplit * n / p.ks, t1 = (tsplit + 1) * n / p.ks;
    const float* pt = p.part + int64_t{ttile} * p.ks * n * 128;
    const int64_t stride = int64_t{n} * 128;
    // Router tiles: the top-k CTA of each row sums its logits itself (below).
    if ((work || (dry && warm != 2)) && ttile >= p.rtiles) {
      // Latent: BF16 round + the stager's quantization of each block of this tile.
      constexpr int vec = NVFP4 ? 16 : 32;
      constexpr int per = 128 / vec;
      const int twrow = ttile * 128;
      const int blk0 = (twrow - a.num_logits) / vec;
      for (int f = threadIdx.x; f < (t1 - t0) * per; f += kThreads) {
        const int t = t0 + f / per, b = f % per;
        float4 in[8];
#pragma unroll
        for (int q = 0; q < vec / 4; ++q)
          in[q] = tile_sum(pt, stride, p.ks, int64_t{t} * 128 + b * vec + 4 * q);
        if (p.yfull && !dry) {
          float4* dst = reinterpret_cast<float4*>(p.y + int64_t{t} * a.ldy + twrow + b * vec);
#pragma unroll
          for (int q = 0; q < vec / 4; ++q) dst[q] = in[q];
        }
        if constexpr (NVFP4) {
          quant_nvfp4(a, t, blk0 + b, in, dry);
        } else {
          quant_mxfp8(a, t, blk0 + b, in, dry);
        }
      }
    }
  }
  if (p.stamps && !dry) {  // microbench: this CTA's latent sum + quantization done
    __syncthreads();
    if (threadIdx.x == 0) STAMP(57);
  }
  if (trow < n) {
    // Every router tile's partials stored: sum this row's logits in rank order
    // (bitwise the tile sum) from the partials, one counter round trip fewer.
    if (threadIdx.x < p.rtiles && !dry) {
      const unsigned* c = p.cnt + kTileCnt + kLine * threadIdx.x;
      while (ld_acquire(c) < static_cast<unsigned>(p.ks)) __nanosleep(32);
    }
    __syncthreads();
    if (threadIdx.x == 0 && !dry) {
      STAMP(4);
      if (p.stamps) p.stamps[blockIdx.x * kStampStride + 27] = clock64();
    }
    const int64_t stride = int64_t{n} * 128;
    // Every partial load is issued before the first store (one L2 round trip, not
    // one per expert slot), then the rank-order sums.
    float v[kPer][kMaxKs];
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
      const int e = j * kThreads + threadIdx.x;
      if (e < a.num_logits && warm != 2) {
        const float* pe = p.part + (int64_t{e / 128} * p.ks * n + trow) * 128 + (e % 128);
#pragma unroll
        for (int q = 0; q < kMaxKs; ++q)
          if (q < p.ks) v[j][q] = __ldcg(pe + q * stride);
      }
    }
    float lg[kPer];
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
      const int e = j * kThreads + threadIdx.x;
      float acc = 0.0F;
      if (e < a.num_logits && warm != 2) {
        acc = v[j][0];
#pragma unroll
        for (int q = 1; q < kMaxKs; ++q)
          if (q < p.ks) acc += v[j][q];
        if (!dry) p.y[int64_t{trow} * a.ldy + e] = acc;
      }
      lg[j] = acc;
    }
    route_block(a, trow, lg, s_bias, s_unb, s_cand,  // warps 1.. return inside
                p.stamps && !dry ? p.stamps + blockIdx.x * kStampStride + 58 : nullptr, warm);
    if (warp == 1 && !dry) {
      asm volatile("tcgen05.fence::after_thread_sync;");
      asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" ::"r"(tmem));
    }
  }
  if (dry) {  // the spare SM's own work below follows the dependency
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (warp == 1) {
      asm volatile("tcgen05.fence::after_thread_sync;");
      asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" ::"r"(tmem));
    }
  }
  // Dead rows [n, capacity): routes -1 only (spread over the grid).
  {
    const int64_t lo = int64_t{n} * a.topk, hi = int64_t{a.capacity} * a.topk;
    for (int64_t i = lo + int64_t{blockIdx.x} * kThreads + threadIdx.x; i < hi;
         i += int64_t{gridDim.x} * kThreads)
      a.idx[i] = -1;
  }
  if (threadIdx.x == 0) {
    STAMP(5);
    // The last CTA of the grid (every CTA is past its waits) resets the counters.
    if (atomicAdd(p.cnt + 2 * kLine, 1u) == gridDim.x - 1) {
      p.cnt[2 * kLine] = 0;
      for (int t = 0; t < p.tiles; ++t) p.cnt[kTileCnt + kLine * t] = 0;
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

struct Plan {
  int na, nt_max, nr_max, ns, smem, xs, xg;
  signed char xslot[kMaxNS], xfree[kMaxNS];
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
  // Token ops of xg K blocks, <= 16 KiB: one W ring slot each.
  pl.xg = std::max(1, std::min(4, kSlot / (n * 128)));
  pl.xs = std::min({kMaxX, (nk_max + pl.xg - 1) / pl.xg, pl.ns});
  // Token slots take the W slots in the order their last weight block leaves for TMEM
  // (the fill visits items in order; every CTA has nt_max TMEM blocks).
  const int nt = pl.nt_max;
  int last[kMaxNS], order[kMaxNS];
  for (int s = 0; s < pl.ns; ++s) {
    last[s] = s < nt ? s + pl.ns * ((nt - 1 - s) / pl.ns) : -1;
    order[s] = s;
  }
  std::sort(order, order + pl.ns, [&](int x, int y) { return last[x] < last[y]; });
  for (int g = 0; g < pl.xs; ++g) {
    const int s = order[g];
    pl.xslot[g] = static_cast<signed char>(s);
    pl.xfree[g] = static_cast<signed char>(last[s] >= 0);
  }
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
void moe_front(const torch::Tensor& w, const torch::Tensor& wimg, const torch::Tensor& plan,
               const torch::Tensor& h, torch::Tensor& y, torch::Tensor& cnt, torch::Tensor& part, const torch::Tensor& bias,
               torch::Tensor& x, torch::Tensor& sf, torch::Tensor& idx, torch::Tensor& sw,
               torch::Tensor& err, const c10::optional<torch::Tensor>& pad,
               int64_t num_logits, int64_t num_experts, bool renormalize,
               double routed_scaling_factor, bool nvfp4, double norm_const, int64_t ks,
               int64_t ns, const torch::Tensor& stamps,
               const c10::optional<torch::Tensor>& l2p,
               const c10::optional<torch::Tensor>& replicas,
               const c10::optional<torch::Tensor>& load,
               const c10::optional<torch::Tensor>& record,
               const c10::optional<torch::Tensor>& unpadded) {
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
  const int64_t tiles = rows_w / 128, kb = k / 64, items = tiles * ks;
  TORCH_CHECK(cnt.scalar_type() == at::kInt && cnt.is_contiguous() &&
                  cnt.numel() >= kTileCnt + kLine * tiles,
              "moe_front: counters");
  TORCH_CHECK(part.scalar_type() == at::kFloat && part.is_contiguous() &&
                  part.numel() >= items * n * 128,
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
  const int64_t grid = sms;  // one CTA per SM (plan: SM -> item)
  TORCH_CHECK(plan.device().is_cpu() && plan.scalar_type() == at::kInt && plan.is_contiguous() &&
                  plan.numel() == 2 * kMaxSmid + 3 && plan[2 * kMaxSmid].item<int>() == items &&
                  wimg.is_cuda() && wimg.scalar_type() == at::kByte &&
                  wimg.numel() >= (kb * tiles + plan[2 * kMaxSmid + 2].item<int>()) * kSlot &&
                  reinterpret_cast<uintptr_t>(wimg.data_ptr()) % 1024 == 0,
              "moe_front: weight image / plan (front_image)");
  TORCH_CHECK(ks >= 1 && ks <= kMaxKs && ks <= kb && grid >= n && items <= grid && pl.nr_max <= kMaxR &&
                  kb / ks >= pl.nt_max &&
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
  route_eplb(a, num_logits, l2p, replicas, load, record, unpadded);
  a.splits = 1;  // the front sums its own split-K partials
  a.split_stride = 0;
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
  for (int g = 0; g < kMaxNS; ++g) {
    p.xslot[g] = g < pl.xs ? pl.xslot[g] : 0;
    p.xfree[g] = g < pl.xs ? pl.xfree[g] : 0;
  }
  p.yfull = y.size(1) >= rows_w ? 1 : 0;
  p.wimg = wimg.data_ptr<uint8_t>();
  p.tiles = (int)tiles;
  const int* pp = plan.data_ptr<int>();
  for (int s = 0; s < kMaxSmid; ++s) {
    p.smid_item[s] = static_cast<short>(pp[s]);
    p.smid_row[s] = static_cast<short>(pp[kMaxSmid + s]);
  }
  p.item_split = pp[2 * kMaxSmid + 1];
  p.item_pad = pp[2 * kMaxSmid + 2];
  CUtensorMap tx;
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
  const cudaError_t rc = nvfp4 ? cudaLaunchKernelEx(&cfg, moe_front_kernel<true>, tx, p)
                              : cudaLaunchKernelEx(&cfg, moe_front_kernel<false>, tx, p);
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
void moe_front(const torch::Tensor& w, const torch::Tensor& wimg, const torch::Tensor& plan,
               const torch::Tensor& h, torch::Tensor& y, torch::Tensor& cnt, torch::Tensor& part, const torch::Tensor& bias,
               torch::Tensor& x, torch::Tensor& sf, torch::Tensor& idx, torch::Tensor& sw,
               torch::Tensor& err, const c10::optional<torch::Tensor>& pad,
               int64_t num_logits, int64_t num_experts, bool renormalize,
               double routed_scaling_factor, bool nvfp4, double norm_const, int64_t ks,
               int64_t ns, const torch::Tensor& stamps,
               const c10::optional<torch::Tensor>& l2p,
               const c10::optional<torch::Tensor>& replicas,
               const c10::optional<torch::Tensor>& load,
               const c10::optional<torch::Tensor>& record,
               const c10::optional<torch::Tensor>& unpadded);
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
    return "#include <cuda.h>\n#include <algorithm>\n" + head + device + sep + tail + host


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


_SLOT = 128 * 64 * 2  # one [128 rows][64 K] BF16 tile (kSlot)
_MAX_SMID = 512  # kMaxSmid
_CHUNK_SLOTS = (2 << 20) // _SLOT  # tiles per 2 MiB chunk (the die homing unit)


@functools.cache
def _front_plan(device: int, items: int, item_slots: tuple, router_items: int) -> tuple:
    """(plan, split item, die-0 chunks, die-1 chunks): every SM's work item, the die-0
    SMs taking items [0, split) and the die-1 SMs the rest (one CTA per SM), every SM's
    top-k row (router items first, then latent items, spare SMs last: they stay
    row-free and warm the tail's code), and the 2 MiB chunks of each die's image region.
    item_slots[i]: item i's 16 KiB tiles."""
    from ..common import dies

    table = dies.sm_domains(device)
    by_die = [[s for s, d in enumerate(table) if d == die] for die in (0, 1)]
    # Spare SMs (no item) on die 0, beside the top-k rows of the router items: they warm
    # the tail's code into the L2 those rows fetch it from.
    spare = max(0, sum(len(b) for b in by_die) - items)
    split = min(len(by_die[0]) - min(spare, len(by_die[0])), items)
    if items - split > len(by_die[1]) or len(table) > _MAX_SMID:
        raise Unsupported("moe", None, f"moe_front: {items} items on {len(table)} SMs")
    plan = torch.full((2 * _MAX_SMID + 3,), -1, dtype=torch.int32)
    for i, s in enumerate(by_die[0][:split]):
        plan[s] = i
    for i, s in enumerate(by_die[1][: items - split]):
        plan[s] = split + i
    sms = [s for s, d in enumerate(table) if d >= 0]
    item = {s: int(plan[s]) for s in sms}
    order = sorted(sms, key=lambda s: (
        0 if 0 <= item[s] < router_items else 2 if item[s] < 0 else 1, item[s]))
    plan[_MAX_SMID : 2 * _MAX_SMID] = _MAX_SMID
    for r, s in enumerate(order):
        plan[_MAX_SMID + s] = r
    slots0 = sum(item_slots[:split])
    chunks0 = -(-slots0 // _CHUNK_SLOTS)
    chunks1 = -(-sum(item_slots[split:]) // _CHUNK_SLOTS)
    plan[2 * _MAX_SMID] = items
    plan[2 * _MAX_SMID + 1] = split
    plan[2 * _MAX_SMID + 2] = chunks0 * _CHUNK_SLOTS - slots0  # die 1 on its own chunk
    return plan, split, chunks0, chunks1


def front_image(
    packed: torch.Tensor, ks: int, num_logits: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """(image, plan) of the front's weight stream: the [128 rows x 64 K] tiles of
    packed ([router | latent down], BF16) in the TMEM fill's 128-byte-swizzled
    shared-memory layout (prewait_gemm.pack_weight), work item (tile, K split) after
    work item; the items the die-0 SMs run live on die 0, the rest on die 1 (from the
    next 2 MiB chunk), so every SM streams die-local weights. plan: the kernel's
    SM -> item table (CPU int32)."""
    from ...prewait_gemm import pack_weight
    from ..common import dies

    rows, k = packed.shape
    kb, tiles = k // 64, rows // 128
    item_slots = tuple(
        (s + 1) * kb // ks - s * kb // ks for _ in range(tiles) for s in range(ks)
    )
    dev = packed.device.index or 0
    plan, split, chunks0, chunks1 = _front_plan(
        dev, tiles * ks, item_slots, num_logits // 128 * ks
    )
    flat = pack_weight(packed).view(torch.uint8).view(-1, _SLOT)
    slots0 = sum(item_slots[:split])
    nbytes = (chunks0 + chunks1) * _CHUNK_SLOTS * _SLOT
    if dies.domain_count(dev) > 1:
        image = dies.localized_empty(nbytes, [0] * chunks0 + [1] * chunks1, packed.device)
    else:
        image = torch.empty(nbytes, dtype=torch.uint8, device=packed.device)
    view = image.view(-1, _SLOT)
    view[:slots0].copy_(flat[:slots0])
    start1 = slots0 + int(plan[2 * _MAX_SMID + 2])
    view[start1 : start1 + flat.shape[0] - slots0].copy_(flat[slots0:])
    return image, plan


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
        splits = {config_for(n, dev)[0] for n in rows}
        if len(splits) != 1:
            raise Unsupported("moe", None, f"moe_front: K splits {sorted(splits)} per rows")
        ks = splits.pop()
        self.part = torch.empty(tiles * ks * n_max * 128, dtype=torch.float32, device=dev)
        # Die-local weight stream: the tile image this layer's front reads (the stock
        # linears keep viewing the row-major packed).
        self.wimg, self.plan = front_image(packed, ks, self.num_logits)
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
            self.packed, self.wimg, self.plan, h, self.y, self.cnt, self.part,
            self.mlp.gate.e_score_correction_bias.data,
            session.x.view(torch.uint8), session.x_sf.view(torch.uint8),
            session.topk_idx, session.topk_weights, rt.route_error, pad,
            self.num_logits, p.num_experts, self.route.renormalize, self.route.scale,
            p.quant_kind == "nvfp4", float(p.input_norm_const),
            *cfg,
            self.no_stamps if stamps is None else stamps,
            *self.route._eplb_args(),
        )  # fmt: skip
        return rt, session, n

    def launch(self, staged, output_rmsnorm=None, shared=None):
        return self.route.launch(staged, output_rmsnorm, shared)
