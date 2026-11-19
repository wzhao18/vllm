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

constexpr int kThreads = kWarps * 32;  // 256: w0/w2/w3 weight TMA, w1 MMA, w5 tokens, w4-7 TMEM
constexpr int kWBox = 128 * 128;       // one 128 x 64 BF16 weight box
// Tokens are the MMA M operand (128 rows; rows >= n are the TMA's zero fill) and a
// 128-row weight tile the N operand. A tcgen05 MMA occupies the tensor pipe for a
// time set by M and its operand bytes, not by tiny N: weights-as-M with N = n tokens
// (the swap-AB form) ran one M128 N32 K16 MMA per ~250 clk, 4x below this form.
constexpr int kXRows = 128;
constexpr int kMaxW = 24;
constexpr int kMaxX = 8;
constexpr int kMaxC = 8;
// Diagnostic stamps per CTA: [0, 8) phases, [8, 72) token-ready per K block, [72, 136)
// weight-ready per K block, [136, 200) weight-issue per weight slot group.
constexpr int kStampStride = 328;

struct Params {
  Args a;            // route_stage's arguments (y = router logits rows, ldy)
  float* y;          // [n, ldy] FP32 router logits
  unsigned* cnt;     // [1] router CTAs done [2] CTAs finished (written after griddepcontrol.wait only)
  unsigned long long* stamps;  // [grid][8] globaltimer or null
  int k, bn, tiles, rtiles, kb, csize, wst, xst, nprod, xg;
  int wg;  // K blocks per weight slot (one barrier, wg x tr x 128 B)
  int tr;  // weight rows per tile = MMA N (128 or 256): x is read once per tr rows
  int rows_w;  // E + L
  int debug;  // microbench probes only (0 in production): bit 0 skips the MMAs
  int yfull;  // test hook: y also holds the FP32 latent columns (y.size(1) >= E + L)
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
// For long waits: one polling thread that backs off, so the waiting warps do not take
// shared-memory/barrier bandwidth from the issuing threads (measured: 128 spinning
// threads stretched every TMA issue and MMA round to ~0.3-0.45 us).
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
    __nanosleep(128);
  }
}
__device__ __forceinline__ void tma_2d(uint32_t dst, const CUtensorMap* map, int c0,
                                       int c1, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(bar)
      : "memory");
}
// 3-D maps {64 columns, rows, K blocks}: one op = box[2] consecutive K blocks of a
// row range, stored as consecutive [rows][64] SWIZZLE_128B tiles.
__device__ __forceinline__ void tma_3d(uint32_t dst, const CUtensorMap* map, int row,
                                       int kb, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(0), "r"(row), "r"(kb), "r"(bar)
      : "memory");
}
__device__ __forceinline__ void tma_3d_ef(uint32_t dst, const CUtensorMap* map, int row,
                                          int kb, uint32_t bar) {
  asm volatile(
      "{ .reg .b64 pol; createpolicy.fractional.L2::evict_first.b64 pol, 1.0; "
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      ".L2::cache_hint [%0], [%1, {%2, %3, %4}], [%5], pol; }" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(0), "r"(row), "r"(kb), "r"(bar)
      : "memory");
}
__device__ __forceinline__ void tma_2d_ef(uint32_t dst, const CUtensorMap* map, int c0,
                                          int c1, uint32_t bar) {
  asm volatile(
      "{ .reg .b64 pol; createpolicy.fractional.L2::evict_first.b64 pol, 1.0; "
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      ".L2::cache_hint [%0], [%1, {%2, %3}], [%4], pol; }" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(bar)
      : "memory");
}
// K-major SWIZZLE_128B operand descriptor (8-row core groups 1024 B apart).
__device__ __forceinline__ uint64_t smem_desc(uint32_t addr) {
  return static_cast<uint64_t>((addr & 0x3FFFF) >> 4) | (1ull << 16) | (64ull << 32) |
         (1ull << 46) | (2ull << 61);
}
__device__ __forceinline__ void mma_bf16(uint32_t tmem, uint64_t a, uint64_t b,
                                         uint32_t idesc, uint32_t acc) {
  asm volatile(
      "{ .reg .pred p; setp.ne.b32 p, %4, 0; "
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p; }" ::"r"(tmem),
      "l"(a), "l"(b), "r"(idesc), "r"(acc));
}
// Warp-uniform issue (hwdb hw28c): the whole warp executes the asm, elect.sync picks
// the issuing lane inside it, so the descriptors stay in uniform registers (a
// lane-0 branch turns them into an R2UR waterfall: ~88 clk per MMA, ~250 measured here).
__device__ __forceinline__ void mma_bf16_el(uint32_t tmem, uint64_t a, uint64_t b,
                                            uint32_t idesc, uint32_t acc) {
  asm volatile(
      "{ .reg .pred P, Q; elect.sync _|P, 0xffffffff; setp.ne.b32 Q, %4, 0; "
      "@P tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, Q; }" ::"r"(tmem),
      "l"(a), "l"(b), "r"(idesc), "r"(acc));
}
__device__ __forceinline__ void mma_commit_el(uint32_t bar) {
  asm volatile(
      "{ .reg .pred P; elect.sync _|P, 0xffffffff; "
      "@P tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0]; }" ::"r"(
          bar)
      : "memory");
}
__device__ __forceinline__ void mma_commit(uint32_t bar) {
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(
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
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;"
               ::: "memory");
}
__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ uint32_t mapa(uint32_t addr, uint32_t rank) {
  uint32_t o;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(o) : "r"(addr), "r"(rank));
  return o;
}
__device__ __forceinline__ float4 ld_cluster_f4(uint32_t addr) {
  float4 v;
  asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
               : "r"(addr)
               : "memory");
  return v;
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
// Per-K-block stamps: SM clock (cheap); phases: globaltimer. Slot 7 / 199 hold the
// clock at entry / end for the conversion.
#define STAMPC(i) \
  do { \
    if (p.stamps) p.stamps[blockIdx.x * kStampStride + (i)] = clock64(); \
  } while (0)
#define STAMP(i) \
  do { \
    if (p.stamps) p.stamps[blockIdx.x * kStampStride + (i)] = globaltimer(); \
  } while (0)

__host__ __device__ inline int ring_w_bytes(int bn, int wst, int tr) {
  return wst * tr * 128 > bn * tr * 4 ? wst * tr * 128 : bn * tr * 4;
}

// Sum of the C ranks' partial float4 at `off` (rank order, deterministic).
__device__ __forceinline__ float4 rank_sum(uint32_t off, int csize) {
  float4 v[kMaxC];
#pragma unroll
  for (int c = 0; c < kMaxC; ++c)
    if (c < csize) v[c] = ld_cluster_f4(mapa(off, c));
  float4 acc = v[0];
#pragma unroll
  for (int c = 1; c < kMaxC; ++c) {
    if (c < csize) {
      acc.x += v[c].x;
      acc.y += v[c].y;
      acc.z += v[c].z;
      acc.w += v[c].w;
    }
  }
  return acc;
}

// <= 96 registers per thread (24 K per CTA): room beside three AttnRes CTAs (13 K each).
template <bool NVFP4>
__global__ void __maxnreg__(96)  // launched with kThreads
    moe_front_kernel(const __grid_constant__ CUtensorMap tw,
                     const __grid_constant__ CUtensorMap tx,
                     const __grid_constant__ Params p) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  __shared__ float s_bias[kWarps * 32 * kPer];
  __shared__ float s_unb[kWarps * 32 * kPer];
  __shared__ uint64_t s_cand[kWarps * 32];
  const Args& a = p.a;
  uint8_t* ring = reinterpret_cast<uint8_t*>(
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const int xbytes = kXRows * 128;
  const int wring = ring_w_bytes(p.bn, p.wst * p.wg, p.tr);
  const int wbox = p.tr * 128;  // one tr x 64 BF16 weight K block
  uint64_t* bars = reinterpret_cast<uint64_t*>(ring + wring + p.xst * p.xg * xbytes);
  // fullW[kMaxW] emptyW[kMaxW] fullX[kMaxX] emptyX[kMaxX] acc
  uint32_t* slot = reinterpret_cast<uint32_t*>(bars + 2 * kMaxW + 2 * kMaxX + 1);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const uint32_t rank = cluster_rank();
  const uint32_t ring_s = smem_u32(ring), xring_s = ring_s + wring;
  const uint32_t fw0 = smem_u32(bars), ew0 = fw0 + 8 * kMaxW, fx0 = ew0 + 8 * kMaxW,
                 ex0 = fx0 + 8 * kMaxX, acc0 = ex0 + 8 * kMaxX;

  if (threadIdx.x == 32) {
    // Descriptor prefetch (as CUTLASS): every TMA below names one of these two maps.
    asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&tw)) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&tx)) : "memory");
  }
  if (threadIdx.x == 0) {
    asm volatile("griddepcontrol.launch_dependents;");
    STAMP(0);
    STAMPC(7);
    for (int s = 0; s < p.wst; ++s) {
      mbar_init(fw0 + 8 * s, 1);
      mbar_init(ew0 + 8 * s, 1);
    }
    for (int s = 0; s < p.xst; ++s) {
      mbar_init(fx0 + 8 * s, 1);
      mbar_init(ex0 + 8 * s, 1);
    }
    mbar_init(acc0, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  // Router correction bias (a weight: before the dependency).
  for (int e = threadIdx.x; e < a.num_logits; e += kThreads) s_bias[e] = a.bias[e];
  const int cols = p.tr;  // D [128 tokens, tr weight rows] FP32
  if (warp == 1) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(
            smem_u32(slot)),
        "r"(cols));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  asm volatile("tcgen05.fence::before_thread_sync;");
  cluster_sync();  // barrier init + the leader's tile visible cluster-wide
  asm volatile("tcgen05.fence::after_thread_sync;");
  const uint32_t tmem = slot[0];
  // Cluster c owns weight tile c: the router tiles [0, rtiles) are the lowest
  // clusters, dispatched first (the top-k CTAs b < n wait only on them). No claim
  // counter: the next call's CTAs may start (transitive PDL) before this grid ends,
  // and nothing before griddepcontrol.wait may touch shared global state.
  const int tile = static_cast<int>(blockIdx.x) / p.csize;
  const int wrow = tile * p.tr;
  const int kb_lo = (static_cast<int>(rank) * p.kb) / p.csize;
  const int kb_hi = ((static_cast<int>(rank) + 1) * p.kb) / p.csize;
  const int nkb = kb_hi - kb_lo;
  // One TMA op per ring slot (3-D maps: wg / xg K blocks per op) and several issuing
  // warps: a TMA issuing thread completes ~one op per ~0.3 us whatever the op size or
  // ring depth (hwdb hw8/hw13; measured here: one 4 KB token op per ~0.3-0.45 us
  // capped the whole mainloop). Weight slots: warps 0, 2, 3 (slot s -> s % nprod);
  // token slots: warps 4.. (slot s -> 4 + s % nxp). A slot's box may run past this
  // rank's K range (the next rank's blocks or zero fill): loaded, never used.
  int prod = -1;
  // Whole warps run the loops (waits on every lane, lane 0 issues: the CUTLASS form;
  // a lone lane with the other 31 parked at a __syncwarp issued each tcgen05 / TMA
  // instruction at ~0.05-0.1 us).
  prod = warp == 0 ? 0 : warp == 2 ? 1 : warp == 3 ? 2 : -1;
  if (prod >= p.nprod) prod = -1;
  const int nxp = p.xst < 4 ? p.xst : 4;
  if (prod >= 0) {
    const int ngw = (nkb + p.wg - 1) / p.wg;
    for (int gw = 0; gw < ngw; ++gw) {
      const int s = gw % p.wst;
      if (s % p.nprod != prod) continue;
      if (gw >= p.wst) mbar_wait(ew0 + 8 * s, ((gw / p.wst) & 1) ^ 1);
      if (lane == 0) {
        if (gw < 64) STAMPC(136 + gw);
        mbar_expect_tx(fw0 + 8 * s, p.wg * wbox);
        tma_3d_ef(ring_s + s * p.wg * wbox, &tw, wrow, kb_lo + gw * p.wg, fw0 + 8 * s);
      }
      __syncwarp();
    }
  } else if (warp >= 4 && warp - 4 < nxp) {
    // Token issuers: after the predecessor grid (it wrote h).
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (warp == 4 && lane == 0) {
      STAMP(1);
      if (p.stamps) {  // which weight slots have landed by the dependency release
        uint32_t mask = 0;
        for (int s = 0; s < p.wst; ++s) {
          uint32_t ok;
          asm volatile(
              "{ .reg .pred q; mbarrier.test_wait.parity.shared::cta.b64 q, [%1], 0; "
              "selp.u32 %0, 1, 0, q; }"
              : "=r"(ok)
              : "r"(fw0 + 8 * s)
              : "memory");
          mask |= ok << s;
        }
        p.stamps[blockIdx.x * kStampStride + 198] = mask;
      }
    }
    const int ng = (nkb + p.xg - 1) / p.xg;
    for (int g = 0; g < ng; ++g) {
      const int s = g % p.xst;
      if (s % nxp != warp - 4) continue;
      if (g >= p.xst) mbar_wait(ex0 + 8 * s, ((g / p.xst) & 1) ^ 1);
      if (lane == 0) {
        if (g < 64) STAMPC(264 + g);
        mbar_expect_tx(fx0 + 8 * s, p.xg * xbytes);
        tma_3d(xring_s + s * p.xg * xbytes, &tx, 0, kb_lo + g * p.xg, fx0 + 8 * s);
      }
      __syncwarp();
    }
  } else if (warp == 1) {
    // kind::f16, BF16 A/B, FP32 D, K-major both, N = bn, M = 128.
    const uint32_t idesc = (1u << 4) | (1u << 7) | (1u << 10) |
                           (static_cast<uint32_t>(p.tr >> 3) << 17) | (8u << 24);
    for (int it = 0; it < nkb; ++it) {
      const int g = it / p.xg, j = it % p.xg;
      const int gw = it / p.wg, jw = it % p.wg;
      const int sw = gw % p.wst, sx = g % p.xst;
      const bool probe = (p.debug & 2) && it > 0;  // probe: MMA issue rate on slot 0 data
      if (j == 0 && !probe) mbar_wait(fx0 + 8 * sx, (g / p.xst) & 1);
      if (lane == 0 && it < 64) STAMPC(8 + it);
      if (jw == 0 && !probe) mbar_wait(fw0 + 8 * sw, (gw / p.wst) & 1);
      asm volatile("tcgen05.fence::after_thread_sync;");
      if (lane == 0) {
        if (it < 64) STAMPC(72 + it);
        if (it + 1 == nkb) STAMP(2);
      }
      // Warp-uniform operands: slot indices derive from the loop counter only.
      const uint64_t ad = smem_desc(ring_s + (probe ? 0 : (sw * p.wg + jw) * wbox));
      const uint64_t bd = smem_desc(xring_s + (probe ? 0 : (sx * p.xg + j) * xbytes));
      if (!(p.debug & 1)) {
#pragma unroll
        for (int kk = 0; kk < 4; ++kk)
          mma_bf16_el(tmem, bd + 2 * kk, ad + 2 * kk, idesc, (it == 0 && kk == 0) ? 0u : 1u);
      }
      if (lane == 0 && it < 64) STAMPC(136 + 64 + it);
      if (jw == p.wg - 1 || it + 1 == nkb) mma_commit_el(ew0 + 8 * sw);
      if (j == p.xg - 1 || it + 1 == nkb) mma_commit_el(ex0 + 8 * sx);
      __syncwarp();
    }
    mma_commit_el(acc0);
  }
  __syncwarp();
  asm volatile("griddepcontrol.wait;" ::: "memory");  // before any global write
  if (warp >= 4) {
    // TMEM -> own shared memory partial [n][128] FP32 (TMEM lane = token row).
    const int q = warp & 3, tok = q * 32 + lane;
    if (lane == 0) mbar_wait_sleep(acc0, 0);
    __syncwarp();
    if (threadIdx.x == 128) STAMP(3);
    asm volatile("tcgen05.fence::after_thread_sync;");
    float* part = reinterpret_cast<float*>(ring);
    const uint32_t ta = tmem + (static_cast<uint32_t>(q * 32) << 16);
    if (q * 32 < a.n) {  // warp-uniform (tcgen05.ld is .aligned)
      for (int c = 0; c < p.tr; c += 16) {
        float v[16];
        tmem_ld16(ta + c, v);
        if (tok < a.n) {
          float4* dst = reinterpret_cast<float4*>(part + tok * p.tr + c);
#pragma unroll
          for (int j = 0; j < 4; ++j)
            dst[j] = make_float4(v[4 * j], v[4 * j + 1], v[4 * j + 2], v[4 * j + 3]);
        }
      }
    }
  }
  asm volatile("tcgen05.fence::before_thread_sync;");
  cluster_sync();  // every rank's partial is in its shared memory
  const int n = a.n;
  {
    const int t0 = (static_cast<int>(rank) * n) / p.csize;
    const int t1 = ((static_cast<int>(rank) + 1) * n) / p.csize;
    // Router columns [0, rc) and latent columns [rc, lc) of this tile (a tile
    // may hold both: E is a multiple of 128, not of tr).
    const int rc = max(0, min(p.tr, a.num_logits - wrow));
    const int lc = min(p.tr, p.rows_w - wrow);
    if (rc > 0) {
      // Router logits: FP32 rows of y.
      const int per4 = rc / 4;
      const int nf = (t1 - t0) * per4;
      for (int f = threadIdx.x; f < nf; f += kThreads) {
        const int t = t0 + f / per4, r4 = (f % per4) * 4;
        const float4 acc = rank_sum(ring_s + (t * p.tr + r4) * 4, p.csize);
        *reinterpret_cast<float4*>(p.y + int64_t{t} * a.ldy + wrow + r4) = acc;
      }
    }
    if (lc > rc) {
      // Latent: BF16 round + the stager's quantization of each block of this tile.
      constexpr int vec = NVFP4 ? 16 : 32;
      const int per = (lc - rc) / vec;
      const int nf = (t1 - t0) * per;
      const int blk0 = (wrow + rc - a.num_logits) / vec;
      for (int f = threadIdx.x; f < nf; f += kThreads) {
        const int t = t0 + f / per, b = f % per;
        const int c0 = rc + b * vec;
        float4 in[8];
#pragma unroll
        for (int q = 0; q < vec / 4; ++q)
          in[q] = rank_sum(ring_s + (t * p.tr + c0 + 4 * q) * 4, p.csize);
        if (p.yfull) {
          float4* dst = reinterpret_cast<float4*>(p.y + int64_t{t} * a.ldy + wrow + c0);
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
  if (threadIdx.x == 128) STAMP(4);
  asm volatile("tcgen05.fence::before_thread_sync;");
  cluster_sync();  // peers are done reading this CTA's partial
  if (warp == 1) {
    asm volatile("tcgen05.fence::after_thread_sync;");
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem),
                 "r"(cols));
  }
  if (tile < p.rtiles && threadIdx.x == 0) {
    // This CTA's logits rows are stored (the cluster barrier ordered every thread's
    // stores before thread 0); publish them at gpu scope.
    __threadfence();
    atomicAdd(p.cnt + 1, 1u);
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
    if (threadIdx.x == 0) {
      const unsigned need = static_cast<unsigned>(p.rtiles * p.csize);
      while (ld_acquire(p.cnt + 1) < need) {
      }
      STAMP(5);
    }
    __syncthreads();
    const float* yrow = p.y + int64_t{row} * a.ldy;
    float lg[kPer];
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
      const int e = j * kThreads + threadIdx.x;
      lg[j] = e < a.num_logits ? __ldcg(yrow + e) : 0.0F;
    }
    route_block(a, row, lg, s_bias, s_unb, s_cand);  // warps 1.. return inside
  }
  if (threadIdx.x == 0) {
    STAMP(6);
    STAMPC(199);
    // The last CTA of the grid (every top-k CTA is past its wait) resets the
    // counters for the next call.
    if (atomicAdd(p.cnt + 2, 1u) == gridDim.x - 1) {
      p.cnt[1] = 0;
      p.cnt[2] = 0;
    }
  }
}

static void make_map(CUtensorMap* m, const void* ptr, int rows, int k, int box_rows,
                     int box_kb) {
  // {64 columns, rows, K blocks}: K block j of row r at r * k + 64 j.
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

int64_t smem_bytes(int64_t bn, int64_t wst, int64_t xst, int64_t xg, int64_t tr) {
  // wst: weight K blocks resident in the ring (slots x wg)
  return 1024 + ring_w_bytes(bn, wst, tr) + xst * xg * kXRows * 128 +
         (2 * kMaxW + 2 * kMaxX + 1) * 8 + 16;
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
               torch::Tensor& cnt, const torch::Tensor& bias, torch::Tensor& x,
               torch::Tensor& sf, torch::Tensor& idx, torch::Tensor& sw,
               torch::Tensor& err, const c10::optional<torch::Tensor>& pad,
               int64_t num_logits, int64_t num_experts, bool renormalize,
               double routed_scaling_factor, bool nvfp4, double norm_const,
               int64_t csize, int64_t wst, int64_t xst, int64_t nprod, int64_t xg,
               int64_t wg, int64_t tr, const torch::Tensor& stamps) {
  using namespace k3front;
  TORCH_CHECK(w.scalar_type() == at::kBFloat16 && h.scalar_type() == at::kBFloat16 &&
                  w.is_contiguous() && h.is_contiguous(),
              "moe_front: BF16 contiguous weight and tokens");
  const int64_t n = h.size(0), k = h.size(1);
  const int64_t rows_w = w.size(0);
  const int64_t hidden = rows_w - num_logits;
  TORCH_CHECK(w.size(1) == k && k % 64 == 0 && rows_w % 128 == 0 &&
                  num_logits % 128 == 0 && num_logits <= kWarps * 32 * kPer &&
                  hidden > 0 && n >= 1 && n <= 256,
              "moe_front: shapes");
  TORCH_CHECK(y.scalar_type() == at::kFloat && y.dim() == 2 && y.stride(1) == 1 &&
                  y.size(0) >= n && y.size(1) >= num_logits && y.stride(0) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(y.data_ptr()) % 16 == 0,
              "moe_front: logits buffer");
  TORCH_CHECK(cnt.scalar_type() == at::kInt && cnt.numel() >= 3, "moe_front: counters");
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
  const int64_t kb = k / 64;
  TORCH_CHECK(tr == 128 || tr == 256, "moe_front: tile rows 128 or 256");
  const int64_t tiles = (rows_w + tr - 1) / tr;
  TORCH_CHECK(csize >= 1 && csize <= kMaxC && kb >= csize && n >= csize &&
                  tiles * csize >= n && wst >= 1 && wst <= kMaxW && xst >= 1 &&
                  xst <= kMaxX && nprod >= 1 && nprod <= 3 && nprod <= wst && xg >= 1 &&
                  xg <= 8 && wg >= 1 && wg <= 4,
              "moe_front: launch configuration");
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
  p.cnt = reinterpret_cast<unsigned*>(cnt.data_ptr<int32_t>());
  // numel < 8: probe bits only (timed skeleton runs), no stamps.
  p.stamps = stamps.numel() >= 8 ? reinterpret_cast<unsigned long long*>(stamps.data_ptr())
                                 : nullptr;
  p.k = (int)k;
  p.bn = (int)((n + 15) / 16 * 16);
  p.tiles = (int)tiles;
  p.rtiles = (int)((num_logits + tr - 1) / tr);  // tiles holding router rows
  p.tr = (int)tr;
  p.rows_w = (int)rows_w;
  p.kb = (int)kb;
  p.csize = (int)csize;
  p.wst = (int)wst;
  p.xst = (int)xst;
  p.nprod = (int)nprod;
  p.xg = (int)xg;
  p.wg = (int)wg;
  p.debug = (int)(stamps.numel() > 0 ? stamps.numel() % 8 : 0);
  p.yfull = y.size(1) >= rows_w ? 1 : 0;
  CUtensorMap tw, tx;
  make_map(&tw, w.data_ptr(), (int)rows_w, (int)k, (int)tr, (int)wg);
  make_map(&tx, h.data_ptr(), (int)n, (int)k, kXRows, (int)xg);
  const int smem = (int)smem_bytes(p.bn, wst * wg, xst, xg, tr);
  static int configured = 0;
  if (smem > configured) {
    set_attrs(smem);
    configured = smem;
  }
  const c10::cuda::CUDAGuard guard(h.device());
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(static_cast<unsigned>(tiles * csize));
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[2];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  at[1].id = cudaLaunchAttributeClusterDimension;
  at[1].val.clusterDim.x = static_cast<unsigned>(csize);
  at[1].val.clusterDim.y = 1;
  at[1].val.clusterDim.z = 1;
  cfg.attrs = at;
  cfg.numAttrs = 2;
  const cudaError_t rc = nvfp4 ? cudaLaunchKernelEx(&cfg, moe_front_kernel<true>, tw, tx, p)
                              : cudaLaunchKernelEx(&cfg, moe_front_kernel<false>, tw, tx, p);
  TORCH_CHECK(rc == cudaSuccess, "moe_front launch: ", cudaGetErrorString(rc));
}

int64_t moe_front_smem(int64_t bn, int64_t wst, int64_t xst, int64_t xg, int64_t tr) {
  return k3front::smem_bytes(bn, wst, xst, xg, tr);
}

// (registers per thread, local memory bytes per thread) of the NVFP4 kernel.
std::vector<int64_t> moe_front_attrs() {
  cudaFuncAttributes fa;
  TORCH_CHECK(cudaFuncGetAttributes(&fa, k3front::moe_front_kernel<true>) == cudaSuccess);
  return {fa.numRegs, (int64_t)fa.localSizeBytes};
}

// Co-resident clusters of csize CTAs at this shared memory size.
int64_t moe_front_max_clusters(int64_t csize, int64_t smem) {
  using namespace k3front;
  set_attrs(static_cast<int>(smem));
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(static_cast<unsigned>(csize * 64));
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = static_cast<size_t>(smem);
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = static_cast<unsigned>(csize);
  at[0].val.clusterDim.y = 1;
  at[0].val.clusterDim.z = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  int nc = -1;
  const cudaError_t e = cudaOccupancyMaxActiveClusters(&nc, moe_front_kernel<true>, &cfg);
  return e == cudaSuccess ? nc : -static_cast<int64_t>(e);
}
"""

_CPP_SRC = """
void moe_front(const torch::Tensor& w, const torch::Tensor& h, torch::Tensor& y,
               torch::Tensor& cnt, const torch::Tensor& bias, torch::Tensor& x,
               torch::Tensor& sf, torch::Tensor& idx, torch::Tensor& sw,
               torch::Tensor& err, const c10::optional<torch::Tensor>& pad,
               int64_t num_logits, int64_t num_experts, bool renormalize,
               double routed_scaling_factor, bool nvfp4, double norm_const,
               int64_t csize, int64_t wst, int64_t xst, int64_t nprod, int64_t xg,
               int64_t wg, int64_t tr, const torch::Tensor& stamps);
int64_t moe_front_smem(int64_t bn, int64_t wst, int64_t xst, int64_t xg, int64_t tr);
int64_t moe_front_max_clusters(int64_t csize, int64_t smem);
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
                functions=["moe_front", "moe_front_smem", "moe_front_max_clusters", "moe_front_attrs"],
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


# Launch shape per bucket of shard rows: (cluster size, weight ring slots, token ring
# slots, weight issuers, K blocks per token slot, K blocks per weight slot). Chosen by the microbench
# (runs/D-MOEFRONT). C = 4: all 35 clusters co-resident (46 fit at this shared memory;
# C = 5 / 6 fit 32 / 28 and run a second wave). The token ring is as deep as the
# weight ring: tokens come from L2 one K block per slot, and a shallow token ring
# (4 slots) capped the mainloop at ~3 K blocks per us.
# (csize, wst, xst, nprod, xg, wg, tile rows)
CONFIGS = {32: (4, 4, 2, 3, 2, 2, 128), 64: (4, 4, 2, 3, 2, 2, 128), 96: (4, 4, 2, 3, 2, 2, 128)}


def config_for(n: int):
    for rows in sorted(CONFIGS):
        if n <= rows:
            return CONFIGS[rows]
    return None


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
            if config_for(n) is None:
                raise Unsupported("moe", n, "moe_front: no launch configuration")
        dev = packed.device
        n_max = max(rows)
        self.y = torch.empty((n_max, self.num_logits), dtype=torch.float32, device=dev)
        self.cnt = torch.zeros(4, dtype=torch.int32, device=dev)
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
        cfg = config_for(n)
        _module().moe_front(
            self.packed, h, self.y, self.cnt,
            self.mlp.gate.e_score_correction_bias.data,
            session.x.view(torch.uint8), session.x_sf.view(torch.uint8),
            session.topk_idx, session.topk_weights, rt.route_error, pad,
            self.num_logits, p.num_experts, self.route.renormalize, self.route.scale,
            p.quant_kind == "nvfp4", float(p.input_norm_const),
            *cfg,
            self.no_stamps if stamps is None else stamps,
        )  # fmt: skip
        return rt, session, n

    def launch(self, staged, output_rmsnorm=None) -> torch.Tensor:
        return self.route.launch(staged, output_rmsnorm)
