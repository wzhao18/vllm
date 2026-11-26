# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sequence-parallel all-gather of the Kimi K3 decoder layers over NVLink
symmetric memory.

Under sequence parallelism every decoder layer gathers the TP ranks' token
shards of its attention input ([T / tp, hidden] per rank -> [T, hidden]).
Above the size the stock custom (IPC) all-gather takes, that went to NCCL,
which runs it as RING / SIMPLE over 16 channels: 16 CTAs at ~25 GB/s each,
0.40 TB/s of the 1.25 TB/s NVLink ingress (TP4 C512 prefill at 17.4K tokens:
463 us per layer, 94 layers per step).

Here each rank pushes its shard with 16-byte stores straight into every
rank's symmetric output buffer (one persistent kernel over all SMs), then one
symmetric-memory barrier makes all shards visible: 191 us at 17.4K tokens
(0.98 TB/s). The output is a view of the buffer, so the gather does no extra
copy. The result is bitwise the NCCL all-gather (a pure copy). On the
40-layer TP4 SP prefill step: -11.7 ms (-5.8%) at 17.4K tokens, -5.4 ms
(-4.7%) at 8.2K.

Allocation (``sp_all_gather_rows``): a prefill or unified instance allocates,
at model init (so the KV sizing accounts for it), two slots of the token shard
of a max_num_batched_tokens step (896 MiB per GPU at 32K tokens, TP4) when that
shard is above the custom all-gather's size. No step of the instance gathers
more, so a larger shard raises. A P/D decode instance (KV consumer only)
allocates nothing: its decode steps are small (C512: 128 rows per rank, inside
the custom all-gather; the specialized decode's SP step has its own
communication), a buffer for its rare local prefills (preemption recompute) up
to max_num_batched_tokens would cost KV capacity, and a smaller one would not
cover them. Without a buffer (also: no symmetric memory over the TP group,
logged) every gather takes the stock all-gather.

The final gather of the model (``KimiK3Model.forward``, with the drafter's
auxiliary hidden states packed beside the hidden states, so three times wider)
keeps the stock path: a slot of its width would add 1.5x the buffer for one
gather of 41 (~0.8 ms of a 190 ms step).

MXFP8 gather (``kimi_sp_all_gather_mxfp8``): a KDA layer's in_proj is an
online-MXFP8 linear that quantizes the gathered BF16 rows with FlashInfer's
``mxfp8_quantize`` (E4M3 + F8_128x4-swizzled E8M0 scales, 1x32 blocks along
the row). Each scale depends on its row's own values, so the ranks quantize
their shards (the AttnRes epilogue, bit for bit ``mxfp8_quantize``) and push
the FP8 rows + scale bytes (1.03 B per element instead of 2). With a shard of
a multiple of 128 rows the F8_128x4 row tiles of the gathered tensor are the
ranks' local tiles in rank order (the row-tile index is the outermost of the
layout), so the gathered scale bytes are the shards' local swizzled scales
concatenated: bitwise the quantization of the gathered BF16. Other shard sizes,
and shards the custom (Lamport) all-gather takes (BF16 only: FP8 bytes can
alias its sentinel), keep the BF16 gather (``mxfp8_gather_rows_ok``). The FP8
rows and scales share the BF16 slot (1 + 1/32 <= 2 bytes per element): rows
from byte 0, scales from byte ``world * max_rows * hidden``.

TMA push of BF16 shards (the DSA layers' gather; SM 10.x GPUs): every SM
TMA-loads 32 KiB chunks of the shard into shared memory and bulk-stores each
into one rank's slot (chunk-major, ranks rotated). The last CTA to finish (a
local count) adds 1 (release, system scope) to the slot's arrival counter of
every rank, a word at the end of each rank's signal pad (no new memory), then
waits until its own counter holds world and takes that back off, so the push
returns with the gather complete and no barrier follows. Every
value is device-side and the counter returns to 0, so CUDA graphs capture it.
The slot reuse rule below holds as with the barrier: a rank's wait on gather
n - 1 returned only after every peer started its push of it, which its stream
issues after every read of gather n - 2. The bytes are the stock gather's
(pure copy). The kernel is built and loaded at buffer allocation.

Buffer reuse: the two slots alternate over the gathers of a forward
(``begin_forward`` resets the count; the slot sequence is fixed per forward,
so CUDA graphs capture it). The push of a gather overwrites a peer's slot only
after the barrier of an earlier gather on the other slot, which that peer
reached after its stream finished every read of the slot's previous gather.
So a gathered tensor stays valid until the gather two later, and it is
consumed inside its layer. Across forwards: with an even number of gathers per
forward the last uses slot 1 and the next forward's first slot 0; with an odd
number both use slot 0, so the first push of such a forward waits on one more
barrier first (one per forward, a few us).
"""

import hashlib
import os
import threading
from typing import Any

import torch

from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.ops.cp_common import direct_cp_enabled, symm_mem

logger = init_logger(__name__)

# Persistent grid of one program per SM, 8192-element (16 KiB) tiles, 4 warps:
# 98 / 191 / 350 us at 2050 / 4350 / 8192 rows x 7168 per rank on TP4 VR200
# (non-persistent tiles: 117-141 / 221-264 / 404-484 us).
_BLOCK = 8192
_NUM_WARPS = 4
_NUM_SLOTS = 2


@triton.jit
def _push_shard_kernel(
    src,
    dst_addresses,
    dst_offset,
    numel,
    rank,
    WORLD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Copy ``numel`` elements of ``src`` to ``dst_offset`` of every peer
    buffer in ``dst_addresses`` (absolute addresses, [WORLD]). Persistent: one
    program per SM strides over the tiles. Peers are visited from ``rank + 1``
    on so the ranks' first stores spread over the peers."""
    for tile in range(tl.program_id(0), tl.cdiv(numel, BLOCK), tl.num_programs(0)):
        offs = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel
        x = tl.load(src + offs, mask=mask)
        for i in tl.static_range(WORLD):
            peer = (rank + 1 + i) % WORLD
            dst = tl.load(dst_addresses + peer).to(
                tl.pointer_type(src.dtype.element_ty)
            )
            tl.store(dst + dst_offset + offs, x, mask=mask)


@triton.jit
def _push_shard2_kernel(
    src_a,
    src_b,
    dst_addresses,
    dst_offset_a,
    dst_offset_b,
    numel_a,
    numel_b,
    rank,
    WORLD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """``_push_shard_kernel`` over two segments in one persistent grid: the
    tiles of ``src_a`` (``numel_a`` elements to ``dst_offset_a``), then those
    of ``src_b``. Same element type for both (32-bit words of the MXFP8 rows
    and scale bytes)."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    for tile in range(pid, tl.cdiv(numel_a, BLOCK), nprog):
        offs = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel_a
        x = tl.load(src_a + offs, mask=mask)
        for i in tl.static_range(WORLD):
            peer = (rank + 1 + i) % WORLD
            dst = tl.load(dst_addresses + peer).to(
                tl.pointer_type(src_a.dtype.element_ty)
            )
            tl.store(dst + dst_offset_a + offs, x, mask=mask)
    # Segment b continues the round robin where segment a's tiles end.
    start_b = (pid + nprog - tl.cdiv(numel_a, BLOCK) % nprog) % nprog
    for tile in range(start_b, tl.cdiv(numel_b, BLOCK), nprog):
        offs = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel_b
        x = tl.load(src_b + offs, mask=mask)
        for i in tl.static_range(WORLD):
            peer = (rank + 1 + i) % WORLD
            dst = tl.load(dst_addresses + peer).to(
                tl.pointer_type(src_b.dtype.element_ty)
            )
            tl.store(dst + dst_offset_b + offs, x, mask=mask)


# MXFP8: 1x32 blocks along the row, F8_128x4 scale tiles of 128 rows.
MXFP8_BLOCK = 32
MXFP8_ROW_TILE = 128
_WORD_BLOCK = _BLOCK // 2  # 32-bit words per tile (16 KiB, as the BF16 tiles)


# TMA bulk push of a BF16 shard (PR-3b): see the module docstring.
_TMA_CHUNK = 32 << 10  # bytes per TMA unit (2-stage smem ring: 64 KiB)
_TMA_STAGES = 2
_TMA_SPIN_NS = 30_000_000_000  # 30 s: a hang traps with its values printed

_TMA_CUDA = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdint.h>
#include <map>
#include <utility>

__device__ __forceinline__ uint64_t tp_now() { uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint32_t tp_sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint32_t tp_ld_acq(uint64_t a) {
  uint32_t v; asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory"); return v;
}

// One issuing thread per CTA. Unit u = (chunk u / W of this rank's shard, destination rank (rank + 1 + u) % W): TMA-load
// the chunk into smem, bulk-store it into that rank's slot. Unit u runs on CTA u % G (NS-stage smem ring). Once all of
// its stores completed, a CTA counts itself done.
// The last CTA to finish (a local count) then adds 1 to the slot's arrival counter of every rank (release, system
// scope), waits until its own counter holds W (every rank, itself included) and takes W back off it.
template <int NS>
__global__ void __launch_bounds__(32, 1) sp_ag_tma_push_kernel(
    const uint8_t* __restrict__ src, const int64_t* __restrict__ dst_slot, int64_t dst_off, int64_t nbytes,
    const int64_t* __restrict__ ctr, uint64_t done, int rank, int W, int chunk, uint64_t limit) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ __align__(8) uint64_t bar[NS];
  if (threadIdx.x != 0 || nbytes == 0) return;
  for (int i = 0; i < NS; ++i) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(tp_sa(&bar[i])));
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  const int G = gridDim.x, g = blockIdx.x;
  const int64_t nch = (nbytes + chunk - 1) / chunk, total = nch * W;
  const int64_t n = g < total ? (total - g + G - 1) / G : 0;
  uint64_t dsts[NS];
  uint32_t nbs[NS];
  uint32_t ph = 0;
  auto issue = [&](int64_t k) {
    const int64_t u = g + k * (int64_t)G, off = (u / W) * chunk;
    const int d = (rank + 1 + (int)(u % W)) % W;
    const uint32_t nb = (uint32_t)min((int64_t)chunk, nbytes - off);
    const int s = (int)(k % NS);
    dsts[s] = (uint64_t)dst_slot[d] + dst_off + off;
    nbs[s] = nb;
    asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}"
                 :: "r"(tp_sa(&bar[s])), "r"(nb) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(tp_sa(smem + (size_t)s * chunk)), "l"(src + off), "r"(nb), "r"(tp_sa(&bar[s])) : "memory");
  };
  for (int64_t k = 0; k < NS - 1 && k < n; ++k) issue(k);
  for (int64_t k = 0; k < n; ++k) {
    const int s = (int)(k % NS);
    const uint32_t a = tp_sa(&bar[s]), p = (ph >> s) & 1u;
    asm volatile("{\n.reg .pred q;\nW_%=:\nmbarrier.try_wait.parity.shared::cta.b64 q, [%0], %1;\n@!q bra W_%=;\n}"
                 :: "r"(a), "r"(p) : "memory");
    ph ^= 1u << s;
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                 :: "l"(dsts[s]), "r"(tp_sa(smem + (size_t)s * chunk)), "r"(nbs[s]) : "memory");
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    // The store of unit k - 1 has read its smem stage once at most one group (unit k) is pending.
    asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory");
    if (k + NS - 1 < n) issue(k + NS - 1);
  }
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");  // every store of this CTA performed
  asm volatile("fence.proxy.async.global;" ::: "memory");
  __threadfence_system();
  // The last CTA of this grid (local count) signals every rank once: W remote adds per rank instead of W x G (a
  // single word takes them one at a time).
  uint32_t prev;
  asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;" : "=r"(prev) : "l"(done) : "memory");
  if (prev != (uint32_t)(G - 1)) return;
  *(volatile uint32_t*)done = 0u;  // every CTA of this grid has added
  __threadfence_system();
  for (int i = 0; i < W; ++i) {
    const uint64_t c = (uint64_t)ctr[(rank + 1 + i) % W];
    asm volatile("red.release.sys.global.add.u32 [%0], 1;" :: "l"(c) : "memory");
  }
  const uint64_t own = (uint64_t)ctr[rank];
  const uint32_t target = (uint32_t)W;
  const uint64_t t0 = tp_now();
  uint32_t v;
  while ((int32_t)((v = tp_ld_acq(own)) - target) < 0) {
    __nanosleep(32);
    if (tp_now() - t0 > limit) {
      printf("K31_SPIN_TIMEOUT kernel=sp_ag_tma_push rank=%d target=%u seen=%u\n", rank, target, v);
      __trap();
    }
  }
  // No add of a later gather on this slot can precede this: a rank pushes the slot's next gather only after the
  // gather in between completed, which needs this rank's own push of it (after this kernel on the stream).
  asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" :: "l"(own), "r"(0u - target) : "memory");
}

static cudaStream_t tp_cur() { return c10::cuda::getCurrentCUDAStream().stream(); }

// PX probe (TLX, never merged): the same push over up to PX_MAXSEG source segments in one grid. Unit u = (global chunk
// u / W over the concatenated segments, destination (rank + 1 + u % W) % W); completion as sp_ag_tma_push_kernel.
#define PX_MAXSEG 8
struct PxSegs { int64_t src[PX_MAXSEG]; int64_t nb[PX_MAXSEG]; int64_t off[PX_MAXSEG]; int64_t cum[PX_MAXSEG + 1]; int n; };
template <int NS>
__global__ void __launch_bounds__(32, 1) px_tma_push_kernel(
    const PxSegs sg, const int64_t* __restrict__ dst_slot, const int64_t* __restrict__ ctr, uint64_t done, int rank,
    int W, int chunk, uint64_t limit) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ __align__(8) uint64_t bar[NS];
  const int64_t nch = sg.cum[sg.n];
  if (threadIdx.x != 0 || nch == 0) return;
  for (int i = 0; i < NS; ++i) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(tp_sa(&bar[i])));
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  const int G = gridDim.x, g = blockIdx.x;
  const int64_t total = nch * W;
  const int64_t n = g < total ? (total - g + G - 1) / G : 0;
  uint64_t dsts[NS];
  uint32_t nbs[NS];
  uint32_t ph = 0;
  auto issue = [&](int64_t k) {
    const int64_t u = g + k * (int64_t)G, c = u / W;
    int si = 0;
    while (si + 1 < sg.n && c >= sg.cum[si + 1]) ++si;
    const int64_t off = (c - sg.cum[si]) * chunk;
    const int d = (rank + 1 + (int)(u % W)) % W;
    const uint32_t nb = (uint32_t)min((int64_t)chunk, sg.nb[si] - off);
    const int s = (int)(k % NS);
    dsts[s] = (uint64_t)dst_slot[d] + sg.off[si] + off;
    nbs[s] = nb;
    asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}"
                 :: "r"(tp_sa(&bar[s])), "r"(nb) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(tp_sa(smem + (size_t)s * chunk)), "l"((const uint8_t*)sg.src[si] + off), "r"(nb),
                    "r"(tp_sa(&bar[s])) : "memory");
  };
  for (int64_t k = 0; k < NS - 1 && k < n; ++k) issue(k);
  for (int64_t k = 0; k < n; ++k) {
    const int s = (int)(k % NS);
    const uint32_t a = tp_sa(&bar[s]), p = (ph >> s) & 1u;
    asm volatile("{\n.reg .pred q;\nW_%=:\nmbarrier.try_wait.parity.shared::cta.b64 q, [%0], %1;\n@!q bra W_%=;\n}"
                 :: "r"(a), "r"(p) : "memory");
    ph ^= 1u << s;
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                 :: "l"(dsts[s]), "r"(tp_sa(smem + (size_t)s * chunk)), "r"(nbs[s]) : "memory");
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory");
    if (k + NS - 1 < n) issue(k + NS - 1);
  }
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  asm volatile("fence.proxy.async.global;" ::: "memory");
  __threadfence_system();
  uint32_t prev;
  asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;" : "=r"(prev) : "l"(done) : "memory");
  if (prev != (uint32_t)(G - 1)) return;
  *(volatile uint32_t*)done = 0u;
  __threadfence_system();
  for (int i = 0; i < W; ++i) {
    const uint64_t c = (uint64_t)ctr[(rank + 1 + i) % W];
    asm volatile("red.release.sys.global.add.u32 [%0], 1;" :: "l"(c) : "memory");
  }
  const uint64_t own = (uint64_t)ctr[rank];
  const uint32_t target = (uint32_t)W;
  const uint64_t t0 = tp_now();
  uint32_t v;
  while ((int32_t)((v = tp_ld_acq(own)) - target) < 0) {
    __nanosleep(32);
    if (tp_now() - t0 > limit) {
      printf("K31_SPIN_TIMEOUT kernel=px_tma_push rank=%d target=%u seen=%u\n", rank, target, v);
      __trap();
    }
  }
  asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" :: "l"(own), "r"(0u - target) : "memory");
}


void px_tma_push(std::vector<int64_t> srcs, std::vector<int64_t> nbs, std::vector<int64_t> offs, torch::Tensor dst_slot,
                 torch::Tensor ctr, int64_t done, int64_t rank, int64_t W, int64_t chunk, int64_t grid, int64_t limit) {
  TORCH_CHECK(srcs.size() == nbs.size() && nbs.size() == offs.size() && srcs.size() <= PX_MAXSEG, "px_tma_push: segments");
  TORCH_CHECK(dst_slot.numel() == W && ctr.numel() == W, "px_tma_push: [W] address tensors");
  PxSegs sg{};
  sg.n = (int)srcs.size();
  sg.cum[0] = 0;
  for (int i = 0; i < sg.n; ++i) {
    TORCH_CHECK(nbs[i] > 0 && nbs[i] % 16 == 0 && offs[i] % 16 == 0 && srcs[i] % 16 == 0, "px_tma_push: 16-byte aligned");
    sg.src[i] = srcs[i]; sg.nb[i] = nbs[i]; sg.off[i] = offs[i];
    sg.cum[i + 1] = sg.cum[i] + (nbs[i] + chunk - 1) / chunk;
  }
  auto k = px_tma_push_kernel<2>;
  const int smem = (int)(2 * chunk);
  static int smem_set = 0;
  if (smem_set < smem) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    smem_set = smem;
  }
  k<<<(unsigned)grid, 32, smem, tp_cur()>>>(sg, dst_slot.data_ptr<int64_t>(), ctr.data_ptr<int64_t>(), (uint64_t)done,
                                            (int)rank, (int)W, (int)chunk, (uint64_t)limit);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// PX arm C (TLX, never merged): head->token all-to-all of the KDA o_proj input. Source (this rank, head group g) =
// [W * R, 1536] E4M3 rows + F8_128x4 scales ([W * R / 128 row tiles] x 12 blocks x 512 B). Owner d gets rows
// [d R, (d + 1) R) at column g * 1536 of its [R, 12288] buffer: a strided 2D destination, written with
// cp.async.bulk.tensor 2D stores through one tensor map per destination (box 192 x u64 by BR rows); the scale slice of a
// row tile is 12 whole 512 B blocks, contiguous at block g * 12 of the owner's 96 (one 1D bulk store per row tile).
// Unit u = (j = u / W, destination (rank + 1 + u % W) % W): j < R / BR a data chunk, else scale tile j - R / BR.
// Completion as sp_ag_tma_push_kernel (counted, on the slot's arrival counters).
struct __align__(64) PxA2a { CUtensorMap m[8]; int64_t sf[8]; };
template <int NS>
__global__ void __launch_bounds__(32, 1) px_a2a_kernel(
    const __grid_constant__ PxA2a a, const uint8_t* __restrict__ src, const uint8_t* __restrict__ src_sf, int R, int gh,
    const int64_t* __restrict__ ctr, uint64_t done, int rank, int W, int br, int chunk, uint64_t limit) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ __align__(8) uint64_t bar[NS];
  if (threadIdx.x != 0 || R == 0) return;
  for (int i = 0; i < NS; ++i) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(tp_sa(&bar[i])));
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  const int G = gridDim.x, g = blockIdx.x;
  const int nd = R / br, ns = R / 128;
  const int64_t total = (int64_t)(nd + ns) * W;
  const int64_t n = g < total ? (total - g + G - 1) / G : 0;
  int kind[NS], dd[NS], jj[NS];
  uint32_t nbs[NS];
  uint32_t ph = 0;
  auto issue = [&](int64_t k) {
    const int64_t u = g + k * (int64_t)G;
    const int j = (int)(u / W), d = (rank + 1 + (int)(u % W)) % W, s = (int)(k % NS);
    const uint8_t* p;
    uint32_t nb;
    if (j < nd) { p = src + ((int64_t)d * R + (int64_t)j * br) * 1536; nb = (uint32_t)br * 1536u; kind[s] = 0; jj[s] = j; }
    else { p = src_sf + ((int64_t)d * ns + (j - nd)) * 6144; nb = 6144u; kind[s] = 1; jj[s] = j - nd; }
    dd[s] = d; nbs[s] = nb;
    asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}"
                 :: "r"(tp_sa(&bar[s])), "r"(nb) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(tp_sa(smem + (size_t)s * chunk)), "l"(p), "r"(nb), "r"(tp_sa(&bar[s])) : "memory");
  };
  for (int64_t k = 0; k < NS - 1 && k < n; ++k) issue(k);
  for (int64_t k = 0; k < n; ++k) {
    const int s = (int)(k % NS);
    const uint32_t ab = tp_sa(&bar[s]), p = (ph >> s) & 1u;
    asm volatile("{\n.reg .pred q;\nW_%=:\nmbarrier.try_wait.parity.shared::cta.b64 q, [%0], %1;\n@!q bra W_%=;\n}"
                 :: "r"(ab), "r"(p) : "memory");
    ph ^= 1u << s;
    if (kind[s] == 0) {
      asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                   :: "l"(reinterpret_cast<uint64_t>(&a.m[dd[s]])), "r"(gh * 192), "r"(jj[s] * br),
                      "r"(tp_sa(smem + (size_t)s * chunk)) : "memory");
    } else {
      asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                   :: "l"((uint64_t)a.sf[dd[s]] + (uint64_t)jj[s] * 49152u + (uint64_t)gh * 6144u),
                      "r"(tp_sa(smem + (size_t)s * chunk)), "r"(nbs[s]) : "memory");
    }
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory");
    if (k + NS - 1 < n) issue(k + NS - 1);
  }
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  asm volatile("fence.proxy.async.global;" ::: "memory");
  __threadfence_system();
  uint32_t prev;
  asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;" : "=r"(prev) : "l"(done) : "memory");
  if (prev != (uint32_t)(G - 1)) return;
  *(volatile uint32_t*)done = 0u;
  __threadfence_system();
  for (int i = 0; i < W; ++i) {
    const uint64_t c = (uint64_t)ctr[(rank + 1 + i) % W];
    asm volatile("red.release.sys.global.add.u32 [%0], 1;" :: "l"(c) : "memory");
  }
  const uint64_t own = (uint64_t)ctr[rank];
  const uint32_t target = (uint32_t)W;
  const uint64_t t0 = tp_now();
  uint32_t v;
  while ((int32_t)((v = tp_ld_acq(own)) - target) < 0) {
    __nanosleep(32);
    if (tp_now() - t0 > limit) {
      printf("K31_SPIN_TIMEOUT kernel=px_a2a rank=%d target=%u seen=%u\n", rank, target, v);
      __trap();
    }
  }
  asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" :: "l"(own), "r"(0u - target) : "memory");
}

// PX arm D marker: an empty kernel whose name (px_mark_kernel<ID>) tags the stand-in GEMMs after it in the trace.
template <int ID>
__global__ void px_mark_kernel() {}

typedef CUresult (*PxEncodeFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                               const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                               CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static PxEncodeFn px_encode_fn() {
  static PxEncodeFn fn = nullptr;
  if (fn == nullptr) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q;
#if CUDART_VERSION >= 12050
    C10_CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q));
#else
    C10_CUDA_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q));
#endif
    TORCH_CHECK(q == cudaDriverEntryPointSuccess && p != nullptr, "px: cuTensorMapEncodeTiled not found");
    fn = (PxEncodeFn)p;
  }
  return fn;
}

void px_a2a(int64_t src, int64_t src_sf, int64_t R, int64_t gh, std::vector<int64_t> dst, std::vector<int64_t> dst_sf,
            torch::Tensor ctr, int64_t done, int64_t rank, int64_t W, int64_t br, int64_t grid, int64_t limit) {
  TORCH_CHECK((int64_t)dst.size() == W && (int64_t)dst_sf.size() == W && W <= 8 && ctr.numel() == W, "px_a2a: W");
  TORCH_CHECK(R % 128 == 0 && R % br == 0 && br <= 256 && src % 16 == 0 && src_sf % 16 == 0, "px_a2a: shape");
  static std::map<std::pair<int64_t, int64_t>, PxA2a> cache;
  const auto key = std::make_pair(R * 64 + br, dst[0]);
  auto it = cache.find(key);
  if (it == cache.end()) {
    PxA2a a;
    memset(&a, 0, sizeof(a));
    for (int d = 0; d < W; ++d) {
      TORCH_CHECK(dst[d] % 16 == 0 && dst_sf[d] % 16 == 0, "px_a2a: 16-byte aligned destinations");
      const cuuint64_t dims[2] = {(cuuint64_t)(12288 / 8), (cuuint64_t)R};
      const cuuint64_t strides[1] = {(cuuint64_t)12288};
      const cuuint32_t box[2] = {192u, (cuuint32_t)br};
      const cuuint32_t es[2] = {1u, 1u};
      CUresult r = px_encode_fn()(&a.m[d], CU_TENSOR_MAP_DATA_TYPE_UINT64, 2, (void*)dst[d], dims, strides, box, es,
                                  CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                  CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      TORCH_CHECK(r == CUDA_SUCCESS, "px_a2a: cuTensorMapEncodeTiled failed ", (int)r);
      a.sf[d] = dst_sf[d];
    }
    it = cache.emplace(key, a).first;
  }
  const int chunk = (int)std::max<int64_t>(br * 1536, 6144);
  auto k = px_a2a_kernel<2>;
  const int smem = 2 * chunk;
  static int smem_set = 0;
  if (smem_set < smem) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    smem_set = smem;
  }
  k<<<(unsigned)grid, 32, smem, tp_cur()>>>(it->second, (const uint8_t*)src, (const uint8_t*)src_sf, (int)R, (int)gh,
                                            ctr.data_ptr<int64_t>(), (uint64_t)done, (int)rank, (int)W, (int)br, chunk,
                                            (uint64_t)limit);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void px_mark(int64_t id) {
  switch (id) {
    case 1: px_mark_kernel<1><<<1, 32, 0, tp_cur()>>>(); break;
    case 2: px_mark_kernel<2><<<1, 32, 0, tp_cur()>>>(); break;
    case 3: px_mark_kernel<3><<<1, 32, 0, tp_cur()>>>(); break;
    case 4: px_mark_kernel<4><<<1, 32, 0, tp_cur()>>>(); break;
    case 5: px_mark_kernel<5><<<1, 32, 0, tp_cur()>>>(); break;
    default: px_mark_kernel<0><<<1, 32, 0, tp_cur()>>>(); break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sp_ag_tma_push(torch::Tensor src, torch::Tensor dst_slot, int64_t dst_off, torch::Tensor ctr, int64_t done, int64_t rank,
                    int64_t W, int64_t chunk, int64_t grid, int64_t limit) {
  TORCH_CHECK(src.is_cuda() && src.is_contiguous(), "sp_ag_tma_push: contiguous CUDA source");
  const int64_t nbytes = src.numel() * src.element_size();
  TORCH_CHECK(nbytes % 16 == 0 && dst_off % 16 == 0 && ((uintptr_t)src.data_ptr()) % 16 == 0 && chunk % 16 == 0,
              "sp_ag_tma_push: 16-byte aligned shard and offsets");
  TORCH_CHECK(dst_slot.numel() == W && ctr.numel() == W && dst_slot.dtype() == torch::kInt64 &&
              ctr.dtype() == torch::kInt64, "sp_ag_tma_push: [W] int64 address tensors");
  auto k = sp_ag_tma_push_kernel<2>;
  const int smem = (int)(2 * chunk);
  static int smem_set = 0;
  if (smem_set < smem) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    smem_set = smem;
  }
  k<<<(unsigned)grid, 32, smem, tp_cur()>>>((const uint8_t*)src.data_ptr(), dst_slot.data_ptr<int64_t>(), dst_off,
                                            nbytes, ctr.data_ptr<int64_t>(), (uint64_t)done, (int)rank, (int)W, (int)chunk,
                                            (uint64_t)limit);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_TMA_CPP = r"""
#include <torch/extension.h>
void sp_ag_tma_push(torch::Tensor, torch::Tensor, int64_t, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t,
                    int64_t);
void px_tma_push(std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>, torch::Tensor, torch::Tensor, int64_t,
                 int64_t, int64_t, int64_t, int64_t, int64_t);
void px_a2a(int64_t, int64_t, int64_t, int64_t, std::vector<int64_t>, std::vector<int64_t>, torch::Tensor, int64_t, int64_t,
            int64_t, int64_t, int64_t, int64_t);
void px_mark(int64_t);
"""

_tma_lock = threading.Lock()
_tma_ext = None


def _tma_module():
    """The TMA push extension (built once per source into VLLM_CACHE_ROOT)."""
    global _tma_ext
    if _tma_ext is not None:
        return _tma_ext
    with _tma_lock:
        if _tma_ext is None:
            from torch.utils.cpp_extension import load_inline

            import vllm.envs as envs

            h = hashlib.sha256((_TMA_CUDA + _TMA_CPP).encode())
            h.update(f"{torch.__version__} {torch.version.cuda}".encode())
            tag = h.hexdigest()[:16]
            build = os.path.join(envs.VLLM_CACHE_ROOT, "k3_sp_ag_tma", tag)
            os.makedirs(build, exist_ok=True)
            _tma_ext = load_inline(
                name=f"k3_sp_ag_tma_{tag}",
                cpp_sources=_TMA_CPP,
                cuda_sources=_TMA_CUDA,
                functions=["sp_ag_tma_push", "px_tma_push", "px_a2a", "px_mark"],
                extra_cuda_cflags=["-O3", "-gencode", "arch=compute_100f,code=sm_100f"],
                build_directory=build,
                verbose=False,
            )
    return _tma_ext


def tma_push_capable(device) -> bool:
    """The TMA push needs a Blackwell-family GPU (sm_100f code: SM 10.x)."""
    return torch.cuda.get_device_capability(device)[0] == 10


# ---------------------------------------------------------------------------------------------------------------
# PX probe overlay (TLX day-1 in-step probe, designs/tp8comm-sol.md sec 4; NEVER MERGED). K31_PX = comma flags:
#   kda_tma : the KDA MXFP8 gather goes through the TMA push (rows + scales, one launch, counted completion, no barrier)
#   kda_w   : + the KDA layer's own W_o shard bytes (1536 x 7168 MXFP8 + E8M0 = 11,354,112 B per source; 79.5 MB ingress)
#   dsa_w   : the DSA BF16 TMA gather also carries W_qb|W_g|W_o shard bytes (51,118,080 B per source; 357.8 MB ingress)
# Weight bytes are real parameter bytes of the layer's attention module, written into the dead tail of the SAME slot:
# bytes [world * max_rows * hidden * (1 + 1/32), 2 * world * max_rows * hidden) of a slot are written by no gather
# (BF16 gathers end at 2 * world * rows * hidden <= world * max_rows * hidden * 2 only when rows = max_rows; the
# init check below requires rows_max_step <= max_rows / 2, true at MNBT 65536 for p2k/p4k) and read by nobody.
# The model's math is unchanged (bitwise); no memory is allocated.
_PX = frozenset(f for f in os.environ.get("K31_PX", "").split(",") if f)
_PX_KDA_W_BYTES = 1536 * 7168 * 33 // 32
_PX_DSA_W_BYTES = 1536 * 2304 * 2 + 7168 * 1536 * 2 + 1536 * 7168 * 2
_px_src: list[tuple[int, int]] | None = None  # (address, bytes) of the next gather's shadow segments
_px_logged: set[str] = set()


def px_enabled(flag: str) -> bool:
    return flag in _PX


def px_set_shadow(module: torch.nn.Module | None, kind: str) -> None:
    """Announce the shadow weight bytes of the next gather (kind 'kda' or 'dsa'): parameter bytes of ``module``
    (o_proj for KDA; q_b|g|o projections for DSA), cycled to the design's per-source byte count, 16-byte aligned."""
    global _px_src
    _px_src = None
    if module is None or not ((kind == "kda" and "kda_w" in _PX) or (kind == "dsa" and "dsa_w" in _PX)):
        return
    target = _PX_KDA_W_BYTES if kind == "kda" else _PX_DSA_W_BYTES
    cache = getattr(module, "_px_shadow_cache", None)
    if cache is None:
        keys = ("o_proj",) if kind == "kda" else ("q_b", "wq_b", "g_proj", "gate", "o_proj")
        # Real bytes of the layer's tensors, by storage (packed / non-contiguous weights included), key matches first,
        # then the largest; one storage is cycled only if the module holds fewer bytes than the target.
        seen, spans = set(), []
        for n, t in list(module.named_parameters()) + list(module.named_buffers()):
            if not t.is_cuda:
                continue
            st = t.untyped_storage()
            if st.data_ptr() in seen or st.nbytes() < 4096:
                continue
            seen.add(st.data_ptr())
            spans.append((not any(k in n for k in keys), -st.nbytes(), n, st.data_ptr(), st.nbytes(), tuple(t.shape)))
        spans.sort()
        params = [(x[2], x[5]) for x in spans]
        segs, left, i = [], target, 0
        while left > 0:
            _, _, n, p0, nbytes, _ = spans[i % len(spans)]
            addr = (p0 + 15) // 16 * 16
            nb = min(left, (p0 + nbytes - addr) // 16 * 16)
            if nb >= 16:
                segs.append((addr, nb))
                left -= nb
            i += 1
            assert len(segs) <= 6, f"PX shadow {kind}: {target} B needs > 6 segments from {params[:6]}"
        cache = segs
        module._px_shadow_cache = cache
        if kind not in _px_logged:
            _px_logged.add(kind)
            logger.warning(
                "K31_PX shadow %s: %d B per source from %s", kind, sum(b for _, b in segs),
                params[:6],
            )
    _px_src = cache


def _px_take() -> list[tuple[int, int]]:
    global _px_src
    s, _px_src = (_px_src or []), None
    return s


# PX arms C / D (rev 5 follow-up, 2026-11-26). Regions of a slot's dead tail (bytes from px_tail; PX shadow bytes of
# arms A / B occupy [0, 409 MB) in DSA layers and [0, 91 MB) in KDA layers):
#   a2a  : the head->token all-to-all destination [R, 12288] E4M3 + [R / 128, 96 x 512 B] scales (C, KDA layers);
#   src  : the o_norm output quantized to MXFP8, [W R, 1536] E4M3 + scales (C's source; the GemmRsAr still runs on BF16);
#   wsf  : constant UE8M0 weight scales of D's [7168, 12288] KDA o_proj stand-in (last 3 MiB of the tail);
#   D outputs (discarded) go to the OTHER slot's tail from byte 0.
#   a2a  : flag "a2a"  -> arm C, every KDA layer;   dgemm : flag "dgemm" -> arm D on 8 KDA + 4 DSA layers per forward.
_PX_A2A_OFF = 96 << 20
_PX_SRC_OFF = 152 << 20
_PX_WSF_OFF_FROM_END = 3 << 20
_PX_WSF_BYTES = 7168 * 12288 // 32
_PX_BR = 16  # rows per 2D box (192 x u64 = 1536 B wide): 24 KiB per unit
_PX_KDA_D_EVERY, _PX_KDA_D_AT = 9, 4  # KDA layers 4, 13, ..., 67 of 69
_PX_DSA_D_EVERY, _PX_DSA_D_AT = 6, 2  # DSA layers 2, 8, 14, 20 of 24


def px_kda_out(core_attn_out: torch.Tensor) -> None:
    """Arm C: the KDA o_norm output ([W R, 1536] BF16) as TLX's head->token all-to-all, into the slot tail."""
    g = _sp_all_gather
    if g is not None and "a2a" in _PX:
        g.px_kda_a2a(core_attn_out)


def px_after_attn(module: torch.nn.Module, kind: str, x: Any) -> None:
    """Arm D: token-local stand-in GEMMs on R rows after the attention output (GemmRsAr) of sampled layers."""
    g = _sp_all_gather
    if g is not None and "dgemm" in _PX:
        g.px_dgemm(module, kind, x)


class SymmSpAllGather:
    """Two-slot symmetric output buffer of the TP group, sized for
    ``max_rows`` rows per rank (the token shard of the largest step)."""

    def __init__(
        self, max_rows: int, hidden_size: int, dtype: torch.dtype, device
    ) -> None:
        tp = get_tp_group()
        self.world = tp.world_size
        self.rank = tp.rank_in_group
        self.max_rows = max_rows
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.slot_numel = self.world * max_rows * hidden_size
        self.num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.storage = symm_mem.empty(
            _NUM_SLOTS * self.slot_numel, dtype=dtype, device=device
        )
        self.handle = symm_mem.rendezvous(self.storage, tp.device_group.group_name)
        assert self.handle is not None, "SP all-gather rendezvous returned None"
        numel = _NUM_SLOTS * self.slot_numel
        bases = [
            self.handle.get_buffer(peer, [numel], dtype, 0).data_ptr()
            for peer in range(self.world)
        ]
        esize = torch.empty((), dtype=dtype).element_size()
        self.px_bases = bases  # PX probe: host copies of the peers' buffer addresses
        self.px_slot_bytes = self.slot_numel * esize
        self.px_slot = 0
        self.px_R = None
        self.px_n = {"kda": 0, "dsa": 0}
        self.px_rg: dict = {}
        # [slot, peer] absolute address of the slot in the peer's buffer.
        self.peer_ptrs = torch.tensor(
            [
                [base + slot * self.slot_numel * esize for base in bases]
                for slot in range(_NUM_SLOTS)
            ],
            dtype=torch.int64,
            device=self.storage.device,
        )
        # Gathers of the current forward: announced and pushed so far.
        self.num_gathers = 0
        self.count = 0
        # TMA push of BF16 shards (module docstring): per slot, one arrival
        # counter word at the end of each rank's signal pad (the barrier uses
        # its first words); [slot, peer] absolute addresses.
        self.tma = tma_push_capable(device)
        if self.tma:
            pads, pad_size = self.handle.signal_pad_ptrs, self.handle.signal_pad_size
            self.ctr_ptrs = torch.tensor(
                [
                    [pad + pad_size - 4 * (_NUM_SLOTS - slot) for pad in pads]
                    for slot in range(_NUM_SLOTS)
                ],
                dtype=torch.int64,
                device=self.storage.device,
            )
            # Per slot, this rank's count of push CTAs done (the two words
            # below the counters; only this rank touches them).
            own_pad = pads[self.rank] + pad_size
            self.done_ptrs = [own_pad - 4 * (2 * _NUM_SLOTS - slot) for slot in range(_NUM_SLOTS)]
            # Build and load the kernel now (an empty push): CUDA lazy loading
            # would load it at its first launch, inside a step.
            _tma_module().sp_ag_tma_push(
                self.storage[:0], self.peer_ptrs[0], 0, self.ctr_ptrs[0], self.done_ptrs[0],
                self.rank, self.world, _TMA_CHUNK, 1, _TMA_SPIN_NS,
            )  # fmt: skip
            if _PX:
                _tma_module().px_tma_push(
                    [], [], [], self.peer_ptrs[0], self.ctr_ptrs[0], self.done_ptrs[0],
                    self.rank, self.world, _TMA_CHUNK, 1, _TMA_SPIN_NS,
                )  # fmt: skip
                _tma_module().px_mark(0)  # loads the module (px_a2a lives in it too)
        # PX: dead tail of each slot (bytes), after the MXFP8 scale region; shadow bytes land at tail + rank * B.
        esz = torch.empty((), dtype=dtype).element_size()
        self.px_tail = self.world * max_rows * hidden_size * (1 + 1 / MXFP8_BLOCK)
        self.px_tail = int(self.px_tail + 15) // 16 * 16
        self.px_tail_bytes = self.slot_numel * esz - self.px_tail
        if _PX:
            need = self.world * max(_PX_KDA_W_BYTES, _PX_DSA_W_BYTES if "dsa_w" in _PX else 0)
            assert self.tma, "K31_PX needs the TMA push (SM 10.x)"
            assert need <= self.px_tail_bytes, f"K31_PX: shadow {need} B > slot tail {self.px_tail_bytes} B"
            logger.warning("K31_PX=%s: slot tail %d B from byte %d (max_rows %d), allocates nothing",
                           ",".join(sorted(_PX)), self.px_tail_bytes, self.px_tail, max_rows)
        torch.accelerator.synchronize()
        self.handle.barrier()

    def begin_forward(self, num_gathers: int) -> None:
        self.num_gathers = num_gathers
        self.count = 0
        self.px_n = {"kda": 0, "dsa": 0}
        self.px_R = None

    def _check_shard(self, rows: int, hidden: int, dtype: torch.dtype) -> int:
        """Validate a gather of ``rows`` x ``hidden`` ``dtype`` rows per rank
        and take its slot."""
        if hidden != self.hidden_size or dtype != self.dtype:
            raise ValueError(
                f"SP all-gather buffer holds {self.hidden_size}-wide {self.dtype} "
                f"rows, got {rows} x {hidden} {dtype}"
            )
        if rows > self.max_rows:
            # The buffer covers every step of the instance
            # (max_num_batched_tokens): a larger shard is a bug.
            raise ValueError(
                f"SP all-gather shard of {rows} rows exceeds the buffer's "
                f"{self.max_rows} rows per rank (max_num_batched_tokens)"
            )
        # The slot schedule (module docstring) holds only for the gathers the
        # forward announced.
        assert self.count < self.num_gathers, (
            f"SP all-gather {self.count + 1} of a forward that announced "
            f"{self.num_gathers} (begin_forward)"
        )
        slot = self.count % _NUM_SLOTS
        if self.count == 0 and self.num_gathers % _NUM_SLOTS:
            # The previous forward's last push used slot 0 as well.
            self.handle.barrier()
        self.count += 1
        self.px_slot = slot
        return slot

    def _px_push(self, slot: int, segs: list[tuple[int, int, int]]) -> None:
        """One PX push of (src address, bytes, byte offset in each rank's slot) segments; counted completion."""
        shadow = _px_take()
        for j, (a, nb) in enumerate(shadow):
            off = self.px_tail + self.rank * sum(b for _, b in shadow) + sum(b for _, b in shadow[:j])
            segs.append((a, nb, off))
        units = sum(cdiv(nb, _TMA_CHUNK) for _, nb, _ in segs) * self.world
        _tma_module().px_tma_push(
            [a for a, _, _ in segs], [nb for _, nb, _ in segs], [o for _, _, o in segs],
            self.peer_ptrs[slot], self.ctr_ptrs[slot], self.done_ptrs[slot], self.rank, self.world,
            _TMA_CHUNK, min(self.num_sms, units), _TMA_SPIN_NS,
        )  # fmt: skip

    def _px_raw(self, slot: int, off: int, nbytes: int) -> torch.Tensor:
        """uint8 view of this rank's slot tail bytes [off, off + nbytes) (asserted inside the tail)."""
        assert 0 <= off and off + nbytes <= self.px_tail_bytes, f"PX region {off}+{nbytes} > tail {self.px_tail_bytes}"
        o = slot * self.px_slot_bytes + self.px_tail + off
        return self.storage.view(torch.uint8)[o : o + nbytes]

    def px_kda_a2a(self, x: torch.Tensor) -> None:
        from vllm.models.kimi_k3.nvidia.ops import mxfp8_act

        M, W = x.shape[0], self.world
        self.px_R = None
        if x.dim() != 2 or x.shape[1] != 1536 or M % (W * 128) or 2 * (M // W) > self.max_rows or not x.is_contiguous():
            return  # init profile run (gathers reach the tail) or an unsupported shape: no a2a
        R, slot = M // W, self.px_slot
        sf_src = M * 1536
        src = self._px_raw(slot, _PX_SRC_OFF, sf_src + M * 48)
        buf = object.__new__(mxfp8_act.Mxfp8Buffer)
        buf.rows, buf.k, buf.padded_cols, buf.orig_dtype = M, 1536, 1536 // MXFP8_BLOCK, x.dtype
        buf.data = src[:sf_src].view(torch.float8_e4m3fn).view(M, 1536)
        buf.scale = src[sf_src:]
        mxfp8_act.quantize(x, out=buf)
        self._px_raw(slot, _PX_A2A_OFF, R * 12288 + R * 384)  # bounds check of the destination
        dst = [b + slot * self.px_slot_bytes + self.px_tail + _PX_A2A_OFF for b in self.px_bases]
        units = (R // _PX_BR + R // 128) * W
        _tma_module().px_a2a(
            src.data_ptr(), src.data_ptr() + sf_src, R, self.rank, dst, [d + R * 12288 for d in dst],
            self.ctr_ptrs[slot], self.done_ptrs[slot], self.rank, W, _PX_BR, min(self.num_sms, units), _TMA_SPIN_NS,
        )  # fmt: skip
        self.px_R, self.px_a2a_slot = R, slot

    def px_dgemm(self, module: torch.nn.Module, kind: str, x: Any) -> None:
        from types import SimpleNamespace

        n = self.px_n[kind]
        self.px_n[kind] = n + 1
        bf = torch.bfloat16
        mark = _tma_module().px_mark
        if kind == "kda":
            if n % _PX_KDA_D_EVERY != _PX_KDA_D_AT or self.px_R is None:
                return
            from vllm.models.kimi_k3.nvidia.specialized_prefill import attn_res_resid

            R, slot = self.px_R, self.px_a2a_slot
            a = self._px_raw(slot, _PX_A2A_OFF, R * 12288 + R * 384)
            qa = SimpleNamespace(data=a[: R * 12288].view(torch.float8_e4m3fn).view(R, 12288), scale=a[R * 12288 :])
            wsf = self._px_raw(slot, self.px_tail_bytes - _PX_WSF_OFF_FROM_END, _PX_WSF_BYTES)
            wsf.fill_(119)  # 2^-8: real o_proj weight scales sit near it; profile-run gathers overwrite the tail
            rg = self.px_rg.get(slot)
            if rg is None:
                w = self._px_raw(slot, 0, 7168 * 12288).view(torch.float8_e4m3fn).view(7168, 12288)
                rg = self.px_rg[slot] = attn_res_resid.ResidGemm.from_rows(w, wsf)
            out = self._px_raw(slot ^ 1, 0, R * 7168 * 2).view(bf).view(R, 7168)
            mark(1)
            rg.store_into(qa, out)
            mark(0)
            return
        if n % _PX_DSA_D_EVERY != _PX_DSA_D_AT or not isinstance(x, torch.Tensor) or x.dtype != bf:
            return
        M, W = x.shape[0], self.world
        if M % W or 2 * (M // W) > self.max_rows or x.shape[1] != 7168 or not x.is_contiguous():
            return
        R, slot = M // W, self.px_slot
        o = 0
        ws = []
        for nn_, k_ in ((18432, 1536), (12288, 7168), (7168, 12288)):
            ws.append(self._px_raw(slot, o, nn_ * k_ * 2).view(bf).view(nn_, k_))
            o += nn_ * k_ * 2
        outb = self._px_raw(slot ^ 1, 0, R * 18432 * 2)

        def out(cols: int) -> torch.Tensor:
            return outb[: R * cols * 2].view(bf).view(R, cols)

        lins = [getattr(module, "fused_qkv_a_proj", None), getattr(getattr(module, "dsa_indexer", None), "wkv_index", None)]
        lins = [m for m in lins if m is not None]
        if "dsa_d" not in _px_logged:
            _px_logged.add("dsa_d")
            logger.warning("K31_PX D: DSA R-row input linears %s; q_b|g|o stand-ins BF16 %s on R=%d",
                           [(type(m.quant_method).__name__, tuple(m.weight.shape), str(m.weight.dtype)) for m in lins],
                           [tuple(w.shape) for w in ws], R)
        xr = x[:R]
        mark(2)
        for m in lins:
            if m.weight.dtype == bf and m.weight.dim() == 2 and m.weight.shape[1] == 7168:
                torch.mm(xr, m.weight.t(), out=out(m.weight.shape[0]))
            else:
                m(xr)
        mark(3)
        torch.mm(x[:R, :1536], ws[0].t(), out=out(18432))
        mark(4)
        torch.mm(xr, ws[1].t(), out=out(12288))
        mark(5)
        torch.mm(x.view(-1)[: R * 12288].view(R, 12288), ws[2].t(), out=out(7168))
        mark(0)

    def gather_mxfp8(
        self, data: torch.Tensor, scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather an MXFP8 shard: ``data`` [rows, hidden] E4M3 (rows a
        multiple of 128) and ``scale`` its F8_128x4-swizzled E8M0 bytes
        (rows * hidden / 32). Returns the gathered [world * rows, hidden]
        E4M3 rows and the [world * rows * hidden / 32] uint8 swizzled scales
        (views of the slot; module docstring)."""
        rows, hidden = data.shape
        if rows % MXFP8_ROW_TILE or data.element_size() != 1:
            raise ValueError(
                f"SP MXFP8 all-gather: {rows} x {hidden} {data.dtype} shard "
                f"(1-byte rows, a multiple of {MXFP8_ROW_TILE})"
            )
        sf_bytes = rows * hidden // MXFP8_BLOCK
        scale = scale.reshape(-1)
        if scale.numel() * scale.element_size() != sf_bytes:
            raise ValueError(
                f"SP MXFP8 all-gather: {scale.numel()} scale elements for a "
                f"{rows} x {hidden} shard (expected {sf_bytes} bytes)"
            )
        slot = self._check_shard(rows, hidden, self.dtype)
        data_words = data.contiguous().view(torch.int32).reshape(-1)
        sf_words = scale.contiguous().view(torch.int32)
        # Slot layout in bytes: rows from 0, scales from world * max_rows *
        # hidden (inside the BF16 slot of 2 * world * max_rows * hidden).
        sf_base = self.world * self.max_rows * hidden
        if "kda_tma" in _PX and data_words.data_ptr() % 16 == 0 and sf_words.data_ptr() % 16 == 0:
            self._px_push(slot, [
                (data_words.data_ptr(), data_words.numel() * 4, self.rank * data_words.numel() * 4),
                (sf_words.data_ptr(), sf_bytes, sf_base + self.rank * sf_bytes),
            ])  # fmt: skip
            start = slot * self.slot_numel * self.storage.element_size()
            raw = self.storage.view(torch.uint8)
            out = raw[start : start + self.world * rows * hidden]
            sf = raw[start + sf_base : start + sf_base + self.world * sf_bytes]
            return out.view(data.dtype).view(self.world * rows, hidden), sf
        _px_take()
        _push_shard2_kernel[
            (
                min(
                    self.num_sms,
                    cdiv(data_words.numel(), _WORD_BLOCK)
                    + cdiv(sf_words.numel(), _WORD_BLOCK),
                ),
            )
        ](
            data_words,
            sf_words,
            self.peer_ptrs[slot],
            self.rank * data_words.numel(),
            (sf_base + self.rank * sf_bytes) // 4,
            data_words.numel(),
            sf_words.numel(),
            self.rank,
            WORLD=self.world,
            BLOCK=_WORD_BLOCK,
            num_warps=_NUM_WARPS,
        )
        self.handle.barrier()
        start = slot * self.slot_numel * self.storage.element_size()
        raw = self.storage.view(torch.uint8)
        out = raw[start : start + self.world * rows * hidden]
        sf = raw[start + sf_base : start + sf_base + self.world * sf_bytes]
        return out.view(data.dtype).view(self.world * rows, hidden), sf

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        rows, hidden = x.shape
        slot = self._check_shard(rows, hidden, x.dtype)
        x = x.contiguous()
        numel = rows * hidden
        nbytes = numel * x.element_size()
        if self.world * nbytes > self.px_tail:
            _px_take()  # a gather this large (init-time profile run at MNBT rows) reaches the tail: no shadow bytes
        if self.tma and nbytes % 16 == 0 and x.data_ptr() % 16 == 0 and _px_src:
            self._px_push(slot, [(x.data_ptr(), nbytes, self.rank * nbytes)])
            start = slot * self.slot_numel
            return self.storage[start : start + self.world * numel].view(
                self.world * rows, hidden
            )
        _px_take()
        if self.tma and nbytes % 16 == 0 and x.data_ptr() % 16 == 0:
            # Every rank adds to this rank's slot counter once all its bytes
            # landed: the push returns with the gather complete, and no
            # barrier is needed (module docstring).
            _tma_module().sp_ag_tma_push(
                x, self.peer_ptrs[slot], self.rank * nbytes, self.ctr_ptrs[slot],
                self.done_ptrs[slot], self.rank, self.world, _TMA_CHUNK,
                min(self.num_sms, cdiv(nbytes, _TMA_CHUNK) * self.world), _TMA_SPIN_NS,
            )  # fmt: skip
            start = slot * self.slot_numel
            return self.storage[start : start + self.world * numel].view(
                self.world * rows, hidden
            )
        src, block = x.reshape(-1), _BLOCK
        if (hidden * x.element_size()) % 4 == 0:
            # Push 32-bit words (the same bytes): Triton's 16-bit copy code
            # moves 0.86 TB/s against 1.06 for 32-bit words at 8 ranks.
            src, block = x.view(torch.int32).reshape(-1), _WORD_BLOCK
        words = src.numel()
        _push_shard_kernel[(min(self.num_sms, cdiv(words, block)),)](
            src,
            self.peer_ptrs[slot],
            self.rank * words,
            words,
            self.rank,
            WORLD=self.world,
            BLOCK=block,
            num_warps=_NUM_WARPS,
        )
        self.handle.barrier()
        start = slot * self.slot_numel
        return self.storage[start : start + self.world * numel].view(
            self.world * rows, hidden
        )


_sp_all_gather: SymmSpAllGather | None = None


def is_decode_instance(vllm_config: Any) -> bool:
    """A P/D decode instance: KV consumer, not producer."""
    kv_transfer = getattr(vllm_config, "kv_transfer_config", None)
    return (
        kv_transfer is not None
        and kv_transfer.is_kv_consumer
        and not kv_transfer.is_kv_producer
    )


def sp_all_gather_rows(vllm_config: Any, hidden_size: int, dtype: torch.dtype) -> int:
    """Rows per rank the push buffer holds for this instance (module
    docstring): the token shard of a max_num_batched_tokens step on a prefill or
    unified instance, or 0 (no buffer) on a P/D decode instance or when that
    shard fits the custom all-gather, which then takes every step's gather."""
    if is_decode_instance(vllm_config):
        return 0
    tp_size = vllm_config.parallel_config.tensor_parallel_size
    rows = cdiv(vllm_config.scheduler_config.max_num_batched_tokens, tp_size)
    return 0 if _fits_custom_all_gather(rows, hidden_size, dtype) else rows


def _fits_custom_all_gather(rows: int, hidden_size: int, dtype: torch.dtype) -> bool:
    communicator = get_tp_group().device_communicator
    ca_comm = getattr(communicator, "ca_comm", None)
    return ca_comm is not None and ca_comm.should_custom_all_gather(
        torch.empty((rows, hidden_size), dtype=dtype, device="meta")
    )


def init_sp_all_gather(vllm_config: Any, hidden_size: int, dtype: torch.dtype) -> bool:
    """Collectively allocate the SP all-gather buffer this instance needs
    (``sp_all_gather_rows``; none needed: False). Capability check only:
    without NVLink symmetric memory over the TP group the stock all-gather
    stays (logged)."""
    max_rows = sp_all_gather_rows(vllm_config, hidden_size, dtype)
    if max_rows == 0:
        logger.info_once(
            "Kimi K3 SP all-gather: no symmetric-memory buffer on this instance "
            "(P/D decode, or every step fits the custom all-gather); stock "
            "all-gather."
        )
        return False
    return init_sp_all_gather_buffer(max_rows, hidden_size, dtype)


def init_sp_all_gather_buffer(
    max_rows: int, hidden_size: int, dtype: torch.dtype
) -> bool:
    """Collectively allocate the two-slot buffer for ``max_rows`` rows per rank
    (False: no symmetric memory over the TP group, stock all-gather, logged)."""
    global _sp_all_gather
    if _sp_all_gather is not None:
        assert (
            _sp_all_gather.max_rows >= max_rows
            and _sp_all_gather.hidden_size == hidden_size
            and _sp_all_gather.dtype == dtype
        )
        return True
    tp = get_tp_group()
    if symm_mem is None or not direct_cp_enabled(tp, dtype, None):
        logger.warning_once(
            "Kimi K3 SP all-gather: the TP group has no symmetric memory; "
            "using the stock all-gather."
        )
        return False
    device = torch.device("cuda", torch.accelerator.current_device_index())
    _sp_all_gather = SymmSpAllGather(max_rows, hidden_size, dtype, device)
    logger.info_once(
        "Kimi K3 SP all-gather: symmetric-memory push, %d rows x %d per rank, %.0f MiB",
        max_rows,
        hidden_size,
        _sp_all_gather.storage.numel() * _sp_all_gather.storage.element_size() / 2**20,
    )
    return True


def begin_forward(num_gathers: int) -> None:
    """Announce a forward of ``num_gathers`` decoder-layer gathers (one per
    layer, so the same count every forward, which the cross-forward slot rule
    assumes); restarts the slot schedule (module docstring)."""
    if _sp_all_gather is not None:
        _sp_all_gather.begin_forward(num_gathers)


def kimi_sp_all_gather(x: torch.Tensor) -> torch.Tensor:
    """The decoder layer's SP all-gather. Without the buffer (P/D decode
    instance, small steps only, or no symmetric memory) the stock all-gather.
    With it: shards the stock custom (IPC) all-gather takes (small, e.g.
    graph-captured steps) keep it, larger ones take the symmetric-memory push;
    a shard above the buffer raises (``SymmSpAllGather.__call__``)."""
    from vllm.models.common.ops.sequence_parallel import (
        _custom_collective,
        sp_all_gather,
    )

    gather = _sp_all_gather
    if gather is None:
        return sp_all_gather(x)
    output = _custom_collective("custom_all_gather", x)
    if output is not None:
        return output
    return gather(x)


def mxfp8_gather_rows_ok(rows: int, hidden: int) -> bool:
    """A decoder-layer gather of ``rows`` x ``hidden`` BF16 rows per rank may
    go as MXFP8 (``kimi_sp_all_gather_mxfp8``): the symmetric-memory buffer
    takes the shard (the stock custom / Lamport all-gather would not: it stays
    BF16) and ``rows`` is a positive multiple of the 128-row scale tile."""
    gather = _sp_all_gather
    if (
        gather is None
        or gather.dtype != torch.bfloat16
        or hidden != gather.hidden_size
        or hidden % (4 * MXFP8_BLOCK)
        or rows <= 0
        or rows % MXFP8_ROW_TILE
        or rows > gather.max_rows
    ):
        return False
    return not _takes_custom_all_gather(rows, hidden, gather.dtype)


def _takes_custom_all_gather(rows: int, hidden: int, dtype: torch.dtype) -> bool:
    """``kimi_sp_all_gather`` sends this BF16 shard to the stock custom
    all-gather (the predicate of ``custom_all_gather``)."""
    communicator = get_tp_group().device_communicator
    ca_comm = getattr(communicator, "ca_comm", None)
    return ca_comm is not None and ca_comm.should_custom_all_gather(
        torch.empty((rows, hidden), dtype=dtype, device="meta")
    )


def kimi_sp_all_gather_mxfp8(qa: Any, num_tokens: int) -> Any:
    """The decoder layer's SP all-gather of an MXFP8 activation (module
    docstring) for a shard ``mxfp8_gather_rows_ok`` accepted: the gathered
    :class:`QuantizedActivation` of the first ``num_tokens`` rows (SP padding
    removed), bitwise ``mxfp8_quantize(all_gather(bf16))[:num_tokens]`` on
    every real row."""
    from dataclasses import replace

    gather = _sp_all_gather
    assert gather is not None
    rows, hidden = qa.data.shape
    data, sf = gather.gather_mxfp8(qa.data, qa.scale)
    sf_rows = cdiv(num_tokens, MXFP8_ROW_TILE) * MXFP8_ROW_TILE
    return replace(
        qa,
        data=data[:num_tokens],
        scale=sf[: sf_rows * hidden // MXFP8_BLOCK],
        orig_shape=torch.Size((num_tokens, hidden)),
    )
