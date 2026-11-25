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

Overlapped MXFP8 gather (PR-3, ``gather_mxfp8_overlapped``): on an eager
prefill step whose KDA in_proj runs the fused sm_107 Q|K|V GEMM
(``specialized_prefill.kda_conv_in_proj``), the gather does not run before the
GEMM. The rank copies its own shard into its slot, raises its arrival word (the
gather index) and hands the GEMM an ``AgPending``. The GEMM starts on the own
shard and walks the M clusters from there (rank, rank + 1, ...); a pusher grid
launched as its PDL secondary lands on the SMs the GEMM grid leaves idle (28 of
212 at the TP8 tactic) and TMA-pushes this rank's shard into every peer's slot,
peer rank - 1 first (the order each peer's GEMM needs it), one 1024-row cluster
tile at a time, each tile closed by a counted, release-ordered add on the
peer's tile counter. The GEMM's TMA producer waits for that count before
loading a peer tile. A pusher writes a peer's slot only after the peer's
arrival word reached this gather (the peer's stream finished every read of the
slot's previous gather), so no barrier is needed; a stock gather that follows
an overlapped one waits for every peer's arrival first. The bytes are the same
as the stock gather's (pure copy), and the GEMM computes every output tile
exactly as before (only the tile order changes). Every spin is bounded and
traps. Capture, non-fused consumers and other shapes keep the stock gather (a
consumer that is not the fused GEMM completes a pending gather first).
"""

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


_AG_CHUNK = 64 << 10  # pusher TMA chunk (64 KiB, 3-slot ring = 192 KiB smem)
_AG_NS = 3
_AG_SPIN_NS = 2_000_000_000  # 2 s: trap instead of hanging
_AG_FLAG_WORDS = 1024  # [0] arrival word, [64:] per-cluster-tile counters

_AG_CUDA = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ uint64_t ag_now() { uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint32_t ag_sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint32_t ag_ld_acq(uint64_t a) { uint32_t v; asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory"); return v; }

__device__ __forceinline__ void ag_spin_ge(uint64_t addr, uint32_t target, uint64_t limit) {
  const uint64_t t0 = ag_now();
  while ((int32_t)(ag_ld_acq(addr) - target) < 0) {
    __nanosleep(128);
    if (ag_now() - t0 > limit) { printf("K31_SPIN_TIMEOUT kernel=sp_ag_push site=arrival\n"); __trap(); }
  }
}

// One issuing thread per CTA. Units u = (j, b): peer d = rank - j (j = 1..W-1),
// cluster tile b of this rank's shard; unit chunks = the tile's F8_128x4 scale
// bytes, then its rows, in 64 KiB pieces; chunk c of the gather goes to CTA
// c % G. A unit's chunks of this CTA are closed with one counted add on the
// peer's counter of that cluster tile once their stores completed.
template <int NS>
__global__ void __launch_bounds__(32, 1) sp_ag_push_kernel(
    uint64_t own_slot, const int64_t* __restrict__ peer_slot, const int64_t* __restrict__ peer_flags,
    int rank, int W, int64_t rows, int64_t hidden, int64_t sf_base, int64_t cl_rows, int chunk,
    uint32_t arrive_target, uint64_t limit) {
  extern __shared__ __align__(1024) uint8_t smem[];
  __shared__ __align__(8) uint64_t bar[NS];
  if (threadIdx.x != 0) return;
  for (int i = 0; i < NS; ++i) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(ag_sa(&bar[i])));
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  const int G = gridDim.x, g = blockIdx.x;
  const int64_t nblk = rows / cl_rows;
  const int64_t sfb = cl_rows * hidden / 32, db = cl_rows * hidden;
  const int64_t nsf = (sfb + chunk - 1) / chunk, nd = (db + chunk - 1) / chunk, cpu = nsf + nd;
  const int64_t nunits = (int64_t)(W - 1) * nblk, total = nunits * cpu;
  const int64_t cl_base = (int64_t)rank * rows / cl_rows;  // global cluster index of own tile 0
  // this CTA's chunks: c = g, g + G, ...
  const int64_t n = g < total ? (total - g + G - 1) / G : 0;
  struct Pend { int64_t unit; uint32_t cnt; int64_t last; };
  Pend q[16]; int qh = 0, qt = 0;
  int64_t cur_unit = -1; uint32_t cur_cnt = 0; uint32_t seen = 0;
  uint32_t ph = 0;
  auto unit_of = [&](int64_t k) { return (g + k * (int64_t)G) / cpu; };
  auto signal = [&](int64_t unit, uint32_t cnt) {
    const int j = 1 + (int)(unit / nblk); const int64_t b = unit % nblk;
    const int d = (rank - j + W) % W;
    const uint64_t ctr = (uint64_t)peer_flags[d] + 4ull * (64 + cl_base + b);
    asm volatile("fence.proxy.async.global;" ::: "memory");
    asm volatile("red.release.sys.global.add.u32 [%0], %1;" :: "l"(ctr), "r"(cnt) : "memory");
  };
  auto addr_of = [&](int64_t k, uint64_t& src, uint64_t& dst, uint32_t& nb) {
    const int64_t c = g + k * (int64_t)G, unit = c / cpu, i = c % cpu;
    const int j = 1 + (int)(unit / nblk); const int64_t b = unit % nblk;
    const int d = (rank - j + W) % W;
    int64_t off, bytes;
    if (i < nsf) {
      const int64_t o = i * chunk; bytes = min((int64_t)chunk, sfb - o);
      off = sf_base + ((int64_t)rank * rows / cl_rows + b) * sfb + o;
    } else {
      const int64_t o = (i - nsf) * chunk; bytes = min((int64_t)chunk, db - o);
      off = ((int64_t)rank * rows + b * cl_rows) * hidden + o;
    }
    if (!((seen >> d) & 1u)) {  // the peer finished reading this slot's previous gather
      ag_spin_ge((uint64_t)peer_flags[d], arrive_target, limit);
      seen |= 1u << d;
    }
    src = own_slot + off; dst = (uint64_t)peer_slot[d] + off; nb = (uint32_t)bytes;
  };
  uint64_t dsts[NS]; uint32_t nbs[NS];
  auto issue = [&](int64_t k) {
    uint64_t src, dst; uint32_t nb; addr_of(k, src, dst, nb);
    const int s = (int)(k % NS); dsts[s] = dst; nbs[s] = nb;
    asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}" :: "r"(ag_sa(&bar[s])), "r"(nb) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(ag_sa(smem + (size_t)s * chunk)), "l"(src), "r"(nb), "r"(ag_sa(&bar[s])) : "memory");
  };
  for (int64_t k = 0; k < NS - 1 && k < n; ++k) issue(k);
  for (int64_t k = 0; k < n; ++k) {
    const int s = (int)(k % NS);
    const uint32_t a = ag_sa(&bar[s]), p = (ph >> s) & 1u;
    asm volatile("{\n.reg .pred q;\nW_%=:\nmbarrier.try_wait.parity.shared::cta.b64 q, [%0], %1;\n@!q bra W_%=;\n}" :: "r"(a), "r"(p) : "memory");
    ph ^= 1u << s;
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" :: "l"(dsts[s]), "r"(ag_sa(smem + (size_t)s * chunk)), "r"(nbs[s]) : "memory");
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    const int64_t u = unit_of(k);
    if (u != cur_unit) {
      if (cur_unit >= 0) { q[qt & 15] = {cur_unit, cur_cnt, k - 1}; ++qt; }
      cur_unit = u; cur_cnt = 0;
    }
    ++cur_cnt;
    // stores up to k - 4 have completed (writes performed): close their units
    asm volatile("cp.async.bulk.wait_group 4;" ::: "memory");
    while (qh != qt && q[qh & 15].last <= k - 4) { signal(q[qh & 15].unit, q[qh & 15].cnt); ++qh; }
    asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory");  // slot of store k - 1 reusable
    if (k + NS - 1 < n) issue(k + NS - 1);
  }
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  while (qh != qt) { signal(q[qh & 15].unit, q[qh & 15].cnt); ++qh; }
  if (cur_unit >= 0) signal(cur_unit, cur_cnt);
  // PDL secondary of the GEMM: complete only after it (later kernels see both).
  asm volatile("griddepcontrol.wait;" ::: "memory");
}

// Wait (one thread) until every peer cluster tile's counter reaches target.
__global__ void sp_ag_wait_kernel(uint64_t own_flags, int64_t ncl, int64_t lo, int64_t hi, uint32_t target, uint64_t limit) {
  for (int64_t c = 0; c < ncl; ++c)
    if (c < lo || c >= hi) {
      const uint64_t a = own_flags + 4ull * (64 + c);
      const uint64_t t0 = ag_now();
      while ((int32_t)(ag_ld_acq(a) - target) < 0) {
        __nanosleep(128);
        if (ag_now() - t0 > limit) { printf("K31_SPIN_TIMEOUT kernel=sp_ag_wait site=tile\n"); __trap(); }
      }
    }
  __threadfence_system();
}
// Wait until every peer's arrival word reached target.
__global__ void sp_ag_arrivals_kernel(const int64_t* peer_flags, int W, uint32_t target, uint64_t limit) {
  for (int d = 0; d < W; ++d) ag_spin_ge((uint64_t)peer_flags[d], target, limit);
}
// Raise the arrival word. An overlapped gather also tops up its own tile
// counters and those past its tiles by `add` (the pushers add it to every peer
// tile), so every counter gains the same count per overlapped gather whatever
// the step's row count: one target (gather ordinal x add) for all tiles.
__global__ void sp_ag_arrive_kernel(uint64_t own_flags, uint32_t v, int64_t lo, int64_t hi, int64_t ncl,
                                    int64_t maxcl, uint32_t add) {
  if (add)
    for (int64_t c = 0; c < maxcl; ++c)
      if ((c >= lo && c < hi) || c >= ncl) atomicAdd((unsigned*)(own_flags + 4ull * (64 + c)), add);
  __threadfence_system();
  asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(own_flags), "r"(v) : "memory");
}

static cudaStream_t ag_cur() { return c10::cuda::getCurrentCUDAStream().stream(); }

void sp_ag_push(int64_t own_slot, torch::Tensor peer_slot, torch::Tensor peer_flags, int64_t rank, int64_t W,
                int64_t rows, int64_t hidden, int64_t sf_base, int64_t cl_rows, int64_t chunk,
                int64_t arrive_target, int64_t grid, bool pdl, int64_t limit) {
  const size_t smem = (size_t)3 * chunk;
  auto k = sp_ag_push_kernel<3>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)grid); cfg.blockDim = dim3(32); cfg.dynamicSmemBytes = smem; cfg.stream = ag_cur();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cfg.attrs = at; cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k, (uint64_t)own_slot, (const int64_t*)peer_slot.data_ptr<int64_t>(),
      (const int64_t*)peer_flags.data_ptr<int64_t>(), (int)rank, (int)W, rows, hidden, sf_base, cl_rows, (int)chunk,
      (uint32_t)arrive_target, (uint64_t)limit));
}
void sp_ag_wait(int64_t own_flags, int64_t ncl, int64_t lo, int64_t hi, int64_t target, int64_t limit) {
  sp_ag_wait_kernel<<<1, 1, 0, ag_cur()>>>((uint64_t)own_flags, ncl, lo, hi, (uint32_t)target, (uint64_t)limit);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void sp_ag_arrivals(torch::Tensor peer_flags, int64_t W, int64_t target, int64_t limit) {
  sp_ag_arrivals_kernel<<<1, 1, 0, ag_cur()>>>((const int64_t*)peer_flags.data_ptr<int64_t>(), (int)W, (uint32_t)target, (uint64_t)limit);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void sp_ag_arrive(int64_t own_flags, int64_t v, int64_t lo, int64_t hi, int64_t ncl, int64_t maxcl, int64_t add) {
  sp_ag_arrive_kernel<<<1, 1, 0, ag_cur()>>>((uint64_t)own_flags, (uint32_t)v, lo, hi, ncl, maxcl, (uint32_t)add);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_AG_CPP = r"""
#include <torch/extension.h>
void sp_ag_push(int64_t, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                int64_t, int64_t, int64_t, bool, int64_t);
void sp_ag_wait(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
void sp_ag_arrivals(torch::Tensor, int64_t, int64_t, int64_t);
void sp_ag_arrive(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
"""

_ag_ext = None


def _ag_module():
    global _ag_ext
    if _ag_ext is None:
        import hashlib

        from torch.utils.cpp_extension import load_inline

        tag = hashlib.sha256((_AG_CUDA + _AG_CPP).encode()).hexdigest()[:12]
        _ag_ext = load_inline(
            name=f"k31_sp_ag_push_{tag}",
            cpp_sources=_AG_CPP,
            cuda_sources=_AG_CUDA,
            functions=["sp_ag_push", "sp_ag_wait", "sp_ag_arrivals", "sp_ag_arrive"],
            extra_cuda_cflags=["-O3", "-gencode", "arch=compute_100f,code=sm_100f"],
            verbose=False,
        )
    return _ag_ext


class AgPending:
    """An overlapped MXFP8 gather whose peer shards have not been pushed yet
    (module docstring). The fused Q|K|V GEMM takes it (``ctr_ptr``, ``target``,
    ``rot``, ``lo``, ``hi``) and calls ``launch_pushers`` right after its own
    launch; any other consumer calls ``complete`` first."""

    def __init__(self, gather: "SymmSpAllGather", slot: int, rows: int, hidden: int,
                 cl_rows: int, target: int, arrive: int) -> None:  # fmt: skip
        self.gather, self.slot, self.rows, self.hidden = gather, slot, rows, hidden
        self.cl_rows = cl_rows
        self.target = target
        self.arrive = arrive
        self.ncl = gather.world * rows // cl_rows
        self.lo = gather.rank * rows // cl_rows
        self.hi = self.lo + rows // cl_rows
        self.rot = self.lo
        self.ctr_ptr = gather.flags.data_ptr() + 4 * 64
        self.done = False

    def _push(self, pdl: bool) -> None:
        if self.done:
            raise RuntimeError("SP all-gather: pending gather pushed twice")
        self.done = True
        g = self.gather
        _ag_module().sp_ag_push(
            g.storage.data_ptr() + self.slot * g.slot_numel * g.storage.element_size(),
            g.peer_ptrs[self.slot], g.peer_flags, g.rank, g.world, self.rows, self.hidden,
            g.world * g.max_rows * self.hidden, self.cl_rows, _AG_CHUNK, self.arrive,
            g.idle_sms, pdl, _AG_SPIN_NS,
        )  # fmt: skip
        if g.pending is self:
            g.pending = None

    def launch_pushers(self) -> None:
        """Launch the pusher grid as the PDL secondary of the kernel just
        launched on this stream (the fused GEMM)."""
        self._push(True)

    def complete(self) -> None:
        """Make the gathered tensor whole on this stream (non-fused consumer)."""
        self._push(False)
        _ag_module().sp_ag_wait(
            self.gather.flags.data_ptr(), self.ncl, self.lo, self.hi, self.target, _AG_SPIN_NS
        )


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
        # Overlapped MXFP8 gather (module docstring): 4 KiB of flags per rank
        # ([0] arrival word = gather index, [64:] cluster-tile counters).
        self.flags = symm_mem.empty(_AG_FLAG_WORDS, dtype=torch.int32, device=device)
        self.flags.zero_()
        self.flags_handle = symm_mem.rendezvous(self.flags, tp.device_group.group_name)
        self.peer_flags = torch.tensor(
            [
                self.flags_handle.get_buffer(p, [_AG_FLAG_WORDS], torch.int32, 0).data_ptr()
                for p in range(self.world)
            ],
            dtype=torch.int64,
            device=device,
        )
        # Set by the fused Q|K|V GEMM at weight loading (enable_overlap).
        self.idle_sms = 0
        self.cl_rows = 0
        self.gidx = 0  # gathers so far (all forwards): the arrival word
        self.kidx = 0  # overlapped gathers so far: the tile counters' epoch
        self.last_overlapped = False
        self.pending: AgPending | None = None
        torch.accelerator.synchronize()
        self.handle.barrier()

    def begin_forward(self, num_gathers: int) -> None:
        self.num_gathers = num_gathers
        self.count = 0

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
        if self.pending is not None:
            raise RuntimeError(
                "SP all-gather: the previous overlapped gather was never consumed"
            )
        slot = self.count % _NUM_SLOTS
        if self.count == 0 and self.num_gathers % _NUM_SLOTS:
            # The previous forward's last push used slot 0 as well.
            self.handle.barrier()
        self.count += 1
        self.gidx += 1
        return slot

    def _arrive(self, lo: int = 0, hi: int = 0, ncl: int = 0, add: int = 0) -> None:
        """Arrival word = this gather's index: this rank's stream finished every
        read of the slot's previous gather (pushers wait for it)."""
        if self.idle_sms:
            _ag_module().sp_ag_arrive(
                self.flags.data_ptr(), self.gidx, lo, hi, ncl, _AG_FLAG_WORDS - 64, add
            )

    def _stock_push_prologue(self) -> None:
        """A stock push after an overlapped gather: no barrier since, so wait
        for every peer's arrival at this gather before writing its slot."""
        self._arrive()
        if self.last_overlapped:
            _ag_module().sp_ag_arrivals(self.peer_flags, self.world, self.gidx, _AG_SPIN_NS)
        self.last_overlapped = False

    def gather_mxfp8_overlapped(
        self, data: torch.Tensor, scale: torch.Tensor, cl_rows: int
    ) -> tuple[torch.Tensor, torch.Tensor, AgPending]:
        """``gather_mxfp8`` with the peers' shards still to be pushed by the
        consumer (module docstring): copies this rank's shard into its slot,
        returns the gathered views and the ``AgPending``."""
        rows, hidden = data.shape
        sf_bytes = rows * hidden // MXFP8_BLOCK
        scale = scale.reshape(-1)
        if (
            rows % cl_rows
            or data.element_size() != 1
            or scale.numel() * scale.element_size() != sf_bytes
        ):
            raise ValueError(f"SP overlapped all-gather: {rows} x {hidden} shard")
        slot = self._check_shard(rows, hidden, self.dtype)
        sf_base = self.world * self.max_rows * hidden
        start = slot * self.slot_numel * self.storage.element_size()
        raw = self.storage.view(torch.uint8)
        own = raw[start + self.rank * rows * hidden : start + (self.rank + 1) * rows * hidden]
        own.copy_(data.reshape(-1).view(torch.uint8))
        own_sf = raw[
            start + sf_base + self.rank * sf_bytes : start + sf_base + (self.rank + 1) * sf_bytes
        ]
        own_sf.copy_(scale.view(torch.uint8))
        self.kidx += 1
        cpu = _ag_chunks_per_tile(cl_rows, hidden)
        pending = AgPending(self, slot, rows, hidden, cl_rows,
                            target=self.kidx * cpu, arrive=self.gidx)  # fmt: skip
        if pending.ncl > _AG_FLAG_WORDS - 64:
            raise ValueError(f"SP overlapped all-gather: {pending.ncl} tiles")
        self._arrive(pending.lo, pending.hi, pending.ncl, cpu)
        self.pending = pending
        self.last_overlapped = True
        out = raw[start : start + self.world * rows * hidden]
        sf = raw[start + sf_base : start + sf_base + self.world * sf_bytes]
        return out.view(data.dtype).view(self.world * rows, hidden), sf, pending

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
        self._stock_push_prologue()
        data_words = data.contiguous().view(torch.int32).reshape(-1)
        sf_words = scale.contiguous().view(torch.int32)
        # Slot layout in bytes: rows from 0, scales from world * max_rows *
        # hidden (inside the BF16 slot of 2 * world * max_rows * hidden).
        sf_base = self.world * self.max_rows * hidden
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
        self._stock_push_prologue()
        x = x.contiguous()
        numel = rows * hidden
        _push_shard_kernel[(min(self.num_sms, cdiv(numel, _BLOCK)),)](
            x,
            self.peer_ptrs[slot],
            self.rank * numel,
            numel,
            self.rank,
            WORLD=self.world,
            BLOCK=_BLOCK,
            num_warps=_NUM_WARPS,
        )
        self.handle.barrier()
        start = slot * self.slot_numel
        return self.storage[start : start + self.world * numel].view(
            self.world * rows, hidden
        )


def _ag_chunks_per_tile(cl_rows: int, hidden: int) -> int:
    """Pusher chunks of one cluster tile (its scale bytes, then its rows):
    the count each tile's counter gains per overlapped gather."""
    return cdiv(cl_rows * hidden // MXFP8_BLOCK, _AG_CHUNK) + cdiv(cl_rows * hidden, _AG_CHUNK)


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


def kimi_sp_all_gather_mxfp8_overlapped(qa: Any, num_tokens: int, cl_rows: int) -> Any | None:
    """The overlapped form of ``kimi_sp_all_gather_mxfp8`` (module docstring),
    or None when this gather cannot take it (no idle SMs, CUDA graph capture,
    shard not a multiple of the GEMM's cluster tile, SP padding rows): the
    caller then takes the stock gather."""
    from dataclasses import replace

    gather = _sp_all_gather
    assert gather is not None
    rows, hidden = qa.data.shape
    if (
        not gather.idle_sms
        or torch.cuda.is_current_stream_capturing()
        or rows % cl_rows
        or num_tokens != gather.world * rows
    ):
        return None
    data, sf, _ = gather.gather_mxfp8_overlapped(qa.data, qa.scale, cl_rows)
    return replace(
        qa,
        data=data,
        scale=sf,
        orig_shape=torch.Size((num_tokens, hidden)),
    )


def enable_overlap(idle_sms: int, cl_rows: int) -> None:
    """The fused KDA Q|K|V GEMM can take overlapped gathers: its grid leaves
    ``idle_sms`` SMs for the pushers, its cluster M tile is ``cl_rows`` rows
    (same value on every rank: same GPU, same tactic)."""
    gather = _sp_all_gather
    if gather is None or gather.dtype != torch.bfloat16:
        return
    if gather.gidx:
        raise RuntimeError("SP all-gather: overlap enabled after the first gather")
    gather.idle_sms, gather.cl_rows = idle_sms, cl_rows
    logger.info_once(
        "Kimi K3 SP all-gather: KDA in_proj gathers overlap the Q|K|V GEMM "
        "(pushers on %d idle SMs, %d-row tiles)", idle_sms, cl_rows,
    )


def overlap_cl_rows() -> int:
    """Cluster tile rows of the overlapped gather (0: not enabled)."""
    gather = _sp_all_gather
    return gather.cl_rows if gather is not None and gather.idle_sms else 0


def take_pending(qa: Any) -> "AgPending | None":
    """The pending overlapped gather whose gathered rows ``qa`` holds, if any
    (handed to the consumer once)."""
    gather = _sp_all_gather
    if gather is None or gather.pending is None or qa is None:
        return None
    p = gather.pending
    start = gather.storage.data_ptr() + p.slot * gather.slot_numel * gather.storage.element_size()
    return p if qa.data.data_ptr() == start else None


def complete_pending() -> None:
    """Complete a pending overlapped gather on the current stream (a consumer
    other than the fused GEMM)."""
    gather = _sp_all_gather
    if gather is not None and gather.pending is not None:
        gather.pending.complete()


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
