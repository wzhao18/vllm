# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Histogram-exchange DCP top-k merge for DSA prefill chunks (CuTe DSL).

Under DCP with interleave 1, rank ``r`` owns global token ``g`` iff
``g % W == r``; every rank scores all chunk rows against its own tokens and
keeps its local top-``K``. The generic merge all-gathers every candidate as
an 8-byte ``(score, id)`` pair (``W * K * 8`` bytes per row, ~1.9 GB per
rank for a 16K-token chunk) and radix-selects ``W * K`` keys per row.

SM100 MQA logits are bf16, so a candidate's score is a 16-bit ordered key and
the global ``K``-th key can be found from histograms instead. Each rank keeps
its candidates and exchanges four small per-row summaries:

1. ``keys``: ordered key of each candidate; histogram of the key high byte.
2. ``refine``: from the summed high-byte histograms, the high byte of the
   ``K``-th key; histogram of the low byte of the candidates inside it.
3. ``tie_hist``: the exact ``K``-th key ``t`` and how many candidates equal to
   ``t`` are kept; histogram of those tied candidates by local index bin.
4. ``tie_bits``: the bin holding the last kept tied candidate; bitmap of the
   tied candidates inside it.
5. ``select``: keep keys above ``t`` and the tied candidates up to the last
   kept one, writing their global ids.

The selected set is exactly the generic merge's: the top-``K`` candidates by
(bf16 score desc, global id asc), since with interleave 1 the global id
``local * W + rank`` orders ties by (local index, rank). Rows with at most
``K`` candidates keep all of them; the output is ``-1`` padded. Each rank
receives ``(W - 1) * (3 * 512 + 4 * tie_words)`` bytes per row.
"""

from dataclasses import dataclass
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Int16, Int32, Uint16, Uint32
from quack.compile_utils import make_fake_tensor

from vllm.cute_utils import recast_val
from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_cutedsl_helper import (
    CuTeDSLLaunchSpec,
    VllmCuTeDSLJitKernel,
)
from vllm.models.kimi_k3.nvidia.dsa_kernels.dcp_prefill_topk_merge import NUM_BINS

_WARP = 32
_NUM_WARPS = NUM_BINS // _WARP
# Per-row state handed from round to round, int32 columns of ``meta``.
_META_WORDS = 8
_HIGH_BYTE, _NEED_HIGH, _THRESHOLD, _NEED_TIED, _TIE_BIN, _NEED_IN_BIN = range(6)
_STAGES = ("keys", "refine", "tie_hist", "tie_bits", "select")


@cute.jit
def _ordered_key(score: BFloat16) -> Int32:
    """16-bit key of a bf16 score that orders like the score."""
    bits = Uint32(recast_val(score, Uint16))
    key = bits | Uint32(0x8000)
    if (bits & Uint32(0x8000)) != Uint32(0):
        key = ~bits & Uint32(0xFFFF)
    return recast_val(key, Int32)


@cute.jit
def _block_scan_inclusive(
    val: Int32, lane: Int32, warp: Int32, warp_totals: cute.Tensor
) -> Int32:
    """Inclusive sum over the CTA's threads; ``warp_totals`` is smem
    [_NUM_WARPS]. Ends synced."""
    for i in cutlass.range_constexpr(5):
        partial = cute.arch.shuffle_sync_up(val, offset=1 << i, mask_and_clamp=0)
        if lane >= Int32(1 << i):
            val += partial
    if lane == Int32(_WARP - 1):
        warp_totals[warp] = val
    cute.arch.sync_threads()
    if warp == Int32(0):
        total = Int32(0)
        if lane < Int32(_NUM_WARPS):
            total = warp_totals[lane]
        prefix = total
        for i in cutlass.range_constexpr(5):
            partial = cute.arch.shuffle_sync_up(prefix, offset=1 << i, mask_and_clamp=0)
            if lane >= Int32(1 << i):
                prefix += partial
        if lane < Int32(_NUM_WARPS):
            warp_totals[lane] = prefix - total
    cute.arch.sync_threads()
    val += warp_totals[warp]
    cute.arch.sync_threads()
    return val


@cute.jit
def _find_boundary(
    gathered: cute.Tensor,
    row: Int32,
    need: Int32,
    world_size: cutlass.Constexpr[int],
    tid: Int32,
    lane: Int32,
    warp: Int32,
    warp_totals: cute.Tensor,
    state: cute.Tensor,
):
    """Sum the ranks' histograms ``gathered[:, row]`` and walk the bins from
    the top: ``state[0]`` = the bin where the running count reaches ``need``,
    ``state[1]`` = how many of that bin are needed, ``state[2]`` = the total.
    ``state[0:2]`` keep their values when the total is below ``need``. Thread
    ``t`` owns bin ``NUM_BINS - 1 - t``. Ends synced."""
    bin_id = Int32(NUM_BINS - 1) - tid
    count = Int32(0)
    for r in cutlass.range_constexpr(world_size):
        count += Int32(gathered[r, row, bin_id])
    inclusive = _block_scan_inclusive(count, lane, warp, warp_totals)
    if tid == Int32(NUM_BINS - 1):
        state[2] = inclusive
    if count > Int32(0) and inclusive >= need and inclusive - count < need:
        state[0] = bin_id
        state[1] = need - (inclusive - count)
    cute.arch.sync_threads()


def _smem_type(num_hist_words: int) -> Any:
    @cute.struct
    class SharedStorage:
        hist: cute.struct.MemRange[Int32, num_hist_words]
        state: cute.struct.MemRange[Int32, 4]
        warp_totals: cute.struct.MemRange[Int32, _NUM_WARPS]

    return SharedStorage


class DCPPrefillMergeKernel(VllmCuTeDSLJitKernel["DCPPrefillMergeKernel.CompileKey"]):
    """The five per-row stages of the merge; one CTA of ``NUM_BINS`` threads
    per row, candidates ``j * NUM_BINS + tid`` per thread."""

    @dataclass(frozen=True)
    class CompileKey:
        stage: str
        topk: int
        world_size: int
        tie_words: int

    @staticmethod
    def kernel(compile_key: CompileKey) -> Any:
        return _BUILDERS[compile_key.stage](compile_key)

    def dispatch(  # type: ignore[override]
        self, *, stage: str, topk: int, world_size: int, tie_words: int
    ) -> CompileKey:
        return self.CompileKey(
            stage=stage, topk=topk, world_size=world_size, tie_words=tie_words
        )

    def get_warmup_keys(
        self, *, topk: int, world_size: int, tie_words: int
    ) -> list[CompileKey]:
        return self._trace_dispatch(self.dispatch)(
            stage=_STAGES, topk=topk, world_size=world_size, tie_words=tie_words
        )

    def warmup_inputs(self, compile_key: CompileKey) -> tuple[Any, ...]:
        rows = cute.sym_int()
        topk = compile_key.topk
        world_size = compile_key.world_size
        tie_words = compile_key.tie_words

        def fake(dtype: Any, *shape: Any) -> Any:
            return make_fake_tensor(dtype, shape, divisibility=1)

        candidates = fake(Int32, rows, topk)
        meta = fake(Int32, rows, _META_WORDS)
        hist = fake(Int16, rows, NUM_BINS)
        gathered_hist = fake(Int16, world_size, rows, NUM_BINS)
        return {
            "keys": lambda: (
                fake(BFloat16, rows, cute.sym_int()),
                candidates,
                fake(Int32, rows, topk),
                hist,
            ),
            "refine": lambda: (gathered_hist, candidates, meta, hist),
            "tie_hist": lambda: (
                gathered_hist,
                meta,
                candidates,
                fake(Int32, rows, topk),
                hist,
            ),
            "tie_bits": lambda: (
                gathered_hist,
                meta,
                candidates,
                fake(Int32, rows, topk),
                fake(Int32, rows, tie_words),
            ),
            "select": lambda: (
                fake(Int32, world_size, rows, tie_words),
                meta,
                candidates,
                fake(Int32, rows, topk),
                Int32(0),
            ),
        }[compile_key.stage]()

    def _key(self, stage: str, candidates: torch.Tensor, **kwargs: int) -> CompileKey:
        return self.dispatch(stage=stage, topk=candidates.shape[1], **kwargs)

    @kernel_launcher
    def keys(
        self,
        logits: torch.Tensor,
        topk_indices: torch.Tensor,
        keys: torch.Tensor,
        hist: torch.Tensor,
        *,
        world_size: int,
        tie_words: int,
    ) -> CuTeDSLLaunchSpec[CompileKey]:
        key = self._key(
            "keys", topk_indices, world_size=world_size, tie_words=tie_words
        )
        return key, (logits, topk_indices, keys, hist)

    @kernel_launcher
    def refine(
        self,
        gathered_hist: torch.Tensor,
        keys: torch.Tensor,
        meta: torch.Tensor,
        hist: torch.Tensor,
        *,
        tie_words: int,
    ) -> CuTeDSLLaunchSpec[CompileKey]:
        key = self._key(
            "refine", keys, world_size=gathered_hist.shape[0], tie_words=tie_words
        )
        return key, (gathered_hist, keys, meta, hist)

    @kernel_launcher
    def tie_hist(
        self,
        gathered_hist: torch.Tensor,
        meta: torch.Tensor,
        keys: torch.Tensor,
        topk_indices: torch.Tensor,
        hist: torch.Tensor,
        *,
        tie_words: int,
    ) -> CuTeDSLLaunchSpec[CompileKey]:
        key = self._key(
            "tie_hist", keys, world_size=gathered_hist.shape[0], tie_words=tie_words
        )
        return key, (gathered_hist, meta, keys, topk_indices, hist)

    @kernel_launcher
    def tie_bits(
        self,
        gathered_hist: torch.Tensor,
        meta: torch.Tensor,
        keys: torch.Tensor,
        topk_indices: torch.Tensor,
        bits: torch.Tensor,
    ) -> CuTeDSLLaunchSpec[CompileKey]:
        key = self._key(
            "tie_bits",
            keys,
            world_size=gathered_hist.shape[0],
            tie_words=bits.shape[1],
        )
        return key, (gathered_hist, meta, keys, topk_indices, bits)

    @kernel_launcher
    def select(
        self,
        gathered_bits: torch.Tensor,
        meta: torch.Tensor,
        keys: torch.Tensor,
        topk_indices: torch.Tensor,
        rank: int,
    ) -> CuTeDSLLaunchSpec[CompileKey]:
        key = self._key(
            "select",
            keys,
            world_size=gathered_bits.shape[0],
            tie_words=gathered_bits.shape[2],
        )
        return key, (gathered_bits, meta, keys, topk_indices, rank)


def _build_keys(compile_key: DCPPrefillMergeKernel.CompileKey) -> Any:
    """Round 1: ``keys[t, j]`` = ordered key of candidate ``j`` (-1 if none)
    and the histogram of the key high bytes. Local indices past the tie
    bitmap (impossible for rows within ``max_model_len``) count as none, so
    the tie rounds stay inside shared memory."""
    cols_per_thread = compile_key.topk // NUM_BINS
    max_local = NUM_BINS * 32 * compile_key.tie_words
    shared_storage = _smem_type(NUM_BINS)

    @cute.kernel
    def device_kernel(
        logits: cute.Tensor,
        topk_indices: cute.Tensor,
        keys: cute.Tensor,
        hist_out: cute.Tensor,
    ):
        row, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        storage = cutlass.utils.SmemAllocator().allocate(shared_storage, 8)
        hist = storage.hist.get_tensor(cute.make_layout((NUM_BINS,)))
        hist[tid] = Int32(0)
        cute.arch.sync_threads()
        num_cols = cutlass.min(Int32(logits.shape[1]), Int32(max_local))
        for j in cutlass.range_constexpr(cols_per_thread):
            col = Int32(j * NUM_BINS) + tid
            local = topk_indices[row, col]
            key = Int32(-1)
            if local >= Int32(0) and local < num_cols:
                key = _ordered_key(logits[row, local])
                cute.arch.atomic_add(
                    hist.iterator + (key >> Int32(8)),
                    Int32(1),
                    sem="relaxed",
                    scope="cta",
                )
            keys[row, col] = key
        cute.arch.sync_threads()
        hist_out[row, tid] = Int16(hist[tid])

    @cute.jit
    def host_entrypoint(logits, topk_indices, keys, hist_out, stream):
        device_kernel(logits, topk_indices, keys, hist_out).launch(
            grid=(keys.shape[0], 1, 1), block=(NUM_BINS, 1, 1), stream=stream
        )

    return host_entrypoint


def _build_refine(compile_key: DCPPrefillMergeKernel.CompileKey) -> Any:
    """Round 2: the high byte of the K-th key (-1 when every candidate is
    kept) and the low-byte histogram of the candidates inside it."""
    topk = compile_key.topk
    world_size = compile_key.world_size
    cols_per_thread = topk // NUM_BINS
    shared_storage = _smem_type(NUM_BINS)

    @cute.kernel
    def device_kernel(
        gathered_hist: cute.Tensor,
        keys: cute.Tensor,
        meta: cute.Tensor,
        hist_out: cute.Tensor,
    ):
        row, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        storage = cutlass.utils.SmemAllocator().allocate(shared_storage, 8)
        hist = storage.hist.get_tensor(cute.make_layout((NUM_BINS,)))
        state = storage.state.get_tensor(cute.make_layout((4,)))
        warp_totals = storage.warp_totals.get_tensor(cute.make_layout((_NUM_WARPS,)))
        hist[tid] = Int32(0)
        if tid == Int32(0):
            state[0] = Int32(-1)
            state[1] = Int32(0)
        cute.arch.sync_threads()
        _find_boundary(
            gathered_hist,
            row,
            Int32(topk),
            world_size,
            tid,
            lane,
            warp,
            warp_totals,
            state,
        )
        high = Int32(-1)
        if state[2] > Int32(topk):
            high = state[0]
        if high >= Int32(0):
            for j in cutlass.range_constexpr(cols_per_thread):
                key = keys[row, Int32(j * NUM_BINS) + tid]
                if key >= Int32(0) and (key >> Int32(8)) == high:
                    cute.arch.atomic_add(
                        hist.iterator + (key & Int32(0xFF)),
                        Int32(1),
                        sem="relaxed",
                        scope="cta",
                    )
        cute.arch.sync_threads()
        hist_out[row, tid] = Int16(hist[tid])
        if tid == Int32(0):
            meta[row, _HIGH_BYTE] = high
            meta[row, _NEED_HIGH] = state[1]

    @cute.jit
    def host_entrypoint(gathered_hist, keys, meta, hist_out, stream):
        device_kernel(gathered_hist, keys, meta, hist_out).launch(
            grid=(keys.shape[0], 1, 1), block=(NUM_BINS, 1, 1), stream=stream
        )

    return host_entrypoint


def _build_tie_hist(compile_key: DCPPrefillMergeKernel.CompileKey) -> Any:
    """Round 3: the K-th key ``t`` and the count of kept candidates equal to
    it; histogram of this rank's ``t``-tied candidates by local index bin,
    stored reversed so the top-down walk visits lower local indices first."""
    world_size = compile_key.world_size
    cols_per_thread = compile_key.topk // NUM_BINS
    bin_width = 32 * compile_key.tie_words
    shared_storage = _smem_type(NUM_BINS)

    @cute.kernel
    def device_kernel(
        gathered_hist: cute.Tensor,
        meta: cute.Tensor,
        keys: cute.Tensor,
        topk_indices: cute.Tensor,
        hist_out: cute.Tensor,
    ):
        row, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        storage = cutlass.utils.SmemAllocator().allocate(shared_storage, 8)
        hist = storage.hist.get_tensor(cute.make_layout((NUM_BINS,)))
        state = storage.state.get_tensor(cute.make_layout((4,)))
        warp_totals = storage.warp_totals.get_tensor(cute.make_layout((_NUM_WARPS,)))
        hist[tid] = Int32(0)
        high = meta[row, _HIGH_BYTE]
        threshold = Int32(-1)
        need_tied = Int32(0)
        cute.arch.sync_threads()
        if high >= Int32(0):
            _find_boundary(
                gathered_hist,
                row,
                meta[row, _NEED_HIGH],
                world_size,
                tid,
                lane,
                warp,
                warp_totals,
                state,
            )
            threshold = (high << Int32(8)) | state[0]
            need_tied = state[1]
            for j in cutlass.range_constexpr(cols_per_thread):
                col = Int32(j * NUM_BINS) + tid
                if keys[row, col] == threshold:
                    tie_bin = topk_indices[row, col] // Int32(bin_width)
                    cute.arch.atomic_add(
                        hist.iterator + (Int32(NUM_BINS - 1) - tie_bin),
                        Int32(1),
                        sem="relaxed",
                        scope="cta",
                    )
        cute.arch.sync_threads()
        hist_out[row, tid] = Int16(hist[tid])
        if tid == Int32(0):
            meta[row, _THRESHOLD] = threshold
            meta[row, _NEED_TIED] = need_tied

    @cute.jit
    def host_entrypoint(gathered_hist, meta, keys, topk_indices, hist_out, stream):
        device_kernel(gathered_hist, meta, keys, topk_indices, hist_out).launch(
            grid=(keys.shape[0], 1, 1), block=(NUM_BINS, 1, 1), stream=stream
        )

    return host_entrypoint


def _build_tie_bits(compile_key: DCPPrefillMergeKernel.CompileKey) -> Any:
    """Round 4: the local index bin holding the last kept tied candidate and
    how many tied candidates of that bin are kept; this rank's bitmap of the
    tied candidates in the bin."""
    world_size = compile_key.world_size
    cols_per_thread = compile_key.topk // NUM_BINS
    tie_words = compile_key.tie_words
    bin_width = 32 * tie_words
    shared_storage = _smem_type(tie_words)

    @cute.kernel
    def device_kernel(
        gathered_hist: cute.Tensor,
        meta: cute.Tensor,
        keys: cute.Tensor,
        topk_indices: cute.Tensor,
        bits_out: cute.Tensor,
    ):
        row, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        storage = cutlass.utils.SmemAllocator().allocate(shared_storage, 8)
        bits = storage.hist.get_tensor(cute.make_layout((tie_words,)))
        state = storage.state.get_tensor(cute.make_layout((4,)))
        warp_totals = storage.warp_totals.get_tensor(cute.make_layout((_NUM_WARPS,)))
        if tid < Int32(tie_words):
            bits[tid] = Int32(0)
        high = meta[row, _HIGH_BYTE]
        threshold = meta[row, _THRESHOLD]
        tie_bin = Int32(-1)
        need_in_bin = Int32(0)
        cute.arch.sync_threads()
        if high >= Int32(0):
            _find_boundary(
                gathered_hist,
                row,
                meta[row, _NEED_TIED],
                world_size,
                tid,
                lane,
                warp,
                warp_totals,
                state,
            )
            tie_bin = Int32(NUM_BINS - 1) - state[0]
            need_in_bin = state[1]
            for j in cutlass.range_constexpr(cols_per_thread):
                col = Int32(j * NUM_BINS) + tid
                if keys[row, col] == threshold:
                    offset = topk_indices[row, col] - tie_bin * Int32(bin_width)
                    if offset >= Int32(0) and offset < Int32(bin_width):
                        # Distinct bits, so the add is an or.
                        cute.arch.atomic_add(
                            bits.iterator + (offset >> Int32(5)),
                            Int32(1) << (offset & Int32(31)),
                            sem="relaxed",
                            scope="cta",
                        )
        cute.arch.sync_threads()
        if tid < Int32(tie_words):
            bits_out[row, tid] = bits[tid]
        if tid == Int32(0):
            meta[row, _TIE_BIN] = tie_bin
            meta[row, _NEED_IN_BIN] = need_in_bin

    @cute.jit
    def host_entrypoint(gathered_hist, meta, keys, topk_indices, bits_out, stream):
        device_kernel(gathered_hist, meta, keys, topk_indices, bits_out).launch(
            grid=(keys.shape[0], 1, 1), block=(NUM_BINS, 1, 1), stream=stream
        )

    return host_entrypoint


def _build_select(compile_key: DCPPrefillMergeKernel.CompileKey) -> Any:
    """Round 5: find the last kept tied candidate ``(local*, rank*)`` from the
    gathered bitmaps (ties ordered by (local, rank), i.e. by global id), then
    overwrite ``topk_indices`` with the global ids of this row's kept
    candidates followed by ``-1``."""
    topk = compile_key.topk
    world_size = compile_key.world_size
    cols_per_thread = topk // NUM_BINS
    bin_width = 32 * compile_key.tie_words
    positions_per_thread = -(-bin_width // NUM_BINS)
    shared_storage = _smem_type(1)

    @cute.kernel
    def device_kernel(
        gathered_bits: cute.Tensor,
        meta: cute.Tensor,
        keys: cute.Tensor,
        topk_indices: cute.Tensor,
        rank: Int32,
    ):
        row, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        storage = cutlass.utils.SmemAllocator().allocate(shared_storage, 8)
        state = storage.state.get_tensor(cute.make_layout((4,)))
        warp_totals = storage.warp_totals.get_tensor(cute.make_layout((_NUM_WARPS,)))
        high = meta[row, _HIGH_BYTE]
        threshold = meta[row, _THRESHOLD]
        tie_bin = meta[row, _TIE_BIN]
        need_in_bin = meta[row, _NEED_IN_BIN]
        # Read the candidates before any thread overwrites them below.
        row_keys = cute.make_rmem_tensor((cols_per_thread,), Int32)
        row_locals = cute.make_rmem_tensor((cols_per_thread,), Int32)
        for j in cutlass.range_constexpr(cols_per_thread):
            col = Int32(j * NUM_BINS) + tid
            row_keys[j] = keys[row, col]
            row_locals[j] = topk_indices[row, col]
        if tid == Int32(0):
            state[0] = Int32(-1)
            state[1] = Int32(-1)
        cute.arch.sync_threads()
        if high >= Int32(0):
            # Thread t walks bin positions [t * ppt, (t + 1) * ppt), each
            # over the ranks: the tied candidates in ascending global id.
            count = Int32(0)
            for p in cutlass.range_constexpr(positions_per_thread):
                pos = tid * Int32(positions_per_thread) + Int32(p)
                if pos < Int32(bin_width):
                    for r in cutlass.range_constexpr(world_size):
                        word = gathered_bits[r, row, pos >> Int32(5)]
                        count += (word >> (pos & Int32(31))) & Int32(1)
            inclusive = _block_scan_inclusive(count, lane, warp, warp_totals)
            if count > Int32(0) and inclusive >= need_in_bin:
                remaining = need_in_bin - (inclusive - count)
                if remaining > Int32(0):
                    for p in cutlass.range_constexpr(positions_per_thread):
                        pos = tid * Int32(positions_per_thread) + Int32(p)
                        if pos < Int32(bin_width):
                            for r in cutlass.range_constexpr(world_size):
                                word = gathered_bits[r, row, pos >> Int32(5)]
                                if remaining > Int32(0) and (
                                    (word >> (pos & Int32(31))) & Int32(1)
                                ) != Int32(0):
                                    remaining -= Int32(1)
                                    if remaining == Int32(0):
                                        state[0] = tie_bin * Int32(bin_width) + pos
                                        state[1] = Int32(r)
            cute.arch.sync_threads()
        last_local = state[0]
        last_rank = state[1]
        num_kept = Int32(0)
        for j in cutlass.range_constexpr(cols_per_thread):
            key = row_keys[j]
            local = row_locals[j]
            kept = key > threshold
            if key == threshold:
                kept = local < last_local
                if local == last_local:
                    kept = rank <= last_rank
            if key < Int32(0):
                kept = False
            if kept:
                num_kept += Int32(1)
            else:
                row_locals[j] = Int32(-1)
        inclusive_kept = _block_scan_inclusive(num_kept, lane, warp, warp_totals)
        if tid == Int32(NUM_BINS - 1):
            state[2] = inclusive_kept
        cute.arch.sync_threads()
        total_kept = state[2]
        out_pos = inclusive_kept - num_kept
        for j in cutlass.range_constexpr(cols_per_thread):
            local = row_locals[j]
            if local >= Int32(0):
                topk_indices[row, out_pos] = local * Int32(world_size) + rank
                out_pos += Int32(1)
        for j in cutlass.range_constexpr(cols_per_thread):
            col = Int32(j * NUM_BINS) + tid
            if col >= total_kept:
                topk_indices[row, col] = Int32(-1)

    @cute.jit
    def host_entrypoint(gathered_bits, meta, keys, topk_indices, rank, stream):
        device_kernel(gathered_bits, meta, keys, topk_indices, rank).launch(
            grid=(keys.shape[0], 1, 1), block=(NUM_BINS, 1, 1), stream=stream
        )

    return host_entrypoint


_BUILDERS = {
    "keys": _build_keys,
    "refine": _build_refine,
    "tie_hist": _build_tie_hist,
    "tie_bits": _build_tie_bits,
    "select": _build_select,
}
_DCP_PREFILL_MERGE_KERNEL = DCPPrefillMergeKernel()


def register_dcp_prefill_merge_warmup(topk: int, world_size: int, tie_words: int):
    _DCP_PREFILL_MERGE_KERNEL.register_warmup(
        topk=topk, world_size=world_size, tie_words=tie_words
    )


def _all_gather_rows(group: Any, x: torch.Tensor) -> torch.Tensor:
    """[T, n] per rank -> [W, T, n] (rank-major; NCCL has no int16, so the
    rows travel as int32 words)."""
    gathered = group.all_gather(x.view(torch.int32), dim=0)
    return gathered.view(x.dtype).view(group.world_size, *x.shape)


def dcp_prefill_topk_merge(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    tie_words: int,
    group: Any = None,
) -> None:
    """Merge each DCP rank's local top-K into the global top-K, in place.

    Takes ``_merge_dcp_topk_global``'s inputs with interleave 1 and no row
    starts: ``topk_indices`` ``[T, K]`` holds this rank's local top-K columns
    of the bf16 ``logits`` ``[T, S]``; ``logits`` may be empty (no local
    tokens in the chunk). Unlike it, each rank writes only its own share of
    the global top-K, as global token ids followed by ``-1``: the DCP sparse
    attention keeps only this rank's ids anyway. PCP + DCP, whose gathered
    prefill attends every rank's ids, never takes this path.
    """
    if group is None:
        from vllm.distributed import get_dcp_group

        group = get_dcp_group()
    num_rows, topk = topk_indices.shape
    if num_rows == 0:
        return
    if logits.dtype != torch.bfloat16:
        assert logits.shape[1] == 0, "the histogram merge reads bf16 logits"
        logits = topk_indices.new_empty((num_rows, 0), dtype=torch.bfloat16)
    kernel = _DCP_PREFILL_MERGE_KERNEL
    world_size = group.world_size
    device = topk_indices.device
    keys = torch.empty((num_rows, topk), dtype=torch.int32, device=device)
    hist = torch.empty((num_rows, NUM_BINS), dtype=torch.int16, device=device)
    meta = torch.empty((num_rows, _META_WORDS), dtype=torch.int32, device=device)
    bits = torch.empty((num_rows, tie_words), dtype=torch.int32, device=device)
    kernel.keys(
        logits, topk_indices, keys, hist, world_size=world_size, tie_words=tie_words
    )
    kernel.refine(_all_gather_rows(group, hist), keys, meta, hist, tie_words=tie_words)
    kernel.tie_hist(
        _all_gather_rows(group, hist),
        meta,
        keys,
        topk_indices,
        hist,
        tie_words=tie_words,
    )
    kernel.tie_bits(_all_gather_rows(group, hist), meta, keys, topk_indices, bits)
    kernel.select(
        _all_gather_rows(group, bits), meta, keys, topk_indices, group.rank_in_group
    )
