# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MLA rows of a DCP ShardQ step's context, moved off the critical path.

The gathered-context workspace (``context.py``) holds every rank's context rows.
The indexer needs only the indexer rows (68 of the 644 bytes per token) before
it runs; the MLA rows (576 bytes) are read by the sparse attention only, after
the indexer. So a step whose context fits lays the workspace's MLA pages out
rank-major, the layout the top-k remap's prefill-workspace mode addresses
(``triton_convert_req_index_to_global_index`` with ``HAS_PREFILL_WORKSPACE``):
rank ``r``'s region of ``region_rows`` rows holds, for each prefill request
``q`` from row ``starts[q]`` on, the request's rows of global tokens
``t % W == r`` at ``t // W``, its context rows and then its new tokens' rows.

Per layer, each rank

1. after the barrier that opens the indexer's gather (every peer is done
   reading the previous layer's workspace), copies its own context rows from
   its DCP-local paged cache into its own region (``pack``: local HBM only);
2. lets the copy engine copy that region's context rows into the same region
   of every peer's workspace, on a side stream, while the indexer runs; then
   writes the layer's epoch into each peer's flag word (``cuStreamWriteValue32``,
   ordered after the copies, with its memory barrier);
3. before the sparse attention, waits on its own stream until every peer's
   flag word holds the epoch (``cuStreamWaitValue32``).

No SM runs a copy or spins on a flag. Every rank writes all new tokens' rows
into its own workspace itself (``kv_insert``), at the rows ``slot_mapping``
gives, so the copies carry context rows only. A step of several request groups,
or one whose rank-major rows exceed a region, keeps the token-ordered layout
and the SM push of ``gather.py``.
"""

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    LaunchSpec,
    TritonWarmupTensor,
    VllmTritonJitKernel,
)
from vllm.triton_utils import tl, triton


@dataclass
class RankMajorPlan:
    """One step's rank-major MLA rows (host values are identical on every
    rank: every rank derives the plan from the same batch)."""

    world: int
    rank: int
    # Rows of each rank's region.
    region_rows: int
    # int64 [R]: first row of each prefill request in every region.
    starts: np.ndarray
    # int64 [R, W]: context rows of each prefill request on each rank.
    context_rows: np.ndarray
    # int32 [num_reqs]: ``starts`` by the step's request index (decode rows 0),
    # the remap's prefill_workspace_starts.
    starts_by_req: torch.Tensor
    # int32 [R]: ``starts`` and int32 [R] context lengths, for ``pack``.
    starts_gpu: torch.Tensor
    context_lens_gpu: torch.Tensor
    # Most context rows of any request on any rank.
    max_context_rows: int
    # int64 [num_prefill_tokens]: the workspace row of each new prefill token
    # (set by the context builder).
    slot_mapping: torch.Tensor | None = None


def plan_rank_major(
    seq_lens: np.ndarray,
    query_lens: np.ndarray,
    num_decodes: int,
    pages: int,
    page_tokens: int,
    world: int,
    rank: int,
    to_device,
) -> RankMajorPlan | None:
    """The rank-major plan of a step's prefill requests (``seq_lens``,
    ``query_lens``: theirs only) over a workspace of ``pages`` MLA pages, or
    None when some region would overflow."""
    region_rows = pages * page_tokens // world
    rows = -(-seq_lens // world)  # rank 0 holds the most rows of a request
    starts = np.concatenate(([0], np.cumsum(rows)[:-1])).astype(np.int64)
    if int(starts[-1] + rows[-1]) > region_rows:
        return None
    context_lens = seq_lens - query_lens
    ranks = np.arange(world)
    context_rows = np.maximum(context_lens[:, None] - ranks[None, :] + world - 1, 0)
    context_rows //= world
    starts_by_req = np.zeros(num_decodes + len(seq_lens), dtype=np.int64)
    starts_by_req[num_decodes:] = starts
    return RankMajorPlan(
        world=world,
        rank=rank,
        region_rows=region_rows,
        starts=starts,
        context_rows=context_rows,
        starts_by_req=to_device(starts_by_req, torch.int32),
        starts_gpu=to_device(starts, torch.int32),
        context_lens_gpu=to_device(context_lens, torch.int32),
        max_context_rows=int(context_rows.max(initial=0)),
    )


def rank_major_slots(
    plan: RankMajorPlan, seq_lens: np.ndarray, query_lens: np.ndarray
) -> np.ndarray:
    """int64 workspace row of each new prefill token (all requests, in order)."""
    world = plan.world
    context_lens = seq_lens - query_lens
    tokens = np.concatenate(
        [np.arange(c, s) for c, s in zip(context_lens.tolist(), seq_lens.tolist())]
    ) if len(seq_lens) else np.empty(0, np.int64)
    starts = np.repeat(plan.starts, query_lens)
    return (tokens % world) * plan.region_rows + starts + tokens // world


class RankMajorPackKernel(VllmTritonJitKernel["RankMajorPackKernel.CompileKey"]):
    """Copy this rank's context MLA rows of each prefill request from the
    DCP-local paged cache into this rank's region of the workspace."""

    ROWS = 16

    @dataclass(frozen=True)
    class CompileKey:
        row_words: int
        world: int

    @staticmethod
    @triton.jit(
        do_not_specialize=[
            "rank",
            "region_row0",
            "rows_per_block",
            "block_stride",
            "table_stride",
            "first_req",
        ]
    )
    def dcp_shardq_rank_major_pack_kernel(
        cache,
        block_table,
        context_lens,
        starts,
        workspace,
        rank,
        region_row0,
        rows_per_block,
        block_stride,
        table_stride,
        first_req,
        WORLD: tl.constexpr,
        ROW_WORDS: tl.constexpr,
        WORDS: tl.constexpr,
        ROWS: tl.constexpr,
    ):
        req = tl.program_id(0)
        rows = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
        context_len = tl.load(context_lens + req)
        num_rows = tl.maximum(context_len - rank + WORLD - 1, 0) // WORLD
        live = rows < num_rows
        block = tl.load(
            block_table
            + (first_req + req).to(tl.int64) * table_stride
            + rows // rows_per_block,
            mask=live,
            other=0,
        ).to(tl.int64)
        src = block * block_stride + (rows % rows_per_block).to(tl.int64) * ROW_WORDS
        start = tl.load(starts + req).to(tl.int64)
        dst = (region_row0 + start + rows.to(tl.int64)) * ROW_WORDS
        cols = tl.arange(0, WORDS)[None, :]
        mask = live[:, None] & (cols < ROW_WORDS)
        value = tl.load(cache + src[:, None] + cols, mask=mask)
        tl.store(workspace + dst[:, None] + cols, value, mask=mask)

    # A descriptive CUDA symbol for traces.
    kernel = dcp_shardq_rank_major_pack_kernel

    def dispatch(  # type: ignore[override]
        self, *, row_words: int, world: int
    ) -> CompileKey:
        return self.CompileKey(row_words=row_words, world=world)

    def get_warmup_keys(  # type: ignore[override]
        self, *, row_bytes: int, world: int
    ) -> list[CompileKey]:
        return [self.dispatch(row_words=row_bytes // 4, world=world)]

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        return dict(
            cache=TritonWarmupTensor(torch.int32),
            block_table=TritonWarmupTensor(torch.int32),
            context_lens=TritonWarmupTensor(torch.int32, shape=(2,)),
            starts=TritonWarmupTensor(torch.int32),
            workspace=TritonWarmupTensor(torch.int32),
            rank=0,
            region_row0=0,
            rows_per_block=64,
            block_stride=7,
            table_stride=7,
            first_req=0,
            world=compile_key.world,
            row_words=compile_key.row_words,
            max_rows=1,
        )

    @kernel_launcher
    def __call__(
        self,
        cache: torch.Tensor,
        block_table: torch.Tensor,
        context_lens: torch.Tensor,
        starts: torch.Tensor,
        workspace: torch.Tensor,
        rank: int,
        region_row0: int,
        rows_per_block: int,
        block_stride: int,
        table_stride: int,
        first_req: int,
        world: int,
        row_words: int,
        max_rows: int,
    ) -> LaunchSpec:
        grid = (context_lens.shape[0], triton.cdiv(max_rows, self.ROWS))
        return grid, dict(
            WORLD=world,
            ROW_WORDS=row_words,
            WORDS=triton.next_power_of_2(row_words),
            ROWS=self.ROWS,
            num_warps=4,
        )


RANK_MAJOR_PACK_KERNEL = RankMajorPackKernel()


def pack(
    plan: RankMajorPlan,
    mla_cache: torch.Tensor,
    mla_block_table: torch.Tensor,
    first_req: int,
    workspace_mla: torch.Tensor,
) -> None:
    """Copy this rank's context rows of the plan's requests (block table rows
    ``first_req`` on) from ``mla_cache`` ([blocks, rows_per_block, row] bytes)
    into its region of ``workspace_mla`` (the workspace's MLA rows, bytes)."""
    if plan.max_context_rows == 0:
        return
    row_bytes = mla_cache.shape[-1]
    assert row_bytes % 4 == 0 and mla_cache.stride(1) == row_bytes
    assert mla_cache.stride(0) % 4 == 0 and mla_cache.element_size() == 1
    RANK_MAJOR_PACK_KERNEL(
        mla_cache.view(torch.int32),
        mla_block_table,
        plan.context_lens_gpu,
        plan.starts_gpu,
        workspace_mla.view(torch.int32),
        plan.rank,
        plan.rank * plan.region_rows,
        mla_cache.shape[1],
        mla_cache.stride(0) // 4,
        mla_block_table.stride(0),
        first_req,
        plan.world,
        row_bytes // 4,
        plan.max_context_rows,
    )


# Signalling -------------------------------------------------------------


@dataclass
class _Flags:
    # int32 [world] on every rank: word p is rank p's epoch of its last copy
    # into this rank.
    local: torch.Tensor
    # This rank's word on every peer.
    peer_addrs: list[int]


_flags: dict[str, _Flags] = {}
_epochs: dict[str, itertools.count] = {}
_side_streams: dict[torch.device, torch.cuda.Stream] = {}


def reserve_flags(group, world: int, rank: int, device: torch.device) -> _Flags:
    """The flag words of ``group`` (collective on first use)."""
    from vllm.v1.attention.ops.shardq import _symm_buffer

    key = group.group_name
    flags = _flags.get(key)
    if flags is None:
        buf = _symm_buffer(group, "dcp_shardq_flags", world, torch.int32, device)
        buf.storage[:world].zero_()
        torch.accelerator.synchronize()
        # Every rank zeroed its words before any peer can write them.
        buf.handle.barrier()
        torch.accelerator.synchronize()
        peer_addrs = [
            int(buf.handle.get_buffer(peer, [world], torch.int32, 0).data_ptr())
            + rank * 4
            for peer in range(world)
        ]
        flags = _flags[key] = _Flags(buf.storage[:world], peer_addrs)
        _epochs[key] = itertools.count(1)
    return flags


def _side_stream(device: torch.device) -> torch.cuda.Stream:
    stream = _side_streams.get(device)
    if stream is None:
        stream = _side_streams[device] = torch.cuda.Stream(device=device)
    return stream


def _check(result) -> None:
    from cuda.bindings import driver

    err = result[0] if isinstance(result, tuple) else result
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver call failed: {err}")


class InFlight:
    """One layer's copies into the peers, and how to wait for theirs."""

    def __init__(self, flags: _Flags, epoch: int, rank: int, done: torch.cuda.Event):
        self.flags = flags
        self.epoch = epoch
        self.rank = rank
        self.done = done
        self.waited = False

    def wait(self) -> None:
        """Order the current stream after every peer's copies into this rank
        (and this rank's own copies, so the next layer may reuse its region)."""
        if self.waited:
            return
        from cuda.bindings import driver

        stream = torch.cuda.current_stream()
        handle = driver.CUstream(stream.cuda_stream)
        base = self.flags.local.data_ptr()
        for peer in range(self.flags.local.shape[0]):
            if peer == self.rank:
                continue
            _check(
                driver.cuStreamWaitValue32(
                    handle,
                    driver.CUdeviceptr(base + 4 * peer),
                    self.epoch,
                    driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ,
                )
            )
        stream.wait_event(self.done)
        self.waited = True


def copy_to_peers(
    plan: RankMajorPlan,
    group,
    workspace_peer_ptrs: list[int],
    row_bytes: int,
    after: torch.cuda.Event,
    device: torch.device,
) -> InFlight:
    """On the side stream, once ``after`` (the pack) completed: copy this
    rank's region's context rows into every peer's workspace with the copy
    engine, then publish the epoch into each peer's flag word for this rank."""
    from cuda.bindings import driver

    flags = reserve_flags(group, plan.world, plan.rank, device)
    # cuStreamWaitValue32's GEQ compares (int32)(word - epoch) >= 0: wraps.
    epoch = next(_epochs[group.group_name]) & 0xFFFFFFFF
    side = _side_stream(device)
    side.wait_event(after)
    handle = driver.CUstream(side.cuda_stream)
    rank = plan.rank
    region = rank * plan.region_rows
    spans = [
        ((region + int(start)) * row_bytes, int(rows) * row_bytes)
        for start, rows in zip(plan.starts.tolist(), plan.context_rows[:, rank].tolist())
        if rows
    ]
    local = workspace_peer_ptrs[rank]
    for peer in range(plan.world):
        if peer == rank:
            continue
        remote = workspace_peer_ptrs[peer]
        for offset, nbytes in spans:
            _check(
                driver.cuMemcpyDtoDAsync(
                    driver.CUdeviceptr(remote + offset),
                    driver.CUdeviceptr(local + offset),
                    nbytes,
                    handle,
                )
            )
        _check(
            driver.cuStreamWriteValue32(
                handle,
                driver.CUdeviceptr(flags.peer_addrs[peer]),
                epoch,
                driver.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
            )
        )
    done = torch.cuda.Event()
    done.record(side)
    return InFlight(flags, epoch, rank, done)
