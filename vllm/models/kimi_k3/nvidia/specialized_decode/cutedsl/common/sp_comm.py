# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sequence-parallel (SP) mailboxes of the TP group (C512: TP4 = DCP4 = SP4) for the
SP decode's attention blocks (``runtime.sp_attention``): the all-gather of the
row-sharded normed input into the in-projection operand, and the reduce-scatter of
the o-projection partials back to the row shards.

Rank r owns global rows [r n, (r + 1) n), n = R / TP (``sp_shard`` order). Every
buffer is Lamport-armed (``SENTINEL`` words) and polled 16 bytes at a time:

- ``gather`` [parities, rows_cap, width] (multicast): ``SpAllGather`` multicast-
  stores this rank's n shard rows into rows [r n, (r + 1) n) of every rank's
  ``gather[parity]``, then polls all R rows; the next kernel (the in-projection)
  reads them plainly. ``SpReduceScatter`` of the same parity re-arms rows [0, R)
  after the attention core (the in-projection has consumed them by then).
- ``recv`` [parities, TP, rows_cap / TP, width] (peer-mapped, no multicast): slot
  [src] holds rank src's o-projection partial (BF16) of MY rows. Producers (the
  fused o-projection, ``oproj.sp.SpOprojWarp``, or ``SpReduceScatter(push=True)``
  from a local partial) store each 16-byte chunk to its row owner's slot [rank];
  the owner polls the TP slots, sums them in FP32 in rank order 0..TP-1, rounds
  once to BF16 and re-arms them.

Precision is the stock SP reduce-scatter's (``custom_all_reduce`` ->
``mnnvl_lamport_reduce_scatter``, the backend at R <= 512 x 7168 BF16 = 7.3 MB <=
16 MB): BF16 partials upcast to FP32, summed in rank order 0..TP-1, rounded once
(RN), a word equal to the sentinel replaced by 0 (``positive_zero32``: stock's
``sanitize_lamport_payload``, applied to published partials and to the sum alike).
Given the same partials the shard is bitwise stock's. The all-gather publishes the
same sanitized words as stock's ``mnnvl_lamport_all_gather``.

Parities: block ``i`` uses parity ``i % 2`` of the step's SP blocks, which
alternate within a step. A rank reaches block i + 2's all-gather / pushes only after
its block i + 1 reduce saw every rank's block i + 1 pushes, which each rank issues
after its own block i reduce-scatter (where it re-armed parity i % 2), so two
parities suffice. Across steps the stock final ``sp_all_gather`` separates the last
block from the next step's first (equal parities when the block count is odd).
Every kernel here and between them waits for its predecessor (griddepcontrol.wait)
before touching these buffers, so stream order carries this ordering through PDL.

Polling kernels (``SpAllGather``, ``SpReduceScatter``) wait on other ranks' stores
issued by the same kernel there: their grid is at most one CTA per SM and checked
co-resident at build (``host.check_residency``), so no rank's storing CTA can be
kept off the GPU by its own polling CTAs.

NQ-TUNE (no VR200 measurement yet; GENERIC everywhere): grid size and chunks per
thread, polling before griddepcontrol.wait, fusing the all-gather poll into the
in-projection's operand loader (``source`` / ``SpGatherSource``), and folding the
gather re-arm into that loader."""

from dataclasses import dataclass
from functools import cache

import cutlass
import cutlass.cute as cute
import torch
import torch.distributed._symmetric_memory as symm_mem
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_stream

from ...shapes import Choice, Path, Unsupported
from ...symm import multicast_rendezvous, peer_rendezvous
from .host import check_residency, fake, stream, to_cute
from .mailbox import rearm
from .primitives import (
    add_rn,
    asm,
    at,
    launch,
    load_vector,
    multicast_words,
    pack_pair,
    published,
    store_relaxed,
    store_words,
    unpack_bf16,
    zeros,
)
from .twoshot import poll_slots

SENTINEL = -0x80000000  # the Lamport arm word (int32); 0x80000000 as Uint32
THREADS = 256
# Polling kernels: 16-byte chunks per thread at least (sets the grid at small R).
CHUNKS_PER_THREAD = 2
# The poll's pending mask is one Uint32: at most this many chunks per thread.
MAX_ROUNDS = 32


@asm(Uint32, pure=True)
def positive_zero32(word):
    """``word``, or 0 where it is the Lamport sentinel 0x80000000 (BF16 pair (+0,
    -0)): stock's ``sanitize_lamport_payload``. A published 16-byte chunk never holds
    the sentinel, so a reader that polls it cannot mistake data for unpublished."""
    code = "{ .reg .pred p; setp.eq.u32 p,$1,0x80000000; selp.b32 $0,0,$1,p; }"
    return code, "=r,r", (Uint32(word),)


@cute.jit
def poll_chunk(pointer):
    """The four words of the 16-byte Lamport chunk at ``pointer`` (BF16, global),
    reloaded until none is the sentinel. For a source consumer (``SpGatherSource``)."""
    words = load_vector(pointer, cache="volatile")
    held = cute.make_rmem_tensor(4, Uint32)
    for j in cutlass.range_constexpr(4):
        held[j] = words[j]
    while not published(held[j] for j in range(4)):
        words = load_vector(pointer, cache="volatile")
        for j in cutlass.range_constexpr(4):
            held[j] = words[j]
    return tuple(held[j] for j in range(4))


def sanitized(words):
    return tuple(positive_zero32(w) for w in words)


def rounds(items, grid):
    """Chunks per thread of ``items`` 16-byte chunks over ``grid`` CTAs."""
    return -(-items // (grid * THREADS))


def polling_grid(rows, width, sms):
    """CTAs of a polling kernel over ``rows`` x ``width`` BF16: at most one per SM
    (co-resident), at least CHUNKS_PER_THREAD chunks per thread."""
    chunks = rows * width // 8
    return max(1, min(sms, -(-chunks // (THREADS * CHUNKS_PER_THREAD))))


@dataclass(frozen=True)
class SpGatherSource:
    """An all-gather a consumer polls itself (``SpComm.publish`` stored this rank's
    shard; nothing waited): ``gather`` = gather[parity, :rows] ([rows, width] BF16,
    stride (width, 1)). Every 16-byte chunk must be read with ``poll_chunk`` (or an
    equivalent Lamport poll), never plainly; the consumer must not re-arm it
    (``SpReduceScatter`` of the same parity does)."""

    gather: torch.Tensor
    rows: int


class SpAllGather:
    """Multicast-store this rank's shard x [n, width] into rows [rank n, (rank + 1) n)
    of every rank's gather [cap, width] (sanitized words), then (``poll``) Lamport-
    poll gather rows [0, TP n) until every chunk landed. Args: x, the local gather
    view, its multicast address, rank."""

    def __init__(self, tp, grid, poll=True):
        self.tp, self.grid, self.poll = tp, grid, poll

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.grid, THREADS)

    @cute.kernel
    def kernel(self, args):
        x, gather, mc, rank = args
        n: cutlass.Constexpr = x.shape[0]
        width: cutlass.Constexpr = x.shape[1]
        per: cutlass.Constexpr = width // 8  # chunks per row
        total: cutlass.Constexpr = self.grid * THREADS
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        g = cta * THREADS + tid
        # x's producer is done; this parity's gather rows were re-armed.
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        for i in cutlass.range_constexpr(rounds(n * per, self.grid)):
            c = g + i * total
            if c < n * per:
                row, column = c // per, c % per * 8
                words = sanitized(load_vector(at(x, row, column)))
                offset = Int64(rank * n + row) * width + column
                multicast_words(mc + offset * 2, words)
        if cutlass.const_expr(self.poll):
            R: cutlass.Constexpr = self.tp * n
            count: cutlass.Constexpr = rounds(R * per, self.grid)
            pending = Uint32(0)
            for i in cutlass.range_constexpr(count):
                if g + i * total < R * per:
                    pending |= Uint32(1 << i)
            while pending != 0:
                again = Uint32(0)
                for i in cutlass.range_constexpr(count):
                    if (pending >> i) & 1:
                        c = g + i * total
                        words = load_vector(
                            at(gather, c // per, c % per * 8), cache="volatile"
                        )
                        if not published(words):
                            again |= Uint32(1 << i)
                pending = again


class SpReduceScatter:
    """The owner half of the SP reduce-scatter, R = TP n rows of ``width``:

    - ``push``: first store this rank's BF16 partial [R, width] (local, plain) chunk
      by chunk into its row owner's receive slot [rank] (sanitized words);
    - poll this rank's TP slots of its n rows, sum them in FP32 in rank order, round
      once (RN, ``positive_zero32``), store the shard row, re-arm the slots;
    - re-arm gather rows [0, R) of the same parity.

    Args: partial [R, width] (read only with ``push``), the local receive view
    [TP, cap / TP, width], the owners' receive bases [TP] (int64), out [n, width],
    the local gather view [cap, width], rank."""

    def __init__(self, tp, rows, grid, push):
        self.tp, self.rows, self.grid, self.push = tp, rows, grid, push

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.grid, THREADS)

    @cute.kernel
    def kernel(self, args):
        partial, recv, peers, out, gather, rank = args
        tp: cutlass.Constexpr = self.tp
        R: cutlass.Constexpr = self.rows
        n: cutlass.Constexpr = R // tp
        width: cutlass.Constexpr = out.shape[1]
        per: cutlass.Constexpr = width // 8
        total: cutlass.Constexpr = self.grid * THREADS
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        g = cta * THREADS + tid
        # The partial's producer is done; the in-projection read this parity's gather.
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(self.push):
            for i in cutlass.range_constexpr(rounds(R * per, self.grid)):
                c = g + i * total
                if c < R * per:
                    row, column = c // per, c % per * 8
                    words = sanitized(load_vector(at(partial, row, column)))
                    owner = row // n
                    offset = cute.crd2idx((rank, row - owner * n, column), recv.layout)
                    store_relaxed(Int64(peers[owner]) + Int64(offset) * 2, words)
        for i in cutlass.range_constexpr(rounds(n * per, self.grid)):
            c = g + i * total
            if c < n * per:
                row, column = c // per, c % per * 8
                raw = zeros((tp, 4), Uint32)
                poll_slots(recv, raw, row, column)
                values = cute.make_rmem_tensor(8, Float32)
                for j in cutlass.range_constexpr(4):
                    low, high = unpack_bf16(raw[0, j])
                    for slot in cutlass.range_constexpr(1, tp):
                        a, b = unpack_bf16(raw[slot, j])
                        low, high = add_rn(low, a), add_rn(high, b)
                    values[2 * j], values[2 * j + 1] = low, high
                words = tuple(
                    positive_zero32(
                        pack_pair(values[2 * j], values[2 * j + 1], policy=None)
                    )
                    for j in range(4)
                )
                store_words(at(out, row, column), words)
                for slot in cutlass.range_constexpr(tp):
                    rearm(at(recv, slot, row, column))
        for i in cutlass.range_constexpr(rounds(R * per, self.grid)):
            c = g + i * total
            if c < R * per:
                rearm(at(gather, c // per, c % per * 8))


def _check_rounds(count, what):
    if count > MAX_ROUNDS:
        raise Unsupported(
            "sp_comm", None, f"{what}: {count} chunks per thread > {MAX_ROUNDS}"
        )


@cache
def compile_all_gather(n, tp, cap, width, grid, poll, device):
    """``SpAllGather`` of n shard rows (TP n gathered) over ``grid`` CTAs."""
    _check_rounds(rounds(tp * n * width // 8, grid), "SP all-gather poll")
    with torch.accelerator.device_index(device):
        x, gather = fake((n, width)), fake((cap, width))
        kernel = SpAllGather(tp, grid, poll)
        compiled = cute.compile(
            kernel, x, gather, Int64(0), Int32(0), make_fake_stream()
        )
        if poll:  # the CTAs poll stores of every rank's grid: one co-resident wave
            what = f"Kimi K3 SP all-gather ({tp} x {n} rows)"
            check_residency(compiled, device, grid, THREADS, what)
    return compiled


@cache
def compile_reduce_scatter(rows, tp, cap, width, grid, push, device):
    """``SpReduceScatter`` of ``rows`` (= TP n) over ``grid`` CTAs."""
    _check_rounds(rounds(rows * width // 8, grid), "SP reduce-scatter")
    with torch.accelerator.device_index(device):
        partial = fake((rows, width))
        recv = fake((tp, cap // tp, width))
        peers = fake((tp,), None, torch.int64)
        out, gather = fake((rows // tp, width)), fake((cap, width))
        kernel = SpReduceScatter(tp, rows, grid, push)
        args = (partial, recv, peers, out, gather, Int32(0))
        compiled = cute.compile(kernel, *args, make_fake_stream())
        what = f"Kimi K3 SP reduce-scatter ({rows} rows, TP{tp})"
        check_residency(compiled, device, grid, THREADS, what)
    return compiled


class SpComm:
    """The TP group's SP mailboxes (module docstring): ``gather`` [parities,
    rows_cap, width] (multicast), ``recv`` [parities, TP, rows_cap / TP, width]
    (peer-mapped), ``peers`` [parities, TP] (every rank's recv[parity] base) and the
    local ``partial`` [rows_cap, width] staging of a non-fused o-projection. One per
    model, shared by every SP block; collective to build (every rank of ``group``)."""

    def __init__(
        self, group, device, tp: int, rank: int, *, width: int = 7168,
        rows_cap: int = 512, parities: int = 2,
    ):  # fmt: skip
        if rows_cap % tp or width % 8:
            raise Unsupported(
                "sp_comm", None, f"rows_cap {rows_cap} / width {width} over TP{tp}"
            )
        self.group, self.device = group, torch.device(device)
        self.tp, self.rank, self.width = tp, rank, width
        self.rows_cap, self.parities = rows_cap, parities
        self.sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        bf16 = dict(dtype=torch.bfloat16, device=self.device)
        self.gather = symm_mem.empty((parities, rows_cap, width), **bf16)
        self.recv = symm_mem.empty((parities, tp, rows_cap // tp, width), **bf16)
        _, self.gather_mc = multicast_rendezvous(
            self.gather, group, "Kimi K3 SP all-gather"
        )
        if not self.gather_mc:  # the all-gather stores through it: no fallback
            raise Unsupported("sp_comm", None, "no NVLink multicast")
        handle = peer_rendezvous(self.recv, group)
        size = self.recv[0].numel() * 2
        ptrs = handle.buffer_ptrs
        bases = [[int(b) + p * size for b in ptrs] for p in range(parities)]
        self.peers = torch.tensor(bases, dtype=torch.int64, device=self.device)
        self.partial = torch.empty((rows_cap, width), **bf16)
        for t in (self.gather, self.recv):
            t.view(torch.int32).fill_(SENTINEL)
        torch.accelerator.synchronize(self.device)
        torch.distributed.barrier(group=group)
        # cute views per parity: local gather, its multicast address, local recv, peers
        self._views = [
            (
                to_cute(self.gather[p]),
                Int64(self.gather_mc + p * self.gather[0].numel() * 2),
                to_cute(self.recv[p]),
                to_cute(self.peers[p]),
            )
            for p in range(parities)
        ]
        self._rank = Int32(rank)

    def nbytes(self) -> int:
        return sum(t.numel() * 2 for t in (self.gather, self.recv, self.partial))

    def shard(self, rows: int) -> int:
        """The shard rows n = rows / TP; Unsupported for rows this box cannot carry."""
        if rows % self.tp or not 0 < rows <= self.rows_cap:
            raise Unsupported(
                "sp_comm", rows, f"not a multiple of TP{self.tp} in 1..{self.rows_cap}"
            )
        return rows // self.tp

    def grid(self, rows: int) -> int:
        return polling_grid(rows, self.width, self.sms)

    def prepare(self, rows: int) -> None:
        """Compile (and residency-check) the all-gather of ``rows`` at build."""
        n = self.shard(rows)
        index = self.device.index
        compile_all_gather(
            n, self.tp, self.rows_cap, self.width, self.grid(rows), True, index
        )

    def reduce_scatter_code(self, rows: int, push: bool):
        """The compiled (residency-checked) ``SpReduceScatter`` of ``rows``."""
        self.shard(rows)
        return compile_reduce_scatter(
            rows, self.tp, self.rows_cap, self.width, self.grid(rows), push,
            self.device.index,
        )  # fmt: skip

    def views(self, parity: int):
        """(local gather, its multicast address, local recv, peers) of ``parity``
        as kernel arguments."""
        return self._views[parity]

    def choice(self, rows: int) -> Choice:
        self.shard(rows)
        return Choice(Path.GENERIC, "multicast", note="sp all-gather (NQ-TUNE)")

    def _check_shard(self, x_shard: torch.Tensor) -> int:
        n = x_shard.shape[0]
        if (
            x_shard.dtype != torch.bfloat16
            or x_shard.shape[1] != self.width
            or not x_shard.is_contiguous()
        ):
            raise Unsupported(
                "sp_comm", n * self.tp, f"x_shard {tuple(x_shard.shape)} "
                f"{x_shard.dtype} (want contiguous [n, {self.width}] BF16)",
            )  # fmt: skip
        self.shard(n * self.tp)
        return n

    def _gather(self, x_shard: torch.Tensor, parity: int, poll: bool) -> None:
        n = self._check_shard(x_shard)
        rows = n * self.tp
        gather, mc, _, _ = self._views[parity]
        code = compile_all_gather(
            n, self.tp, self.rows_cap, self.width, self.grid(rows), poll,
            self.device.index,
        )  # fmt: skip
        code(to_cute(x_shard), gather, mc, self._rank, stream(self.device))

    def all_gather(self, x_shard: torch.Tensor, parity: int) -> torch.Tensor:
        """gather[parity, :TP n] ([R, width], stride (width, 1)) holding every rank's
        shard in rank order once this call's kernel ends (stream-ordered)."""
        self._gather(x_shard, parity, True)
        return self.gather[parity, : x_shard.shape[0] * self.tp]

    def publish(self, x_shard: torch.Tensor, parity: int) -> None:
        """The store half of ``all_gather`` only (phase 1b: the consumer polls
        ``source(parity, rows)`` itself)."""
        self._gather(x_shard, parity, False)

    def source(self, parity: int, rows: int) -> SpGatherSource:
        self.shard(rows)
        return SpGatherSource(self.gather[parity, :rows], rows)

    def reduce_scatter(
        self, out_shard: torch.Tensor, parity: int, rows: int, partial=None
    ) -> torch.Tensor:
        """The owner half (``SpReduceScatter``) into out_shard [n, width]: the TP
        slots of this rank's rows summed in rank order (already pushed, e.g. by the
        fused o-projection), or, given ``partial`` [rows, width] (local BF16), this
        rank's pushes first. Re-arms gather[parity][:rows]."""
        n = self.shard(rows)
        if (
            out_shard.dtype != torch.bfloat16
            or out_shard.shape != (n, self.width)
            or not out_shard.is_contiguous()
        ):
            raise Unsupported(
                "oproj_rs", rows, f"out_shard {tuple(out_shard.shape)} (want [{n}, "
                f"{self.width}] contiguous)",
            )  # fmt: skip
        gather, _, recv, peers = self._views[parity]
        push = partial is not None
        source = partial if push else self.partial[:rows]
        code = self.reduce_scatter_code(rows, push)
        s = stream(self.device)
        code(to_cute(source), recv, peers, to_cute(out_shard), gather, self._rank, s)
        return out_shard
