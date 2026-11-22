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

from vllm.models.kimi_k3.nvidia.k31_spin_guard import spin_start, spin_tick

from ...shapes import Choice, Path, Unsupported
from ...symm import multicast_rendezvous, peer_rendezvous
from . import mx_rows
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
# SpGatherCopy: 16-byte chunks per thread in flight per batch.
GATHER_BATCH = 8
# SpGatherPush: CTAs per early push (beside the step's kernels), 16-byte chunks
# per 7168-wide row.
PUSH_CTAS = 32
WIDTH_CHUNK = 7168 // 8


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
            sgn, sgt = spin_start()
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
                sgn, sgt = spin_tick(
                    sgn, sgt, observed=pending,
                    target=0, rank=rank, size=self.tp
                )


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
        self.reduce_owned(recv, out, gather, g)

    @cute.jit
    def reduce_owned(self, recv, out, gather, g):
        """Poll, sum in rank order, store and re-arm this rank's n rows; re-arm the
        gather rows [0, R) of the same parity (``g``: this thread's global index)."""
        tp: cutlass.Constexpr = self.tp
        R: cutlass.Constexpr = self.rows
        n: cutlass.Constexpr = R // tp
        width: cutlass.Constexpr = out.shape[1]
        per: cutlass.Constexpr = width // 8
        total: cutlass.Constexpr = self.grid * THREADS
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


class SpEmbedScatter(SpReduceScatter):
    """The SP decode's input embedding straight into the row shards: the vocab-
    parallel gather (stock ``VocabParallelEmbedding``: this rank's vocabulary shard
    rows, zeros for ids it does not hold) pushed as the reduce-scatter's partial,
    then ``SpReduceScatter``'s owner half. Replaces the masked gather, the TP
    all-reduce of all R rows and the shard slice. Exactly one rank holds a valid id's
    row, so the rank-order FP32 sum is that row bit for bit (a -0 becomes +0, as in
    the all-reduce's x + 0).

    Args: ids [R] (int32 / int64), weight [shard, width] (this rank's vocabulary
    rows), start (the shard's first id), vocab (ids >= vocab are zero), then
    ``SpReduceScatter``'s recv, peers, out, gather, rank."""

    def __init__(self, tp, rows, grid):
        super().__init__(tp, rows, grid, True)

    @cute.kernel
    def kernel(self, args):
        ids, weight, start, vocab, recv, peers, out, gather, rank = args
        tp: cutlass.Constexpr = self.tp
        R: cutlass.Constexpr = self.rows
        n: cutlass.Constexpr = R // tp
        width: cutlass.Constexpr = out.shape[1]
        per: cutlass.Constexpr = width // 8
        total: cutlass.Constexpr = self.grid * THREADS
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        g = cta * THREADS + tid
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        for i in cutlass.range_constexpr(rounds(R * per, self.grid)):
            c = g + i * total
            if c < R * per:
                row, column = c // per, c % per * 8
                token = Int64(ids[row])
                local = token - start
                held = zeros((4,), Uint32)
                valid = (token >= 0) & (token < vocab)
                if valid & (local >= 0) & (local < weight.shape[0]):
                    words = sanitized(load_vector(at(weight, local, column)))
                    for j in cutlass.range_constexpr(4):
                        held[j] = words[j]
                owner = row // n
                offset = cute.crd2idx((rank, row - owner * n, column), recv.layout)
                words = tuple(held[j] for j in range(4))
                store_relaxed(Int64(peers[owner]) + Int64(offset) * 2, words)
        self.reduce_owned(recv, out, gather, g)


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


@cache
def compile_embed_scatter(rows, tp, cap, width, grid, shard, ids_dtype, device):
    """``SpEmbedScatter`` of ``rows`` token ids over ``grid`` CTAs."""
    _check_rounds(rounds(rows * width // 8, grid), "SP embedding scatter")
    with torch.accelerator.device_index(device):
        size = torch.empty((), dtype=ids_dtype).element_size()
        ids = fake((rows,), None, ids_dtype, size)
        weight = fake((shard, width))
        recv = fake((tp, cap // tp, width))
        peers = fake((tp,), None, torch.int64)
        out, gather = fake((rows // tp, width)), fake((cap, width))
        args = (ids, weight, Int64(0), Int64(0), recv, peers, out, gather, Int32(0))
        compiled = cute.compile(
            SpEmbedScatter(tp, rows, grid), *args, make_fake_stream()
        )
        what = f"Kimi K3 SP embedding scatter ({rows} rows, TP{tp})"
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
        # Each parity's box: rows_cap BF16 rows, then the E8M0 scale region of the
        # MXFP8 form of the same all-gather (``mx_view``: an MXFP8 in-projection's
        # input, FP8 rows over the box's first bytes; ``common.mx_rows``).
        self.mx_scale_bytes = mx_rows.scale_bytes(rows_cap, width)
        extra = -(-self.mx_scale_bytes // (2 * width))
        box = symm_mem.empty((parities, rows_cap + extra, width), **bf16)
        self.box = box
        self.gather = box[:, :rows_cap]
        self.recv = symm_mem.empty((parities, tp, rows_cap // tp, width), **bf16)
        _, self.gather_mc = multicast_rendezvous(box, group, "Kimi K3 SP all-gather")
        if not self.gather_mc:  # the all-gather stores through it: no fallback
            raise Unsupported("sp_comm", None, "no NVLink multicast")
        handle = peer_rendezvous(self.recv, group)
        size = self.recv[0].numel() * 2
        ptrs = handle.buffer_ptrs
        bases = [[int(b) + p * size for b in ptrs] for p in range(parities)]
        self.peers = torch.tensor(bases, dtype=torch.int64, device=self.device)
        self.partial = torch.empty((rows_cap, width), **bf16)
        for t in (self.box, self.recv):
            t.view(torch.int32).fill_(SENTINEL)
        # The MXFP8 scale regions have their own sentinel (E8M0 0xFF is never a
        # scale; 0x80000000 may be four).
        self.box[:, rows_cap:].view(torch.int32).fill_(mx_rows.SCALE_SENTINEL - 2**32)
        torch.accelerator.synchronize(self.device)
        torch.distributed.barrier(group=group)
        # cute views per parity: local gather, its multicast address, local recv, peers
        pitch = box[0].numel() * 2  # bytes per parity
        scales = rows_cap * width * 2  # the scale region's byte offset in a box
        self._views = [
            (
                to_cute(self.gather[p]),
                Int64(self.gather_mc + p * pitch),
                to_cute(self.recv[p]),
                to_cute(self.peers[p]),
            )
            for p in range(parities)
        ]
        # MXFP8 views per parity: (data mc, scale mc, local data, local scales)
        self._mx = [
            tuple(
                Int64(base + p * pitch + offset)
                for base, offset in (
                    (self.gather_mc, 0), (self.gather_mc, scales),
                    (box.data_ptr(), 0), (box.data_ptr(), scales),
                )
            )
            for p in range(parities)
        ]  # fmt: skip
        self._mx_views: dict = {}
        self._rank = Int32(rank)

    def nbytes(self) -> int:
        return sum(t.numel() * 2 for t in (self.box, self.recv, self.partial))

    def mx_addresses(self, parity: int):
        """(data multicast, scale multicast, local data, local scales) addresses of
        the MXFP8 form of ``parity``'s box (kernel arguments)."""
        return self._mx[parity]

    def mx_view(self, parity: int, rows: int):
        """The MXFP8 all-gather of ``rows`` rows in ``parity``'s box as an
        ``ops.mxfp8_act.Mxfp8Buffer`` (FP8 [rows, width] over the box's first bytes,
        F8_128x4 scales of whole 128-row tiles; padding rows are never read into
        stored outputs). Read plainly once its gather kernel ended."""
        key = parity, rows
        view = self._mx_views.get(key)
        if view is None:
            from ....ops import mxfp8_act

            self.shard(rows)
            flat = self.box[parity].view(torch.uint8).reshape(-1)
            view = object.__new__(mxfp8_act.Mxfp8Buffer)
            view.rows, view.k = rows, self.width
            view.padded_cols = self.width // mx_rows.BLOCK
            view.data = flat[: rows * self.width].view(torch.float8_e4m3fn).view(
                rows, self.width
            )
            start = self.rows_cap * self.width * 2
            view.scale = flat[start : start + mx_rows.scale_bytes(rows, self.width)]
            view.orig_dtype = torch.bfloat16
            self._mx_views[key] = view
        return view

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

    def embed_scatter(
        self, ids, weight, start: int, vocab: int, out_shard, parity: int
    ) -> torch.Tensor:
        """``SpEmbedScatter``: this rank's shard [n, width] of the embedding of the
        R = len(ids) tokens (``weight``: this rank's vocabulary rows from id
        ``start``) into ``out_shard``. Re-arms gather[parity][:R]."""
        rows = ids.shape[0]
        n = self.shard(rows)
        if out_shard.shape != (n, self.width) or not out_shard.is_contiguous():
            raise Unsupported("embed", rows, f"out_shard {tuple(out_shard.shape)}")
        gather, _, recv, peers = self._views[parity]
        code = compile_embed_scatter(
            rows, self.tp, self.rows_cap, self.width, self.grid(rows),
            weight.shape[0], ids.dtype, self.device.index,
        )  # fmt: skip
        size = ids.element_size()
        args = (to_cute(ids, size), to_cute(weight), Int64(start), Int64(vocab))
        code(*args, recv, peers, to_cute(out_shard), gather, self._rank,
             stream(self.device))  # fmt: skip
        return out_shard

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


class SpGatherPush:
    """The early part of ``SpGatherCopy``'s push: multicast-store this rank's n rows of
    ``x`` [n, w] (sanitized words) into columns [col, col + w) of rows [rank n,
    (rank + 1) n) of every rank's box [cap, W], for packed columns that are final
    before the step's end (``SpFinalGather.push``). No poll: ``SpGatherCopy`` polls
    every column. Args: x [n, w], the box's multicast address, rank."""

    def __init__(self, grid, width, col):
        self.grid, self.width, self.col = grid, width, col

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.grid, THREADS)

    @cute.kernel
    def kernel(self, args):
        x, mc, rank = args
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        push_rows(x, mc, rank, self.col, self.width, cta * THREADS + tid,
                  self.grid * THREADS)  # fmt: skip


@cute.jit
def push_rows(x, mc, rank, col: cutlass.Constexpr, width: cutlass.Constexpr, g, total):
    """Multicast-store rows [0, n) of ``x`` [n, w] at rows rank n + row, columns col +
    .. of a box of ``width`` columns: GATHER_BATCH chunks per thread in flight (all
    loads, then all stores: a load -> store chain per chunk is latency bound)."""
    n: cutlass.Constexpr = x.shape[0]
    per: cutlass.Constexpr = x.shape[1] // 8
    U: cutlass.Constexpr = GATHER_BATCH
    held = cute.make_rmem_tensor(4 * U, Uint32)
    for base in range(g, n * per, total * U):
        for i in cutlass.range_constexpr(U):
            c = base + i * total
            if c < n * per:
                words = load_vector(at(x, c // per, c % per * 8))
                for j in cutlass.range_constexpr(4):
                    held[4 * i + j] = words[j]
        for i in cutlass.range_constexpr(U):
            c = base + i * total
            if c < n * per:
                row, column = c // per, c % per * 8
                words = sanitized(tuple(held[4 * i + j] for j in range(4)))
                offset = Int64(rank * n + row) * width + col + column
                multicast_words(mc + offset * 2, words)


class SpGatherCopy:
    """The step's final row all-gather (stock ``sp_all_gather`` of the packed
    [hidden | aux...] shard rows, width W): multicast-store this rank's n rows of each
    source (``cols``: its first packed column; the other columns were pushed earlier
    by ``SpGatherPush``) into rows [rank n, (rank + 1) n) of every rank's box [cap, W],
    then poll every chunk of rows [0, TP n), copy it into ``out`` [TP n, W] and re-arm
    it. One kernel; its CTAs poll stores of every rank's grid: at most one CTA per SM
    (co-resident). Args: the sources [n, w_i], the local box view, its multicast
    address, out, rank."""

    def __init__(self, tp, grid, cols=(0,)):
        self.tp, self.grid, self.cols = tp, grid, cols

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.grid, THREADS)

    @cute.kernel
    def kernel(self, args):
        S: cutlass.Constexpr = len(self.cols)
        xs = args[:S]
        box, mc, out, rank = args[S:]
        n: cutlass.Constexpr = xs[0].shape[0]
        width: cutlass.Constexpr = out.shape[1]
        per: cutlass.Constexpr = width // 8
        total: cutlass.Constexpr = self.grid * THREADS
        R: cutlass.Constexpr = self.tp * n
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        g = cta * THREADS + tid
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        for s in cutlass.range_constexpr(S):
            push_rows(xs[s], mc, rank, self.cols[s], width, g, total)
        U: cutlass.Constexpr = GATHER_BATCH
        held = cute.make_rmem_tensor(4 * U, Uint32)
        for base in range(g, R * per, total * U):
            pending = Uint32(0)
            for i in cutlass.range_constexpr(U):
                c = base + i * total
                if c < R * per:
                    pending |= Uint32(1 << i)
            sgn, sgt = spin_start()
            while pending != 0:
                again = Uint32(0)
                for i in cutlass.range_constexpr(U):
                    if (pending >> i) & 1:
                        c = base + i * total
                        words = load_vector(
                            at(box, c // per, c % per * 8), cache="volatile"
                        )
                        for j in cutlass.range_constexpr(4):
                            held[4 * i + j] = words[j]
                        if not published(words):
                            again |= Uint32(1 << i)
                pending = again
                sgn, sgt = spin_tick(
                    sgn, sgt, observed=pending,
                    target=0, rank=rank, size=self.tp
                )
            for i in cutlass.range_constexpr(U):
                c = base + i * total
                if c < R * per:
                    row, column = c // per, c % per * 8
                    store_words(
                        at(out, row, column), tuple(held[4 * i + j] for j in range(4))
                    )
                    rearm(at(box, row, column))


@cache
def compile_gather_copy(n, tp, cap, width, grid, device, cols=(0,), widths=None):
    widths = widths or (width,)
    with torch.accelerator.device_index(device):
        xs = tuple(fake((n, w)) for w in widths)
        box, out = fake((cap, width)), fake((tp * n, width))
        kernel = SpGatherCopy(tp, grid, cols)
        compiled = cute.compile(
            kernel, *xs, box, Int64(0), out, Int32(0), make_fake_stream()
        )
        what = f"Kimi K3 SP final gather ({tp} x {n} rows x {width})"
        check_residency(compiled, device, grid, THREADS, what)
    return compiled


@cache
def compile_gather_push(n, w, width, col, grid, device):
    with torch.accelerator.device_index(device):
        kernel = SpGatherPush(grid, width, col)
        return cute.compile(
            kernel, fake((n, w)), Int64(0), Int32(0), make_fake_stream()
        )


class SpFinalGather:
    """The final all-gather's multicast box [rows_cap, width] (Lamport-armed; the
    kernel re-arms what it reads). Collective to build. One box: a rank may start
    the next gather only once every rank has read this one (a decode step's
    collectives separate two final gathers)."""

    def __init__(self, group, device, tp, rank, *, width, rows_cap):
        if rows_cap % tp or width % 8:
            raise Unsupported(
                "final_gather", None, f"rows_cap {rows_cap} / width {width} TP{tp}"
            )
        self.device, self.tp, self.rank = torch.device(device), tp, rank
        self.width, self.rows_cap = width, rows_cap
        self.sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        self.box = symm_mem.empty(
            (rows_cap, width), dtype=torch.bfloat16, device=self.device
        )
        _, self.mc = multicast_rendezvous(self.box, group, "Kimi K3 SP final gather")
        if not self.mc:
            raise Unsupported("final_gather", None, "no NVLink multicast")
        self.box.view(torch.int32).fill_(SENTINEL)
        torch.accelerator.synchronize(self.device)
        torch.distributed.barrier(group=group)
        self._box, self._rank = to_cute(self.box), Int32(rank)

    def grid(self, rows):
        chunks = rows * self.width // 8
        return max(1, min(self.sms, -(-chunks // (THREADS * 8))))

    def prepare(self, rows, cols=(0,), widths=None):
        n = rows // self.tp
        return compile_gather_copy(
            n, self.tp, self.rows_cap, self.width, self.grid(rows),
            self.device.index, tuple(cols), None if widths is None else tuple(widths),
        )  # fmt: skip

    def push_grid(self, rows):
        """A few CTAs: the early pushes run beside the step's own kernels."""
        chunks = rows // self.tp * WIDTH_CHUNK
        return max(1, min(PUSH_CTAS, -(-chunks // (THREADS * GATHER_BATCH))))

    def prepare_push(self, rows, col, w):
        n = rows // self.tp
        return compile_gather_push(
            n, w, self.width, col, self.push_grid(rows), self.device.index
        )

    def _check(self, x_shard, w, rows):
        if (
            x_shard.shape[1] != w
            or not x_shard.is_contiguous()
            or x_shard.dtype != torch.bfloat16
            or rows > self.rows_cap
        ):
            raise Unsupported(
                "final_gather", rows, f"x_shard {tuple(x_shard.shape)} (width "
                f"{w} of {self.width}, cap {self.rows_cap})",
            )  # fmt: skip

    def push(self, x_shard: torch.Tensor, col: int) -> None:
        """Multicast this rank's rows of packed columns [col, col + w) now (the
        column block is final); the step's ``run`` polls them with the rest."""
        n, w = x_shard.shape
        rows = n * self.tp
        self._check(x_shard, w, rows)
        code = self.prepare_push(rows, col, w)
        code(to_cute(x_shard), Int64(self.mc), self._rank, stream(self.device))

    def run(self, x_shard: torch.Tensor, cols=None) -> torch.Tensor:
        """The packed all-gather [TP n, W]: ``x_shard`` [n, W], or the column blocks
        not pushed yet: ``cols`` = ((tensor [n, w], first column), ...)."""
        full = cols is None
        if full:
            cols = ((x_shard, 0),)
        n = cols[0][0].shape[0]
        rows = n * self.tp
        for t, _ in cols:
            self._check(t, self.width if full else t.shape[1], rows)
        out = cols[0][0].new_empty((rows, self.width))
        offsets = tuple(c for _, c in cols)
        widths = None if full else tuple(t.shape[1] for t, _ in cols)
        code = self.prepare(rows, offsets, widths)
        code(
            *(to_cute(t) for t, _ in cols), self._box, Int64(self.mc), to_cute(out),
            self._rank, stream(self.device),
        )  # fmt: skip
        return out
