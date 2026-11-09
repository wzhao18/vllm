# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DCP top-k of a K3.1 DSA layer at stock Kimi K3 precision: FP32 logits, so no
one-round (#63) merge (stock gates it on BF16 logits), but stock's FP32 rule
(sparse_attn_indexer._merge_dcp_topk_global, dcp_topk_score_bits(FP32) = 32):

  local top-K   IndexerSelect ("candidates"): each rank's exact top-K of its FP32
                logits (ties to the lowest local index), each as a 64-bit word,
                high half the score's order-preserving 32-bit key, low half
                0xFFFFFFFF - its global id; zero words past a short row's count
  exchange      DcpTopk (dcp_round.py): each rank's local top-N (N ~ 1.5 K / W)
                pushed by peer stores into every rank's symmetric box, the exact
                K-th word of the W * N, and every rank whose N words are all
                selected while it holds more read whole from its symmetric copy
                -> this rank's selected keys as its physical KV rows (a compacted
                prefix in its local key order, -1 after) and their counts. No
                NCCL all-gather (DCP16: 16 x fewer bytes than the K words, and no
                ring latency; 4-node C1 step 35.5 us -> a few us per DSA layer).
  DcpMerge      (the all-gather reference, tests) per row, the exact top-K of the
                W * K all-gathered words (a 64-bit radix select: the 32 score bits,
                then the id bits; every global id is on one rank only, so the
                words are distinct)

The global top-K is exact: a key of it has at most K - 1 keys above it overall, so
at most K - 1 on its own rank, and is in that rank's local top-K (dcp_round.py: and
in the union of the exchange). The selection (score desc, global id asc) is the
stock merge's stable key at 32 score bits."""

import functools

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_stream

from ..common.host import fake, stream, to_cute
from ..common.primitives import asm, load_words
from .indexer import (
    SELECT_THREADS,
    SELECT_WARPS,
    block_exclusive,
    compile_select,
    shared_add,
    table_entry,
    topdown,
)


@asm()
def require_below(value, bound):
    """Trap unless ``value`` < ``bound`` (signed)."""
    return "{ .reg .pred p; setp.ge.s32 p,$0,$1; @p trap; }", "r,r", (value, bound)


# 32-bit key digits from the top: (shift, width).
DIGITS = ((21, 11), (10, 11), (0, 10))


class DcpMerge:
    """Per row (one CTA), the exact top-``topk`` of the ``world`` ranks' gathered
    candidate words [world, rows, topk] (int32 pairs, rank-major) -> this rank's
    physical KV rows ``rows_out`` [rows, >= topk] (compacted, -1 after), their
    counts ``valid_out`` [rows] and, with ``emit_ids``, the selected global ids
    ``ids_out`` [rows, topk] (rank-major, each rank's in its candidate order; -1
    past the count; tests).

    A word's high half is its score's order key (0: empty), its low half
    0xFFFFFFFF - its global id. Each thread keeps the score keys of its W * K /
    1024 words in registers: a radix select (11, 11, 10 bits) finds the K-th
    score and the ties at it to take; only when the ties outnumber them does a
    second one over those ties' low halves (read again) pick the lowest ids.

    It never launches the next kernel early (PDL): every kernel after it reads its
    rows or counts before its own PDL wait (FlashInfer's trtllm-gen and prims_ts
    sparse MLA, prepared.HELD_KERNELS; the DSA sparse core polls first). A selected
    key past the block table traps (fast or raise: never a silent -1 row)."""

    def __init__(self, rows, world, topk, interleave=1, emit_ids=False):
        if topk % SELECT_THREADS:
            raise ValueError(f"DcpMerge: top-k {topk} not a multiple of 1024")
        self.rows, self.world, self.topk = rows, world, topk
        self.interleave, self.emit_ids = interleave, emit_ids

    @cute.jit
    def __call__(
        self,
        gathered: Int64,  # int32 [world, rows, 2 * topk]
        req_id: cute.Tensor,  # int32 [>= rows]
        table_address: Int64,
        table_pitch: Int32,
        table_width: Int32,
        block_size: Int32,
        stride_rows: Int32,
        rank: Int32,
        rows_out: cute.Tensor,
        valid_out: cute.Tensor,
        ids_out: cute.Tensor,
        s,
    ):
        table = (table_address, table_pitch, table_width)
        args = (gathered, req_id, table, block_size, stride_rows, rank)
        self.kernel(args, rows_out, valid_out, ids_out).launch(
            grid=(self.rows, 1, 1),
            block=(SELECT_THREADS, 1, 1),
            stream=s,
            use_pdl=True,
        )

    @cute.jit
    def address(self, gathered, r, row, j):
        """The int32 pair of candidate ``j`` of rank ``r``'s row ``row``."""
        width: cutlass.Constexpr = 2 * self.topk
        offset = (Int64(r) * Int64(self.rows) + Int64(row)) * Int64(width)
        return gathered + (offset + Int64(2 * j)) * 4

    @cute.jit
    def radix(self, ties: cutlass.Constexpr[bool], need, sm, items, tid, warp, lane):
        """The 32-bit radix select of the ``need``-th largest of the thread's
        items -> (that value, how many equal to it to take). ``ties`` False: the
        items' score keys (``items``: (keys, ...)); True: the low halves of the
        words whose score key is the threshold (``items``: (keys, threshold,
        gathered, row), the low halves read again)."""
        hist, scan, info = sm
        prefix = Uint32(0)
        for d in cutlass.range_constexpr(3):
            shift, width = DIGITS[d]
            for b in cutlass.range(tid, 1 << width, SELECT_THREADS, unroll=1):
                hist[b] = Uint32(0)
            cute.arch.sync_threads()
            per_rank: cutlass.Constexpr = self.topk // SELECT_THREADS
            for i in cutlass.range_constexpr(self.world * per_rank):
                keys = items[0]
                key = keys[i]
                valid = Int32(0)
                if cutlass.const_expr(ties):
                    _, threshold, gathered, row = items
                    if keys[i] == threshold:
                        j = (i % per_rank) * SELECT_THREADS + tid
                        at = self.address(gathered, i // per_rank, row, j)
                        key = Uint32(load_words(at, count=1)[0])
                        valid = Int32(1)
                else:
                    if key != Uint32(0):
                        valid = Int32(1)
                if valid != Int32(0):
                    bin_ = (key >> Uint32(shift)) & Uint32((1 << width) - 1)
                    if cutlass.const_expr(d == 0):
                        shared_add(hist.iterator + bin_, 1)
                    else:
                        if (key >> Uint32(shift + width)) == prefix:
                            shared_add(hist.iterator + bin_, 1)
            cute.arch.sync_threads()
            topdown(hist, 1 << width, need, info, 2 + 2 * d, scan, tid, warp, lane)
            prefix = (prefix << Uint32(width)) | Uint32(info[2 + 2 * d])
            need = need - Uint32(info[3 + 2 * d])
        return prefix, need

    @cute.kernel
    def kernel(self, args, rows_out, valid_out, ids_out):
        gathered, req_id, table, block_size, stride_rows, rank = args
        K, W = self.topk, self.world
        per_rank: cutlass.Constexpr = K // SELECT_THREADS
        count: cutlass.Constexpr = W * per_rank
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32

        smem = utils.SmemAllocator()
        hist = smem.allocate_tensor(Uint32, cute.make_layout(2048), 16)
        scan = smem.allocate_tensor(Uint32, cute.make_layout(SELECT_WARPS), 16)
        info = smem.allocate_tensor(Int32, cute.make_layout(16), 16)
        sm = hist, scan, info
        cute.arch.griddepcontrol_wait()  # (no early launch: see the class doc)
        # Item i of this thread: rank i // per_rank, candidate (i % per_rank) x
        # 1024 + tid (coalesced). Its score key only.
        keys = []
        mine = Uint32(0)
        for i in cutlass.range_constexpr(count):
            j = (i % per_rank) * SELECT_THREADS + tid
            at = self.address(gathered, i // per_rank, row, j)
            key = Uint32(load_words(at + 4, count=1)[0])
            keys.append(key)
            if key != Uint32(0):
                mine += Uint32(1)
        _, total = block_exclusive(mine, scan, warp, lane)
        # The selection: score key > T, or == T with low half >= L (every nonzero
        # word when at most K are: T = 0, L = 0).
        threshold = Uint32(0)
        low = Uint32(0)
        if total > Uint32(K):
            scores = (keys,)
            threshold, take = self.radix(False, Uint32(K), sm, scores, tid, warp, lane)
            ties = hist[Int32(threshold & Uint32(0x3FF))]
            cute.arch.sync_threads()
            if ties != take:  # the lowest global ids among the ties
                tied = (keys, threshold, gathered, row)
                low, _ = self.radix(True, take, sm, tied, tid, warp, lane)
        # This rank's selected keys (its own candidates), in its local key order.
        base = Uint32(0)
        for j0 in cutlass.range_constexpr(per_rank):
            j = j0 * SELECT_THREADS + tid
            lo, hi = load_words(self.address(gathered, rank, row, j), count=2)
            take = self.selected(Uint32(hi), Uint32(lo), threshold, low)
            pos, n = block_exclusive(take, scan, warp, lane)
            # (at most K: the words are distinct, every global id on one rank)
            if take != Uint32(0) and base + pos < Uint32(K):
                gid = Int32(Uint32(0xFFFFFFFF) - Uint32(lo))
                span: cutlass.Constexpr = self.interleave
                local = (gid // (span * W)) * span + gid % span
                block = local // block_size
                require_below(block, table[2])  # a key past the block table
                page = table_entry(table, req_id[row], block)
                slot = page * stride_rows + local % block_size
                rows_out[row, Int32(base + pos)] = slot
            base += n
        base = cutlass.min(base, Uint32(K))
        if tid == 0:
            valid_out[row] = Int32(base)
        for e in cutlass.range(Int32(base) + tid, K, SELECT_THREADS, unroll=1):
            rows_out[row, e] = Int32(-1)
        if cutlass.const_expr(self.emit_ids):
            # Deterministic: rank-major, each rank's candidates in order.
            filled = Uint32(0)
            for i in cutlass.range_constexpr(count):
                j = (i % per_rank) * SELECT_THREADS + tid
                at = self.address(gathered, i // per_rank, row, j)
                lo = Uint32(load_words(at, count=1)[0])
                take = self.selected(keys[i], lo, threshold, low)
                pos, n = block_exclusive(take, scan, warp, lane)
                if take != Uint32(0) and filled + pos < Uint32(K):
                    gid = Int32(Uint32(0xFFFFFFFF) - lo)
                    ids_out[row, Int32(filled + pos)] = gid
                filled += n
            filled = cutlass.min(filled, Uint32(K))
            for e in cutlass.range(Int32(filled) + tid, K, SELECT_THREADS, unroll=1):
                ids_out[row, e] = Int32(-1)

    @cute.jit
    def selected(self, hi, lo, threshold, low) -> Uint32:
        """1 if a word (score key ``hi``, low half ``lo``) is selected, else 0."""
        take = Uint32(0)
        if hi != Uint32(0):
            if hi > threshold:
                take = Uint32(1)
            if hi == threshold and lo >= low:
                take = Uint32(1)
        return take


@functools.cache
def compile_merge(rows, world, topk, interleave, emit_ids, device):
    merge = DcpMerge(rows, world, topk, interleave, emit_ids)
    with torch.accelerator.device_index(device):
        req = fake((rows,), (1,), torch.int32, 4)
        out = fake((rows, topk), (topk, 1), torch.int32, 4)
        valid = fake((rows,), (1,), torch.int32, 4)
        ids = fake((rows, topk), (topk, 1), torch.int32, 4)
        scalars = Int64(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0)
        s = make_fake_stream()
        args = Int64(0), req, *scalars, out, valid, ids, s
        return cute.compile(merge, *args)


@functools.cache
def candidate_rows(device, rows, topk):
    """The local candidates [rows, 2 topk] int32 of every DSA layer of a device
    (stream-ordered: each layer's merge reads its gather before the next layer's
    select writes them)."""
    return torch.zeros((rows, 2 * topk), dtype=torch.int32, device=device)


class DcpTopk:
    """The DCP top-k of one DSA layer's indexer (``world`` ranks, this one
    ``rank``; ``interleave``: the DCP interleave, or a callable giving it once it is
    final, after the block size setup): ``run`` takes FP32 logits rows of this
    rank's keys to this rank's physical KV rows of the exact global top-K and their
    counts. Collective over ``group`` (default: the DCP group's device group): every
    rank must call ``warm`` (which builds the exchange's symmetric box, dcp_round.py)
    and ``run`` with the same rows, in the same order."""

    def __init__(self, topk, world, rank, interleave, device, max_rows, group=None):
        from .dcp_round import round1_words
        from .indexer import select_unsupported

        reason = select_unsupported(topk)
        if reason is not None:
            raise ValueError(f"Kimi K3.1 specialized DSA DCP top-k: {reason}")
        self.topk, self.world, self.rank = topk, world, rank
        self._interleave, self.device, self.group = interleave, device, group
        self.dummy = torch.zeros(1, dtype=torch.int32, device=device)
        self.candidates = candidate_rows(device, max_rows, topk)
        self.local_topk = round1_words(topk, world)
        self.box = None  # the exchange's DcpRoundBox (``warm``: collective)

    def warm(self, rows, emit_ids=False):
        """Build the exchange's symmetric box (collective, once per device, rows and
        group) and compile ``run``'s kernels for steps of ``rows`` rows now (a first
        compile inside a CUDA graph capture invalidates it)."""
        self.check_rows(rows)
        self.exchange()
        self.kernels(rows, emit_ids)

    def exchange(self):
        """The DcpRoundBox this top-k exchanges through (built here on first use)."""
        if self.box is None:
            from .dcp_round import round_box

            group = self.group
            if group is None:
                from vllm.distributed.parallel_state import get_dcp_group

                group = get_dcp_group()
            group = getattr(group, "device_group", group)
            max_rows = self.candidates.shape[0]
            args = self.device, max_rows, self.topk, self.world, self.rank, group
            self.box = round_box(*args)
        return self.box

    def check_rows(self, rows):
        """The select and merge grids write ``rows`` rows of the shared candidates."""
        if not 0 < rows <= self.candidates.shape[0]:
            raise ValueError(
                f"DCP top-k: {rows} rows, the candidates hold "
                f"{self.candidates.shape[0]}"
            )

    @property
    def interleave(self) -> int:
        span = self._interleave
        return int(span() if callable(span) else span)

    def kernels(self, rows, emit_ids):
        """(IndexerSelect "candidates", DcpPack, DcpSelect) of ``rows`` rows. The
        select launches DcpPack early and DcpPack DcpSelect (each waits for its
        predecessor before reading anything)."""
        from .dcp_round import compile_pack, compile_round_select

        index = self.device.index
        args = self.topk, "candidates", index, 1
        select = compile_select(
            rows, 1, *args, release=True, world=self.world, interleave=self.interleave
        )
        shape = rows, self.world, self.topk, self.local_topk
        max_rows = self.candidates.shape[0]
        pack = compile_pack(*shape, max_rows, index)
        merge = compile_round_select(
            *shape, max_rows, self.interleave, emit_ids, index
        )
        return select, pack, merge

    def run(
        self,
        logits,
        row_ends,
        req_id,
        block_table,
        block_size,
        stride_rows,
        rows_out,
        valid_out,
        ids_out=None,
    ):
        """``logits`` [>= the rows with keys, >= max(row_ends)] FP32 (16-byte
        aligned rows): this rank's local keys; ``row_ends`` [rows] int32 their
        counts per row (0: a row past the step's requests, its logits unread);
        ``req_id`` / ``block_table`` / ``block_size`` / ``stride_rows``: the MLA
        layer's metadata and flat KV row stride. Writes ``rows_out`` [rows, >= K],
        ``valid_out`` [rows] and (tests) ``ids_out`` [rows, K]. Nothing after it
        starts before it has finished (no early PDL launch)."""
        rows = row_ends.shape[0]
        self.check_rows(rows)
        if rows_out.shape[0] < rows or valid_out.shape[0] < rows:
            raise ValueError(f"DCP top-k: outputs of {rows_out.shape[0]} rows < {rows}")
        if logits.dtype != torch.float32:
            raise TypeError(f"DCP top-k: FP32 logits (stock K3), got {logits.dtype}")
        if logits.stride(1) != 1 or logits.stride(0) % 4 or logits.data_ptr() % 16:
            raise ValueError(f"DCP top-k: 16-byte aligned rows, got {logits.stride()}")
        if block_table.stride(1) != 1 or rows_out.stride(0) != self.topk:
            raise ValueError("DCP top-k: contiguous block table rows, K-wide rows_out")
        if ids_out is not None and ids_out.stride(0) != self.topk:
            raise ValueError("DCP top-k: K-wide ids_out")
        box = self.box
        if box is None:
            raise RuntimeError(
                "DCP top-k: run before warm (the exchange's symmetric box is built "
                "collectively by warm, outside any capture)"
            )
        select, pack, merge = self.kernels(rows, ids_out is not None)
        candidates = self.candidates[:rows]
        s = stream(self.device)
        dummy = to_cute(self.dummy, 4)
        select(
            Int64(logits.data_ptr()),
            Int32(logits.stride(0)),
            dummy,
            to_cute(row_ends[:rows], 4),
            Int64(0),
            Int32(0),
            Int32(0),
            Int32(self.rank),
            to_cute(candidates, 4),
            to_cute(valid_out[:rows], 4),
            to_cute(candidates, 4),
            s,
        )
        ordinal = Int64(box.ordinal.data_ptr())  # int64 [max_rows]
        copies = int(box.copies.data_ptr())
        pack(
            Int64(candidates.data_ptr()),
            ordinal,
            to_cute(box.box_peers, 8),
            Int64(copies),
            Int32(self.rank),
            s,
        )
        ids = ids_out if ids_out is not None else rows_out
        merge(
            Int64(box.box.data_ptr()),
            to_cute(box.copy_peers, 8),
            to_cute(box.ready_peers, 8),
            ordinal,
            Int64(candidates.data_ptr()),
            to_cute(req_id[:rows], 4),
            Int64(block_table.data_ptr()),
            Int32(block_table.stride(0)),
            Int32(block_table.shape[1]),
            Int32(block_size),
            Int32(stride_rows),
            Int32(self.rank),
            to_cute(rows_out[:rows, : self.topk], 4),
            to_cute(valid_out[:rows], 4),
            to_cute(ids[:rows, : self.topk], 4),
            s,
        )
