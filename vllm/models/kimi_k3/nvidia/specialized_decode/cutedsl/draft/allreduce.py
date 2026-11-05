# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSpark drafter's all-reduce + residual + RMSNorm sites as Lamport kernels:
one-shot for at most 8 rows (B1: 7 query rows), two-shot for at most 64 (B8: 56).

Stock (FlashInfer's fused all-reduce, two-shot at TP16): a two-shot all-reduce kernel
then a Lamport RMSNorm kernel, ~13 us per site at B1 on Rubin, 11 sites per draft step
(the embedding, two per layer, the final norm).

``ArNorm``: one cluster of 8 CTAs per row (896 columns each). Every rank multicasts
its BF16 partial row into slot ``rank`` of every rank's Lamport mailbox [parity, TP,
CAP, 7168]; each CTA polls the TP slots of its columns, sums them in FP32 in rank
order and rounds once (FlashInfer's fp32_acc all-reduce), adds the residual in FP32
and rounds (the new residual, written in place of the partial), and RMS-normalizes
with the row's sum of squares (warp, CTA, then DSMEM cluster sums; rsqrtf as
FlashInfer's fusion; x * r * w rounded once). Re-arms what it read. The embedding site
polls one slot instead: the target's embedding rows multicast by their owner rank
(common.embedding.RingEmbeddingGather), no residual. Parity alternates by site.

``polled`` sites read no partial: the projection before them (o_proj, down_proj) is a
``RingProjection`` whose epilogue multicasts its BF16 partial rows (the same split-K
tiles, rounding and negative-zero rule as the stored partial) into slot ``rank``
itself; the new residual is written in place. ``early`` (deferred) polled sites run
everything before their PDL wait and wait at the end: their ring producers release
them only after their own wait (``RingSplitKGemm.trigger_after_wait``), so every
kernel before the producer has completed when the site starts; the site reads, besides
the slots (Lamport / validity landed, written by this layer's producers), only the
residual (written in place by the previous site), the norm weights (static) and its
own output buffer, all older than the producer. The site still completes only after
its wait (after the producer), so the next kernel's wait covers both.

``gate``: before the CTA polls, lane s of warp 0 waits for one fragment of source s
(this CTA's last 8 columns); every fragment is still checked by its owning thread. At
16 ranks the many outstanding polls of an ungated sweep delay the remote stores
landing in the lines they poll.

``ArNormTwoShot`` (common.twoshot, the target's two-shot exchange; ``two_shot`` rows):
every rank pushes its BF16 partial row chunks to their column owners (peer stores;
rank o owns columns [o W, (o + 1) W), W = 7168 / TP); the owners sum the TP slots in
FP32 in rank order, round once and multicast the sum into every rank's gather row
(each CTA of the row's cluster reduces 1/8 of the owned chunks); then, per CTA as
``ArNorm``, the gather row is polled, the residual added and the row normalized. Each
GPU moves ~2 x the rows instead of TP x. The same sums and rounding points as
``ArNorm`` (and FlashInfer's fused all-reduce).

``ArNormRows`` (wide rows, above 64: one 8-CTA cluster per row would take ~10 waves at
224 rows): one CTA of 8 x 128 threads per row (the cluster's CTAs as thread groups),
the same push, owner reduce, residual add, warp sums, group sums and cross-group order
as ``ArNormTwoShot`` / ``ArNorm.finish``: bitwise their outputs, in one wave while
every row's CTA is resident. Above that a row's CTA waits only on the same row's CTAs
of the other ranks (owner sums of pushes those CTAs issue before any wait), so with
in-order block dispatch the lowest unfinished rows are resident on every rank and
progress; the wide box is sized from the deployment's largest rows (no static cap).
"""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
import torch.distributed._symmetric_memory as symm_mem
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass.cute.runtime import make_fake_stream

from ..common.embedding import compile_ring_gather
from ..common.host import (
    architecture,
    fake,
    prepare_shared_carveout,
    stream,
    to_cute,
)
from ..common.mailbox import SENTINEL, poll, rearm
from ..common.primitives import (
    alloc,
    asm,
    at,
    bf16,
    bf16x2,
    fma,
    launch,
    load_words,
    multicast_words,
    no_neg_zero,
    published,
    push_float,
    store_words,
    wait,
    warp_sum,
)
from ..common.twoshot import TwoShot, push
from ..common.twoshot import reduce as two_shot_reduce
from ..mla.qvalid import copy_valid, expect_valid, init_valid, wait_valid
from .context import div_rn, rsqrt_approx

HIDDEN = 7168
CAP = 64  # one-shot rows per step (B8: 56 query rows)
TWO_CAP = 64  # two-shot rows per step
# Wide (``ArNormRows``) rows per step: the deployment's largest (``PreparedDraftAr``
# sizes the wide box in 64-row tiles; hb192 measured to 576).
WIDE_CAP = None
CLUSTER = 8  # CTAs per row
COLUMNS = HIDDEN // CLUSTER  # 896: 112 threads x 8 columns
# Benchmarks only: one-shot sites write CTA (0, 0)'s %globaltimer at entry, after the
# PDL wait, after the poll and at the end into PreparedDraftAr.stamps (4 per site).
STAMPS = False


@asm(Int64)
def globaltimer():
    return "mov.u64 $0, %globaltimer;", "=l,~{memory}", ()


@cute.jit
def wait_published(pointer):
    """Spin until the 16-byte fragment at ``pointer`` holds no sentinel word."""
    words = load_words(pointer, count=4, volatile=True)
    while not published(words):
        words = load_words(pointer, count=4, volatile=True)


class ArNorm:
    """``sources``: TP (all-reduce of every rank's partial, published here) or 1 (the
    embedding ring: one owner-published row, no residual)."""

    def __init__(
        self,
        rows,
        sources,
        tp,
        polled=False,
        early=False,
        trigger=False,
        stamps=False,
        gate=False,
        valid=False,
    ):
        self.rows, self.sources, self.tp = rows, sources, tp
        self.stamps = stamps
        self.reduce = sources > 1
        self.polled = polled and self.reduce
        # ``valid`` (Rubin): the CTA's slices of the slots (TP, or the embedding ring's
        # one) land in shared memory through validity-reporting bulk copies
        # (mla.qvalid; every writer of the box publishes one 16 B store per chunk,
        # never a 0x80000000 word: ``latent``, no_neg_zero), re-issued by thread 0
        # until no copied chunk is armed, instead of every thread's LSU polls; the
        # same words and sums. Every site of a valid batch reads the box this way.
        self.valid = valid
        self.gate = gate and sources > 1 and not self.valid
        self.early = early and self.polled
        # ``trigger``: release the PDL dependents right after the wait (their weight
        # loads then overlap this site; they read its outputs after their own wait).
        self.trigger = trigger
        self.settle = False  # a CTA barrier between the polls and the norm (two-shot)

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, (CLUSTER, self.rows, 1), 128, cluster=CLUSTER)

    @cute.kernel
    def kernel(self, args):
        partial, residual, gamma, eps, norm_out, box, mc, parity, rank = args[:9]
        part, row, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        self.stamp(args, 0)
        sums, peers, bar, stage, vbar = self.setup()
        live = tid < COLUMNS // 8
        col = part * COLUMNS + min(tid, COLUMNS // 8 - 1) * 8
        slot = HIDDEN * CAP  # elements per source slot
        local = box.iterator + (parity * self.tp * CAP + row) * HIDDEN + col
        if cutlass.const_expr(not self.early):
            cute.arch.griddepcontrol_wait()
        self.stamp(args, 1)
        if cutlass.const_expr(self.trigger):
            cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(self.reduce and not self.polled):  # noqa: SIM102
            if live:
                words = load_words(
                    partial.iterator + row * HIDDEN + col, count=4, mutable=True
                )
                words = tuple(no_neg_zero(w) for w in words)
                offset = ((parity * self.tp + rank) * CAP + row) * HIDDEN + col
                multicast_words(mc + Int64(offset) * 2, words, kind="f32")
        # The norm weights and (after the PDL wait) the residual load while the slots
        # are in flight: their round trips leave the critical path.
        weights = load_words(gamma.iterator + col, count=4)
        res = (Uint32(0),) * 4
        if cutlass.const_expr(self.reduce):  # (deferred: older than the producer)
            res = load_words(
                residual.iterator + row * HIDDEN + col, count=4, mutable=True
            )
        if cutlass.const_expr(self.gate):
            if tid < self.sources:
                last = (parity * self.tp * CAP + row) * HIDDEN + part * COLUMNS
                wait_published(box.iterator + last + COLUMNS - 8 + tid * slot)
            cute.arch.sync_threads()
        if cutlass.const_expr(self.valid):
            if tid < 32:
                first = (parity * self.tp * CAP + row) * HIDDEN + part * COLUMNS
                self.land(box.iterator + first, stage, vbar, tid)
            cute.arch.sync_threads()
        # Only the owning threads poll: a slot is re-armed once its owner read it.
        got = cute.make_rmem_tensor((self.sources, 4), Uint32)
        got.fill(0)
        x = [Float32(0)] * 8
        total = Float32(0)
        if live:
            if cutlass.const_expr(self.valid):
                polled = self.staged(stage, tid)
            else:
                polled = poll(local, slot, self.sources)
            self.stamp(args, 2)
            for s in cutlass.range_constexpr(self.sources):
                rearm(local + s * slot)
                for k in cutlass.range_constexpr(4):
                    got[s, k] = polled[s, k]
            acc = [Float32(0)] * 8
            for s in cutlass.range_constexpr(self.sources):
                for k in cutlass.range_constexpr(4):
                    for h in (False, True):
                        e = 2 * k + int(h)
                        value = bf16(got[s, k], h)
                        acc[e] = value if s == 0 else acc[e] + value
            if cutlass.const_expr(self.reduce):
                words = []
                for k in cutlass.range_constexpr(4):
                    pair = bf16x2(acc[2 * k], acc[2 * k + 1])  # the all-reduce output
                    lo = bf16(pair, False) + bf16(res[k], False)
                    hi = bf16(pair, True) + bf16(res[k], True)
                    words.append(bf16x2(lo, hi))
                target = residual if self.polled else partial  # the new residual
                store_words(target.iterator + row * HIDDEN + col, tuple(words))
            else:
                words = [bf16x2(acc[2 * k], acc[2 * k + 1]) for k in range(4)]
                store_words(residual.iterator + row * HIDDEN + col, tuple(words))
            for k in cutlass.range_constexpr(4):
                for h in (False, True):
                    x[2 * k + int(h)] = bf16(words[k], h)
            for e in cutlass.range_constexpr(8):
                total = fma(x[e], x[e], total)
        self.finish(
            x, total, live, col, row, part, weights, eps, norm_out, sums, peers, bar
        )
        self.stamp(args, 3)
        if cutlass.const_expr(self.early):
            cute.arch.griddepcontrol_wait()

    @cute.jit
    def stamp(self, args, slot: cutlass.Constexpr[int]):
        """Benchmarks (``stamps``): CTA (0, 0) thread 0's %globaltimer into slot
        ``slot`` of the site's 4 stamps (args[9]: their address)."""
        if cutlass.const_expr(self.stamps):
            part, row, _ = cute.arch.block_idx()
            tid, _, _ = cute.arch.thread_idx()
            if (part == 0) & (row == 0) & (tid == 0):
                address = args[9] + slot * 8
                value = globaltimer()
                store_words(
                    address,
                    (Uint32(value & Int64(0xFFFFFFFF)), Uint32(value >> 32)),
                )

    @cute.jit
    def setup(self):
        """(sums, peers, bar, stage, vbar) shared memory; arms the cluster
        sum-of-squares barrier (``valid``: the TP slots' stage and its layout::v1
        barrier, else None)."""
        tid, _, _ = cute.arch.thread_idx()
        smem = cutlass.utils.SmemAllocator()
        sums = alloc(smem, Float32, 4)
        peers = alloc(smem, Float32, CLUSTER)
        bar = smem.allocate(Uint64, 8)
        stage, vbar = None, None
        if cutlass.const_expr(self.valid):
            stage = alloc(smem, Uint32, self.sources * COLUMNS // 2, 128)
            vbar = smem.allocate(Uint64, 8)
        if tid == 0:
            cute.arch.mbarrier_init(bar, 1)
            if cutlass.const_expr(self.valid):
                init_valid(vbar)
            cute.arch.mbarrier_init_fence()
            cute.arch.mbarrier_arrive_and_expect_tx(bar, CLUSTER * 4)
        if cutlass.const_expr(self.valid):
            cute.arch.sync_warp()  # warp 0 waits on vbar
        cute.arch.cluster_arrive_relaxed()
        return sums, peers, bar, stage, vbar

    @cute.jit
    def land(self, first, stage, vbar, tid):
        """Warp 0 (``valid``): rounds of thread 0's bulk validity copies of the CTA's
        COLUMNS of every TP slot (``first``: slot 0's) into ``stage`` until no copied
        16 B chunk was armed; lane 0's report decides each round for the warp."""
        nbytes: cutlass.Constexpr = COLUMNS * 2
        phase, armed = Int32(0), Int32(1)
        while armed != 0:
            if tid == 0:
                expect_valid(vbar, self.sources * nbytes)
                for s in cutlass.range_constexpr(self.sources):
                    source = first + s * HIDDEN * CAP
                    copy_valid(
                        stage.iterator + s * (COLUMNS // 2), source, nbytes, vbar
                    )
            armed = cute.arch.shuffle_sync(wait_valid(vbar, phase), 0)
            phase = phase ^ 1

    @cute.jit
    def staged(self, stage, tid):
        """This thread's 16 B of every TP slot from the landed stage, as ``poll``'s
        [sources, 4] words."""
        got = cute.make_rmem_tensor((self.sources, 4), Uint32)
        for s in cutlass.range_constexpr(self.sources):
            pointer = stage.iterator + s * (COLUMNS // 2) + 4 * tid
            words = load_words(pointer, count=4, shared=True)
            for k in cutlass.range_constexpr(4):
                got[s, k] = words[k]
        return got

    @cute.jit
    def finish(self, x, total, live, col, row, part, weights, eps, norm_out, sums,
               peers, bar):  # fmt: skip
        """The row's sum of squares (warp, CTA, DSMEM cluster), then this thread's 8
        normalized columns (rsqrtf as FlashInfer's fusion; x * r * w rounded once);
        ``weights``: this thread's 8 norm weights (BF16 words)."""
        tid, _, _ = cute.arch.thread_idx()
        lane, warp = tid % 32, tid // 32
        if cutlass.const_expr(self.settle):
            # The warps whose rows landed early do not shuffle while siblings poll.
            cute.arch.sync_threads()
        total = warp_sum(total)
        if lane == 0:
            sums[warp] = total
        cute.arch.sync_threads()
        cute.arch.cluster_wait()  # every CTA's barrier is initialized
        if tid < CLUSTER:
            mine = (sums[0] + sums[1]) + (sums[2] + sums[3])
            push_float(peers.iterator + part, mine, bar, tid)
        wait(bar)
        variance = Float32(0)
        for c in cutlass.range_constexpr(CLUSTER):
            variance += peers[c]
        scale = rsqrt_approx(div_rn(variance, Float32(HIDDEN)) + eps)
        if live:
            normed = []
            for k in cutlass.range_constexpr(4):
                pair = []
                for h in (False, True):
                    pair.append(x[2 * k + int(h)] * scale * bf16(weights[k], h))
                normed.append(bf16x2(pair[0], pair[1]))
            store_words(norm_out.iterator + row * HIDDEN + col, tuple(normed))
        if cutlass.const_expr(not self.trigger):
            cute.arch.griddepcontrol_launch_dependents()
        # No CTA exits before every DSMEM push into the cluster has landed.
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()


class ArNormTwoShot(ArNorm):
    """``ArNorm`` over the two-shot exchange (common.twoshot): ``sources`` TP (push the
    partial chunks to their owners, the owned chunks reduced into every gather row)
    or 1 (the embedding ring's rows already in the gather row). ``polled``: the
    projection before the site pushed its partial chunks itself (no partial read);
    the new residual is written in place. ``trigger`` as ``ArNorm``'s. The CTA's
    gather-row pollers start only once its owners have multicast (the gather row
    cannot land earlier, and pollers spinning while the reduce-scatter lands slow the
    landing), and its warps settle at a barrier before the norm's shuffles."""

    def __init__(self, rows, sources, tp, polled=False, trigger=False):
        super().__init__(rows, sources, tp, polled, trigger=trigger)
        self.settle = True
        self.owned = HIDDEN // tp
        self.owned_chunks = self.owned // 8
        self.share = -(-self.owned_chunks // CLUSTER)  # owned chunks per CTA
        assert self.share <= 128

    @cute.kernel
    def kernel(self, args):
        partial, residual, gamma, eps, norm_out, recv, peers, gather, mc, rank = args
        part, row, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        sums, cluster, bar, _, _ = self.setup()
        live = tid < COLUMNS // 8
        col = part * COLUMNS + min(tid, COLUMNS // 8 - 1) * 8
        cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(self.trigger):
            cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(self.reduce):
            if cutlass.const_expr(not self.polled):  # noqa: SIM102
                if live:
                    words = load_words(
                        partial.iterator + row * HIDDEN + col, count=4, mutable=True
                    )
                    words = tuple(no_neg_zero(w) for w in words)
                    push(recv, peers, rank, row, col, words)
            if tid < self.share:
                chunk = part * self.share + tid
                if chunk < self.owned_chunks:
                    column = rank * self.owned + chunk * 8
                    two_shot_reduce(recv, gather, mc, rank, row, column, batched=True)
            cute.arch.sync_threads()  # the gather row's pollers after the owners
        # The norm weights and the residual load while the gather row is in flight.
        weights = load_words(gamma.iterator + col, count=4)
        res = (Uint32(0),) * 4
        if cutlass.const_expr(self.reduce):
            res = load_words(
                residual.iterator + row * HIDDEN + col, count=4, mutable=True
            )
        x = [Float32(0)] * 8
        total = Float32(0)
        if live:
            source = at(gather, 0, row, col)
            polled = poll(source, 0, 1)
            rearm(source)
            if cutlass.const_expr(self.reduce):
                words = []
                for k in cutlass.range_constexpr(4):
                    lo = bf16(polled[0, k], False) + bf16(res[k], False)
                    hi = bf16(polled[0, k], True) + bf16(res[k], True)
                    words.append(bf16x2(lo, hi))
                target = residual if self.polled else partial  # the new residual
                store_words(target.iterator + row * HIDDEN + col, tuple(words))
            else:
                words = [polled[0, k] for k in range(4)]
                store_words(residual.iterator + row * HIDDEN + col, tuple(words))
            for k in cutlass.range_constexpr(4):
                for h in (False, True):
                    x[2 * k + int(h)] = bf16(words[k], h)
            for e in cutlass.range_constexpr(8):
                total = fma(x[e], x[e], total)
        self.finish(
            x, total, live, col, row, part, weights, eps, norm_out, sums, cluster, bar
        )


@cache
def compile_two_shot(rows, sources, tp, device, polled=False, trigger=False):
    args = (
        fake((rows, HIDDEN)),
        fake((rows, HIDDEN)),
        fake((HIDDEN,)),
        Float32(0),
        fake((rows, HIDDEN)),
        fake((tp, TWO_CAP, HIDDEN // tp)),
        fake((tp,), None, torch.int64, 8),
        fake((1, TWO_CAP, HIDDEN)),
        Int64(0),
        Int32(0),
    )
    with torch.accelerator.device_index(device):
        kernel = ArNormTwoShot(rows, sources, tp, polled, trigger)
        compiled = cute.compile(kernel, *args, make_fake_stream())
        if trigger:
            # The released dependents (tensor-memory GEMMs of ~200 KB shared memory)
            # place beside the site's CTAs only on SMs configured for the largest
            # shared-memory carve-out.
            compiled = prepare_shared_carveout(compiled, device, 100)
    return compiled


class ArNormRows:
    """``rows`` > 64: one CTA (CLUSTER groups of 128 threads, group g on ``ArNorm``'s
    CTA g's columns) per row over the gather row of ``gather`` [1, cap, HIDDEN].
    ``push`` (sources TP; arguments as ``ArNormTwoShot``'s): the CTA reads the stored
    partial row, pushes its chunks to their owners, reduces this rank's owned chunks
    of the row (``common.twoshot.reduce``) and writes the new residual in place of the
    partial. Without ``push``: sources TP (the gather row was reduced already: the
    residual is added and written in place) or 1 (the embedding ring's rows: the
    gathered row is the residual). ``trigger``: release the PDL dependents right after
    the wait (as ``ArNorm``'s). Re-arms the gather row it read."""

    def __init__(self, rows, sources, tp, trigger=False, push=False):
        self.rows, self.sources, self.tp, self.trigger = rows, sources, tp, trigger
        self.reduce = sources > 1
        self.push = push and self.reduce
        self.owned_chunks = HIDDEN // tp // 8

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.rows, CLUSTER * 128)

    @cute.kernel
    def kernel(self, args):
        if cutlass.const_expr(self.push):
            partial, residual, gamma, eps, norm_out = args[:5]
            recv, peers, gather, mc, rank = args[5:]
        else:
            residual, gamma, eps, norm_out, gather = args
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        group, local = tid // 128, tid % 128
        lane, warp = tid % 32, tid // 32
        smem = cutlass.utils.SmemAllocator()
        sums = alloc(smem, Float32, 4 * CLUSTER)
        live = local < COLUMNS // 8
        col = group * COLUMNS + min(local, COLUMNS // 8 - 1) * 8
        weights = load_words(gamma.iterator + col, count=4)  # static: before the wait
        cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(self.trigger):
            cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(self.push):
            if live:
                words = load_words(
                    partial.iterator + row * HIDDEN + col, count=4, mutable=True
                )
                words = tuple(no_neg_zero(w) for w in words)
                push(recv, peers, rank, row, col, words)
            if tid < self.owned_chunks:
                column = rank * (HIDDEN // self.tp) + tid * 8
                two_shot_reduce(recv, gather, mc, rank, row, column, batched=True)
            cute.arch.sync_threads()  # the gather row's pollers after the owners
        res = (Uint32(0),) * 4
        if cutlass.const_expr(self.reduce):
            res = load_words(
                residual.iterator + row * HIDDEN + col, count=4, mutable=True
            )
        x = [Float32(0)] * 8
        total = Float32(0)
        if live:
            source = at(gather, 0, row, col)
            polled = poll(source, 0, 1)
            rearm(source)
            if cutlass.const_expr(self.reduce):
                words = []
                for k in cutlass.range_constexpr(4):
                    lo = bf16(polled[0, k], False) + bf16(res[k], False)
                    hi = bf16(polled[0, k], True) + bf16(res[k], True)
                    words.append(bf16x2(lo, hi))
            else:
                words = [polled[0, k] for k in range(4)]
            target = partial if self.push else residual  # the new residual
            store_words(target.iterator + row * HIDDEN + col, tuple(words))
            for k in cutlass.range_constexpr(4):
                for h in (False, True):
                    x[2 * k + int(h)] = bf16(words[k], h)
            for e in cutlass.range_constexpr(8):
                total = fma(x[e], x[e], total)
        # As ``ArNorm.finish``: the warps settle, warp sums, each group's (CTA's) sum
        # of its four warps, then the groups in order (the cluster's DSMEM sum).
        cute.arch.sync_threads()
        total = warp_sum(total)
        if lane == 0:
            sums[warp] = total
        cute.arch.sync_threads()
        variance = Float32(0)
        for g in cutlass.range_constexpr(CLUSTER):
            mine = (sums[4 * g] + sums[4 * g + 1]) + (sums[4 * g + 2] + sums[4 * g + 3])
            variance += mine
        scale = rsqrt_approx(div_rn(variance, Float32(HIDDEN)) + eps)
        if live:
            normed = []
            for k in cutlass.range_constexpr(4):
                pair = []
                for h in (False, True):
                    pair.append(x[2 * k + int(h)] * scale * bf16(weights[k], h))
                normed.append(bf16x2(pair[0], pair[1]))
            store_words(norm_out.iterator + row * HIDDEN + col, tuple(normed))
        if cutlass.const_expr(not self.trigger):
            cute.arch.griddepcontrol_launch_dependents()


@cache
def compile_rows_site(rows, sources, tp, cap, device, trigger=False, push=False):
    """``ArNormRows`` over a [1, ``cap``, HIDDEN] gather view (``push``: and the
    [TP, cap, HIDDEN / TP] receive slots, as ``compile_two_shot``)."""
    args = (
        fake((rows, HIDDEN)),
        fake((HIDDEN,)),
        Float32(0),
        fake((rows, HIDDEN)),
        fake((1, cap, HIDDEN)),
    )
    if push:
        args = (
            fake((rows, HIDDEN)),
            fake((rows, HIDDEN)),
            fake((HIDDEN,)),
            Float32(0),
            fake((rows, HIDDEN)),
            fake((tp, cap, HIDDEN // tp)),
            fake((tp,), None, torch.int64, 8),
            fake((1, cap, HIDDEN)),
            Int64(0),
            Int32(0),
        )
    with torch.accelerator.device_index(device):
        kernel = ArNormRows(rows, sources, tp, trigger, push)
        compiled = cute.compile(kernel, *args, make_fake_stream())
        if trigger:  # as ``compile_two_shot``: the released GEMMs place beside it
            compiled = prepare_shared_carveout(compiled, device, 100)
    return compiled


@cache
def compile_ar_norm(
    rows,
    sources,
    tp,
    device,
    polled=False,
    early=False,
    trigger=False,
    stamps=False,
    gate=False,
    valid=False,
):
    args = (
        fake((rows, HIDDEN)),
        fake((rows, HIDDEN)),
        fake((HIDDEN,)),
        Float32(0),
        fake((rows, HIDDEN)),
        fake((2 * tp * CAP * HIDDEN,), None, torch.bfloat16),
        Int64(0),
        Int32(0),
        Int32(0),
    )
    if stamps:
        args += (Int64(0),)
    with torch.accelerator.device_index(device):
        kernel = ArNorm(rows, sources, tp, polled, early, trigger, stamps, gate, valid)
        return cute.compile(kernel, *args, make_fake_stream())


def plain_layer(layer):
    """No output gate and no fused GEMM + reduce: the layer's o_proj / down_proj
    partial rows can be published by producers."""
    attention, mlp = layer.self_attn, layer.mlp
    return not (
        getattr(attention, "g_proj", None) is not None
        or getattr(attention, "use_output_gate", False)
        or getattr(attention, "gemm_rs_ar", None) is not None
        or getattr(mlp, "gemm_rs_ar", None) is not None
        or getattr(mlp, "shard_sequence_parallel", False)
    )


class PreparedDraftAr:
    """The drafter forward (at most 64 rows) with its all-reduce + norm sites on the
    Lamport kernels (one-shot, or two-shot for ``two_shot`` rows); ``run`` returns None
    (nothing launched) when not covered. With ``prepare_producers`` the o_proj /
    down_proj GEMMs of one-shot rows publish their partials into the sites' slots
    (``polled`` sites). ``early``: two-shot rows whose sites release their PDL
    dependents right after their wait (``trigger``). ``valid``: one-shot rows whose
    polled sites land the slots with validity-reporting copies (Rubin). ``wide``:
    rows above 64 (at most WIDE_CAP when set) on ``ArNormRows`` sites (push) over
    their own two-shot box."""

    def __init__(
        self, group, device, tp, rank, rows, vocab, two_shot=(), gated=(), early=(),
        valid=(), wide=(),
    ):  # fmt: skip
        self.tp, self.rank, self.device = tp, rank, device
        cap = WIDE_CAP or max(rows, default=0)
        self.wide_rows = frozenset(r for r in rows if r in wide and 64 < r <= cap)
        self.rows = tuple(
            r for r in rows if r <= min(CAP, TWO_CAP) or r in self.wide_rows
        )
        self.two_rows = frozenset(r for r in self.rows if r in two_shot)
        self.gated = frozenset(r for r in self.rows if r in gated)  # one-shot
        self.early_two = frozenset(r for r in self.two_rows if r in early)
        rubin = torch.cuda.get_device_capability(device) == (10, 7)
        self.valid_ar = frozenset(
            r for r in self.rows if rubin and r in valid and r not in self.two_rows
        )
        index = device.index
        self.reduce, self.embed = {}, {}
        self.box, self.mc, self.view = None, 0, None
        if any(r not in self.two_rows for r in self.rows):  # one-shot sites
            self.box = symm_mem.empty(
                (2 * tp * CAP * HIDDEN,), dtype=torch.bfloat16, device=device
            )
            handle = symm_mem.rendezvous(self.box, group)
            self.mc = int(handle.multicast_ptr or 0)
            if not self.mc:
                raise RuntimeError(
                    "Kimi K3 DSpark drafter all-reduce needs multicast memory"
                )
            self.box.view(torch.int32).fill_(SENTINEL - (1 << 32))
            torch.accelerator.synchronize(device)
            torch.distributed.barrier(group=group)
            self.view = to_cute(self.box)
        # Wide rows: their own two-shot box (rows up to the largest, in whole 64-row
        # tiles), the sites' gather views per parity.
        self.wide_box, self.wide_cap = None, 0
        if self.wide_rows:
            self.wide_cap = -(-max(self.wide_rows) // 64) * 64
            self.wide_box = TwoShot(group, device, tp, rank, HIDDEN, self.wide_cap, 2)
        self.two, self.two_views = None, []
        if self.two_rows:  # two-shot sites
            self.two = TwoShot(group, device, tp, rank, HIDDEN, TWO_CAP, 2)
            for parity in (0, 1):
                recv, peers, gather, mc = self.two.views(parity)
                views = to_cute(recv), to_cute(peers, 8), to_cute(gather), Int64(mc)
                self.two_views.append(views)
        for r in self.rows:
            if r in self.wide_rows:
                wide_cap, trigger = self.wide_cap, r in early
                self.reduce[r] = compile_rows_site(
                    r, tp, tp, wide_cap, index, trigger, push=True
                )
                self.embed[r] = compile_rows_site(r, 1, tp, wide_cap, index, trigger)
            elif r in self.two_rows:
                trigger = r in self.early_two
                self.reduce[r] = compile_two_shot(r, tp, tp, index, trigger=trigger)
                self.embed[r] = compile_two_shot(r, 1, tp, index, trigger=trigger)
            else:
                options = dict(stamps=STAMPS, gate=r in self.gated)
                options.update(valid=r in self.valid_ar)
                self.reduce[r] = compile_ar_norm(r, tp, tp, index, **options)
                self.embed[r] = compile_ar_norm(r, 1, tp, index, **options)
        # Benchmarks (STAMPS): 4 %globaltimer stamps per one-shot site of a forward.
        self.stamps = None
        self.stamp_index = 0
        if STAMPS:
            self.stamps = torch.zeros((64, 4), dtype=torch.int64, device=device)
        self.shard, self.vocab = vocab // tp, vocab
        arch = architecture(device)
        with torch.accelerator.device_index(index):
            self.ring = compile_ring_gather(torch.int32, self.shard, arch)
        # Per layer (o_proj, down_proj) RingProjections, the gate/up + silu_and_mul
        # GEMMs (None: the stock gate_up_proj + act_fn) and the polled sites' kernels
        # by rows (``prepare_producers``).
        self.producers: list[tuple] = []
        self.two_producers: list[tuple] = []  # ``set_two_shot_producers``
        self.two_gate_up: list = []  # ``prepare_two_gate_up``
        self.gate_up: list = []
        self.polled: dict[int, object] = {}
        self.polled_embed: dict[int, object] = {}
        self.shape_rows = torch.empty(
            (CAP, HIDDEN), dtype=torch.bfloat16, device=device
        )

    def prepare_producers(self, layers, tactics, early=(), trigger=False):
        """Ring o_proj / down_proj of every layer for ``tactics`` {rows: {(N, K):
        split-K tactic}}; ``early``: the rows (or True: all) whose sites are
        deferred (their rings trigger after their wait); rows without both tactics
        (or layers with an output gate or a fused GEMM + reduce) keep the stored
        partials. A gate_up_proj tactic (M tile 64) for every such rows runs the
        gate/up GEMM with silu_and_mul in its epilogue (gemm.splitk_situ, SiLU-and-mul
        MLPs only)."""
        from vllm.model_executor.layers.activation import SiluAndMul

        from ..gemm.splitk_ring import RingProjection
        from ..gemm.splitk_situ import PreparedSitu

        if not all(plain_layer(layer) for layer in layers):
            return
        if isinstance(early, bool):  # every row (True) or none
            early = set(self.rows) if early else set()
        rows = {}
        for r in self.rows:
            if r in self.two_rows:  # ring partials go to one-shot slots
                continue
            table = tactics.get(r, {})
            shapes = [tuple(layers[0].self_attn.o_proj.weight.shape)]
            shapes.append(tuple(layers[0].mlp.down_proj.weight.shape))
            if all(shape in table for shape in shapes):
                rows[r] = tuple(table[shape] for shape in shapes)
                if r in early:  # deferred sites: the rings release them after the wait
                    rows[r] = tuple((*t[:4], 1) for t in rows[r])
        if not rows:
            return
        for layer in layers:
            weights = layer.self_attn.o_proj.weight, layer.mlp.down_proj.weight
            self.producers.append(
                tuple(
                    RingProjection(w, {r: t[i] for r, t in rows.items()}).prepare()
                    for i, w in enumerate(weights)
                )
            )
        gate_up = tuple(layers[0].mlp.gate_up_proj.weight.shape)
        fused = {r: tactics[r][gate_up] for r in rows if gate_up in tactics[r]}
        acts = (getattr(layer.mlp, "act_fn", None) for layer in layers)
        silu = all(isinstance(act, SiluAndMul) for act in acts)
        for layer in layers:
            weight = layer.mlp.gate_up_proj.weight
            ok = silu and len(fused) == len(rows) and weight.shape[0] % 64 == 0
            self.gate_up.append(
                PreparedSitu(weight, fused, 1.0, 0.0, silu=True) if ok else None
            )
        index = self.device.index
        for r in rows:
            gate = r in self.gated
            self.polled[r] = compile_ar_norm(
                r, self.tp, self.tp, index, True, r in early, trigger, STAMPS, gate,
                r in self.valid_ar,
            )  # fmt: skip
            self.polled_embed[r] = compile_ar_norm(
                r, 1, self.tp, index, False, False, trigger, STAMPS, False,
                r in self.valid_ar,
            )  # fmt: skip

    def set_two_shot_producers(self, producers, rows):
        """Per layer (o_proj, down_proj) producers of two-shot ``rows`` that push their
        partial row chunks to the column owners themselves
        (``run_two_shot(x, recv, peers, slot)``, common.twoshot.push): those rows'
        sites become ``polled``."""
        self.two_producers = list(producers)
        for r in rows:
            assert r in self.two_rows
            self.polled[r] = compile_two_shot(
                r, self.tp, self.tp, self.device.index, True, r in self.early_two
            )

    def prepare_two_gate_up(self, layers, tactics):
        """The two-shot rows' gate/up GEMMs with silu_and_mul in their epilogue for
        ``tactics`` {rows: tactic}: split-K M tile 64 (gemm.splitk_situ.SplitKSiluMul)
        or tensor-memory (128, 64, ...) (gemm.splitk_situ64): bitwise the gate/up
        projection with the same tactic followed by silu_and_mul. SiLU-and-mul MLPs
        whose two-shot sites are polled only."""
        from vllm.model_executor.layers.activation import SiluAndMul

        from ..gemm.splitk_situ import PreparedSitu

        rows = {
            r: t
            for r, t in tactics.items()
            if r in self.polled and (t[0] == 64 or (len(t) > 4 and t[1] == 64))
        }
        acts = (getattr(layer.mlp, "act_fn", None) for layer in layers)
        if not rows or not all(isinstance(act, SiluAndMul) for act in acts):
            return
        self.two_gate_up = [
            PreparedSitu(layer.mlp.gate_up_proj.weight, rows, 1.0, 0.0, silu=True)
            for layer in layers
        ]

    def slot_address(self, parity, rank=0):
        """Multicast address of one-shot slot ``rank`` of ``parity``."""
        return self.mc + ((parity * self.tp + rank) * CAP) * HIDDEN * 2

    def embed_address(self, rows):
        """Multicast address of the rows the embedding ring fills for ``rows``: slot
        0 of parity 0 (one-shot) or the parity-0 gather row (two-shot)."""
        if rows in self.wide_rows:
            return int(self.wide_box.views(0)[3])
        if rows in self.two_rows:
            return int(self.two_views[0][3])
        return self.slot_address(0)

    def site(self, kernel, partial, residual, norm, out, parity):
        rows = partial.shape[0]
        if rows in self.wide_rows:
            recv, peers, gather, mc = self.wide_box.views(parity)
            weight, eps = to_cute(norm.weight), Float32(norm.variance_epsilon)
            if kernel is self.embed[rows]:  # the embedding ring filled the gather row
                args = (to_cute(residual), weight, eps, to_cute(out), to_cute(gather))
            else:  # push: the stored partial row's chunks, the owners' reduce
                args = (to_cute(partial), to_cute(residual), weight, eps, to_cute(out))
                args += (to_cute(recv), to_cute(peers, 8), to_cute(gather), Int64(mc))
                args += (Int32(self.rank),)
            kernel(*args, stream(self.device))
            return
        args = (to_cute(partial), to_cute(residual), to_cute(norm.weight))
        args += (Float32(norm.variance_epsilon), to_cute(out))
        if partial.shape[0] in self.two_rows:
            args += (*self.two_views[parity], Int32(self.rank))
        else:
            args += (self.view, Int64(self.mc), Int32(parity), Int32(self.rank))
            if self.stamps is not None:
                address = self.stamps.data_ptr() + (self.stamp_index % 64) * 32
                args += (Int64(address),)
                self.stamp_index += 1
        kernel(*args, stream(self.device))

    def publish(self, producer, x, parity):
        """``producer``'s partial rows of ``x`` into this rank's slot of ``parity``."""
        rows = x.shape[0]
        if rows in self.two_rows:
            recv, peers = self.two.views(parity)[:2]
            producer.run_two_shot(x, recv, peers, self.rank)
            return
        address = self.slot_address(parity, self.rank)
        producer.run_ring(x, self.shape_rows[:rows], address)

    def run(self, model, input_ids, positions):
        rows = input_ids.shape[0]
        self.stamp_index = 0
        if (
            rows not in self.reduce
            or not torch.cuda.is_current_stream_capturing()
            or input_ids.dtype != torch.int32
            or not input_ids.is_contiguous()
            or model.embed_tokens is None
        ):
            return None
        weight = model.embed_tokens.weight
        if weight.shape[0] != self.shard or not weight.is_contiguous():
            return None
        empty = lambda: torch.empty(  # noqa: E731
            (rows, HIDDEN), dtype=weight.dtype, device=weight.device
        )
        # Site 0: the embedding rows from their owners (parity 0), normed.
        address = self.embed_address(rows)
        self.ring(input_ids, weight, self.rank * self.shard, self.vocab, address)
        residual, hidden = empty(), empty()
        layers = model.layers
        embed = self.polled_embed.get(rows, self.embed[rows])
        self.site(embed, residual, residual, layers[0].input_layernorm, hidden, 0)
        if rows in self.polled:
            return self.run_polled(model, rows, residual, hidden, positions, empty)
        site = 1
        for index, layer in enumerate(layers):
            if index > 0:
                normed = empty()
                self.site(
                    self.reduce[rows],
                    hidden,
                    residual,
                    layer.input_layernorm,
                    normed,
                    site % 2,
                )
                residual, hidden, site = hidden, normed, site + 1
            attn = layer.self_attn(positions=positions, hidden_states=hidden)
            normed = empty()
            self.site(
                self.reduce[rows],
                attn,
                residual,
                layer.post_attention_layernorm,
                normed,
                site % 2,
            )
            residual, site = attn, site + 1
            hidden = layer.mlp(normed)
        out = empty()
        self.site(self.reduce[rows], hidden, residual, model.final_norm, out, site % 2)
        return out

    def run_polled(self, model, rows, residual, hidden, positions, empty):
        """The layers with ring (or two-shot pushing) o_proj / down_proj partials and
        polled sites; the residual is updated in place."""
        polled, site = self.polled[rows], 1
        two = rows in self.two_rows
        for index, layer in enumerate(model.layers):
            o_proj, down = (self.two_producers if two else self.producers)[index]
            if index > 0:  # the previous layer's down partials are in this site's slots
                hidden = empty()
                norm = layer.input_layernorm
                self.site(polled, residual, residual, norm, hidden, site % 2)
                site += 1
            attn_out, gate = layer.self_attn._forward_q_lora(positions, hidden)
            assert gate is None
            self.publish(o_proj, attn_out, site % 2)
            normed = empty()
            norm = layer.post_attention_layernorm
            self.site(polled, residual, residual, norm, normed, site % 2)
            site += 1
            mlp, gate_up = layer.mlp, None
            if two and self.two_gate_up and rows in self.two_gate_up[index].policies:
                gate_up = self.two_gate_up[index]
            elif not two:
                gate_up = self.gate_up[index]
            if gate_up is not None:
                act = gate_up.run(normed, normed.new_empty((rows, gate_up.half)))
            else:
                act = mlp.act_fn(mlp.gate_up_proj(normed)[0])
            self.publish(down, act, site % 2)
        out = empty()
        self.site(polled, residual, residual, model.final_norm, out, site % 2)
        return out
