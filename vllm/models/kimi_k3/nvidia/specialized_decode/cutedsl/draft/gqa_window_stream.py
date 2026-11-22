# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The release DSpark drafter's sliding-window GQA attention as a persistent stream
over every SM (``GqaWindowStream``): ``GqaWindowUmma``'s per-stage computation (module
notes of ``gqa_window_umma``: Q replicated into the four TMEM lane quarters, warp w
owning keys 16 w .. 16 w + 15 of each 64-key stage, P from TMEM, 16 KB TMA stages
through a ring), with the work split evenly over a fixed grid instead of one CTA per
(request, KV head).

Work: unit u = (request, KV head) owns ``slots`` virtual stages (the most a full window
spans); the grid's CTAs take equal contiguous ranges of the ``units x slots`` virtual
stage list, so each CTA streams one to ``MAXP`` pieces of consecutive units (a piece's
virtual stages beyond its unit's window are empty). Two CTAs per SM: one wave on every
SM at every drafter batch (``GqaWindowUmma`` launches one CTA per unit: 384 CTAs on 212
SMs at B96, two serial chains of 33 stages; small batches leave most SMs idle).

Per piece: q / k norm + RoPE of the unit's query rows (the Q tile) and, if the piece's
stages hold them, the query rows' K / V insert (cache write + tile fixup, as
``GqaWindowUmma``); the stage loop (the ring runs on across pieces: the producer
issues the next piece's loads as slots free); the four quarters' LSE merge (in the Q
tile, quarter 0 to 3 in order); then either the output rows (the piece is the whole
unit) or the piece's partial (O over the common max, the max, the sum) to a global
buffer and an acq_rel counter add per unit: the unit's last piece merges every partial
in piece order and stores the rows (and resets the counter for the next launch, which
reads it only after its PDL wait).

Rounding as ``GqaWindowUmma`` (S FP32, P = exp2(s c - m) in BF16, O FP32, the lazy
running max); the merge order of quarters and pieces is fixed (deterministic), and the
output is rounded once.

Before the PDL wait the metadata as it stands decides the plan (pieces, their stages
and blocks) and the first ring loads; after the wait it is read again and a moved plan
is drained and reissued (never in a decode step: the metadata is written before the
graph).
"""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_stream

from vllm.models.kimi_k3.nvidia.k31_spin_guard import spin_start, spin_tick

from ..common.host import dynamic_fake, fake
from ..common.primitives import (
    alloc,
    asm,
    bf16,
    bf16x2,
    bits_float,
    fence_tmem,
    fma,
    free_tmem,
    launch,
    load_words,
    mma_bf16_block,
    named_barrier,
    shared_address,
    store_words,
    wait,
)
from ..common.resident import _smid, land, wait_landed
from .gqa_window import (
    HEAD,
    LOG2E,
    NEG_INF,
    arm,
    arrive,
    atomic_or_shared,
    ex2,
    fence_async_shared,
    float_word,
    global_timer,
    init_barrier,
    load_i32,
    load_i64,
    norm_rope8,
    store_shared_u32,
    swz,
    tree,
    vote_any,
)
from .gqa_window_umma import (
    KB,
    KTILE,
    NS,
    O_COL,
    P_COL,
    PRODUCER,
    PV_INSTR,
    S_INSTR,
    SLOT,
    THREADS,
    THRESH,
    TMEM_COLS,
    alloc_tmem,
    desc_sw128,
    fmax,
    ld16,
    ld64,
    mma_pv_block,
    st8w,
    st32w,
    st64,
    store_u64,
    tma_load5,
)

SOFTMAX = 128  # the four softmax warps
MAXP = 3  # pieces per CTA (units * slots / CTAs <= 2 slots-per-unit spans)
OPITCH = 68  # FP32 words per merged row in the Q tile (bank spread)


def smem_bytes(ring, q):
    """The kernel's shared memory (its allocations in order, with their alignment)."""
    up = lambda size, align: -(-size // align) * align  # noqa: E731
    size = up(ring * SLOT, 1024) + 128 * 128
    size = up(size, 16) + q * 256
    size = up(size, 16) + q * 4
    size = up(size, 8) + (ring + 4 * NS + 1) * 8
    return up(size, 16) + 4 * 4


def unit_slots(window, q):
    """Virtual stages per unit: the most 64-key stages a full window spans."""
    return -(-(window + q - 2) // KB) + 1


def grid_ctas(units, slots, sms, cap=0):
    """The grid: ``cap`` CTAs (0: two per SM), at most one per virtual stage."""
    return min(cap or 2 * sms, units * slots)


def piece_bounds(units, slots, ctas):
    """(most pieces of one CTA, most pieces of one unit) of the virtual-stage split."""
    total = units * slots
    cta = lambda x: ((x + 1) * ctas - 1) // total  # noqa: E731
    per_cta = max(
        (((c + 1) * total // ctas - 1) // slots) - (c * total // ctas) // slots + 1
        for c in range(ctas)
    )
    per_unit = max(cta(u * slots + slots - 1) - cta(u * slots) + 1 for u in range(units))
    return per_cta, per_unit


@asm(Uint32)
def counter_add(address, value):
    """The unit counter: release of this CTA's partial (after a CTA barrier) and
    acquire of the earlier pieces' (gpu scope)."""
    code = "atom.add.acq_rel.gpu.global.u32 $0, [$1], $2;"
    return code, "=r,l,r,~{memory}", (Int64(address), Uint32(value))


@asm()
def counter_reset(address):
    return "st.relaxed.gpu.global.u32 [$0], 0;", "l,~{memory}", (Int64(address),)


@asm()
def counter_release(address):
    """A non-merging piece's arrival: release of its partial (after a CTA barrier)."""
    code = "red.release.gpu.global.add.u32 [$0], 1;"
    return code, "l,~{memory}", (Int64(address),)


@asm(Uint32)
def counter_acquire(address):
    return "ld.acquire.gpu.global.u32 $0, [$1];", "=r,l,~{memory}", (Int64(address),)


@asm(Int32)
def shfl_idx(value, lane):
    code = "shfl.sync.idx.b32 $0, $1, $2, 31, 0xffffffff;"
    return code, "=r,r,r", (Int32(value), Int32(lane))


class GqaWindowStream:
    """One drafter layer's fused attention streamed over a fixed grid (module notes):
    ``group`` query heads per KV head, ``kv_heads`` per rank, ``page`` tokens per
    cache block (a multiple of 64, >= 128), ``ring`` stages in flight, ``window``
    keys; ``sms`` the device's SM count (the grid is two CTAs per SM). Arguments as
    ``GqaWindowUmma`` plus the partials buffer and the per-unit counters."""

    def __init__(self, group, kv_heads, q, page, ring, window, sms, v_scale=None,
                 round_norm=False, debug=0, redo=False, cap=0, die_sms=None):  # fmt: skip
        assert page % KB == 0 and page >= 128 and window >= 1 and 1 <= q <= 8
        assert group * q <= 32 and ring >= 3
        self.group, self.kv_heads, self.q, self.page = group, kv_heads, q, page
        self.ring, self.window, self.sms, self.v_scale = ring, window, sms, v_scale
        self.round_norm, self.debug, self.debug_redo = round_norm, debug, redo
        self.cap = cap or 2 * sms
        # die split: a resident grid of two CTAs per SM (common.resident); the CTAs of
        # die d stream the units whose KV head die d homes (runtime head -> die mask)
        self.die_sms = die_sms
        if die_sms is not None:
            assert cap in (0, 1, 2 * sms) and sum(die_sms) == sms, (cap, die_sms, sms)
        # (die split, cap 1: one unit per CTA of its die, the die's other CTAs idle)
        self.unit_per_cta = die_sms is not None and cap == 1
        self.rows = group * q
        self.slots = unit_slots(window, q)
        # blocks one piece's stages touch (held one per producer lane)
        self.pages = -(-(window + q - 1 + 2 * KB) // page) + 1
        assert self.pages <= 32, self.pages
        # (merge in the Q tile: two row regions, then the quarters' max / sum)
        assert 2 * self.rows * OPITCH * 4 + 4 * 32 * 2 * 4 <= 128 * 128
        self.pstride = self.rows * (HEAD + 2)  # FP32 words per partial

    @cute.jit
    def __call__(self, *args):
        out = args[8]
        units = (out.shape[0] // self.q) * self.kv_heads
        ctas = min(units * self.slots, Int32(self.cap))
        if cutlass.const_expr(self.die_sms is not None):
            ctas = Int32(2 * self.sms)
        launch(self.kernel, args, ctas, THREADS, max_number_threads=[THREADS, 1, 1])

    @cute.jit
    def stages(self, length):
        """(window start, first 64-key stage, stage count) of a full window."""
        lo = max(length - self.q - (self.window - 1), Int32(0))
        first = Int32(0)
        count = Int32(0)
        if length > 0:
            first = lo // KB
            count = (length - 1) // KB - first + 1
        return lo, first, count

    @cute.jit
    def plan(self, seq_lens, c, ctas, total, piece, die, heads, mask, lane):
        """Piece ``piece`` of CTA ``c`` (of ``ctas`` streaming the ``total`` virtual
        stages of the units of ``heads`` KV heads per request: the heads whose ``mask``
        bit is ``die``, all when ``die`` < 0): (valid, unit, request, head, length,
        window start, first stage, stages, piece index in the unit, pieces of the
        unit)."""
        HS: cutlass.Constexpr = self.slots
        HKV: cutlass.Constexpr = self.kv_heads
        a = c * total // ctas  # (units x slots x CTAs < 2^31: 32-bit math)
        b = (c + 1) * total // ctas
        j = a // HS + piece
        valid = (j * HS < b) & (a < total)
        u_base = j * HS
        j0 = max(a, u_base) - u_base
        j1 = min(b, u_base + HS) - u_base
        request = j // heads
        k = j - request * heads
        head = k
        if die >= 0:
            seen = Int32(0)
            for h in cutlass.range_constexpr(HKV):
                on = ((mask >> h) & 1) == die
                if on & (seen == k):
                    head = Int32(h)
                if on:
                    seen = seen + 1
        u = request * HKV + head
        length = Int32(0)
        if valid & (lane == 0):  # (one load per warp: every CTA reads these lines)
            length = load_i32(seq_lens.iterator + request)
        length = shfl_idx(length, Int32(0))
        lo, first, count = self.stages(length)
        n = Int32(0)
        if valid:
            n = max(min(j1, count) - j0, Int32(0))
        cta_lo = ((u_base + 1) * ctas - 1) // total
        cta_hi = ((u_base + HS) * ctas - 1) // total
        return (valid, u, request, head, length, lo, first + j0, n, c - cta_lo,
                cta_hi - cta_lo + 1)  # fmt: skip

    @cute.kernel
    def kernel(self, args):
        qkv, positions, slots, seq_lens, table, cos_sin, qw, kw, out = args[0:9]
        kv_map, kv_base, row_stride, head_stride = args[9:13]
        eps = args[14]
        partials, counters = args[15], args[16]
        G: cutlass.Constexpr = self.group
        Q: cutlass.Constexpr = self.q
        R: cutlass.Constexpr = self.ring
        N: cutlass.Constexpr = self.page
        NP: cutlass.Constexpr = self.pages
        W: cutlass.Constexpr = self.window
        ROWS: cutlass.Constexpr = self.rows
        HQ: cutlass.Constexpr = self.group * self.kv_heads
        HKV: cutlass.Constexpr = self.kv_heads
        HS: cutlass.Constexpr = self.slots

        c, _, _ = cute.arch.block_idx()
        bid = c
        self.rec(bid, 14, global_timer(), cute.arch.thread_idx()[0] == 0)
        self.rec(bid, 13, Int64(_smid()), cute.arch.thread_idx()[0] == 0)
        ctas, _, _ = cute.arch.grid_dim()
        tid, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        group8 = lane & 7
        upper = group8 >= 4
        table_last = Int32(table.shape[1] - 1)
        blocks_max = args[13] - 1
        units = (out.shape[0] // Q) * HKV
        total = units * HS

        smem = cutlass.utils.SmemAllocator()
        ring = alloc(smem, cutlass.Uint8, R * SLOT, align=1024)
        qtile = alloc(smem, cutlass.Uint8, 128 * 128, align=1024)
        newkv = alloc(smem, cutlass.Uint8, Q * 256, align=16)
        newvalid = alloc(smem, Int32, Q, align=16)
        bars = alloc(smem, cutlass.Int64, R + 4 * NS + 1, align=8)
        # [0] TMEM, [1] redo, [2] die << 16 | die-local CTA, [3] launch (the shared
        # memory sits 8 B under two CTAs per SM: every byte here counts)
        tslot = alloc(smem, Int32, 4, align=16)
        ring_base = Int32(shared_address(ring.iterator))
        q_base = Int32(shared_address(qtile.iterator))
        kv_new = Int32(shared_address(newkv.iterator))
        redo_flag = Int32(shared_address(tslot.iterator)) + 4
        full = Int32(shared_address(bars.iterator))
        sready = full + 8 * R
        pready = sready + 8 * NS
        pvdone = pready + 8 * NS
        sfree = pvdone + 8 * NS
        qready = sfree + 8 * NS
        # merge scratch in the Q tile (free once a piece's last S MMA completed):
        # rows [ROWS, OPITCH] FP32, then (max, sum) per quarter and row
        merged = cute.make_tensor(
            cute.recast_ptr(qtile.iterator, dtype=Float32),
            cute.make_layout(2 * ROWS * OPITCH + 4 * 32 * 2),
        )  # fmt: skip
        stats0 = 2 * ROWS * OPITCH

        if tid == 0:
            for i in cutlass.range_constexpr(R):
                init_barrier(full + 8 * i, 1)
            for b in cutlass.range_constexpr(NS):
                init_barrier(sready + 8 * b, 1)
                init_barrier(pready + 8 * b, 4)
                init_barrier(pvdone + 8 * b, 1)
                init_barrier(sfree + 8 * b, 4)
            init_barrier(qready, 1)
            store_shared_u32(redo_flag, 0)
        # the Q tile's padding rows (32 - ROWS per lane quarter), once
        PAD: cutlass.Constexpr = (32 - ROWS) * 4 * 8
        for it in cutlass.range_constexpr(-(-PAD // THREADS)):
            t = it * THREADS + tid
            if t < PAD:
                quarter, rest = t // ((32 - ROWS) * 8), t % ((32 - ROWS) * 8)
                prow = 32 * quarter + ROWS + rest // 8
                store_words(q_base + swz(prow, rest % 8), (Uint32(0),) * 4,
                            shared=True)  # fmt: skip
        q_w = load_words(qw.iterator + 8 * group8, count=4)
        k_w = load_words(kw.iterator + 8 * group8, count=4)

        # ---- the CTA's place: die split = (die, die-local index) from the resident
        # grid's landing; else (-1, block index) over every unit
        die = Int32(-1)
        heads = Int32(HKV)
        mask = Int32(0)
        launch = Int32(0)
        if cutlass.const_expr(self.die_sms is not None):
            mask = args[20]
            if tid == 0:
                d0, rank, slot, nth = land(args[17], args[18], args[19], 2)
                tslot[2] = (d0 << 16) | (rank * 2 + slot)
                tslot[3] = nth
            cute.arch.sync_threads()
            die = tslot[2] >> 16
            c = tslot[2] & 0xFFFF
            launch = tslot[3]
            ctas = Int32(2 * self.die_sms[0])
            if die != 0:
                ctas = Int32(2 * self.die_sms[1])
            heads = Int32(0)
            for h in cutlass.range_constexpr(HKV):
                if ((mask >> h) & 1) == die:
                    heads = heads + 1
            total = (out.shape[0] // Q) * heads * HS
            if cutlass.const_expr(self.unit_per_cta):
                ctas = min(ctas, (out.shape[0] // Q) * heads)

        # ---- the plan from the metadata as it stands (before the PDL wait)
        pieces0 = [self.plan(seq_lens, c, ctas, total, p, die, heads, mask, lane)
                   for p in range(MAXP)]  # fmt: skip
        pre0 = self.prep_load(qkv, positions, slots, cos_sin, group8, tid, pieces0[0],
                              False)  # fmt: skip
        # producer lanes hold one block table entry per piece (entry ``lane`` from
        # the piece's first page)
        blk0 = [self.block_entry(table, pieces0[p], lane, warp, table_last,
                                 blocks_max) for p in range(MAXP)]  # fmt: skip
        n_tot0 = pieces0[0][7] + pieces0[1][7] + pieces0[2][7]
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        spec_n = min(n_tot0, Int32(R))
        if warp == PRODUCER:
            for g in cutlass.range_constexpr(R):
                self.load_stage(kv_map, full, ring_base, Int32(g), pieces0, blk0,
                                lane, Int32(g) < spec_n)  # fmt: skip
        self.rec(bid, 0, global_timer(), tid == 0)
        cute.arch.griddepcontrol_wait()
        if warp == PRODUCER:
            alloc_tmem(Int32(shared_address(tslot.iterator)), TMEM_COLS)
        self.rec(bid, 1, global_timer(), tid == 0)

        # ---- after the wait: the plan again (confirming it); a moved plan drains the
        # speculated loads and reissues them (those slots' phases then start at 1)
        pieces = [self.plan(seq_lens, c, ctas, total, p, die, heads, mask, lane)
                  for p in range(MAXP)]  # fmt: skip
        blk = [self.block_entry(table, pieces[p], lane, warp, table_last, blocks_max)
               for p in range(MAXP)]  # fmt: skip
        moved = Int32(0)
        for p in cutlass.range_constexpr(MAXP):
            for k in cutlass.range_constexpr(10):
                if pieces[p][k] != pieces0[p][k]:
                    moved = Int32(1)
            if blk[p] != blk0[p]:
                moved = Int32(1)
        if cutlass.const_expr(self.debug_redo):
            moved = Int32(tid == 0)
        if moved != 0:
            atomic_or_shared(redo_flag, 1)
        cute.arch.sync_threads()
        offset_bits = Int32(0)
        n_tot = pieces[0][7] + pieces[1][7] + pieces[2][7]
        if load_words(redo_flag, count=1, shared=True)[0] != 0:
            if warp == PRODUCER:
                if lane < spec_n:
                    wait(full + 8 * lane, 0, scope="cta")
                cute.arch.sync_warp()
                for g in cutlass.range_constexpr(R):
                    self.load_stage(kv_map, full, ring_base, Int32(g), pieces, blk,
                                    lane, Int32(g) < min(n_tot, Int32(R)))  # fmt: skip
            offset_bits = (Int32(1) << spec_n) - 1

        # ---- the first piece's Q / K / V prep
        ld0 = self.prep_refresh(qkv, positions, slots, cos_sin, group8, tid, pieces[0],
                                pre0)  # fmt: skip
        if warp < 4:
            self.prep_store(q_w, k_w, eps, upper, group8, tid, pieces[0], ld0, q_base,
                            kv_new, newvalid, kv_base, row_stride, head_stride)  # fmt: skip
            fence_async_shared()
        fence_tmem(before=True)
        cute.arch.sync_threads()
        fence_tmem()
        tmem = Uint32(tslot[0])
        self.rec(bid, 2, global_timer(), tid == 0)
        self.rec(bid, 12, Int64(n_tot), tid == 0)

        if warp == PRODUCER:
            self.producer(kv_map, full, sready, pready, pvdone, sfree, qready,
                          ring_base, q_base, kv_new, newvalid, tmem, pieces, blk,
                          lane, n_tot, offset_bits, counters)  # fmt: skip
            self.rec(bid, 3, global_timer(), lane == 0)
        else:
            self.softmax(qkv, positions, slots, cos_sin, q_w, k_w, eps, upper, group8,
                         tid, warp, lane, pieces, q_base, kv_new, newvalid, kv_base,
                         row_stride, head_stride, out, tmem, sready, pready, pvdone,
                         sfree, qready, merged, stats0, partials, counters, bid)  # fmt: skip
        if cutlass.const_expr(self.die_sms is not None):
            if tid == 0:  # (no CTA leaves before every CTA of the launch landed)
                wait_landed(args[19], launch, 2 * sum(self.die_sms))
        self.rec(bid, 11, global_timer(), tid == 0)
        fence_tmem(before=True)
        cute.arch.sync_threads()
        if warp == PRODUCER:
            fence_tmem()
            free_tmem(tmem, TMEM_COLS)

    @cute.jit
    def block_entry(self, table, piece, lane, warp, table_last, blocks_max):
        """Block table entry lane of the piece's pages (from its first stage)."""
        valid, request, s0 = piece[0], piece[2], piece[6]
        entry = Int32(0)
        if valid & (lane < self.pages) & (warp == PRODUCER):  # (the producer's only)
            row = table.iterator + request * table.layout.stride[0]
            entry = load_i32(row + min(s0 * KB // self.page + lane, table_last))
            entry = min(max(entry, Int32(0)), blocks_max)
        return entry

    @cute.jit
    def locate(self, g, pieces):
        """(piece, its first global stage) of the CTA's global stage ``g``."""
        n0, n1 = pieces[0][7], pieces[1][7]
        p = Int32(0)
        base = Int32(0)
        if g >= n0:
            p = Int32(1)
            base = n0
        if g >= n0 + n1:
            p = Int32(2)
            base = n0 + n1
        return p, base

    @cute.jit
    def load_stage(self, kv_map, full, ring_base, g, pieces, blk, lane, when):
        """Issue the CTA's global stage ``g`` into ring slot g % R (lane 0 of the
        producer; every lane shuffles the piece's block entry)."""
        N: cutlass.Constexpr = self.page
        p, base = self.locate(g, pieces)
        s0 = pieces[0][6]
        head = pieces[0][3]
        entries = blk[0]
        if p == 1:
            s0 = pieces[1][6]
            head = pieces[1][3]
            entries = blk[1]
        if p == 2:
            s0 = pieces[2][6]
            head = pieces[2][3]
            entries = blk[2]
        j = s0 + (g - base)
        index = j * KB // N - s0 * KB // N
        entry = shfl_idx(entries, min(index, Int32(31)))
        if when & (lane == 0):
            r_i = g % self.ring
            arm(full + 8 * r_i, SLOT)
            tma_load5(ring_base + r_i * SLOT, kv_map, full + 8 * r_i, j * KB % N, head,
                      entry)  # fmt: skip

    @cute.jit
    def piece_rows(self, piece):
        """(window start, length, first query key, lo_max) of a piece."""
        length, lo = piece[4], piece[5]
        return lo, length, length - self.q, max(length - self.window, Int32(0))

    @cute.jit
    def task(self, t, head, request):
        """Prep task ``t``: (K / V row, query row i, qkv row, column): 8 lanes per
        64-value head; Q rows first, then the query rows' K (V at + kv heads)."""
        G: cutlass.Constexpr = self.group
        Q: cutlass.Constexpr = self.q
        ROWS: cutlass.Constexpr = self.rows
        HQ: cutlass.Constexpr = self.group * self.kv_heads
        group8 = t % 8
        is_kv = t >= ROWS * 8
        i = min(t // 8, ROWS - 1) // G
        gq = min(t // 8, ROWS - 1) % G
        col = (head * G + gq) * HEAD + 8 * group8
        if is_kv:
            i = min((t - ROWS * 8) // 8, Q - 1)
            col = HQ * HEAD + head * HEAD + 8 * group8
        return is_kv, i, request * Q + i, col

    @cute.jit
    def rope_into(self, rope, cos_sin, pos, group8):
        row = min(max(Int32(pos), Int32(0)), Int32(cos_sin.shape[0] - 1))
        base = cos_sin.iterator + row * cos_sin.layout.stride[0]
        cos = load_words(base + 8 * (group8 % 4), count=4)
        sin = load_words(base + 32 + 8 * (group8 % 4), count=4)
        for k in cutlass.range_constexpr(4):
            rope[k] = cos[k]
            rope[4 + k] = sin[k]

    @cute.jit
    def prep_load(self, qkv, positions, slots, cos_sin, group8, tid, piece,
                  with_qkv: cutlass.Constexpr[bool]):  # fmt: skip
        """This thread's prep tasks' inputs: (position, cos | sin words, q-or-k | v
        words, slot) per task; ``with_qkv`` False: the position and RoPE rows only
        (before the PDL wait). Any thread may load (rows clamped to the batch)."""
        Q: cutlass.Constexpr = self.q
        HKV: cutlass.Constexpr = self.kv_heads
        TASKS: cutlass.Constexpr = self.rows * 8 + self.q * 8
        request = min(piece[2], Int32(qkv.shape[0] // Q - 1))
        stride0 = qkv.layout.stride[0]
        loaded = []
        for it in cutlass.range_constexpr(-(-TASKS // SOFTMAX)):
            is_kv, i, row, col = self.task(it * SOFTMAX + tid % SOFTMAX, piece[3],
                                           request)  # fmt: skip
            pos = load_i64(positions.iterator + row)
            rope = cute.make_rmem_tensor(8, Uint32)
            self.rope_into(rope, cos_sin, pos, group8)
            words = cute.make_rmem_tensor(8, Uint32)
            slot = Int64(-1)
            if cutlass.const_expr(with_qkv):
                w = load_words(qkv.iterator + row * stride0 + col, count=4,
                               mutable=True)  # fmt: skip
                v = load_words(qkv.iterator + row * stride0 + col + HKV * HEAD,
                               count=4, mutable=True)  # fmt: skip
                for k in cutlass.range_constexpr(4):
                    words[k] = w[k]
                    words[4 + k] = v[k]
                slot = load_i64(slots.iterator + row)
            loaded.append((pos, rope, words, slot))
        return loaded

    @cute.jit
    def prep_refresh(self, qkv, positions, slots, cos_sin, group8, tid, piece,
                     loaded):  # fmt: skip
        """After the PDL wait: the qkv rows and slots, the position again (a moved
        position reloads its RoPE rows)."""
        Q: cutlass.Constexpr = self.q
        HKV: cutlass.Constexpr = self.kv_heads
        request = min(piece[2], Int32(qkv.shape[0] // Q - 1))
        stride0 = qkv.layout.stride[0]
        fresh = []
        for it, (pos0, rope, words, _) in enumerate(loaded):
            is_kv, i, row, col = self.task(it * SOFTMAX + tid % SOFTMAX, piece[3],
                                           request)  # fmt: skip
            pos = load_i64(positions.iterator + row)
            if pos != pos0:
                self.rope_into(rope, cos_sin, pos, group8)
            w = load_words(qkv.iterator + row * stride0 + col, count=4, mutable=True)
            v = load_words(qkv.iterator + row * stride0 + col + HKV * HEAD, count=4,
                           mutable=True)  # fmt: skip
            for k in cutlass.range_constexpr(4):
                words[k] = w[k]
                words[4 + k] = v[k]
            fresh.append((pos, rope, words, load_i64(slots.iterator + row)))
        return fresh

    @cute.jit
    def prep_store(self, q_w, k_w, eps, upper, group8, tid, piece, loaded, q_base,
                   kv_new, newvalid, kv_base, row_stride, head_stride):  # fmt: skip
        """The softmax warps: q / k norm + RoPE of the piece's unit into the Q tile;
        the query rows' K / V (cache insert + ``newkv``) when the piece's stages hold
        their keys."""
        Q: cutlass.Constexpr = self.q
        N: cutlass.Constexpr = self.page
        ROWS: cutlass.Constexpr = self.rows
        TASKS: cutlass.Constexpr = ROWS * 8 + Q * 8
        head, length, s0, n = piece[3], piece[4], piece[6], piece[7]
        q_first = length - Q
        for it, (pos, rope, words, slot) in enumerate(loaded):
            t = it * SOFTMAX + tid
            is_kv, i, row, col = self.task(t, head, Int32(0))
            wts = cute.make_rmem_tensor(4, Uint32)
            for k in cutlass.range_constexpr(4):
                wts[k] = q_w[k]
            if is_kv:
                for k in cutlass.range_constexpr(4):
                    wts[k] = k_w[k]
            x = [bf16(words[e // 2], e % 2 == 1) for e in range(8)]
            roped = norm_rope8(
                x, tuple(wts[k] for k in range(4)), tuple(rope[k] for k in range(4)),
                tuple(rope[4 + k] for k in range(4)), eps, upper, self.round_norm,
            )  # fmt: skip
            if t < ROWS * 8:
                for quarter in cutlass.range_constexpr(4):
                    store_words(q_base + swz(32 * quarter + t // 8, group8),
                                tuple(roped), shared=True)  # fmt: skip
            elif (t < TASKS) & (length > 0) & (q_first + i >= s0 * KB) & (
                q_first + i < (s0 + n) * KB
            ):  # (the piece whose stages hold the query row's key)
                value = [words[4 + k] for k in range(4)]
                if cutlass.const_expr(self.v_scale is not None):
                    scaled = []
                    for k in cutlass.range_constexpr(4):
                        lo_v = bf16(words[4 + k], False) * Float32(self.v_scale)
                        hi_v = bf16(words[4 + k], True) * Float32(self.v_scale)
                        scaled.append(bf16x2(lo_v, hi_v))
                    value = scaled
                store_words(kv_new + i * 256 + 16 * group8, tuple(roped),
                            shared=True)  # fmt: skip
                store_words(kv_new + i * 256 + 128 + 16 * group8, tuple(value),
                            shared=True)  # fmt: skip
                if group8 == 0:
                    newvalid[i] = Int32(slot >= 0)
                if slot >= 0:
                    s_elem = Int64(slot // N) * row_stride
                    s_elem = s_elem + Int64(head) * head_stride
                    s_elem = s_elem + Int64(slot % N) * 128 + 8 * group8
                    s_addr = kv_base + s_elem * 2
                    store_words(s_addr, tuple(roped))
                    store_words(s_addr + 128, tuple(value))

    @cute.jit
    def fixup(self, tile, p0, lane, lo, length, kv_new, newvalid):
        """Rows lane and lane + 32 of a stage: zero outside [lo, length); the computed
        K / V of a query row whose slot was valid."""
        q_first = length - self.q
        zero = (Uint32(0),) * 4
        for h in cutlass.range_constexpr(2):
            row = lane + 32 * h
            position = p0 + row
            if (position < lo) | (position >= length):
                for ch in cutlass.range_constexpr(8):
                    store_words(tile + swz(row, ch), zero, shared=True)
                    store_words(tile + KTILE + swz(row, ch), zero, shared=True)
            elif position >= q_first:
                i = position - q_first
                if newvalid[i] != 0:
                    for ch in cutlass.range_constexpr(8):
                        kq = load_words(kv_new + i * 256 + 16 * ch, count=4,
                                        shared=True)  # fmt: skip
                        vq = load_words(kv_new + i * 256 + 128 + 16 * ch, count=4,
                                        shared=True)  # fmt: skip
                        store_words(tile + swz(row, ch), kq, shared=True)
                        store_words(tile + KTILE + swz(row, ch), vq, shared=True)

    @cute.jit
    def issue_s(self, g, local, piece, ring_base, full, sready, sfree, tmem, q_desc,
                lane, kv_new, newvalid, offset_bits):  # fmt: skip
        """S = Q K^T of global stage ``g`` (the piece's stage ``local``) once its K / V
        landed (edge rows fixed first) and the warps loaded stage g - NS's S."""
        lo, length, q_first, lo_max = self.piece_rows(piece)
        s_i = g % self.ring
        tile = ring_base + s_i * SLOT
        p0 = (piece[6] + local) * KB
        wait(full + 8 * s_i, ((g // self.ring) + ((offset_bits >> s_i) & 1)) & 1,
             scope="cta")  # fmt: skip
        if (p0 < lo_max) | (p0 + KB > q_first):
            self.fixup(tile, p0, lane, lo, length, kv_new, newvalid)
            fence_async_shared()
            cute.arch.sync_warp()
        b = g % NS
        if g >= NS:
            wait(sfree + 8 * b, ((g - NS) // NS) & 1, scope="cta")
        fence_tmem()
        mma_bf16_block(
            tmem + Uint32(64) * Uint32(b), q_desc, desc_sw128(tile, 16),
            Uint32(S_INSTR), Int32(0), sready + 8 * b, Int32(1),
        )  # fmt: skip

    @cute.jit
    def refill(self, kv_map, full, pvdone, ring_base, x, pieces, blk, lane, n_tot):
        """Ring slot of global stage ``x`` (its PV issued): stage x + R once that PV
        completed."""
        nxt = x + self.ring
        if nxt < n_tot:
            wait(pvdone + 8 * (x % NS), (x // NS) & 1, scope="cta")
            self.load_stage(kv_map, full, ring_base, nxt, pieces, blk, lane,
                            nxt < n_tot)  # fmt: skip

    @cute.jit
    def producer(self, kv_map, full, sready, pready, pvdone, sfree, qready, ring_base,
                 q_base, kv_new, newvalid, tmem, pieces, blk, lane, n_tot,
                 offset_bits, counters):  # fmt: skip
        """S of stage t + 1 before stage t's PV waits on its P; ring refills behind the
        PVs, across pieces; a piece's first S waits for its Q tile (``qready``)."""
        q_desc = desc_sw128(q_base, 16)
        g = Int32(0)
        for p in cutlass.range_constexpr(MAXP):
            piece = pieces[p]
            n = piece[7]
            if cutlass.const_expr(p > 0):
                if piece[0]:
                    wait(qready, (p - 1) & 1, scope="cta")
            if n > 0:
                self.issue_s(g, Int32(0), piece, ring_base, full, sready, sfree, tmem,
                             q_desc, lane, kv_new, newvalid, offset_bits)  # fmt: skip
            for t in cutlass.range(n, unroll=1):
                gg = g + t
                if t + 1 < n:
                    self.issue_s(gg + 1, t + 1, piece, ring_base, full, sready, sfree,
                                 tmem, q_desc, lane, kv_new, newvalid,
                                 offset_bits)  # fmt: skip
                b = gg % NS
                wait(pready + 8 * b, (gg // NS) & 1, scope="cta")
                fence_tmem()
                mma_pv_block(
                    tmem + O_COL, tmem + P_COL + Uint32(32) * Uint32(b),
                    desc_sw128(ring_base + (gg % self.ring) * SLOT + KTILE, KTILE),
                    Uint32(PV_INSTR), Int32(t > 0), pvdone + 8 * b,
                )  # fmt: skip
                if t >= 1:
                    self.refill(kv_map, full, pvdone, ring_base, gg - 1, pieces, blk,
                                lane, n_tot)  # fmt: skip
            if n > 0:
                self.refill(kv_map, full, pvdone, ring_base, g + n - 1, pieces, blk,
                            lane, n_tot)  # fmt: skip
            g = g + n
            if piece[0] & (piece[9] > 1) & (piece[8] > 0):
                # a later piece of a split unit: once the softmax warps stored its
                # partial (barrier 2), publish it to the unit's first piece
                named_barrier(2, SOFTMAX + 32)
                if lane == 0:
                    counter_release(Int64(counters) + Int64(piece[1]) * 4)

    @cute.jit
    def softmax(self, qkv, positions, slots, cos_sin, q_w, k_w, eps, upper, group8,
                tid, warp, lane, pieces, q_base, kv_new, newvalid, kv_base,
                row_stride, head_stride, out, tmem, sready, pready, pvdone, sfree,
                qready, merged, stats0, partials, counters, c):  # fmt: skip
        """Warp w = TMEM lane quarter w, keys 16w..16w+15 of each stage; per piece the
        quarters' merge and the piece's rows (or partial)."""
        G: cutlass.Constexpr = self.group
        W: cutlass.Constexpr = self.window
        ROWS: cutlass.Constexpr = self.rows
        c_scale = Float32(LOG2E / (HEAD**0.5))
        r = lane
        lanes = Uint32(32 * warp) << 16
        p_lanes = tmem + lanes + P_COL
        for k in cutlass.range_constexpr(NS):  # every P buffer's columns: zero
            st32w(p_lanes + Uint32(32 * k), (Uint32(0),) * 32)
        g = Int32(0)
        for p in cutlass.range_constexpr(MAXP):
            piece = pieces[p]
            n = piece[7]
            lo, length, q_first, lo_max = self.piece_rows(piece)
            hi_r = length - 1
            lo_r = lo
            if r < ROWS:
                hi_r = q_first + r // G
                lo_r = max(hi_r - (W - 1), Int32(0))
            m_use = Float32(NEG_INF)
            l_run = Float32(0)
            nxt = None
            if cutlass.const_expr(p + 1 < MAXP):  # (the next piece's prep inputs)
                nxt = self.prep_load(qkv, positions, slots, cos_sin, group8, tid,
                                     pieces[p + 1], False)  # fmt: skip
            for t in cutlass.range(n, unroll=1):
                gg = g + t
                b = gg % NS
                s_addr = tmem + lanes + Uint32(64) * Uint32(b)
                wait(sready + 8 * b, (gg // NS) & 1, scope="cta")
                fence_tmem()
                raw = ld16(s_addr + Uint32(16) * Uint32(warp))
                fence_tmem(before=True)
                cute.arch.sync_warp()
                if lane == 0:
                    arrive(sfree + 8 * b)
                ps0 = (piece[6] + t) * KB
                xs = cute.make_rmem_tensor(16, Float32)
                for e in cutlass.range_constexpr(16):
                    xs[e] = raw[e]
                if (ps0 < lo_max) | (ps0 + KB > q_first):
                    for e in cutlass.range_constexpr(16):
                        key = ps0 + 16 * warp + e
                        if (key < lo_r) | (key > hi_r):
                            xs[e] = Float32(NEG_INF)
                mx = tree(fmax, [xs[e] for e in range(16)]) * c_scale
                need = mx > m_use + Float32(THRESH)
                rescale = Int32(0)
                if need & (m_use > Float32(NEG_INF)):
                    rescale = Int32(1)
                if vote_any(rescale) != 0:
                    alpha = Float32(1)
                    if rescale != 0:
                        alpha = ex2(m_use - mx)
                    l_run = l_run * alpha
                    wait(pvdone + 8 * ((gg - 1) % NS), ((gg - 1) // NS) & 1,
                         scope="cta")  # fmt: skip
                    fence_tmem()
                    o_addr = tmem + lanes + O_COL
                    o = ld64(o_addr)
                    st64(o_addr, tuple(o[e] * alpha for e in range(64)))
                if need:
                    m_use = mx
                bias = Float32(0)
                if m_use > Float32(NEG_INF):
                    bias = -m_use
                ps = [ex2(fma(xs[e], c_scale, bias)) for e in range(16)]
                l_run = l_run + tree(lambda x, y: x + y, ps)
                packed = tuple(bf16x2(ps[2 * k], ps[2 * k + 1]) for k in range(8))
                if gg >= NS:  # stage gg - NS's PV read this P buffer
                    wait(pvdone + 8 * b, ((gg - NS) // NS) & 1, scope="cta")
                    fence_tmem()
                p_own = p_lanes + Uint32(32) * Uint32(b) + Uint32(8) * Uint32(warp)
                st8w(p_own, packed)
                fence_tmem(before=True)
                cute.arch.sync_warp()
                if lane == 0:
                    arrive(pready + 8 * b)
            if cutlass.const_expr(p == 0):  # (the successor still waits for the end)
                cute.arch.griddepcontrol_launch_dependents()
            self.rec(c, 4 + 2 * p, global_timer(), tid == 0)
            fresh = None
            if cutlass.const_expr(p + 1 < MAXP):  # (in flight during the merge)
                fresh = self.prep_refresh(qkv, positions, slots, cos_sin, group8, tid,
                                          pieces[p + 1], nxt)  # fmt: skip
            if piece[0]:
                self.merge(warp, lane, tid, piece, n, g, tmem, lanes, pvdone, m_use,
                           l_run, q_base, merged, stats0, partials, counters, out,
                           c)  # fmt: skip
            self.rec(c, 5 + 2 * p, global_timer(), tid == 0)
            g = g + n
            if cutlass.const_expr(p + 1 < MAXP):
                if pieces[p + 1][0]:
                    named_barrier(1, SOFTMAX)  # (the merge read the Q tile)
                    self.prep_store(q_w, k_w, eps, upper, group8, tid, pieces[p + 1],
                                    fresh, q_base, kv_new, newvalid, kv_base,
                                    row_stride, head_stride)  # fmt: skip
                    fence_async_shared()
                    named_barrier(1, SOFTMAX)
                    if tid == 0:
                        arrive(qready)

    @cute.jit
    def merge(self, warp, lane, tid, piece, n, g, tmem, lanes, pvdone, m_use, l_run,
              q_base, merged, stats0, partials, counters, out, c):  # fmt: skip
        """The four quarters' rows into two row regions of the Q tile (quarters 2, 3
        stored, then 0, 1 added: region w % 2); then the unit's output rows (one
        piece), or the piece's partial: a later piece of the unit (k > 0) stores it
        and leaves the publication to the producer warp (barrier 2); the unit's first
        piece (k = 0, its CTA's last) waits for the others (acquire) and merges every
        partial in piece order."""
        ROWS: cutlass.Constexpr = self.rows
        PS: cutlass.Constexpr = self.pstride
        PM: cutlass.Constexpr = self.pmax
        RB: cutlass.Constexpr = ROWS * OPITCH * 4  # bytes per row region
        u, k_p, p_u = piece[1], piece[8], piece[9]
        r = lane
        if n > 0:
            last = g + n - 1
            wait(pvdone + 8 * (last % NS), (last // NS) & 1, scope="cta")
            fence_tmem()
        named_barrier(1, SOFTMAX)  # every warp past its last S: the Q tile is free
        if r < ROWS:
            merged[stats0 + (warp * 32 + r) * 2] = m_use
            merged[stats0 + (warp * 32 + r) * 2 + 1] = l_run
        named_barrier(1, SOFTMAX)
        top = Float32(NEG_INF)
        total = Float32(0)
        weight = Float32(0)
        if r < ROWS:
            ms = [merged[stats0 + (w * 32 + r) * 2] for w in range(4)]
            top = fmax(fmax(ms[0], ms[1]), fmax(ms[2], ms[3]))
            use = top
            if top == Float32(NEG_INF):
                use = Float32(0)
            for w in cutlass.range_constexpr(4):
                wt = ex2(ms[w] - use)
                total = fma(merged[stats0 + (w * 32 + r) * 2 + 1], wt, total)
            weight = ex2(m_use - use)
        row_addr = q_base + (warp % 2) * RB + r * (OPITCH * 4)
        for step in cutlass.range_constexpr(2):
            if warp // 2 == 1 - step:  # (step 0: quarters 2, 3; step 1: 0, 1)
                if n > 0:
                    for ch in cutlass.range_constexpr(4):
                        o = ld16(tmem + lanes + O_COL + Uint32(16 * ch))
                        if r < ROWS:
                            for q4 in cutlass.range_constexpr(4):
                                at = row_addr + 64 * ch + 16 * q4
                                if cutlass.const_expr(step == 0):
                                    acc = tuple(float_word(o[4 * q4 + e] * weight)
                                                for e in range(4))  # fmt: skip
                                else:
                                    cur = load_words(at, count=4, shared=True)
                                    acc = tuple(
                                        float_word(fma(o[4 * q4 + e], weight,
                                                       bits_float(cur[e])))
                                        for e in range(4)
                                    )  # fmt: skip
                                store_words(at, acc, shared=True)
                if cutlass.const_expr(step == 0):
                    if (n == 0) & (r < ROWS):  # (no PV in this piece: zero rows)
                        for q4 in cutlass.range_constexpr(16):
                            store_words(row_addr + 16 * q4, (Uint32(0),) * 4,
                                        shared=True)  # fmt: skip
            named_barrier(1, SOFTMAX)
        TASKS: cutlass.Constexpr = ROWS * 8
        if p_u == 1:
            for it in cutlass.range_constexpr(-(-TASKS // SOFTMAX)):
                task = it * SOFTMAX + tid
                if task < TASKS:
                    rr, d8 = task // 8, task % 8
                    values = self.own_row(q_base, rr, d8)
                    _, total_r = self.own_stats(merged, stats0, rr)
                    self.store_row(out, piece, rr, d8, values, total_r)
        else:
            unit_base = Int64(partials) + Int64(u) * (PM * PS * 4)
            ctr = Int64(counters) + Int64(u) * 4
            if k_p != 0:
                base = unit_base + Int64(k_p) * (PS * 4)
                for it in cutlass.range_constexpr(-(-TASKS // SOFTMAX)):
                    task = it * SOFTMAX + tid
                    if task < TASKS:
                        rr, d8 = task // 8, task % 8
                        values = self.own_row(q_base, rr, d8)
                        dst = base + (rr * 64 + 8 * d8) * 4
                        store_words(dst, tuple(float_word(values[e]) for e in range(4)))
                        store_words(dst + 16, tuple(float_word(values[4 + e])
                                                    for e in range(4)))  # fmt: skip
                if tid < ROWS:
                    st = base + (ROWS * 64 + 2 * tid) * 4
                    top_t, total_t = self.own_stats(merged, stats0, tid)
                    store_words(st, (float_word(top_t), float_word(total_t)))
                named_barrier(2, SOFTMAX + 32, arrive=True)  # (the producer publishes)
            else:
                if tid == 0:
                    seen = counter_acquire(ctr)
                    sgn, sgt = spin_start()
                    while seen != Uint32(p_u - 1):
                        seen = counter_acquire(ctr)
                        sgn, sgt = spin_tick(sgn, sgt, observed=seen, target=0)
                named_barrier(1, SOFTMAX)
                self.rec(c, 10, global_timer(), tid == 0)
                for it in cutlass.range_constexpr(-(-TASKS // SOFTMAX)):
                    task = it * SOFTMAX + tid
                    if task < TASKS:
                        rr, d8 = task // 8, task % 8
                        self.final_row(out, piece, rr, d8, unit_base, p_u, q_base,
                                       merged, stats0)  # fmt: skip
                if tid == 0:
                    counter_reset(ctr)

    @cute.jit
    def own_row(self, q_base, rr, d8):
        """Dims 8 d8.. of merged row ``rr``: region 0 + region 1 (the quarters)."""
        RB: cutlass.Constexpr = self.rows * OPITCH * 4
        at = q_base + (rr * OPITCH + 8 * d8) * 4
        a = load_words(at, count=4, shared=True) + load_words(at + 16, count=4,
                                                              shared=True)  # fmt: skip
        b = load_words(at + RB, count=4, shared=True) + load_words(
            at + RB + 16, count=4, shared=True)  # fmt: skip
        return [bits_float(a[e]) + bits_float(b[e]) for e in range(8)]

    @cute.jit
    def own_stats(self, merged, stats0, rr):
        """(max, sum) of merged row ``rr``: the quarters' max and their sums at it."""
        ms = [merged[stats0 + (w * 32 + rr) * 2] for w in range(4)]
        top = fmax(fmax(ms[0], ms[1]), fmax(ms[2], ms[3]))
        use = top
        if top == Float32(NEG_INF):
            use = Float32(0)
        total = Float32(0)
        for w in cutlass.range_constexpr(4):
            total = fma(merged[stats0 + (w * 32 + rr) * 2 + 1], ex2(ms[w] - use), total)
        return top, total

    @cute.jit
    def final_row(self, out, piece, rr, d8, unit_base, p_u, q_base, merged, stats0):
        """Output dims 8 d8.. of row ``rr``: the unit's pieces in order (piece 0 from
        the Q tile, the others' partials two at a time), an online LSE merge."""
        ROWS: cutlass.Constexpr = self.rows
        PS: cutlass.Constexpr = self.pstride
        PM: cutlass.Constexpr = self.pmax
        top, total = self.own_stats(merged, stats0, rr)
        own = self.own_row(q_base, rr, d8)
        values = cute.make_rmem_tensor(8, Float32)
        for e in cutlass.range_constexpr(8):
            values[e] = own[e]
        buf = cute.make_rmem_tensor(20, Float32)
        for k0 in cutlass.range_constexpr(1, PM, 2):
            for kk in cutlass.range_constexpr(2):
                k = k0 + kk
                if cutlass.const_expr(k < PM):
                    if k < p_u:
                        pb = unit_base + k * (PS * 4)
                        st = load_words(pb + (ROWS * 64 + 2 * rr) * 4, count=2,
                                        scope="gpu")  # fmt: skip
                        w8 = load_words(pb + (rr * 64 + 8 * d8) * 4, count=4,
                                        scope="gpu")  # fmt: skip
                        w8b = load_words(pb + (rr * 64 + 8 * d8 + 4) * 4, count=4,
                                         scope="gpu")  # fmt: skip
                        buf[10 * kk] = bits_float(st[0])
                        buf[10 * kk + 1] = bits_float(st[1])
                        for e in cutlass.range_constexpr(4):
                            buf[10 * kk + 2 + e] = bits_float(w8[e])
                            buf[10 * kk + 6 + e] = bits_float(w8b[e])
            for kk in cutlass.range_constexpr(2):
                k = k0 + kk
                if cutlass.const_expr(k < PM):
                    if k < p_u:
                        m_k = buf[10 * kk]
                        if m_k > top:  # (rescale the running rows to the new max)
                            alpha = ex2(top - m_k)
                            if top == Float32(NEG_INF):
                                alpha = Float32(0)
                            total = total * alpha
                            for e in cutlass.range_constexpr(8):
                                values[e] = values[e] * alpha
                            top = m_k
                        wt = ex2(m_k - top)
                        if m_k == Float32(NEG_INF):
                            wt = Float32(0)
                        total = fma(buf[10 * kk + 1], wt, total)
                        for e in cutlass.range_constexpr(8):
                            values[e] = fma(buf[10 * kk + 2 + e], wt, values[e])
        self.store_row(out, piece, rr, d8, [values[e] for e in range(8)], total)

    @cute.jit
    def store_row(self, out, piece, row, d8, values, total):
        """Output dims 8 d8.. of tile row ``row``: the merged sums over the total."""
        G: cutlass.Constexpr = self.group
        request, head = piece[2], piece[3]
        inv = Float32(0)
        if total > Float32(0):
            inv = Float32(1) / total
        words = tuple(
            bf16x2(values[2 * k] * inv, values[2 * k + 1] * inv) for k in range(4)
        )
        o_row = request * self.q + row // G
        o_col = (head * G + row % G) * HEAD + 8 * d8
        store_words(out.iterator + o_row * out.layout.stride[0] + o_col, words)

    @cute.jit
    def rec(self, cta, k: cutlass.Constexpr[int], value, cond):
        if cutlass.const_expr(self.debug != 0):
            if cond:
                store_u64(Int64(self.debug) + (cta * 16 + k) * 8, value)


@cache
def compile_window_stream(
    group, kv_heads, q, page, ring, window, v_scale, max_position, device, pmax,
    round_norm=False, debug=0, redo=False, cap=0, die_sms=None,
):  # fmt: skip
    """``GqaWindowStream`` for any request count (row-dynamic views); ``GqaWindow``'s
    arguments plus the partials buffer [units, pmax, rows x 66] FP32 and the unit
    counters (int32, zero) addresses."""
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    kernel = GqaWindowStream(
        group, kv_heads, q, page, ring, window, sms, v_scale, round_norm, debug, redo,
        cap, die_sms,
    )
    kernel.pmax = pmax
    i64, i32 = torch.int64, torch.int32
    count = cute.sym_int()
    args = (
        dynamic_fake(2),  # qkv [rows, width]
        fake((count,), None, i64, 8),  # positions
        fake((count,), None, i64, 8),  # slots
        fake((cute.sym_int(),), None, i32, 4),  # seq_lens
        fake((cute.sym_int(), cute.sym_int()), (cute.sym_int(), 1), i32, 4),
        fake((max_position, 2 * 32)),  # cos_sin (BF16)
        fake((HEAD,)),  # q_norm weight
        fake((HEAD,)),  # k_norm weight
        dynamic_fake(2),  # out [rows, heads x 64]
        Int64(0),  # the layer cache's tensor map address
        Int64(0),  # the layer cache's base address
        Int64(0),  # row (block) stride, elements
        Int64(0),  # head stride, elements
        Int32(1),  # blocks
        Float32(1e-5),  # eps
        Int64(0),  # partials
        Int64(0),  # counters
    )
    if die_sms is not None:  # (the resident grid: SM table, tickets, landings; mask)
        args += (Int64(0), Int64(0), Int64(0), Int32(0))
    with torch.accelerator.device_index(device):
        return cute.compile(kernel, *args, make_fake_stream())
