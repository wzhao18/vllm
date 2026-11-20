# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KdaCoreQ: KdaCore with QGROUPS (3 or 4) item groups per CTA and a 4-row scan.

KdaCore's two groups of four warps (thread = 8 rows x 16 keys, 229 registers) leave
two warps per scheduler and the scan latency-bound (IPC ~1 per SM, ncu r5f). Here a
thread holds 4 rows x 32 keys: thread c of a group is (b, l) = (c // 4, c % 4) with
rows 4 b + [0, 4) and keys 4 (l + 4 ch') + [0, 4), ch' = ch ^ (b & 1) over 8 chunks
of 16 keys (the b-parity swap puts the 8 lanes of a shared-memory phase on 8
distinct 16-byte bank groups). Acc slot rr of lane l holds row 4 b + (rr ^ l): a
2-stage select-free transposed butterfly over 4 lanes leaves row 4 b + l = c in
thread c (KdaCore's solve / gated norm / MXFP8 epilogue unchanged). The math is
KdaFlow's (fold association, verify formulas); the dot products' FP32 reduction
order differs (32 keys per thread, a 4-lane tree), within the chain tolerance.

Per warp: a ring of ``nsub`` 4 KiB state sub-blocks (32 rows x 32 keys, 2-D TMA)
and ``sb`` 4 KiB store boxes (TMA tensor stores); operand blocks double-buffered per
group. Static head-major item ranges only (no die-matched claim yet)."""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64

from ..common.primitives import alloc, asm, at, launch, zeros
from . import core, flow
from .core import (
    BOX,
    D_AT,
    G_AT,
    OPB_BYTES,
    OPB_FLOATS,
    U_AT,
    W_AT,
    X_AT,
    bulk_commit,
    clock64,
    prefetch_tensormap,
    tma_load4,
    tma_store4,
    wait_group_all,
)
from .flow import (
    GROUP_THREADS,
    HEAD,
    MAX_ITEMS,
    META,
    PENDING,
    RECORD,
    REQUEST,
    SLOT,
    SUMS,
    K,
    store_shared_f32x4,
)
from .specialized import fence_async_shared, load_shared_f32x4

QGROUPS = 3  # 3: 168 registers, 3 x 4 KiB sub-blocks + 2 store boxes per warp
QROWS, QLANES, QCHUNKS = 4, 4, 8
LAYOUT = {3: (3, 2), 4: (2, 1)}  # groups -> (sub-blocks, store boxes) per warp


@asm()
def wait_group_read0():
    return "cp.async.bulk.wait_group.read 0;", "~{memory}", ()


def shared_bytes(groups):
    nsub, sb = LAYOUT[groups]
    warps = 4 * groups
    ring = warps * (nsub + sb) * BOX * BOX * 4
    blocks = groups * 2 * OPB_BYTES
    return ring + blocks + MAX_ITEMS * META * 4 + groups * SUMS * 16 + 64 * 8 + 4096


class KdaCoreQ(core.KdaCore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert not self.dies, "KdaCoreQ: static item ranges only"
        self.G = QGROUPS
        self.nsub, self.sb = LAYOUT[self.G]

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.ctas, GROUP_THREADS * self.G, min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, args):
        self.bodyq(args)

    @cute.jit
    def bodyq(self, args):
        k0 = Int64(0)
        if cutlass.const_expr(self.trace):
            k0 = clock64()
        T: cutlass.Constexpr = self.tokens
        H: cutlass.Constexpr = self.heads
        G: cutlass.Constexpr = self.G
        NS: cutlass.Constexpr = self.nsub
        W: cutlass.Constexpr = 4 * G
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        requests: cutlass.Constexpr = args[10].shape[0] // T
        total_items: cutlass.Constexpr = H * requests
        first = Int32(total_items * cta // self.ctas)
        count = Int32(total_items * (cta + 1) // self.ctas) - first

        smem = cutlass.utils.SmemAllocator()
        lr = alloc(smem, Float32, (W, NS, BOX * BOX), align=1024)
        ring = alloc(smem, Float32, (W, self.sb, BOX * BOX), align=128)
        opb = alloc(smem, Float32, (G, 2, OPB_FLOATS), align=128)
        table = alloc(smem, Int32, (MAX_ITEMS, META))
        sums = alloc(smem, Float32, (G, SUMS, 4))
        bars = smem.allocate(cutlass.Uint64, W * NS + 2 * G)
        lfull, opb_full = bars, bars + W * NS
        if tid == 0:
            for s in cutlass.range_constexpr(W * NS):
                cute.arch.mbarrier_init(lfull + s, 1)
            for b in cutlass.range_constexpr(2 * G):
                cute.arch.mbarrier_init(opb_full + b, 1)
            cute.arch.mbarrier_init_fence()
        if tid < count:
            head, request, slot, _, _, pending, rec = self.item_info(args, first + tid)
            table[tid, SLOT] = slot
            table[tid, HEAD] = head
            table[tid, REQUEST] = request
            table[tid, PENDING] = pending
            table[tid, RECORD] = rec
        cute.arch.sync_warp()
        cute.arch.barrier()
        g = tid // GROUP_THREADS
        c = tid % GROUP_THREADS
        wid = tid // 32
        mine = (count - g + G - 1) // G
        tmap = args[self.opb_at + 1].iterator
        if tid % 32 == 0:  # the state first (no predecessor dependence)
            prefetch_tensormap(tmap)
            for m in cutlass.range_constexpr(NS):
                if m < 4 * mine:
                    self.load_q(lr, lfull, wid, m, table, g + G * (m // 4), m % 4, tmap)
        cute.arch.sync_warp()
        cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(not self.mx):
            cute.arch.griddepcontrol_launch_dependents()
        if c == 0:
            if g < count:
                self.load_block(args, table, g, opb, opb_full, g, 0)
        cute.arch.sync_warp()
        self.items_q(args, g, c, count, mine, lr, ring, opb, table, sums, lfull,
                     opb_full, k0)  # fmt: skip
        if tid % 32 == 0:
            wait_group_all()

    @cute.jit
    def load_q(self, lr, lfull, wid, m, table, j, s, tmap):
        """Sub-block m of warp ``wid``: item j's rows 32 (wid % 4) + [0, 32), keys
        32 s + [0, 32), into ring slot m % nsub; a padded item arrives empty."""
        slot, head = table[j, SLOT], table[j, HEAD]
        barrier = lfull + wid * self.nsub + m % self.nsub
        if slot > 0:
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, BOX * BOX * 4)
            tma_load4(at(lr, wid, m % self.nsub, 0), tmap, barrier, Int32(BOX * s),
                      Int32(32 * (wid % 4)), head, slot)  # fmt: skip
        else:
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, 0)

    @cute.jit
    def items_q(self, args, g, c, count, mine, lr, ring, opb, table, sums, lfull,
                opb_full, k0):  # fmt: skip
        T: cutlass.Constexpr = self.tokens
        NV: cutlass.Constexpr = 2 * T
        G: cutlass.Constexpr = self.G
        NS: cutlass.Constexpr = self.nsub
        lane, warp = c % 32, c // 32
        l, b = c % QLANES, c // QLANES
        base = QROWS * b
        nw = args[12][c]
        po = cute.make_rmem_tensor(T, Float32)
        pg = cute.make_rmem_tensor(T, Float32)
        pm = cute.make_rmem_tensor(3, Int32)
        for t in cutlass.range_constexpr(T):
            po[t] = Float32(0)
        pm[0] = Int32(0)
        sums_g = sums[g, None, None]
        wid = 4 * g + warp
        tmap = args[self.opb_at + 1].iterator
        tr = zeros(8, Int64)  # TRACE: opb wait, finish, page wait, store wait, scan, solve
        t0 = Int64(0)
        if cutlass.const_expr(self.trace):
            t0 = clock64()
        for jj in cutlass.range(mine, unroll=1):
            j = g + G * jj
            buf = jj % 2
            slot, head = table[j, SLOT], table[j, HEAD]
            pending, rec = table[j, PENDING], table[j, RECORD]
            token = table[j, REQUEST] * T
            if cutlass.const_expr(self.trace):
                tr[6] = clock64()
            self.bwait(opb_full + 2 * g + buf, (jj // 2) % 2, "opb_full")
            if cutlass.const_expr(self.trace):
                tb = clock64()
                tr[0] = tr[0] + (tb - tr[6])
                tr[6] = tb
            block = opb[g, buf, None]
            if pm[0] != 0:
                self.finish_alone(args, g, c, sums_g, po, pg, pm, nw)
                pm[0] = Int32(0)
            else:
                self.group_sync(g)
            if cutlass.const_expr(self.trace):
                tb = clock64()
                tr[1] = tr[1] + (tb - tr[6])
                tr[6] = tb
            gate = cute.make_tensor(
                cute.recast_ptr(at(block, G_AT), dtype=BFloat16), cute.make_layout(T * K)
            )
            for t in cutlass.range_constexpr(T):
                pg[t] = Float32(gate[t * K + c])
            if c == 0:
                if jj + 1 < mine:
                    self.load_block(args, table, j + G, opb, opb_full, g, 1 - buf)
            if slot > 0:
                stored = pending > 0
                if cutlass.const_expr("store" in self.skip):
                    stored = pending < 0
                acc = cute.make_rmem_tensor((QROWS, NV), Float32)
                self.scan_q(acc, block, lr, lfull, ring, wid, warp, lane, pending, l,
                            b, base, slot, head, stored, jj, mine, table, g, tmap, tr)  # fmt: skip
                if cutlass.const_expr(self.trace):
                    tb = clock64()
                    tr[4] = tr[4] + (tb - tr[6])
                    tr[6] = tb
                values = self.reduce_q(acc, l)
                self.core_solve(values, block, c, args[18][rec, head, None, None], po)
                pm[0], pm[1], pm[2] = Int32(1), token, head
                if cutlass.const_expr(self.trace):
                    tr[5] = tr[5] + (clock64() - tr[6])
            else:  # a padded item: pass its (empty) sub-blocks on
                for s in cutlass.range_constexpr(4):
                    m = 4 * jj + s
                    self.bwait(lfull + wid * NS + m % NS, (m // NS) % 2, "lfull")
                    cute.arch.sync_warp()
                    if lane == 0:
                        if m + NS < 4 * mine:
                            self.load_q(lr, lfull, wid, m + NS, table,
                                        g + G * ((m + NS) // 4), (m + NS) % 4, tmap)  # fmt: skip
                    cute.arch.sync_warp()
                for t in cutlass.range_constexpr(T):
                    self.store_output(args, token + t, head, c, BFloat16(0))
        if pm[0] != 0:
            self.finish_alone(args, g, c, sums_g, po, pg, pm, nw)
        if cutlass.const_expr(self.trace):  # noqa: SIM102
            if c % 32 == 0:
                cta, _, _ = cute.arch.block_idx()
                if cta % 53 == 0:
                    cute.printf("QTRACE cta {} g {} w {} items {} pro {} total {} opbw {} fin {} pagew {} storew {} scan {} solve {}", cta, g, c // 32, mine, t0 - k0, clock64() - t0, tr[0], tr[1], tr[2], tr[3], tr[4], tr[5])  # noqa: E501

    @cute.jit
    def scan_q(self, acc, block, lr, lfull, ring, wid, warp, lane, pending, l, b, base,
               slot, head, stored, jj, mine, table, g, tmap, tr):  # fmt: skip
        T: cutlass.Constexpr = self.tokens
        NV: cutlass.Constexpr = 2 * T
        G: cutlass.Constexpr = self.G
        NS: cutlass.Constexpr = self.nsub
        SB: cutlass.Constexpr = self.sb
        start = Float32(0)
        if pending == 0:
            start = Float32(-0.0)
        us = cute.make_rmem_tensor((QROWS, T), Float32)
        for rr in cutlass.range_constexpr(QROWS):
            row = base + (rr ^ l)
            for t in cutlass.range_constexpr(T):
                us[rr, t] = block[U_AT + t * K + row]
        for rr in cutlass.range_constexpr(QROWS):
            for m in cutlass.range_constexpr(NV):
                acc[rr, m] = Float32(0)
        flip = b & 1
        for ch in cutlass.range_constexpr(QCHUNKS):
            s: cutlass.Constexpr = ch // 2
            m = 4 * jj + s
            box = ring[wid, m % SB, None]
            if cutlass.const_expr(ch % 2 == 0):
                w0 = Int64(0)
                if cutlass.const_expr(self.trace):
                    w0 = clock64()
                self.bwait(lfull + wid * NS + m % NS, (m // NS) % 2, "lfull")
                if cutlass.const_expr(self.trace):
                    tr[2] = tr[2] + (clock64() - w0)
                if cutlass.const_expr("tma" not in self.skip):  # noqa: SIM102
                    if stored:
                        if lane == 0:  # the box's previous store has read it
                            if cutlass.const_expr(SB == 1):
                                wait_group_read0()
                            else:
                                core.wait_group_read1()
                        cute.arch.sync_warp()
            sub = lr[wid, m % NS, None]
            self.chunk_q(acc, us, block, sub, box, l, base, warp, start, ch, flip,
                         stored)  # fmt: skip
            if cutlass.const_expr(ch % 2 == 1):  # the sub-block is read; the box full
                fence_async_shared()
                cute.arch.sync_warp()
                if lane == 0:
                    if cutlass.const_expr("tma" not in self.skip):  # noqa: SIM102
                        if stored:
                            tma_store4(tmap, at(box, 0), Int32(BOX * s),
                                       Int32(32 * warp), head, slot)  # fmt: skip
                            bulk_commit()
                    if m + NS < 4 * mine:
                        self.load_q(lr, lfull, wid, m + NS, table,
                                    g + G * ((m + NS) // 4), (m + NS) % 4, tmap)  # fmt: skip
                cute.arch.sync_warp()

    @cute.jit
    def chunk_q(self, acc, us, block, sub, box, l, base, warp, start, ch, flip, stored):
        T: cutlass.Constexpr = self.tokens
        NV: cutlass.Constexpr = 2 * T
        cq = ch ^ flip  # this lane's chunk (the pair's other one for odd b)
        quad = l + 4 * cq  # 4-key group index: keys 4 quad + [0, 4)
        key = 4 * quad
        half = cq % 2  # the chunk's 16-key half of the 32-key sub-block
        d4 = load_shared_f32x4(at(block, D_AT + key))
        w4 = [load_shared_f32x4(at(block, W_AT + t * K + key)) for t in range(T)]
        xs = cute.make_rmem_tensor((4, NV), Float32)
        lane8, chunk8 = quad % 8, quad // 8  # KdaFlow's swizzled scan-vector order
        for e in cutlass.range_constexpr(4):
            for h in cutlass.range_constexpr(NV // 4):
                pos = (((chunk8 * 4 + e) * (NV // 4) + h) * 8 + lane8) * 4
                x4 = load_shared_f32x4(at(block, X_AT + pos))
                for i in cutlass.range_constexpr(4):
                    xs[e, 4 * h + i] = x4[i]
        for rr in cutlass.range_constexpr(QROWS):
            row = base + (rr ^ l)
            local = (row - 32 * warp) * BOX + 16 * half + 4 * l
            sv = load_shared_f32x4(at(sub, local))
            new = [Float32(0) for _ in range(4)]
            for p in cutlass.range_constexpr(2):
                fc = (start, start)
                for i in cutlass.range_constexpr(T):
                    t = T - 1 - i
                    fc = self.pfma((us[rr, t], us[rr, t]),
                                   (w4[t][2 * p], w4[t][2 * p + 1]), fc)  # fmt: skip
                pair = self.pfma((sv[2 * p], sv[2 * p + 1]),
                                 (d4[2 * p], d4[2 * p + 1]), fc)  # fmt: skip
                new[2 * p], new[2 * p + 1] = pair
            for e in cutlass.range_constexpr(4):
                for p in cutlass.range_constexpr(NV // 2):
                    pair = (acc[rr, 2 * p], acc[rr, 2 * p + 1])
                    pair = self.pfma((new[e], new[e]),
                                     (xs[e, 2 * p], xs[e, 2 * p + 1]), pair)  # fmt: skip
                    acc[rr, 2 * p], acc[rr, 2 * p + 1] = pair
            if stored:
                store_shared_f32x4(at(box, local), new)

    @cute.jit
    def reduce_q(self, acc, l):
        """4 lanes' partials -> lane l holds the 2q sums of row 4 b + l."""
        NV: cutlass.Constexpr = 2 * self.tokens
        rows = [[acc[rr, m] for m in range(NV)] for rr in range(QROWS)]
        for stage in cutlass.range_constexpr(2):
            offset: cutlass.Constexpr = 2 >> stage
            half: cutlass.Constexpr = len(rows) // 2
            rows = [
                [rows[i][m] + cute.arch.shuffle_sync_bfly(rows[i + half][m], offset)
                 for m in range(NV)]
                for i in range(half)
            ]  # fmt: skip
        return rows[0]
