# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KdaCore: KdaFlow's state pass (fold + verify scan, solve, gated norm) on prepared
operand blocks, with the checkpoint leaving through a per-warp shared-memory store
ring and TMA tensor stores (no prep, no register-path STG).

The prep (convolution, norms, decays, couplings, fold weights) runs before this
kernel and leaves one operand block per item (``OPB_FLOATS`` FP32 words, head-major
item rows):

    [0, 2TK)        the 2q scan vectors G_t k_t, G_t q_t (KdaFlow's swizzled order)
    [2TK, 3TK)      v (FP32 of the BF16 values)
    [3TK, 4TK)      the fold weights w_t
    [4TK, 5TK)      the pending corrections u_t (masked: +0 / -0 for t >= n)
    [5TK, 5TK + K)  D_n
    then T K BF16 output gates, T x T couplings, T betas

Per item (group g = 4 warps, thread (b, l) = (c // 8, c % 8), KdaFlow's scan layout):
the operand block arrives by one bulk copy (double-buffered, one item ahead); the
state page comes through KdaFlow's 3-slot page ring (each warp refills its quarter
after its scan); the folded checkpoint goes register -> store ring (4 KiB per warp
and chunk, two per warp) -> one TMA tensor store per warp and chunk, so the scan warps
never stall on the global store queue. Acc slot rr of lane l holds row
8 b + (rr ^ l): the transposed butterfly needs no selects and adds the same partials
in the same tree as KdaFlow (bit for bit), and the ring writes stay bank-conflict
free (lanes of a phase hit 8 distinct 16-byte bank groups).

Math: KdaFlow's, bit for bit (scan, fold, solve, the gated RMSNorm with the same
block reduction tree)."""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64

from ..common.primitives import alloc, asm, at, launch, zeros
from . import flow
from .flow import (
    CHUNKS,
    GROUP_THREADS,
    GROUPS,
    HEAD,
    LANES,
    PENDING,
    RECORD,
    REQUEST,
    ROWS,
    SLOT,
    SLOTS,
    SUMS,
    THREADS,
    K,
    META,
    MAX_ITEMS,
    couplings,
    store_shared_f32x4,
)
from .flow import clock64
from .specialized import (
    PAGE_BYTES,
    COPY_BYTES,
    fence_async_shared,
    load_shared_f32x4,
)

TOKENS = 4
OPB_FLOATS = 5 * TOKENS * K + K + TOKENS * K // 2 + TOKENS * TOKENS + TOKENS
OPB_BYTES = OPB_FLOATS * 4
X_AT, V_AT, W_AT, U_AT = 0, 2 * TOKENS * K, 3 * TOKENS * K, 4 * TOKENS * K
D_AT = 5 * TOKENS * K
G_AT = D_AT + K  # BF16 pairs
CE_AT = G_AT + TOKENS * K // 2
BETA_AT = CE_AT + TOKENS * TOKENS
WARPS = THREADS // 32
BOX = 32  # TMA box: 32 rows x 32 keys FP32 (4 KiB)
STORE_BYTES = BOX * BOX * 4
# The state stream: per warp a ring of NSUB sub-blocks (its 32 rows x 64 keys, two
# boxes, 8 KiB); a warp holds only the sub-block it scans, the rest are in flight.
NSUB = 3
SUB_FLOATS = 2 * BOX * BOX


@asm()
def tma_load4(dst, tensor_map, barrier, c0, c1, c2, c3):
    code = "cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx"
    code += "::bytes [$0], [$1, {$2, $3, $4, $5}], [$6];"
    return code, "r,l,r,r,r,r,r,~{memory}", (dst, tensor_map, c0, c1, c2, c3, barrier)


@asm()
def tma_store4(tensor_map, src, c0, c1, c2, c3):
    code = "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
    code += " [$0, {$2, $3, $4, $5}], [$1];"
    return code, "l,r,r,r,r,r,~{memory}", (tensor_map, src, c0, c1, c2, c3)


@asm()
def bulk_commit():
    return "cp.async.bulk.commit_group;", "~{memory}", ()


@asm()
def wait_group_read1():
    return "cp.async.bulk.wait_group.read 1;", "~{memory}", ()


@asm()
def wait_group_all():
    return "cp.async.bulk.wait_group 0;", "~{memory}", ()


@asm()
def prefetch_tensormap(address):
    return "prefetch.tensormap [$0];", "l", (address,)


def shared_bytes():
    pages = WARPS * NSUB * SUB_FLOATS * 4
    stores = WARPS * 2 * STORE_BYTES
    blocks = GROUPS * 2 * OPB_BYTES
    tables = MAX_ITEMS * (META + 8) * 4 + GROUPS * SUMS * 16
    return pages + stores + blocks + tables + 64 * 8 + 4096


def state_map(state):
    """The 4-D TMA map of the FP32 state cache [slots, heads, K, K] (keys, rows,
    heads, slots; box 32 x 32), a 128-byte device tensor."""
    from cuda.bindings import driver as cuda

    from ..common.host import tensor_maps

    assert state.dtype == torch.float32 and state.stride(-1) == 1
    assert state.stride(-2) == K and state.stride(1) == K * K
    dims = (K, K, state.shape[1], state.shape[0])
    strides = (K * 4, K * K * 4, state.stride(0) * 4)
    spec = (
        state.data_ptr(),
        cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        dims,
        strides,
        (BOX, BOX, 1, 1),
        cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
        cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
    )
    return tensor_maps([spec], state.device)


class KdaCore(flow.KdaFlow):
    """KdaFlow's arguments, then (after the deferred, die and mx operands) the operand
    blocks [items, OPB_FLOATS] FP32 and the state cache's TMA map (``state_map``)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.tokens == TOKENS
        self.opb_at = self.mx_at + 2 * int(self.mx)

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.ctas, THREADS, min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, args):
        self.body(args)

    @cute.jit
    def body(self, args):
        k0 = Int64(0)
        if cutlass.const_expr(self.trace):
            k0 = clock64()
        T: cutlass.Constexpr = self.tokens
        H: cutlass.Constexpr = self.heads
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        requests: cutlass.Constexpr = args[10].shape[0] // T
        total_items: cutlass.Constexpr = H * requests
        first = Int32(total_items * cta // self.ctas)
        count = Int32(total_items * (cta + 1) // self.ctas) - first

        smem = cutlass.utils.SmemAllocator()
        lr = alloc(smem, Float32, (WARPS, NSUB, SUB_FLOATS), align=1024)
        scratch = cute.make_tensor(
            cute.recast_ptr(at(lr, 0, 0, 0), dtype=Int32), cute.make_layout(1024)
        )
        if cutlass.const_expr(self.dies):  # die-matched items
            count = self.die_claim(args, tid, cta, scratch)
        ring = alloc(smem, Float32, (WARPS, 2, BOX * BOX), align=128)
        opb = alloc(smem, Float32, (GROUPS, 2, OPB_FLOATS), align=128)
        table = alloc(smem, Int32, (MAX_ITEMS, META))
        sums = alloc(smem, Float32, (GROUPS, SUMS, 4))
        bars = smem.allocate(cutlass.Uint64, WARPS * NSUB + 2 * GROUPS)
        lfull, opb_full = bars, bars + WARPS * NSUB
        if tid == 0:
            for s in cutlass.range_constexpr(WARPS * NSUB):
                cute.arch.mbarrier_init(lfull + s, 1)
            for b in cutlass.range_constexpr(2 * GROUPS):
                cute.arch.mbarrier_init(opb_full + b, 1)
            cute.arch.mbarrier_init_fence()
        h0 = first // requests
        if cutlass.const_expr(self.dies):
            h0 = scratch[flow.SC_MINE] // requests
        if tid < count:
            item = first + tid
            if cutlass.const_expr(self.dies):
                item = scratch[flow.SC_MINE + tid]
            head, request, slot, conv_slot, conv_offset, pending, rec = self.item_info(
                args, item
            )
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
        mine = (count - g + 1) // 2
        tmap = args[self.opb_at + 1].iterator
        # The state does not depend on the predecessor: the first sub-blocks go first.
        if tid % 32 == 0:
            prefetch_tensormap(tmap)
            for m in cutlass.range_constexpr(NSUB):
                if m < 2 * mine:
                    self.load_sub(lr, lfull, wid, m, table, g + GROUPS * (m // 2),
                                  m % 2, tmap)  # fmt: skip
        cute.arch.sync_warp()
        cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(not self.mx):
            cute.arch.griddepcontrol_launch_dependents()
        if c == 0:
            if g < count:
                self.load_block(args, table, g, opb, opb_full, g, 0)
        cute.arch.sync_warp()
        self.core_items(args, g, c, count, lr, ring, opb, table, sums, lfull,
                        opb_full, k0)  # fmt: skip
        if tid % 32 == 0:
            wait_group_all()  # the checkpoint stores have left shared memory

    @cute.jit
    def load_block(self, args, table, j, opb, opb_full, g, buf):
        """Item j's operand block into group g's buffer ``buf`` (one bulk copy)."""
        T: cutlass.Constexpr = self.tokens
        requests: cutlass.Constexpr = args[10].shape[0] // T
        item = table[j, HEAD] * requests + table[j, REQUEST]
        barrier = opb_full + 2 * g + buf
        cute.arch.mbarrier_arrive_and_expect_tx(barrier, OPB_BYTES)
        flow.copy_g2s(at(opb, g, buf, 0), at(args[self.opb_at], item, 0), OPB_BYTES,
                      barrier)  # fmt: skip

    @cute.jit
    def load_sub(self, lr, lfull, wid, m, table, j, half, tmap):
        """Sub-block m of warp ``wid``'s stream (item j's rows 32 (wid % 4) + [0, 32),
        keys 64 half + [0, 64)) into ring slot m % NSUB; a padded item arrives empty."""
        slot, head = table[j, SLOT], table[j, HEAD]
        barrier = lfull + wid * NSUB + m % NSUB
        if slot > 0:
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, SUB_FLOATS * 4)
            for b in cutlass.range_constexpr(2):
                tma_load4(at(lr, wid, m % NSUB, b * BOX * BOX), tmap, barrier,
                          Int32(64 * half + BOX * b), Int32(32 * (wid % 4)), head,
                          slot)  # fmt: skip
        else:
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, 0)

    @cute.jit
    def core_items(self, args, g, c, count, lr, ring, opb, table, sums, lfull,
                   opb_full, k0):  # fmt: skip
        state = args[9]
        T: cutlass.Constexpr = self.tokens
        NV: cutlass.Constexpr = 2 * T
        lane, warp = c % 32, c // 32
        lane_in, base = c % LANES, (c // LANES) * ROWS
        own = base + lane_in
        nw = args[12][c]
        mine = (count - g + 1) // 2
        po = cute.make_rmem_tensor(T, Float32)
        pg = cute.make_rmem_tensor(T, Float32)
        pm = cute.make_rmem_tensor(3, Int32)
        for t in cutlass.range_constexpr(T):
            po[t] = Float32(0)
        pm[0] = Int32(0)
        sums_g = sums[g, None, None]
        wid = 4 * g + warp
        tr = zeros(8, Int64)  # TRACE: opb wait, finish, page wait, store wait, scan, solve
        t0 = Int64(0)
        if cutlass.const_expr(self.trace):
            t0 = clock64()
        for jj in cutlass.range(mine, unroll=1):
            j = g + GROUPS * jj
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
            if pm[0] != 0:  # the previous item's gated norm (its own block reduction)
                self.finish_alone(args, g, c, sums_g, po, pg, pm, nw)
                pm[0] = Int32(0)
            else:
                self.group_sync(g)
            if cutlass.const_expr(self.trace):
                tb = clock64()
                tr[1] = tr[1] + (tb - tr[6])
                tr[6] = tb
            for t in cutlass.range_constexpr(T):  # this item's output gates
                gate = cute.make_tensor(
                    cute.recast_ptr(at(block, G_AT), dtype=BFloat16),
                    cute.make_layout(T * K),
                )
                pg[t] = Float32(gate[t * K + c])
            if c == 0:  # the group's next item's block (the other buffer is free)
                if jj + 1 < mine:
                    self.load_block(args, table, j + GROUPS, opb, opb_full, g, 1 - buf)
            if slot > 0:
                stored = pending > 0
                if cutlass.const_expr("store" in self.skip):  # diagnostics
                    stored = pending < 0
                acc = cute.make_rmem_tensor((ROWS, NV), Float32)
                self.core_scan(args, acc, block, lr, lfull, ring, wid, warp, lane,
                               pending, lane_in, base, slot, head, stored, jj, mine,
                               table, g, tr)  # fmt: skip
                if cutlass.const_expr(self.trace):
                    tb = clock64()
                    tr[4] = tr[4] + (tb - tr[6])
                    tr[6] = tb
                values = self.xor_reduce(acc, lane_in)
                self.core_solve(values, block, own, args[18][rec, head, None, None], po)
                if cutlass.const_expr(self.trace):
                    tr[5] = tr[5] + (clock64() - tr[6])
                pm[0], pm[1], pm[2] = Int32(1), token, head
            else:  # a padded item: pass its (empty) sub-blocks on
                tmap = args[self.opb_at + 1].iterator
                for half in cutlass.range_constexpr(2):
                    m = 2 * jj + half
                    self.bwait(lfull + wid * NSUB + m % NSUB, (m // NSUB) % 2, "lfull")
                    cute.arch.sync_warp()
                    if lane == 0:
                        if m + NSUB < 2 * mine:
                            self.load_sub(lr, lfull, wid, m + NSUB, table,
                                          g + GROUPS * ((m + NSUB) // 2),
                                          (m + NSUB) % 2, tmap)  # fmt: skip
                    cute.arch.sync_warp()
                for t in cutlass.range_constexpr(T):
                    self.store_output(args, token + t, head, c, BFloat16(0))
        if pm[0] != 0:  # the group's last item
            self.finish_alone(args, g, c, sums_g, po, pg, pm, nw)
        if cutlass.const_expr(self.trace):  # noqa: SIM102
            if c % 32 == 0:
                cta, _, _ = cute.arch.block_idx()
                if cta % 53 == 0:
                    cute.printf("CTRACE cta {} g {} w {} items {} pro {} total {} opbw {} fin {} pagew {} storew {} scan {} solve {}", cta, g, c // 32, mine, t0 - k0, clock64() - t0, tr[0], tr[1], tr[2], tr[3], tr[4], tr[5])  # noqa: E501

    @cute.jit
    def core_scan(self, args, acc, block, lr, lfull, ring, wid, warp, lane, pending,
                  lane_in, base, slot, head, stored, jj, mine, table, g, tr):  # fmt: skip
        """KdaFlow's scan over this thread's elements (acc slot rr = row
        base + (rr ^ lane_in)); the committed elements go to the warp's store ring and
        leave by one TMA tensor store per chunk (32 rows x 32 keys)."""
        T: cutlass.Constexpr = self.tokens
        NV: cutlass.Constexpr = 2 * T
        start = Float32(0)
        if pending == 0:
            start = Float32(-0.0)  # S + (-0) == S for every S, -0 included
        us = cute.make_rmem_tensor((ROWS, T), Float32)
        for rr in cutlass.range_constexpr(ROWS):
            row = base + (rr ^ lane_in)
            for t in cutlass.range_constexpr(T):
                us[rr, t] = block[U_AT + t * K + row]
        for rr in cutlass.range_constexpr(ROWS):
            for m in cutlass.range_constexpr(NV):
                acc[rr, m] = Float32(0)
        tmap = args[self.opb_at + 1].iterator
        for ch in cutlass.range_constexpr(CHUNKS):
            box = ring[wid, ch % 2, None]
            m = 2 * jj + ch // 2
            if cutlass.const_expr(ch % 2 == 0):  # this half's sub-block has landed
                w0 = Int64(0)
                if cutlass.const_expr(self.trace):
                    w0 = clock64()
                self.bwait(lfull + wid * NSUB + m % NSUB, (m // NSUB) % 2, "lfull")
                if cutlass.const_expr(self.trace):
                    tr[2] = tr[2] + (clock64() - w0)
            sub = lr[wid, m % NSUB, None]
            if cutlass.const_expr("tma" not in self.skip):  # noqa: SIM102
                if stored:
                    w1 = Int64(0)
                    if cutlass.const_expr(self.trace):
                        w1 = clock64()
                    if lane == 0:  # the ring buffer's previous store has read it
                        if cutlass.const_expr("nowait" not in self.skip):
                            wait_group_read1()
                    cute.arch.sync_warp()
                    if cutlass.const_expr(self.trace):
                        tr[3] = tr[3] + (clock64() - w1)
            self.core_chunk(acc, us, block, sub, box, lane_in, base, warp, start, ch,
                            stored)  # fmt: skip
            if cutlass.const_expr(ch % 2 == 1):  # the sub-block is read: refill it
                fence_async_shared()
                cute.arch.sync_warp()
                if lane == 0:
                    if m + NSUB < 2 * mine:
                        self.load_sub(lr, lfull, wid, m + NSUB, table,
                                      g + GROUPS * ((m + NSUB) // 2), (m + NSUB) % 2,
                                      tmap)  # fmt: skip
            if cutlass.const_expr("tma" not in self.skip):  # noqa: SIM102
                if stored:
                    fence_async_shared()
                    cute.arch.sync_warp()
                    if lane == 0:
                        tma_store4(tmap, at(box, 0), Int32(BOX * ch), Int32(32 * warp),
                                   head, slot)  # fmt: skip
                        bulk_commit()
            cute.arch.sync_warp()

    @cute.jit
    def core_chunk(self, acc, us, block, sub, box, lane_in, base, warp, start, ch,
                   stored):  # fmt: skip
        T: cutlass.Constexpr = self.tokens
        NV: cutlass.Constexpr = 2 * T
        key = 4 * (lane_in + LANES * ch)
        d4 = load_shared_f32x4(at(block, D_AT + key))
        w4 = [load_shared_f32x4(at(block, W_AT + t * K + key)) for t in range(T)]
        xs = cute.make_rmem_tensor((4, NV), Float32)
        for e in cutlass.range_constexpr(4):
            for h in cutlass.range_constexpr(NV // 4):
                pos = (((ch * 4 + e) * (NV // 4) + h) * LANES + lane_in) * 4
                x4 = load_shared_f32x4(at(block, X_AT + pos))
                for i in cutlass.range_constexpr(4):
                    xs[e, 4 * h + i] = x4[i]
        for rr in cutlass.range_constexpr(ROWS):
            row = base + (rr ^ lane_in)
            local = (row - 32 * warp) * BOX + 4 * lane_in
            sv = load_shared_f32x4(at(sub, (ch % 2) * BOX * BOX + local))
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
    def xor_reduce(self, acc, lane_in):
        """The 8 lanes' partials -> lane l holds the 2q sums of row base + l (slot rr
        of lane l is row base + (rr ^ l): no selects; KdaFlow's tree)."""
        NV: cutlass.Constexpr = 2 * self.tokens
        rows = [[acc[rr, m] for m in range(NV)] for rr in range(ROWS)]
        for stage in cutlass.range_constexpr(3):
            offset: cutlass.Constexpr = 4 >> stage
            half: cutlass.Constexpr = len(rows) // 2
            merged = []
            for i in cutlass.range_constexpr(half):
                merged.append([
                    rows[i][m] + cute.arch.shuffle_sync_bfly(rows[i + half][m], offset)
                    for m in range(NV)
                ])  # fmt: skip
            rows = merged
        return rows[0]

    @cute.jit
    def core_solve(self, values, block, own, u_dst, po):
        """KdaFlow's ``solve`` on the operand block's v, couplings and betas."""
        T: cutlass.Constexpr = self.tokens
        index = couplings(T)
        u = [Float32(0) for _ in range(T)]
        vs = [block[V_AT + t * K + own] for t in range(T)]
        for t in cutlass.range_constexpr(T):
            dot = values[t]
            for s in cutlass.range_constexpr(t):
                dot = flow.fma(block[CE_AT + index[("c", t, s)]], u[s], dot)
            u[t] = (vs[t] - dot) * block[BETA_AT + t]
            u_dst[t, own] = u[t]
        for t in cutlass.range_constexpr(T):
            o = values[T + t]
            for s in cutlass.range_constexpr(t + 1):
                o = flow.fma(block[CE_AT + index[("e", t, s)]], u[s], o)
            po[t] = Float32(BFloat16(o))
