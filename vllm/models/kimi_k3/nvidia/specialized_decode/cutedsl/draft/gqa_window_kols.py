# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``GqaWindowKol`` as a persistent stream-K grid (``GqaWindowKols``): one CTA per SM,
each CTA an equal share of the window's 128-key stages.

``GqaWindowKol`` (one (request, KV head) unit per CTA) streams a stage faster than an
SM's share of HBM, so its layer time is set by the units-per-SM granularity (B32: 128
units leave 84 SMs idle; B64 / B96: 44 / 172 SMs carry two units) and by the
per-launch setup, ramp and epilogue. Here every CTA takes the global stages [c T / C,
(c + 1) T / C) of the request-major (request, KV head, stage) list (T stages, C CTAs):
its range is a few contiguous pieces of units, streamed through the same K / V rings
and softmax warpgroups without draining between pieces.

Warps: 0-3 the softmax warpgroup (one keeps up with an SM's HBM share once every SM
streams), 4 PV, 5 S^T, 6 / 7 the K / V TMA rings (as ``GqaWindowKol``), 8-11 the piece
warpgroup: per piece (slot k % 2) it loads the block-table entries (the TMA warps
start on them), the Q^T tile and the unit's new K / V rows (+ the cache insert, by the
piece holding the unit's last stage), then after the piece the epilogue: a whole unit
is stored; the other
pieces of a split unit leave unnormalized partials (O [64][32] + m + l, FP32) with a
release flag, and the piece holding the unit's first stage (the last piece of its CTA,
so it finishes last) folds them in while its own piece still streams, then its own,
and stores. The merge is the same LSE combine as the split kernels'.
"""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_stream

from ..common.host import dynamic_fake, fake
from ..common.primitives import (
    alloc,
    asm,
    bf16,
    bf16x2,
    fence_tmem,
    fma,
    free_tmem,
    launch,
    load_words,
    named_barrier,
    shared_address,
    store_words,
    wait,
)
from .gqa_window import (
    HEAD,
    LOG2E,
    NEG_INF,
    arm,
    arrive,
    ex2,
    fence_async_shared,
    global_timer,
    init_barrier,
    load_i32,
    load_i64,
    norm_rope8,
    swz,
    tree,
)
from .gqa_window_kol import (
    BAR_GROW,
    BAR_SM,
    BOX,
    KB,
    KS,
    KT,
    NPB,
    NSB,
    O_COL,
    POS_INF,
    PTB,
    PV_INSTR,
    S_INSTR,
    THRESH,
    TMEM_COLS,
    GqaWindowKol,
    bar_red_or,
    desc_sw64,
    ld32,
    mma_pv8,
    mma_s4,
    shfl_xor,
    st32,
    tma_box,
)
from .gqa_window_umma import alloc_tmem, desc_sw128, fmax

BMAX = 256  # requests per launch
PMAX = 48  # pieces per CTA
THREADS = 384
PVW, SWARP, KTMA, VTMA, EPW = 4, 5, 6, 7, 8
BAR_EP = 6  # the piece warpgroup
STATS = 160  # FP32 per piece slot: l [4 warps][32], m [32]
PSZ = 64 * 32 + 64  # FP32 per partial: O [64 dims][32 rows], m [32], l [32]


@asm()
def commit1(barrier):
    """tcgen05.commit by the elected lane (the lane that issued the MMAs)."""
    commit = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
    code = "{ .reg .pred e; elect.sync _|e,0xffffffff; "
    code += f"@e {commit} [$0]; }}"
    return code, "r", (Uint32(barrier),)


@asm()
def st_release(address, value):
    return "st.release.gpu.global.b32 [$0], $1;", "l,r,~{memory}", (
        Int64(address),
        Int32(value),
    )


@asm(Int32)
def ld_acquire(address):
    return "ld.acquire.gpu.global.b32 $0, [$1];", "=r,l,~{memory}", (Int64(address),)


@asm()
def fence_gpu():
    return "fence.acq_rel.gpu;", "~{memory}", ()


@asm()
def st_f32(address, value):
    return "st.global.f32 [$0], $1;", "l,f,~{memory}", (Int64(address), Float32(value))


@asm(Float32)
def ld_f32(address):
    return "ld.relaxed.gpu.global.f32 $0, [$1];", "=f,l,~{memory}", (Int64(address),)


@asm()
def ld4_f32(address):
    code = "ld.relaxed.gpu.global.v4.f32 {$0,$1,$2,$3}, [$4];"
    return code, "=f,=f,=f,=f,l,~{memory}", (Int64(address),), (Float32,) * 4


@asm(Int32)
def shfl_up_i32(value, offset):
    code = "shfl.sync.up.b32 $0, $1, $2, 0, 0xffffffff;"
    return code, "=r,r,r", (Int32(value), Int32(offset))


@asm()
def st_bf16(address, value):
    code = "{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; st.global.b16 [$0], h; }"
    return code, "l,f,~{memory}", (Int64(address), Float32(value))


class GqaWindowKols(GqaWindowKol):
    """The persistent stream-K form of ``GqaWindowKol`` (module notes) on ``sms``
    CTAs; arguments as ``GqaWindowKol`` + the partials and flags buffers."""

    def __init__(self, group, kv_heads, q, page, ring, window, sms, v_scale=None,
                 round_norm=False, debug=0):  # fmt: skip
        super().__init__(group, kv_heads, q, page, ring, window, v_scale, round_norm,
                         debug)  # fmt: skip
        self.sms = sms
        self.threads = THREADS

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, (self.sms, 1, 1), THREADS,
               max_number_threads=[THREADS, 1, 1])  # fmt: skip

    @cute.jit
    def issue_b(self, kv_map, bar, tile, kv: cutlass.Constexpr[int], head, blocks, boff,
                page_lo, box0, last_box):  # fmt: skip
        """``issue`` with the piece slot's block-table entries at ``blocks[boff:]``."""
        N: cutlass.Constexpr = self.page
        two = box0 + 1 <= last_box
        nbytes = Int32(BOX)
        if two:
            nbytes = Int32(2 * BOX)
        arm(bar, nbytes)
        token = box0 * KB
        tma_box(tile, kv_map, bar, token % N, kv, head,
                blocks[boff + token // N - page_lo])  # fmt: skip
        if two:
            token1 = token + KB
            tma_box(tile + BOX, kv_map, bar, token1 % N, kv, head,
                    blocks[boff + token1 // N - page_lo])  # fmt: skip

    @cute.jit
    def fixup_q(self, tile, kv: cutlass.Constexpr[int], p0, lane, lo, length, kv_new,
                newvalid, nvoff):  # fmt: skip
        """``fixup`` with the piece slot's new rows (kv_new, newvalid[nvoff:])."""
        q_first = length - self.q
        zero = (Uint32(0),) * 4
        for j in cutlass.range_constexpr(4):
            row = lane + 32 * j
            img = tile + (row // KB) * BOX
            rr = row % KB
            position = p0 + row
            if (position < lo) | (position >= length):
                for ch in cutlass.range_constexpr(8):
                    store_words(img + swz(rr, ch), zero, shared=True)
            elif position >= q_first:
                i = position - q_first
                if newvalid[nvoff + i] != 0:
                    for ch in cutlass.range_constexpr(8):
                        src = kv_new + i * 256 + 128 * kv + 16 * ch
                        store_words(img + swz(rr, ch),
                                    load_words(src, count=4, shared=True),
                                    shared=True)  # fmt: skip

    @cute.jit
    def edge(self, p0, lo, length):
        lo_max = max(length - self.window, Int32(0))
        q_first = length - self.q
        return (p0 < lo_max) | (p0 + KS > q_first) | (p0 < lo) | (p0 + KS > length)

    @cute.jit
    def locate(self, s, pre, srt, requests):
        """(request, KV head, stage) of global stage ``s`` (< T)."""
        HKV: cutlass.Constexpr = self.kv_heads
        lo_r = Int32(0)
        hi_r = requests
        for _ in cutlass.range_constexpr(9):
            if hi_r - lo_r > 1:
                mid = (lo_r + hi_r) // 2
                if HKV * pre[mid] <= s:
                    lo_r = mid
                else:
                    hi_r = mid
        rel = s - HKV * pre[lo_r]
        cnt = max(srt[lo_r], Int32(1))
        h = rel // cnt
        return lo_r, h, rel - h * cnt

    @cute.kernel
    def kernel(self, args):
        qkv, positions, slots, seq_lens, table, cos_sin, qw, kw, out = args[0:9]
        kv_map, kv_base, row_stride, head_stride = args[9:13]
        eps = args[14]
        partials, flags = args[15], args[16]
        Q: cutlass.Constexpr = self.q
        R: cutlass.Constexpr = self.ring
        RK: cutlass.Constexpr = self.kring
        N: cutlass.Constexpr = self.page
        NP: cutlass.Constexpr = self.pages
        ROWS: cutlass.Constexpr = self.rows
        HKV: cutlass.Constexpr = self.kv_heads
        C: cutlass.Constexpr = self.sms
        c = Float32(LOG2E / (HEAD**0.5))

        cta, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        requests = out.shape[0] // Q
        table_last = Int32(table.shape[1] - 1)
        blocks_max = args[13] - 1

        smem = cutlass.utils.SmemAllocator()
        kring = alloc(smem, cutlass.Uint8, RK * KT, align=1024)
        vring = alloc(smem, cutlass.Uint8, R * KT, align=1024)
        qtile = alloc(smem, cutlass.Uint8, 2 * 32 * 128, align=1024)
        ptile = alloc(smem, cutlass.Uint8, NPB * PTB, align=1024)
        newkv = alloc(smem, cutlass.Uint8, 2 * Q * 256, align=16)
        newvalid = alloc(smem, Int32, 2 * Q, align=16)
        blocks = alloc(smem, Int32, 2 * NP, align=16)
        lens = alloc(smem, Int32, BMAX, align=16)
        srt = alloc(smem, Int32, BMAX, align=16)
        pre = alloc(smem, Int32, BMAX + 1, align=16)
        pcs = alloc(smem, Int32, PMAX * 5 + 4, align=16)
        stats = alloc(smem, Float32, 2 * STATS + 128, align=16)
        nbar: cutlass.Constexpr = 2 * RK + 2 * R + 2 * NSB + 2 * NPB + 14
        bars = alloc(smem, cutlass.Int64, nbar, align=8)
        tslot = alloc(smem, Int32, 2, align=16)
        k_base = Int32(shared_address(kring.iterator))
        v_base = Int32(shared_address(vring.iterator))
        q_base = Int32(shared_address(qtile.iterator))
        pt_base = Int32(shared_address(ptile.iterator))
        kv_new = Int32(shared_address(newkv.iterator))
        kfull = Int32(shared_address(bars.iterator))
        kfree = kfull + 8 * RK
        vfull = kfree + 8 * RK
        vfree = vfull + 8 * R
        sready = vfree + 8 * R
        sfree = sready + 8 * NSB
        pready = sfree + 8 * NSB
        pvdone = pready + 8 * NPB
        prepped = pvdone + 8 * NPB  # [2] per piece slot
        pfree = prepped + 16
        pvpiece = pfree + 16
        ofree = pvpiece + 16
        sready_p = ofree + 16  # [2] stats ready
        sfree_p = sready_p + 16  # [2] stats consumed
        bready = sfree_p + 16  # [2] the piece's block-table entries loaded

        if tid == 0:
            for i in cutlass.range_constexpr(RK):
                init_barrier(kfull + 8 * i, 1)
                init_barrier(kfree + 8 * i, 1)
            for i in cutlass.range_constexpr(R):
                init_barrier(vfull + 8 * i, 1)
                init_barrier(vfree + 8 * i, 1)
            for b in cutlass.range_constexpr(NSB):
                init_barrier(sready + 8 * b, 1)
                init_barrier(sfree + 8 * b, 4)
            for b in cutlass.range_constexpr(NPB):
                init_barrier(pready + 8 * b, 4)
                init_barrier(pvdone + 8 * b, 1)
            for b in cutlass.range_constexpr(2):
                init_barrier(prepped + 8 * b, 1)
                init_barrier(pfree + 8 * b, 4)
                init_barrier(pvpiece + 8 * b, 1)
                init_barrier(ofree + 8 * b, 1)
                init_barrier(sready_p + 8 * b, 4)
                init_barrier(sfree_p + 8 * b, 1)
                init_barrier(bready + 8 * b, 1)
        # the Q^T tiles' padding rows ROWS..31 (both slots), P^T padding chunks
        PADQ: cutlass.Constexpr = 2 * (32 - ROWS) * 8
        CH: cutlass.Constexpr = self.chunks
        for it in cutlass.range_constexpr(-(-PADQ // THREADS)):
            t = it * THREADS + tid
            if t < PADQ:
                slot_q, rest = t // ((32 - ROWS) * 8), t % ((32 - ROWS) * 8)
                store_words(q_base + slot_q * 4096 + swz(ROWS + rest // 8, rest % 8),
                            (Uint32(0),) * 4, shared=True)  # fmt: skip
        if cutlass.const_expr(CH < 4):
            if tid < 128:
                for b in cutlass.range_constexpr(NPB):
                    for ch in cutlass.range_constexpr(CH, 4):
                        dst = pt_base + b * PTB + tid * 64
                        dst = dst + ((ch ^ ((tid >> 1) & 3)) << 4)
                        store_words(dst, (Uint32(0),) * 4, shared=True)
        cute.arch.mbarrier_init_fence()
        self.rec(cta, 0, 0, global_timer(), tid == 0)
        cute.arch.griddepcontrol_wait()
        if warp == PVW:
            alloc_tmem(Int32(shared_address(tslot.iterator)), TMEM_COLS)
        # ---- the tape: stages per request, prefixes, this CTA's pieces
        if tid < BMAX:
            ln = Int32(0)
            if tid < requests:
                ln = load_i32(seq_lens.iterator + tid)
            lens[tid] = ln
            _, _, _, cnt = self.stages(ln)
            srt[tid] = cnt
        cute.arch.sync_threads()
        if warp == 0:
            vals = [srt[lane * 8 + e] for e in range(8)]
            tot = Int32(0)
            for e in cutlass.range_constexpr(8):
                tot = tot + vals[e]
            inc = tot
            for sh in cutlass.range_constexpr(5):
                up = shfl_up_i32(inc, 1 << sh)
                if lane >= (1 << sh):
                    inc = inc + up
            run = inc - tot
            for e in cutlass.range_constexpr(8):
                pre[lane * 8 + e] = run
                run = run + vals[e]
            if lane == 31:
                pre[BMAX] = run
        cute.arch.sync_threads()
        if tid == 0:
            T = HKV * pre[BMAX]
            s0 = cta * T // C
            s1 = (cta + 1) * T // C
            sw_ = s0
            kpc = Int32(0)
            for _ in cutlass.range(PMAX, unroll=1):
                if sw_ < s1:
                    r_w, h_w, a_w = self.locate(sw_, pre, srt, requests)
                    n_w = min(srt[r_w] - a_w, s1 - sw_)
                    pcs[kpc * 5] = r_w
                    pcs[kpc * 5 + 1] = h_w
                    pcs[kpc * 5 + 2] = a_w
                    pcs[kpc * 5 + 3] = n_w
                    pcs[kpc * 5 + 4] = sw_ - s0
                    kpc = kpc + 1
                    sw_ = sw_ + n_w
            pcs[PMAX * 5] = kpc
            pcs[PMAX * 5 + 1] = T
            pcs[PMAX * 5 + 2] = s0
            pcs[PMAX * 5 + 3] = cta
        fence_tmem(before=True)
        cute.arch.sync_threads()
        fence_tmem()
        tmem = Uint32(tslot[0])
        npieces = pcs[PMAX * 5]
        T = pcs[PMAX * 5 + 1]
        self.rec(cta, 0, 2, global_timer(), tid == 0)

        if warp == PVW:
            for k in cutlass.range(npieces, unroll=1):
                q = k % 2
                r_p, h_p, a_p, n_p, js = (pcs[k * 5 + f] for f in range(5))
                length = lens[r_p]
                lo, first, last, _ = self.stages(length)
                wait(prepped + 8 * q, (k // 2) & 1, scope="cta")
                if k >= 2:
                    wait(ofree + 8 * q, ((k - 2) // 2) & 1, scope="cta")
                for i in cutlass.range(n_p, unroll=1):
                    j = js + i
                    b = j % NPB
                    wait(pready + 8 * b, (j // NPB) & 1, scope="cta")
                    v_i = j % R
                    tile_v = v_base + v_i * KT
                    wait(vfull + 8 * v_i, (j // R) & 1, scope="cta")
                    p0 = (first + 2 * (a_p + i)) * KB
                    if self.edge(p0, lo, length):
                        self.fixup_q(tile_v, 1, p0, lane, lo, length, kv_new + q * Q * 256,
                                     newvalid, q * Q)  # fmt: skip
                        fence_async_shared()
                        cute.arch.sync_warp()
                    fence_tmem()
                    mma_pv8(
                        tmem + Uint32(O_COL) + Uint32(32) * Uint32(q),
                        desc_sw128(tile_v, BOX), desc_sw64(pt_base + b * PTB),
                        Uint32(PV_INSTR), Int32(i >= 1), pvdone + 8 * b, vfree + 8 * v_i,
                    )  # fmt: skip
                commit1(pvpiece + 8 * q)
                cute.arch.sync_warp()
                if lane == 0:
                    arrive(pfree + 8 * q)
            self.rec(cta, 0, 3, global_timer(), lane == 0)
        elif warp == SWARP:
            for k in cutlass.range(npieces, unroll=1):
                q = k % 2
                r_p, h_p, a_p, n_p, js = (pcs[k * 5 + f] for f in range(5))
                length = lens[r_p]
                lo, first, last, _ = self.stages(length)
                q_desc = desc_sw128(q_base + q * 4096, 16)
                wait(prepped + 8 * q, (k // 2) & 1, scope="cta")
                for i in cutlass.range(n_p, unroll=1):
                    j = js + i
                    k_i = j % RK
                    tile_k = k_base + k_i * KT
                    wait(kfull + 8 * k_i, (j // RK) & 1, scope="cta")
                    p0 = (first + 2 * (a_p + i)) * KB
                    if self.edge(p0, lo, length):
                        self.fixup_q(tile_k, 0, p0, lane, lo, length, kv_new + q * Q * 256,
                                     newvalid, q * Q)  # fmt: skip
                        fence_async_shared()
                        cute.arch.sync_warp()
                    b = j % NSB
                    if j >= NSB:
                        wait(sfree + 8 * b, ((j - NSB) // NSB) & 1, scope="cta")
                    fence_tmem()
                    mma_s4(
                        tmem + Uint32(32) * Uint32(b), desc_sw128(tile_k, 16), q_desc,
                        Uint32(S_INSTR), sready + 8 * b, kfree + 8 * k_i,
                    )  # fmt: skip
                commit1(pfree + 8 * q)
        elif (warp == KTMA) | (warp == VTMA):
            for k in cutlass.range(npieces, unroll=1):
                q = k % 2
                r_p, h_p, a_p, n_p, js = (pcs[k * 5 + f] for f in range(5))
                length = lens[r_p]
                lo, first, last, _ = self.stages(length)
                page_lo = first * KB // N
                wait(bready + 8 * q, (k // 2) & 1, scope="cta")
                for i in cutlass.range(n_p, unroll=1):
                    j = js + i
                    if warp == KTMA:
                        if j >= RK:
                            wait(kfree + 8 * ((j - RK) % RK), ((j - RK) // RK) & 1,
                                 scope="cta")  # fmt: skip
                        if lane == 0:
                            self.issue_b(kv_map, kfull + 8 * (j % RK),
                                         k_base + (j % RK) * KT, 0, h_p, blocks, q * NP,
                                         page_lo, first + 2 * (a_p + i), last)  # fmt: skip
                    else:
                        if j >= R:
                            wait(vfree + 8 * ((j - R) % R), ((j - R) // R) & 1,
                                 scope="cta")  # fmt: skip
                        if lane == 0:
                            self.issue_b(kv_map, vfull + 8 * (j % R),
                                         v_base + (j % R) * KT, 1, h_p, blocks, q * NP,
                                         page_lo, first + 2 * (a_p + i), last)  # fmt: skip
                cute.arch.sync_warp()
                if lane == 0:
                    arrive(pfree + 8 * q)
        elif warp < 4:
            self.softmax_s(0, tmem, warp, lane, npieces, pcs, lens, c, sready, sfree,
                           pready, pvdone, sready_p, sfree_p, pt_base, stats)  # fmt: skip
        else:
            self.pieces_ep(
                tmem, cta, warp, lane, npieces, pcs, lens, srt, pre, requests, T, qkv,
                positions, slots, table, cos_sin, qw, kw, out, kv_base, row_stride,
                head_stride, eps, partials, flags, blocks, newvalid, kv_new, q_base,
                stats, prepped, pfree, pvpiece, ofree, sready_p, sfree_p, table_last,
                blocks_max, bready,
            )  # fmt: skip
        self.rec(cta, 0, 11, global_timer(), tid == 0)
        fence_tmem(before=True)
        cute.arch.sync_threads()
        if warp == PVW:
            fence_tmem()
            free_tmem(tmem, TMEM_COLS)

    @cute.jit
    def softmax_s(
        self, g: cutlass.Constexpr[int], tmem, warp, lane, npieces, pcs, lens, c, sready,
        sfree, pready, pvdone, sready_p, sfree_p, pt_base, stats,
    ):  # fmt: skip
        """Softmax warpgroup ``g``: per piece, the piece's stages of parity g (own m,
        row sums, O^T at O_COL + 32 (2 g + slot)); stats per piece slot."""
        G: cutlass.Constexpr = self.group
        Q: cutlass.Constexpr = self.q
        W: cutlass.Constexpr = self.window
        ROWS: cutlass.Constexpr = self.rows
        CH: cutlass.Constexpr = self.chunks
        BG: cutlass.Constexpr = BAR_GROW + 2 * g
        BS: cutlass.Constexpr = BAR_SM + 2 * g
        q4 = warp % 4
        lanes = Uint32(32 * q4) << 16
        key_l = 32 * q4 + lane
        sw = (key_l >> 1) & 3
        bias = cute.make_rmem_tensor(ROWS, Float32)
        lsum = cute.make_rmem_tensor(ROWS, Float32)
        xs = 2 * STATS  # the stage-max exchange
        for k in cutlass.range(npieces, unroll=1):
            q = k % 2
            r_p, h_p, a_p, n_p, js = (pcs[k * 5 + f] for f in range(5))
            length = lens[r_p]
            lo, first, last, _ = self.stages(length)
            q_first = length - Q
            lo_max = max(length - W, Int32(0))
            o_mine = tmem + lanes + Uint32(O_COL) + Uint32(32) * Uint32(q)
            for r in cutlass.range_constexpr(ROWS):
                bias[r] = Float32(POS_INF)
                lsum[r] = Float32(0)
            for ii in cutlass.range(n_p, unroll=1):
                i = ii
                t = js + i
                b = t % NSB
                wait(sready + 8 * b, (t // NSB) & 1, scope="cta")
                fence_tmem()
                raw = ld32(tmem + lanes + Uint32(32) * Uint32(b))
                fence_tmem(before=True)
                cute.arch.sync_warp()
                if lane == 0:
                    arrive(sfree + 8 * b)
                p0 = (first + 2 * (a_p + i)) * KB
                edge = (p0 < lo_max) | (p0 + KS > q_first)
                ok = cute.make_rmem_tensor(Q, Int32)
                for kq in cutlass.range_constexpr(Q):
                    ok[kq] = Int32(1)
                y = cute.make_rmem_tensor(ROWS, Float32)
                for r in cutlass.range_constexpr(ROWS):
                    y[r] = fma(raw[r], c, bias[r])
                if edge:
                    key = p0 + key_l
                    for kq in cutlass.range_constexpr(Q):
                        hi = q_first + kq
                        lo_k = max(hi - (W - 1), Int32(0))
                        ok[kq] = Int32((key >= lo_k) & (key <= hi))
                    for r in cutlass.range_constexpr(ROWS):
                        if ok[r // G] == 0:
                            y[r] = Float32(NEG_INF)
                top = tree(fmax, [y[r] for r in range(ROWS)])
                grow = Int32(top > Float32(THRESH))
                if bar_red_or(Uint32(BG), Uint32(128), grow) != 0:
                    for r in cutlass.range_constexpr(ROWS):
                        mv = raw[r] * c
                        if ok[r // G] == 0:
                            mv = Float32(NEG_INF)
                        for kk in cutlass.range_constexpr(5):
                            mv = fmax(mv, shfl_xor(mv, Int32(16 >> kk)))
                        if lane == 0:
                            stats[xs + 32 * q4 + r] = mv
                    named_barrier(BS, 128)
                    alpha = cute.make_rmem_tensor(ROWS, Float32)
                    for r in cutlass.range_constexpr(ROWS):
                        mx = fmax(
                            fmax(stats[xs + r], stats[xs + 32 + r]),
                            fmax(stats[xs + 64 + r], stats[xs + 96 + r]),
                        )  # fmt: skip
                        m_old = -bias[r]
                        m_new = fmax(m_old, mx)
                        alpha[r] = Float32(1)
                        if m_new > Float32(NEG_INF):
                            alpha[r] = ex2(m_old - m_new)
                            bias[r] = -m_new
                        lsum[r] = lsum[r] * alpha[r]
                        y[r] = fma(raw[r], c, bias[r])
                        if ok[r // G] == 0:
                            y[r] = Float32(NEG_INF)
                    if ii >= 1:
                        wait(pvdone + 8 * ((t - 1) % NPB), ((t - 1) // NPB) & 1,
                             scope="cta")  # fmt: skip
                        fence_tmem()
                        ox = ld32(o_mine)
                        scaled = [ox[r] * alpha[r] if r < ROWS else ox[r] for r in range(32)]
                        st32(o_mine, tuple(scaled))
                    named_barrier(BS, 128)
                ps = [ex2(y[r]) for r in range(ROWS)]
                for r in cutlass.range_constexpr(ROWS):
                    lsum[r] = lsum[r] + ps[r]
                pwords = [
                    bf16x2(ps[2 * kp], ps[2 * kp + 1] if 2 * kp + 1 < ROWS else Float32(0))
                    for kp in range(-(-ROWS // 2))
                ]  # fmt: skip
                pwords = pwords + [Uint32(0)] * (4 * CH - (ROWS + 1) // 2)
                bp = t % NPB
                if t >= NPB:
                    wait(pvdone + 8 * bp, ((t - NPB) // NPB) & 1, scope="cta")
                dst = pt_base + bp * PTB + key_l * 64
                for ch in cutlass.range_constexpr(CH):
                    store_words(dst + ((ch ^ sw) << 4), tuple(pwords[4 * ch : 4 * ch + 4]),
                                shared=True)  # fmt: skip
                fence_async_shared()
                fence_tmem(before=True)
                cute.arch.sync_warp()
                if lane == 0:
                    arrive(pready + 8 * bp)
            # ---- the piece's stats: row sums per warp, m per warpgroup
            for r in cutlass.range_constexpr(ROWS):
                lv = lsum[r]
                for kk in cutlass.range_constexpr(5):
                    lv = lv + shfl_xor(lv, Int32(16 >> kk))
                lsum[r] = lv
            if k >= 2:
                wait(sfree_p + 8 * q, ((k - 2) // 2) & 1, scope="cta")
            if lane == 0:
                for r in cutlass.range_constexpr(ROWS):
                    stats[q * STATS + q4 * 32 + r] = lsum[r]
                    if q4 == 0:
                        stats[q * STATS + 128 + r] = -bias[r]
            cute.arch.sync_warp()
            if lane == 0:
                arrive(sready_p + 8 * q)
        cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(g == 0):
            self.rec(pcs[PMAX * 5 + 3], 0, 10, global_timer(), (q4 == 0) & (lane == 0))

    @cute.jit
    def pieces_ep(
        self, tmem, cta, warp, lane, npieces, pcs, lens, srt, pre, requests, T, qkv,
        positions, slots, table, cos_sin, qw, kw, out, kv_base, row_stride, head_stride,
        eps, partials, flags, blocks, newvalid, kv_new, q_base, stats, prepped, pfree,
        pvpiece, ofree, sready_p, sfree_p, table_last, blocks_max, bready,
    ):  # fmt: skip
        """The piece warpgroup: prep of piece k, then the epilogue of piece k - 1."""
        G: cutlass.Constexpr = self.group
        Q: cutlass.Constexpr = self.q
        N: cutlass.Constexpr = self.page
        NP: cutlass.Constexpr = self.pages
        ROWS: cutlass.Constexpr = self.rows
        HQ: cutlass.Constexpr = self.group * self.kv_heads
        HKV: cutlass.Constexpr = self.kv_heads
        C: cutlass.Constexpr = self.sms
        q4 = warp - EPW
        te = q4 * 32 + lane
        group8 = lane & 7
        upper = group8 >= 4
        lanes = Uint32(32 * q4) << 16
        q_w = load_words(qw.iterator + 8 * group8, count=4)
        k_w = load_words(kw.iterator + 8 * group8, count=4)
        stride0 = qkv.layout.stride[0]
        # zero rows of the requests with no stage (CTA r % C)
        for it in cutlass.range(-(-BMAX // C), unroll=1):
            r_z = cta + it * C
            if r_z < requests:
                if srt[r_z] == 0:
                    for e in cutlass.range(-(-(Q * HQ * 8) // 128), unroll=1):
                        task = e * 128 + te
                        if task < Q * HQ * 8:
                            o_row = r_z * Q + task // (HQ * 8)
                            o_col = (task % (HQ * 8)) * 8
                            store_words(out.iterator + o_row * out.layout.stride[0] + o_col,
                                        (Uint32(0),) * 4)  # fmt: skip
        TASKS: cutlass.Constexpr = ROWS * 8 + Q * 8
        for kk in cutlass.range(npieces + 1, unroll=1):
            if kk < npieces:
                # ---- prep of piece kk into slot kk % 2
                q = kk % 2
                r_p, h_p, a_p, n_p, js = (pcs[kk * 5 + f] for f in range(5))
                length = lens[r_p]
                lo, first, last, cnt = self.stages(length)
                page_lo = first * KB // N
                # the query rows' K / V: in the pieces holding positions >= q_first
                # (the last one or two stages, maybe on two CTAs); the cache insert by
                # the piece holding the unit's last stage
                last_piece = (a_p + n_p) == cnt
                need_new = (first + 2 * (a_p + n_p)) * KB > length - Q
                if kk >= 2:
                    wait(pfree + 8 * q, ((kk - 2) // 2) & 1, scope="cta")
                table_row = table.iterator + r_p * table.layout.stride[0]
                if te < NP:
                    entry = load_i32(table_row + min(page_lo + te, table_last))
                    blocks[q * NP + te] = min(max(entry, Int32(0)), blocks_max)
                named_barrier(BAR_EP, 128)
                if te == 0:
                    arrive(bready + 8 * q)
                # every load first (both task rounds), then the math: the rounds'
                # dependent global loads overlap
                geo = []
                for it in cutlass.range_constexpr(-(-TASKS // 128)):
                    t = it * 128 + te
                    is_kv = t >= ROWS * 8
                    i = min(t // 8, ROWS - 1) // G
                    gq = min(t // 8, ROWS - 1) % G
                    col = (h_p * G + gq) * HEAD + 8 * group8
                    if is_kv:
                        i = min((t - ROWS * 8) // 8, Q - 1)
                        col = HQ * HEAD + h_p * HEAD + 8 * group8
                    geo.append((t, is_kv, i, col, r_p * Q + i))
                poss = [load_i64(positions.iterator + g_[4]) for g_ in geo]
                slots_ = [load_i64(slots.iterator + g_[4]) for g_ in geo]
                qwords = [
                    load_words(qkv.iterator + g_[4] * stride0 + g_[3], count=4,
                               mutable=True)
                    for g_ in geo
                ]  # fmt: skip
                vwords_ = [
                    load_words(qkv.iterator + g_[4] * stride0 + g_[3] + HKV * HEAD,
                               count=4, mutable=True)
                    for g_ in geo
                ]  # fmt: skip
                ropes = [self.rope_rows(cos_sin, Int32(p_), group8) for p_ in poss]
                for it in cutlass.range_constexpr(-(-TASKS // 128)):
                    t, is_kv, i, col, row = geo[it]
                    cos, sin = ropes[it]
                    words = qwords[it]
                    wts = cute.make_rmem_tensor(4, Uint32)
                    for kq in cutlass.range_constexpr(4):
                        wts[kq] = q_w[kq]
                    if is_kv:
                        for kq in cutlass.range_constexpr(4):
                            wts[kq] = k_w[kq]
                    x = [bf16(words[e // 2], e % 2 == 1) for e in range(8)]
                    roped = norm_rope8(
                        x, tuple(wts[kq] for kq in range(4)), tuple(cos), tuple(sin), eps,
                        upper, self.round_norm,
                    )  # fmt: skip
                    if t < ROWS * 8:
                        store_words(q_base + q * 4096 + swz(t // 8, group8), tuple(roped),
                                    shared=True)  # fmt: skip
                    elif (t < TASKS) & (length > 0) & need_new:
                        vwords = vwords_[it]
                        slot = slots_[it]
                        value = list(vwords)
                        if cutlass.const_expr(self.v_scale is not None):
                            scaled = []
                            for kq in cutlass.range_constexpr(4):
                                lo_v = bf16(vwords[kq], False) * Float32(self.v_scale)
                                hi_v = bf16(vwords[kq], True) * Float32(self.v_scale)
                                scaled.append(bf16x2(lo_v, hi_v))
                            value = scaled
                        nk = kv_new + q * Q * 256 + i * 256
                        store_words(nk + 16 * group8, tuple(roped), shared=True)
                        store_words(nk + 128 + 16 * group8, tuple(value), shared=True)
                        if group8 == 0:
                            newvalid[q * Q + i] = Int32(slot >= 0)
                        if (slot >= 0) & last_piece:
                            s_elem = Int64(slot // N) * row_stride
                            s_elem = s_elem + Int64(h_p) * head_stride
                            s_elem = s_elem + Int64(slot % N) * 128 + 8 * group8
                            s_addr = kv_base + s_elem * 2
                            store_words(s_addr, tuple(roped))
                            store_words(s_addr + 128, tuple(value))
                fence_async_shared()
                named_barrier(BAR_EP, 128)
                if te == 0:
                    arrive(prepped + 8 * q)
                self.rec(cta, 0, 4, global_timer(), (te == 0) & (kk == 0))
            if kk >= 1:
                self.epilogue(kk - 1, tmem, cta, q4, lane, te, lanes, pcs, lens, srt, pre,
                              T, out, partials, flags, stats, pvpiece, ofree, sready_p,
                              sfree_p)  # fmt: skip

    @cute.jit
    def epilogue(
        self, k, tmem, cta, q4, lane, te, lanes, pcs, lens, srt, pre, T, out, partials,
        flags, stats, pvpiece, ofree, sready_p, sfree_p,
    ):  # fmt: skip
        """Piece k: a later piece of a split unit publishes its partial; the unit's
        first piece folds in the later pieces' partials (published early), then its
        own (m, l, O^T), and stores; a whole unit is stored."""
        G: cutlass.Constexpr = self.group
        Q: cutlass.Constexpr = self.q
        ROWS: cutlass.Constexpr = self.rows
        HKV: cutlass.Constexpr = self.kv_heads
        C: cutlass.Constexpr = self.sms
        q = k % 2
        r_p, h_p, a_p, n_p, js = (pcs[k * 5 + f] for f in range(5))
        cnt = srt[r_p]
        dim = 16 * q4 + lane
        mrow = cute.make_rmem_tensor(ROWS, Float32)
        lrow = cute.make_rmem_tensor(ROWS, Float32)
        orow = cute.make_rmem_tensor(ROWS, Float32)
        for r in cutlass.range_constexpr(ROWS):
            mrow[r] = Float32(NEG_INF)
            lrow[r] = Float32(0)
            orow[r] = Float32(0)
        if (a_p == 0) & (n_p < cnt):
            u0 = HKV * pre[r_p] + h_p * cnt
            c_last = ((u0 + cnt) * C - 1) // T
            for cp in cutlass.range(cta + 1, c_last + 1, unroll=1):
                # (CTAs with no stage, T < C, hold no piece)
                if cp * T // C < (cp + 1) * T // C:
                    self.merge_from(cp, te, lane, dim, mrow, lrow, orow, partials,
                                    flags)  # fmt: skip
        wait(sready_p + 8 * q, (k // 2) & 1, scope="cta")
        wait(pvpiece + 8 * q, (k // 2) & 1, scope="cta")
        self.rec(cta, 0, 5, global_timer(), te == 0)
        fence_tmem()
        own = ld32(tmem + lanes + Uint32(O_COL) + Uint32(32) * Uint32(q))
        sb = q * STATS
        for r in cutlass.range_constexpr(ROWS):
            mb = stats[sb + 128 + r]
            lb = stats[sb + r] + stats[sb + 32 + r] + stats[sb + 64 + r]
            lb = lb + stats[sb + 96 + r]
            mt = fmax(mrow[r], mb)
            wa = Float32(0)
            wb = Float32(0)
            if mrow[r] > Float32(NEG_INF):
                wa = ex2(mrow[r] - mt)
            if mb > Float32(NEG_INF):
                wb = ex2(mb - mt)
            lrow[r] = fma(wa, lrow[r], wb * lb)
            orow[r] = fma(wa, orow[r], wb * own[r])
            mrow[r] = mt
        fence_tmem(before=True)
        named_barrier(BAR_EP, 128)
        if te == 0:
            arrive(ofree + 8 * q)
            arrive(sfree_p + 8 * q)
        if a_p > 0:
            # ---- a later piece of a split unit (this CTA's first piece): publish
            base = Int64(partials) + Int64(cta) * (PSZ * 4)
            if lane < 16:
                for r in cutlass.range_constexpr(ROWS):
                    st_f32(base + (dim * 32 + r) * 4, orow[r])
            if te == 0:
                for r in cutlass.range_constexpr(ROWS):
                    st_f32(base + (64 * 32 + r) * 4, mrow[r])
                    st_f32(base + (64 * 32 + 32 + r) * 4, lrow[r])
            fence_gpu()
            named_barrier(BAR_EP, 128)
            if te == 0:
                st_release(Int64(flags) + Int64(cta) * 4, 1)
        else:
            self.rec(cta, 0, 6, global_timer(), te == 0)
            if lane < 16:
                for r in cutlass.range_constexpr(ROWS):
                    inv = Float32(0)
                    if lrow[r] > Float32(0):
                        inv = Float32(1) / lrow[r]
                    o_row = r_p * Q + r // G
                    o_col = (h_p * G + r % G) * HEAD + dim
                    addr = out.iterator.toint() + (o_row * out.layout.stride[0] + o_col) * 2
                    st_bf16(Int64(addr), orow[r] * inv)

    @cute.jit
    def merge_from(self, cp, te, lane, dim, mrow, lrow, orow, partials, flags):
        """Fold CTA cp's published partial (its first piece) into (m, l, O)."""
        ROWS: cutlass.Constexpr = self.rows
        seen = ld_acquire(Int64(flags) + Int64(cp) * 4)
        while seen == 0:
            seen = ld_acquire(Int64(flags) + Int64(cp) * 4)
        base = Int64(partials) + Int64(cp) * (PSZ * 4)
        V4: cutlass.Constexpr = -(-ROWS // 4)
        ms = [ld4_f32(base + (64 * 32 + 4 * v) * 4) for v in range(V4)]
        ls = [ld4_f32(base + (64 * 32 + 32 + 4 * v) * 4) for v in range(V4)]
        os_ = [ld4_f32(base + (min(dim, 63) * 32 + 4 * v) * 4) for v in range(V4)]
        for r in cutlass.range_constexpr(ROWS):
            mb = ms[r // 4][r % 4]
            lb = ls[r // 4][r % 4]
            ob = os_[r // 4][r % 4]
            mt = fmax(mrow[r], mb)
            wa = Float32(0)
            wb = Float32(0)
            if mrow[r] > Float32(NEG_INF):
                wa = ex2(mrow[r] - mt)
            if mb > Float32(NEG_INF):
                wb = ex2(mb - mt)
            lrow[r] = fma(wa, lrow[r], wb * lb)
            orow[r] = fma(wa, orow[r], wb * ob)
            mrow[r] = mt
        named_barrier(BAR_EP, 128)
        if te == 0:
            st_release(Int64(flags) + Int64(cp) * 4, 0)


@cache
def compile_window_kols(
    group, kv_heads, q, page, ring, window, v_scale, max_position, device, sms,
    round_norm=False, debug=0,
):  # fmt: skip
    """``GqaWindowKols`` on ``sms`` CTAs; ``GqaWindowKol``'s arguments + the partials
    (FP32 [sms, PSZ]) and flags (int32 [sms], zero; left zero) addresses."""
    kernel = GqaWindowKols(group, kv_heads, q, page, ring, window, sms, v_scale,
                           round_norm, debug)  # fmt: skip
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
        Int64(0),  # the layer cache's tensor map address (kv_map_kol)
        Int64(0),  # the layer cache's base address
        Int64(0),  # row (block) stride, elements
        Int64(0),  # head stride, elements
        Int32(1),  # blocks
        Float32(1e-5),  # eps
        Int64(0),  # partials
        Int64(0),  # flags
    )
    with torch.accelerator.device_index(device):
        return cute.compile(kernel, *args, make_fake_stream())
