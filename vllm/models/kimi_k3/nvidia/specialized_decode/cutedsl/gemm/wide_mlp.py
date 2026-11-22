# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSpark drafter's MLP, y = down(silu_and_mul(x @ W_gu^T)), as ONE die-local,
persistent, stream-K token-wide grid (``wide_sk``'s MMA form and ``spread``
reduction for both GEMMs).

Two separate launches pay a per-launch fixed cost each (weight ring fill, the split-K
exchange at the end, the launch boundary, the separate silu_and_mul kernel). Here
every CTA pair runs its gate_up range, then its down range, on one ring:

- The weight issuer runs straight from the gate_up weight steps into the down weight
  steps (weights do not depend on activations), so the down weights stream into the
  ring while the gate_up partials are exchanged.
- gate_up's exchange is the ``spread`` one (every segment publishes its FP32 partial
  to a die-homed workspace; once a tile's ranges have all published, each range sums
  its share of the tile's 2 KiB units in K order), read with vector loads straight
  into registers (the ring is busy with down weights), and its epilogue is vLLM's
  silu_and_mul of the BF16 gate and up values into an L2-resident [T, I] buffer.
- Each finished 64-column block of that buffer is counted (``ready``); the activation
  issuer loads a down K step as soon as its block's gate_up ranges have all stored
  it (no grid barrier), then the down partials are exchanged the same way.

Stock math: FP32 accumulation of every GEMM (K order fixed: deterministic), BF16
gate_up output, vLLM's silu_and_mul, BF16 down output.
"""

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import from_dlpack

from vllm.models.kimi_k3.nvidia.k31_spin_guard import spin_start, spin_tick

from ..common import dies, tensormap
from ..common.primitives import asm, div_rn
from .wide import (
    _atom_add,
    _bar,
    _fence_after,
    _fence_proxy_global,
    _ld_acquire,
    _lds_v4,
    _nanosleep,
    _now,
    _red_release_add,
    _round_bf16,
    _st_bf16,
    _st_relaxed,
    _st_u64,
    _st_v4,
    mma_columns,
)
from .wide_sk import (
    CHUNK,
    K_TILE,
    _bar_n,
    _claim,
    _fence_before,
    _lds_u32,
    _mapa,
    _smid,
    _st_u32,
    _st_u64_zero,
    _sts_peer,
    _sts_u32,
    _sts_v4,
    _tma2,
    _tma2_stream,
    _tmem_ld16w,
    die_pairs,
    place_halves,
    silu_rows,
    tile_ranges,
)

UNIT = 2048  # bytes of one published unit ([4][32 lanes][4] FP32)
HB_BYTES = 4 * 2 * UNIT  # silu hand-off: 4 (gate, up) warp pairs x 2 slots x 1 unit
MAX_W_STAGES = 20
MLP_STAMPS = 24  # u64 stamp words per CTA (debug bit 8)


@asm((Float32,) * 4)
def _ldg_v4_if(address, pred):
    """ld.global.cg.v4.f32 under ``pred``, else -0.0 (the identity of add.rn)."""
    code = (
        "{ .reg .pred p; setp.ne.s32 p, $5, 0; mov.f32 $0, 0f80000000; "
        "mov.f32 $1, 0f80000000; mov.f32 $2, 0f80000000; mov.f32 $3, 0f80000000; "
        "@p ld.global.cg.v4.f32 {$0, $1, $2, $3}, [$4]; }"
    )
    return code, "=f,=f,=f,=f,l,r", (Int64(address), Int32(pred))


def owned_block(t, ks, total, nd, jf, jl, me, nu, single):
    """[lo, hi) of the unit pairs (index jj * 4 + up_id, nu of them) range jf + me
    reduces of tile t: ranges that also reduce a neighbouring tile (the first one if it
    started in the previous tile, the last one if it runs into the next) weigh 1, the
    others 2, so every range reduces about the same number of units in all. ``single``:
    the tile's only range (c == 1) owns all."""
    c = jl - jf + 1
    b0 = (jf * total // nd < t * ks) * 1
    b1 = ((jl + 1) * total // nd > (t + 1) * ks) * 1
    w = 2 * c - b0 - b1
    lo_w = 2 * me - b0 * (me > 0)
    hi_w = lo_w + 2 - b0 * (me == 0) - b1 * (me == c - 1)
    return (lo_w * nu + w - 1) // w, (hi_w * nu + w - 1) // w


def ring_bytes(rows, x_stages, smem):
    """(weight stages, ring bytes) left by the silu hand-off and ``x_stages``."""
    n_mma, chunks = mma_columns(rows)
    a_bytes = 128 * K_TILE * 2
    b_bytes = chunks * (n_mma // 2) * K_TILE * 2
    w_stages = min(MAX_W_STAGES, (smem - 4096 - HB_BYTES - x_stages * b_bytes) // a_bytes)
    return w_stages, w_stages * a_bytes + x_stages * b_bytes


def supports(rows, hidden, inter, pairs, x_stages=4, smem=None):
    n_mma, chunks = mma_columns(rows)
    smem = smem or utils.get_smem_capacity_in_bytes("sm_100")
    if not (
        rows % 32 == 0 and chunks > 0 and n_mma % 16 == 0 and hidden % 512 == 0
        and inter % 256 == 0 and min(pairs) > 0
    ):  # fmt: skip
        return False
    w_stages = ring_bytes(rows, x_stages, smem)[0]
    # every range holds at least one K step of each GEMM
    tot1, tot2 = 2 * inter // 512 * (hidden // K_TILE), hidden // 512 * (inter // K_TILE)
    return w_stages >= 4 and all(min(tot1, tot2) >= p for p in pairs)


class WideMlp:
    """One (rows, hidden, inter, pairs per die) specialization. ``debug`` bits: 1 no
    MMA (rings, exchanges and epilogues only), 8 globaltimer stamps per CTA
    (``MLP_STAMPS`` u64, see ``kernel``)."""

    def __init__(self, rows, hidden, inter, pairs, w_issuers=3, x_issuers=2, debug=0,
                 x_stages=4, smem=None, ahead=2):  # fmt: skip
        assert supports(rows, hidden, inter, pairs, x_stages, smem)
        self.rows, self.h, self.i = rows, hidden, inter
        self.p0, self.p1 = pairs
        self.pairs = self.p0 + self.p1
        self.debug = debug
        self.nch = rows // 2 // CHUNK  # 16-column units per epilogue thread
        self.n_mma, self.chunks = mma_columns(rows)
        self.mma_tiler = (256, self.n_mma, K_TILE)
        # phase 0: gate_up [2 I, H] (rows interleaved per 64), phase 1: down [H, I]
        self.nn = (2 * inter, hidden)
        self.kk = (hidden, inter)
        self.ks = tuple(k // K_TILE for k in self.kk)
        self.tpd = tuple(n // 512 for n in self.nn)  # tiles per die
        self.total = tuple(t * s for t, s in zip(self.tpd, self.ks))  # K steps per die
        counts = [
            [c for p in pairs for _, c in tile_ranges(tot, p, ks)]
            for tot, ks in zip(self.total, self.ks)
        ]
        self.cmax = tuple(max(c) for c in counts)
        # most units one warp reduces per tile (balanced blocks, see owned_block)
        nu = 4 * self.nch
        kmax = []
        for tot, ks in zip(self.total, self.ks):
            most = 0
            for p in pairs:
                for t, (jf, c) in enumerate(tile_ranges(tot, p, ks)):
                    for me in range(c):
                        lo, hi = (0, nu) if c == 1 else owned_block(
                            t, ks, tot, p, jf, jf + c - 1, me, nu, False)
                        for up in range(4):
                            most = max(most, (hi - up + 3) // 4 - (lo - up + 3) // 4)
            kmax.append(max(most, 1))
        self.kmax = tuple(kmax)
        self.ahead = ahead  # partials in flight per reduction
        self.cta_bytes = 8 * self.nch * UNIT
        strides = []
        for ph in range(2):
            part = self.tpd[ph] * self.cmax[ph] * 2 * self.cta_bytes
            strides.append(-(-part // (2 << 20)) * (2 << 20))
        self.die_stride = tuple(strides)
        # workspace: [phase 0: die 0, die 1][phase 1: die 0, die 1]
        self.ws_off = (0, 2 * self.die_stride[0])
        self.ws_bytes = 2 * (self.die_stride[0] + self.die_stride[1])
        # flags (bytes): claim u64, exit u32, pad, stamps, per phase [die][tile][CTA]
        # published / done counters, then one ready counter per 64-column block of the
        # [T, I] silu buffer
        off = 16 + 2 * self.pairs * MLP_STAMPS * 8
        self.off_cnt, self.off_done = [], []
        for ph in range(2):
            self.off_cnt.append(off)
            self.off_done.append(off + self.tpd[ph] * 16)
            off += 2 * self.tpd[ph] * 16
        self.off_ready = off
        self.nkd = inter // K_TILE
        self.flag_bytes = off + 4 * self.nkd
        self.nw, self.nx = w_issuers, x_issuers
        self.x_stages = x_stages
        smem = smem or utils.get_smem_capacity_in_bytes("sm_100")
        self.w_stages = ring_bytes(rows, x_stages, smem)[0]
        assert 1 <= w_issuers <= self.w_stages and 1 <= x_issuers <= x_stages
        self.a_stage = 128 * K_TILE * 2
        self.b_stage = self.chunks * (self.n_mma // 2) * K_TILE * 2
        # warps: 0-7 epilogue (two per TMEM lane quarter), 8 MMA, 9 weights, 10 x
        self.threads = 32 * 11
        self.tmem_cols = 32
        while self.tmem_cols < rows:
            self.tmem_cols *= 2

    @cute.jit
    def __call__(
        self, wg0: Int64, wg1: Int64, wd0: Int64, wd1: Int64, x: Int64, x1: Int64,
        a: Int64, mA: cute.Tensor, mY: cute.Tensor, mWs: cute.Tensor,
        mFlags: cute.Tensor, table: cute.Tensor, stream: _cuda.CUstream,
    ):  # fmt: skip
        """wg0 / wg1: the die-0 / die-1 halves of the ``silu_rows`` gate_up weight
        [I, H] each, wd0 / wd1 of the down weight [H / 2, I] each; x (x1: its copy
        read by the die-1 CTAs) [T, H]; a / mA: the [T, I] silu buffer; mY [T, H];
        all BF16 row-major. mWs: the workspace, mFlags: int32 counters (zeroed once),
        table [SMs] int32 smid -> die."""
        dtype = cutlass.BFloat16
        major = tcgen05.OperandMajorMode.K
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            dtype, major, major, Float32, tcgen05.CtaGroup.TWO, self.mma_tiler[:2]
        )
        a_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, dtype, self.w_stages
        )
        b_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, dtype, self.x_stages * self.chunks
        )
        bf16 = _cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
        sw128 = _cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
        l2 = _cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B
        h, i, rows = self.h, self.i, self.rows
        box_w, box_x = (K_TILE, 128), (K_TILE, rows // 2)
        gmap0 = tensormap.encode_map(wg0, bf16, (h, i), (h * 2,), box_w, sw128, l2)
        gmap1 = tensormap.encode_map(wg1, bf16, (h, i), (h * 2,), box_w, sw128, l2)
        dmap0 = tensormap.encode_map(wd0, bf16, (i, h // 2), (i * 2,), box_w, sw128, l2)
        dmap1 = tensormap.encode_map(wd1, bf16, (i, h // 2), (i * 2,), box_w, sw128, l2)
        xmap0 = tensormap.encode_map(x, bf16, (h, rows), (h * 2,), box_x, sw128, l2)
        xmap1 = tensormap.encode_map(x1, bf16, (h, rows), (h * 2,), box_x, sw128, l2)
        amap = tensormap.encode_map(a, bf16, (i, rows), (i * 2,), box_x, sw128, l2)
        self.tx_a = self.a_stage * 2
        self.tx_b = self.b_stage * 2

        @cute.struct
        class Storage:
            w_full: cute.struct.MemRange[cutlass.Int64, self.w_stages]
            w_empty: cute.struct.MemRange[cutlass.Int64, self.w_stages]
            x_full: cute.struct.MemRange[cutlass.Int64, self.x_stages]
            x_empty: cute.struct.MemRange[cutlass.Int64, self.x_stages]
            acc_full: cutlass.Int64
            acc_empty: cutlass.Int64
            dealloc: cutlass.Int64
            tmem_holder: cutlass.Int32
            slot: cutlass.Int32
            hb: cute.struct.Align[cute.struct.MemRange[Float32, HB_BYTES // 4], 16]
            sA: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(a_layout.outer)], 1024
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(b_layout.outer)], 1024
            ]

        self.storage = Storage
        self.kernel(
            tiled_mma, gmap0, gmap1, dmap0, dmap1, xmap0, xmap1, amap, a_layout,
            b_layout, mA, mY, mWs, mFlags, table,
        ).launch(  # fmt: skip
            grid=(2 * self.pairs, 1, 1),
            block=[self.threads, 1, 1],
            cluster=(2, 1, 1),
            stream=stream,
            use_pdl=True,
            min_blocks_per_mp=1,
            max_number_threads=[self.threads, 1, 1],
        )

    def range_of(self, ph, j, nd):
        total = self.total[ph]
        return j * total // nd, (j + 1) * total // nd

    @cute.kernel
    def kernel(
        self, tiled_mma: cute.TiledMma, gmap0: tensormap.GridMap,
        gmap1: tensormap.GridMap, dmap0: tensormap.GridMap, dmap1: tensormap.GridMap,
        xmap0: tensormap.GridMap, xmap1: tensormap.GridMap, amap: tensormap.GridMap,
        a_layout: cute.ComposedLayout, b_layout: cute.ComposedLayout, mA: cute.Tensor,
        mY: cute.Tensor, mWs: cute.Tensor, mFlags: cute.Tensor, mTable: cute.Tensor,
    ):  # fmt: skip
        """Stamps (``debug & 8``, u64 per CTA): 0 entry, 1 slot claimed, 2 / 6 first
        gate_up / down accumulator seen, 3 / 7 last partial published, 4 / 8 first
        tile's ranges all published, 5 / 9 reductions stored, 10 first down
        activation step issued, 11 last gate_up MMA committed, 13 slot, 15 end."""
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        bidx, _, _ = cute.arch.block_idx()
        t_entry = _now()
        v = bidx % 2  # rank in the CTA pair; 0 issues the MMAs
        leader = v == 0
        rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        sw, sx, chunks, rows = self.w_stages, self.x_stages, self.chunks, self.rows

        smem = utils.SmemAllocator()
        st = smem.allocate(self.storage)
        w_full, w_empty = st.w_full.data_ptr(), st.w_empty.data_ptr()
        x_full, x_empty = st.x_full.data_ptr(), st.x_empty.data_ptr()
        acc_full, acc_empty = st.acc_full.ptr, st.acc_empty.ptr
        dealloc = st.dealloc.ptr
        tmem_holder = st.tmem_holder.ptr
        slot_addr = Int32(st.slot.ptr.toint())
        hb = Int32(st.hb.data_ptr().toint())
        sA = st.sA.get_tensor(a_layout.outer, swizzle=a_layout.inner)
        sB = st.sB.get_tensor(b_layout.outer, swizzle=b_layout.inner)
        flags = mFlags.iterator.toint()
        stamp = flags + Int64(16) + Int64(bidx) * (MLP_STAMPS * 8)

        if warp == 0:
            with cute.arch.elect_one():
                for m in (gmap0, gmap1, dmap0, dmap1, xmap0, xmap1, amap):
                    tensormap.prefetch_map(tensormap.map_address(m))
                for s in cutlass.range_constexpr(sw):
                    cute.arch.mbarrier_init(w_full + s, 1)
                    cute.arch.mbarrier_init(w_empty + s, 1)
                for q in cutlass.range_constexpr(sx):
                    cute.arch.mbarrier_init(x_full + q, 1)
                    cute.arch.mbarrier_init(x_empty + q, 1)
                cute.arch.mbarrier_init(acc_full, 1)
                cute.arch.mbarrier_init(acc_empty, 16)
                cute.arch.mbarrier_init(dealloc, 32)
        cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()
        # The pair's slot: claimed by the leader, two-ended by die, then shared.
        if warp == 0:
            if leader:
                with cute.arch.elect_one():
                    my_die = mTable[_smid()]
                    inc = Int64(1)
                    if my_die != 0:
                        inc = Int64(1 << 32)
                    lo, hi = _claim(flags, inc)
                    mine = lo
                    if my_die != 0:
                        mine = self.pairs - 1 - hi
                    if lo + hi == self.pairs - 1:
                        _st_u64_zero(flags)
                    _sts_u32(slot_addr, mine)
                    _sts_peer(slot_addr, 1, mine)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        g = _lds_u32(slot_addr)
        if cutlass.const_expr(self.debug & 8):
            if warp == 0:
                if lane == 0:
                    _st_u64(stamp + 8, _now())
                    _st_u32(stamp + 13 * 8, g)
        if warp == 0:
            cute.arch.griddepcontrol_launch_dependents()
        wd = Int32(0)
        j = g
        nd = Int32(self.p0)
        if g >= self.p0:
            wd = Int32(1)
            j = g - self.p0
            nd = Int32(self.p1)
        s0a, s1a = self.range_of(0, j, nd)
        s0b, s1b = self.range_of(1, j, nd)
        n1 = s1a - s0a
        n2 = s1b - s0b

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        acc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, chunks))
        sa_addr = Int32(st.sA.data_ptr().toint())
        sb_addr = Int32(st.sB.data_ptr().toint())
        ks1, ks2 = self.ks
        if warp == 9:  # weights: gate_up steps, then down steps; before the PDL wait
            gmap = tensormap.map_address(gmap0)
            dmap = tensormap.map_address(dmap0)
            if wd != 0:
                gmap = tensormap.map_address(gmap1)
                dmap = tensormap.map_address(dmap1)
            if lane < self.nw:
                for step in cutlass.range(lane, n1 + n2, self.nw, unroll=1):
                    s = step % sw
                    cute.arch.mbarrier_wait(w_empty + s, ((step // sw) & 1) ^ 1)
                    bar = _mapa(Int32((w_full + s).toint()), 0)
                    dst = sa_addr + s * self.a_stage
                    if cutlass.const_expr(self.debug & 4):  # skeleton: no weight bytes
                        if leader:
                            cute.arch.mbarrier_arrive(w_full + s)
                    else:
                        if leader:
                            cute.arch.mbarrier_arrive_and_expect_tx(w_full + s, self.tx_a)
                        if step < n1:
                            gk = s0a + step
                            t = gk // ks1
                            _tma2_stream(dst, gmap, (gk - t * ks1) * K_TILE,
                                         t * 256 + v * 128, bar)  # fmt: skip
                        else:
                            gk = s0b + step - n1
                            t = gk // ks2
                            _tma2_stream(dst, dmap, (gk - t * ks2) * K_TILE,
                                         t * 256 + v * 128, bar)  # fmt: skip
        elif warp == 10:  # activations: x after the PDL wait, then the silu buffer
            xmap = tensormap.map_address(xmap0)
            if mTable[_smid()] != 0:  # the copy homed on this CTA's die
                xmap = tensormap.map_address(xmap1)
            amap_a = tensormap.map_address(amap)
            ready = flags + Int64(self.off_ready)
            tot1, half_kd = self.total[0], self.nkd // 2
            cute.arch.griddepcontrol_wait()
            if lane < self.nx:
                for step in cutlass.range(lane, n1 + n2, self.nx, unroll=1):
                    q = step % sx
                    cute.arch.mbarrier_wait(x_empty + q, ((step // sx) & 1) ^ 1)
                    bar = _mapa(Int32((x_full + q).toint()), 0)
                    dst = sb_addr + q * self.b_stage
                    if step < n1:
                        if cutlass.const_expr(self.debug & 2):  # skeleton: no x bytes
                            if leader:
                                cute.arch.mbarrier_arrive(x_full + q)
                        else:
                            if leader:
                                cute.arch.mbarrier_arrive_and_expect_tx(x_full + q, self.tx_b)
                            _tma2(dst, xmap, ((s0a + step) % ks1) * K_TILE, v * (rows // 2),
                                  bar)  # fmt: skip
                    else:
                        kd = (s0b + step - n1) % ks2
                        # the gate_up tile that stores this 64-column block, its ranges
                        pd = Int32(0)
                        npd = Int32(self.p0)
                        if kd >= half_kd:
                            pd = Int32(1)
                            npd = Int32(self.p1)
                        tt = (kd - pd * half_kd) // 2
                        jf = ((tt * ks1 + 1) * npd + tot1 - 1) // tot1 - 1
                        jl = ((tt + 1) * ks1 * npd + tot1 - 1) // tot1 - 1
                        sgn, sgt = spin_start()
                        while _ld_acquire(ready + Int64(kd) * 4) < jl - jf + 1:
                            _nanosleep()
                            sgn, sgt = spin_tick(sgn, sgt, observed=0, target=0)
                        _fence_proxy_global()
                        if cutlass.const_expr(self.debug & 8):
                            if step == n1:
                                _st_u64(stamp + 10 * 8, _now())
                        if leader:
                            cute.arch.mbarrier_arrive_and_expect_tx(x_full + q, self.tx_b)
                        _tma2(dst, amap_a, kd * K_TILE, v * (rows // 2), bar)
        elif warp == 8:
            cute.arch.barrier(barrier_id=1, number_of_threads=32 * 9)
            tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, alignment=16,
                                                   ptr_to_buffer_holding_addr=tmem_holder)  # fmt: skip
            acc = cute.make_tensor(tmem_ptr, acc_fake.layout)
            if leader:
                kblocks = cute.size(tCrA, mode=[2])
                sg = Int32(0)
                for ph in cutlass.range_constexpr(2):
                    ks = self.ks[ph]
                    s0, s1 = (s0a, s1a) if ph == 0 else (s0b, s1b)
                    base_step = Int32(0) if ph == 0 else n1
                    t0 = s0 // ks
                    nseg = (s1 - 1) // ks - t0 + 1
                    for si in cutlass.range(nseg, unroll=1):
                        t = t0 + si
                        lo_k = t * ks
                        if lo_k < s0:
                            lo_k = s0
                        hi_k = (t + 1) * ks
                        if hi_k > s1:
                            hi_k = s1
                        cute.arch.mbarrier_wait(acc_empty, (sg & 1) ^ 1)
                        _fence_after()  # the epilogue's TMEM reads before these MMAs
                        for kk in cutlass.range(lo_k, hi_k, unroll=1):
                            step = base_step + kk - s0
                            s = step % sw
                            q = step % sx
                            cute.arch.mbarrier_wait(w_full + s, (step // sw) & 1)
                            cute.arch.mbarrier_wait(x_full + q, (step // sx) & 1)
                            if cutlass.const_expr((self.debug & 1) == 0):
                                for c in cutlass.range_constexpr(chunks):
                                    acc_c = acc[(None, None, None, c)]
                                    for kb in cutlass.range_constexpr(kblocks):
                                        if cutlass.const_expr(kb == 0):
                                            tiled_mma.set(tcgen05.Field.ACCUMULATE, kk != lo_k)
                                        else:
                                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                        cute.gemm(tiled_mma, acc_c, tCrA[(None, None, kb, s)],
                                                  tCrB[(None, None, kb, q * chunks + c)],
                                                  acc_c)  # fmt: skip
                            with cute.arch.elect_one():
                                tcgen05.commit(w_empty + s, 3, tcgen05.CtaGroup.TWO)
                                tcgen05.commit(x_empty + q, 3, tcgen05.CtaGroup.TWO)
                        with cute.arch.elect_one():
                            tcgen05.commit(acc_full, 3, tcgen05.CtaGroup.TWO)
                        sg = sg + 1
                    if cutlass.const_expr(self.debug & 8):
                        if cutlass.const_expr(ph == 0):
                            if lane == 0:
                                _st_u64(stamp + 11 * 8, _now())
        elif warp < 8:
            if warp == 0:
                cute.arch.alloc_tmem(self.tmem_cols, tmem_holder, is_two_cta=True)
            cute.arch.barrier(barrier_id=1, number_of_threads=32 * 9)
            tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, alignment=16,
                                                   ptr_to_buffer_holding_addr=tmem_holder)  # fmt: skip
            acc = cute.make_tensor(tmem_ptr, acc_fake.layout)
            if warp < 4:
                self.epilogue(acc, acc_full, acc_empty, hb, warp, lane, v, wd, j, nd, mA,
                              mY, mWs, flags, stamp, 0)  # fmt: skip
            else:
                self.epilogue(acc, acc_full, acc_empty, hb, warp, lane, v, wd, j, nd, mA,
                              mY, mWs, flags, stamp, 1)  # fmt: skip
            if cutlass.const_expr(self.debug & 8):
                if warp == 0:
                    if lane == 0:
                        _st_u64(stamp, t_entry)
                        _st_u64(stamp + 15 * 8, _now())
            if warp == 0:
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=True)
            cute.arch.barrier(barrier_id=2, number_of_threads=256)
            if warp == 0:
                cute.arch.mbarrier_arrive(dealloc, rank_in_cluster ^ 1)
                cute.arch.mbarrier_wait(dealloc, 0)
                cute.arch.dealloc_tmem(tmem_ptr, self.tmem_cols, is_two_cta=True)

    def silu(self, gate_sum, up_sum):
        """vLLM's silu_and_mul of the BF16 gate and up outputs (debug bit 16: gate *
        up, no exp or division), to be stored as BF16."""
        gate = _round_bf16(gate_sum)
        up = _round_bf16(up_sum)
        if cutlass.const_expr(self.debug & 16):
            act = gate
        else:
            act = div_rn(gate, Float32(1) + cute.math.exp(-gate, fastmath=False))
        return Float32(cutlass.BFloat16(act)) * (up + Float32(0))

    def token(self, col):
        """Token of accumulator column ``col`` (chunk-major): MMA chunk c's columns
        [0, h) are CTA 0's rows [c h, c h + h), [h, 2 h) CTA 1's (tokens T / 2 on)."""
        c, j = divmod(col, self.n_mma)
        h = self.n_mma // 2
        return c * h + j if j < h else self.rows // 2 + c * h + (j - h)

    @cute.jit
    def epilogue(
        self, acc: cute.Tensor, acc_full, acc_empty, hb: Int32, ew: Int32, lane: Int32,
        v: Int32, wd: Int32, j: Int32, nd: Int32, mA: cute.Tensor, mY: cute.Tensor,
        mWs: cute.Tensor, flags: Int64, stamp: Int64, half: cutlass.Constexpr[int],
    ):  # fmt: skip
        """Warp ew (0-7) drains TMEM lane quarter ew % 4, columns [h T / 2, (h + 1)
        T / 2), h = ew // 4. Per GEMM: every segment's accumulator goes to registers
        (then back to the MMA warp) and out as nch 2 KiB units, then the tile is
        counted; after the last segment the range sums its share of every tile it
        touched (unit pair u = warps q, q + 2 and one 16-column unit belongs to the
        tile's range u mod c; c ranges, K order) from vector loads, and stores the
        gate_up tile's silu_and_mul (the up warp hands its sums to the gate warp) or
        the down tile's BF16 output."""
        rows, nch, cta = self.rows, self.nch, self.cta_bytes
        quarter = ew % 4
        hcols = rows // 2
        lane_bits = (quarter * 32) << 16
        base = [
            Int32(acc[(None, None, None, c)].iterator.toint()) + lane_bits
            for c in range(self.chunks)
        ]
        wsb = mWs.iterator.toint()
        a_out = mA.iterator.toint()
        y_out = mY.iterator.toint()
        first_col = half * hcols
        lead = ew * 32 + lane == 0
        up_id = half * 2 + quarter % 2
        p = quarter // 2
        hslot = Int32(0)
        sg = Int32(0)
        ready = flags + Int64(self.off_ready)
        for ph in cutlass.range_constexpr(2):
            ks, total, tpd, cmax = self.ks[ph], self.total[ph], self.tpd[ph], self.cmax[ph]
            ws = wsb + Int64(self.ws_off[ph]) + Int64(wd) * self.die_stride[ph]
            cnt0 = flags + Int64(self.off_cnt[ph]) + Int64(wd * tpd * 8)
            done0 = flags + Int64(self.off_done[ph]) + Int64(wd * tpd * 8)
            s0, s1 = self.range_of(ph, j, nd)
            t0 = s0 // ks
            nseg = (s1 - 1) // ks - t0 + 1
            for si in cutlass.range(nseg, unroll=1):
                t = t0 + si
                cute.arch.mbarrier_wait(acc_full, sg & 1)
                if cutlass.const_expr(self.debug & 8):
                    if lead:
                        if si == 0:
                            _st_u64(stamp + (2 + 4 * ph) * 8, _now())
                _fence_after()
                vals = []
                for jj in cutlass.range_constexpr(nch):
                    col = first_col + jj * CHUNK
                    cc, off = divmod(col, self.n_mma)
                    vals += list(_tmem_ld16w(base[cc] + off))
                _fence_before()
                if lane == 0:
                    cute.arch.mbarrier_arrive(acc_empty, Int32(0))
                sg = sg + 1
                jf = ((t * ks + 1) * nd + total - 1) // total - 1
                dst = ws + Int64(((t * cmax + j - jf) * 2 + v) * cta + ew * nch * UNIT)
                dst = dst + Int64(lane * 16)
                for jj in cutlass.range_constexpr(nch):
                    for q4 in cutlass.range_constexpr(4):
                        i0 = jj * CHUNK + q4 * 4
                        _st_v4(dst + (jj * UNIT + q4 * 512), vals[i0], vals[i0 + 1],
                               vals[i0 + 2], vals[i0 + 3])  # fmt: skip
                _bar(2, 256)
                if lead:
                    _red_release_add(cnt0 + Int64(t * 2 + v) * 4, 1)
            if cutlass.const_expr(self.debug & 8):
                if lead:
                    _st_u64(stamp + (3 + 4 * ph) * 8, _now())
            for si in cutlass.range(nseg, unroll=1):
                t = t0 + si
                jf = ((t * ks + 1) * nd + total - 1) // total - 1
                jl = ((t + 1) * ks * nd + total - 1) // total - 1
                c = jl - jf + 1
                me = j - jf
                cnt = cnt0 + Int64(t * 2 + v) * 4
                if lead:
                    sgn, sgt = spin_start()
                    while _ld_acquire(cnt) < c:
                        _nanosleep()
                        sgn, sgt = spin_tick(sgn, sgt, observed=0, target=0)
                _bar(2, 256)
                if cutlass.const_expr(self.debug & 8):
                    if lead:
                        if si == 0:
                            _st_u64(stamp + (4 + 4 * ph) * 8, _now())
                src = ws + Int64((t * cmax * 2 + v) * cta + ew * nch * UNIT)
                src = src + Int64(lane * 16)
                # this warp's owned units: jj = j0, j0 + c, ... < nch (at most kmax);
                # every load is issued before any store (a store would order the next
                # unit's loads behind it: one L2 round trip per unit), one partial
                # ahead of the K-order sums
                nu = 4 * nch
                b0 = Int32(jf * total // nd < t * ks)
                b1 = Int32((jl + 1) * total // nd > (t + 1) * ks)
                wsum = 2 * c - b0 - b1
                lo_w = 2 * me - b0 * Int32(me > 0)
                hi_w = lo_w + 2 - b0 * Int32(me == 0) - b1 * Int32(me == c - 1)
                lo = Int32(0)
                hi = Int32(nu)
                if c > 1:
                    lo = (lo_w * nu + wsum - 1) // wsum
                    hi = (hi_w * nu + wsum - 1) // wsum
                ja = (lo - up_id + 3) // 4
                jb = (hi - up_id + 3) // 4
                kmax = self.kmax[ph]
                okk = [Int32(ja + k < jb) for k in range(kmax)]
                offs = [Int64((ja + k) * UNIT) for k in range(kmax)]

                def loads(i):
                    return [
                        _ldg_v4_if(src + offs[k] + Int64(i * 2 * cta + q4 * 512),
                                   okk[k] * Int32(c > i))  # fmt: skip
                        for k in range(kmax)
                        for q4 in range(4)
                    ]

                ahead = [loads(i) for i in range(min(self.ahead, cmax))]
                sums = [list(v4) for v4 in ahead[0]]
                for i in cutlass.range_constexpr(1, cmax):
                    if cutlass.const_expr(i + self.ahead - 1 < cmax):
                        ahead.append(loads(i + self.ahead - 1))
                    cur = ahead[i]
                    for n4 in cutlass.range_constexpr(kmax * 4):
                        for e in cutlass.range_constexpr(4):
                            sums[n4][e] = sums[n4][e] + cur[n4][e]
                if cutlass.const_expr(self.debug & 8):
                    if ew == 0:
                        if lane == 0:
                            if si == 0:
                                _st_u64(stamp + (16 + 2 * ph) * 8, _now())
                for k in cutlass.range_constexpr(kmax):
                    jj = ja + k
                    if jj < jb:
                        # tokens of this 16-column unit are consecutive (units never
                        # straddle an MMA chunk or a CTA half of one)
                        col0 = first_col + jj * CHUNK
                        cc = col0 // self.n_mma
                        jc = col0 - cc * self.n_mma
                        hm = self.n_mma // 2
                        tok0 = cc * hm + jc
                        if jc >= hm:
                            tok0 = tok0 + (rows // 2 - hm)
                        sk = [sums[k * 4 + q4][e] for q4 in range(4) for e in range(4)]
                        if cutlass.const_expr(ph == 0):
                            # the gate warp (rows 0-63) finishes tokens 0-7 of the unit,
                            # the up warp (the same outputs' up rows) tokens 8-15: each
                            # hands the other half of its sums over
                            hs = hb + (up_id * 2 + hslot) * UNIT + lane * 16
                            col = wd * (self.nn[0] // 4) + t * 128 + v * 64
                            col = col + (quarter % 2) * 32 + lane
                            dst_a = a_out + (Int64(tok0) * self.i + col) * 2
                            if cutlass.const_expr(self.debug & 32):  # no hand-off, no silu
                                for q4 in cutlass.range_constexpr(2):
                                    for e in cutlass.range_constexpr(4):
                                        _st_bf16(dst_a + (p * 8 + q4 * 4 + e) * self.i * 2,
                                                 sk[q4 * 4 + e])  # fmt: skip
                            else:
                                if p == 0:
                                    for q4 in cutlass.range_constexpr(2, 4):
                                        _sts_v4(hs + q4 * 512, sk[q4 * 4], sk[q4 * 4 + 1],
                                                sk[q4 * 4 + 2], sk[q4 * 4 + 3])  # fmt: skip
                                else:
                                    for q4 in cutlass.range_constexpr(2):
                                        _sts_v4(hs + q4 * 512, sk[q4 * 4], sk[q4 * 4 + 1],
                                                sk[q4 * 4 + 2], sk[q4 * 4 + 3])  # fmt: skip
                                _bar_n(3 + up_id, 64)
                                # all eight silu chains first (independent: they
                                # overlap), then the stores
                                if p == 0:
                                    ups = list(_lds_v4(hs)) + list(_lds_v4(hs + 512))
                                    acts = [self.silu(sk[n], ups[n]) for n in range(8)]
                                    for n in cutlass.range_constexpr(8):
                                        _st_bf16(dst_a + n * self.i * 2, acts[n])
                                else:
                                    gates = list(_lds_v4(hs + 1024)) + list(_lds_v4(hs + 1536))
                                    acts = [self.silu(gates[n], sk[8 + n]) for n in range(8)]
                                    for n in cutlass.range_constexpr(8):
                                        _st_bf16(dst_a + (8 + n) * self.i * 2, acts[n])
                            hslot = hslot ^ 1
                        else:
                            row = wd * (self.h // 2) + t * 256 + v * 128 + quarter * 32 + lane
                            dst_y = y_out + (Int64(tok0) * self.h + row) * 2
                            for e in cutlass.range_constexpr(CHUNK):
                                _st_bf16(dst_y + e * self.h * 2, sk[e])
                if cutlass.const_expr(self.debug & 8):
                    if ew == 0:
                        if lane == 0:
                            if si == 0:
                                _st_u64(stamp + (17 + 2 * ph) * 8, _now())
                _bar(2, 256)
                if lead:
                    if cutlass.const_expr(ph == 0):  # this 64-column silu block stored
                        kd = wd * (self.nkd // 2) + t * 2 + v
                        _red_release_add(ready + Int64(kd) * 4, 1)
                    done = done0 + Int64(t * 2 + v) * 4
                    if _atom_add(done, 1) == c - 1:  # the tile's last range resets
                        _st_relaxed(cnt, 0)
                        _st_relaxed(done, 0)
            if cutlass.const_expr(self.debug & 8):
                if lead:
                    _st_u64(stamp + (5 + 4 * ph) * 8, _now())
        if lead:  # the last CTA out resets the silu-block counters
            exitc = flags + Int64(8)
            if _atom_add(exitc, 1) == 2 * self.pairs - 1:
                for kd in cutlass.range(self.nkd, unroll=1):
                    _st_relaxed(ready + Int64(kd) * 4, 0)
                _st_relaxed(exitc, 0)


_COMPILED = {}


class PreparedWideMlp:
    """``WideMlp`` per prepared row count on one gate_up [2 I, H] and one down [H, I]
    BF16 weight, copied into die-homed halves. ``share``: a prepared instance of the
    same shapes and options whose workspace, counters and silu buffer this one reuses
    (launches on one stream never overlap).
    ``run(x, out)``: out [T, H] BF16 = down(silu_and_mul(x @ gate_up^T))."""

    def __init__(self, gate_up, down, rows, w_issuers=3, x_issuers=2, debug=0,
                 x_stages=4, share=None, ahead=2):  # fmt: skip
        self.share = share
        self.doms, self.pairs = die_pairs(gate_up.device)
        n1, self.h = gate_up.shape
        self.i = down.shape[1]
        assert n1 == 2 * self.i and down.shape[0] == self.h, (gate_up.shape, down.shape)
        self.wg = place_halves(silu_rows(gate_up))
        self.wd = place_halves(down)
        self.rows = tuple(rows)
        self.opts = (w_issuers, x_issuers, debug, x_stages, None, ahead)
        self.kernels, self.state = {}, {}

    def prepare(self):
        dev = self.wg.device
        es = self.wg.element_size()
        self.g0 = self.wg.data_ptr()
        self.g1 = self.g0 + self.i * self.h * es
        self.d0 = self.wd.data_ptr()
        self.d1 = self.d0 + self.h // 2 * self.i * es
        self.table = torch.tensor(self.doms, dtype=torch.int32, device=dev)
        self.table_c = from_dlpack(self.table, assumed_align=16)
        for rows in self.rows:
            mlp = WideMlp(rows, self.h, self.i, self.pairs, *self.opts)
            x = torch.empty(rows, self.h, dtype=torch.bfloat16, device=dev)
            y = torch.empty(rows, self.h, dtype=torch.bfloat16, device=dev)
            if self.share is not None:
                assert (self.share.h, self.share.i, self.share.opts) == (
                    self.h, self.i, self.opts)  # fmt: skip
                ws, flags, a = self.share.state[rows][:3]
            else:
                c0, c1 = (s // (2 << 20) for s in mlp.die_stride)
                doms = [0] * c0 + [1] * c0 + [0] * c1 + [1] * c1
                ws = dies.localized_empty(mlp.ws_bytes, doms, dev)
                flags = torch.zeros(mlp.flag_bytes // 4, dtype=torch.int32, device=dev)
                a = torch.empty(rows, self.i, dtype=torch.bfloat16, device=dev)
            a_t = self._t(a)
            ws_t = from_dlpack(ws, assumed_align=16)
            flags_t = from_dlpack(flags, assumed_align=16)
            # Addresses as DSL Int64 values: runtime arguments of the compiled kernel
            # (``run`` passes them every call), not constants folded into its code.
            args = (*(Int64(v) for v in (self.g0, self.g1, self.d0, self.d1,
                                         x.data_ptr(), x.data_ptr(), a.data_ptr())),
                    a_t, self._t(y), ws_t, flags_t, self.table_c)  # fmt: skip
            key = (rows, self.h, self.i, self.pairs, self.opts)
            if key not in _COMPILED:
                stream = _cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)
                _COMPILED[key] = cute.compile(mlp, *args, stream)
            self.kernels[rows] = _COMPILED[key]
            self.state[rows] = (ws, flags, a, a_t, ws_t, flags_t)
        return self

    @staticmethod
    def _t(t):
        return from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=1)

    def stamps(self, rows):
        """[2 * pairs, MLP_STAMPS] int64 stamps of the last ``debug & 8`` launch."""
        flags = self.state[rows][1]
        npairs = sum(self.pairs)
        st = flags[4 : 4 + 4 * npairs * MLP_STAMPS]
        return st.view(torch.int64).view(2 * npairs, MLP_STAMPS)

    def takes(self, x):
        return (
            x.dim() == 2 and x.shape[0] in self.kernels and x.dtype == torch.bfloat16
            and x.is_contiguous() and x.shape[1] == self.h
        )  # fmt: skip

    def run(self, x, out, x1=None):
        """x1: a copy of x homed on die 1 (x then on die 0), else the CTAs of both
        dies read x."""
        rows = x.shape[0]
        _, _, a, a_t, ws_t, flags_t = self.state[rows]
        stream = _cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
        xa = x.data_ptr()
        self.kernels[rows](self.g0, self.g1, self.d0, self.d1, xa,
                           xa if x1 is None else x1.data_ptr(), a.data_ptr(), a_t,
                           self._t(out), ws_t, flags_t, self.table_c, stream)  # fmt: skip
        return out
