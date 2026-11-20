# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Die-local, persistent, stream-K token-wide BF16 GEMM for the DSpark drafter's LM
head (and other large drafter linears) at 96-384 rows.

y[T, N] = x[T, K] @ W[N, K]^T on ``wide.WideGemm``'s MMA form (swap-AB 2-CTA, 256
weight rows on M, all T tokens on N, FP32 accumulator in TMEM), restructured for the
hardware limits of these shapes:

- Die-local weights: W's two N halves are homed on the two dies (one localized
  allocation, 2 MiB chunks); a CTA pair streams only rows homed on the die it runs on
  (``%smid`` -> die table), so the weight stream reads at the die-local rate.
- One persistent wave, stream-K: one CTA pair per SM pair; each die's (tile, K step)
  space is cut into equal contiguous ranges, one per pair of that die. Slots are
  claimed two-ended (die-0 pairs from the bottom of the slot list, die-1 pairs from
  the top), so every slot is claimed exactly once whatever the placement.
- A tile shared by two ranges is finished by the range holding its first K step (the
  head; that segment is its range's last): the range holding the rest publishes its
  FP32 partial first thing, the head adds it after its own sum (K order:
  deterministic) and rounds once to BF16, as the stock GEMM's FP32 accumulation
  (order may differ).
- Many TMA issuing threads: one issuing thread keeps about one op in flight
  (~0.5-1 us each under load), so every lane of the weight warp and of the activation
  warp is its own issuer (K-step residue); a CTA's activation rows for a K step are
  one box (both MMA chunks: a token permutation the epilogue undoes).
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

from ..common import dies, tensormap
from ..common.primitives import asm
from .wide import (
    _now,
    _st_u64,
    _bar,
    _bulk_load,
    _fence_after,
    _fence_proxy_global,
    _ld_acquire,
    _nanosleep,
    _red_release_add,
    _st_bf16,
    _st_f32,
    _st_relaxed,
    mma_columns,
)

STAMPS = 16  # u64 stamp words per CTA (debug bit 8)
K_TILE = 64  # K per ring stage (one 128-byte swizzle atom)
CHUNK = 16  # tokens per TMEM load (tcgen05.ld 32x32b.x16)
MAX_CONTRIBUTORS = 1  # partial ranges per tile (beyond the head's)


@asm(Int32)
def _smid():
    return "mov.u32 $0, %smid;", "=r,~{memory}", ()


@asm((Int32, Int32))
def _claim(address, inc):
    code = (
        "{ .reg .b64 o; atom.relaxed.gpu.global.add.u64 o, [$2], $3; "
        "mov.b64 {$0, $1}, o; }"
    )
    return code, "=r,=r,l,l,~{memory}", (Int64(address), Int64(inc))


@asm()
def _st_u64_zero(address):
    return "st.relaxed.gpu.global.u64 [$0], 0;", "l,~{memory}", (Int64(address),)


@asm()
def _sts_u32(address, value):
    return "st.shared.u32 [$0], $1;", "r,r,~{memory}", (Int32(address), Int32(value))


@asm()
def _sts_peer(address, peer, value):
    code = (
        "{ .reg .b32 a; mapa.shared::cluster.u32 a, $0, $1; "
        "st.shared::cluster.u32 [a], $2; }"
    )
    return code, "r,r,r,~{memory}", (Int32(address), Int32(peer), Int32(value))


@asm(Int32)
def _lds_u32(address):
    return "ld.shared.u32 $0, [$1];", "=r,r,~{memory}", (Int32(address),)


@asm()
def _st_u32(address, value):
    return "st.global.u32 [$0], $1;", "l,r,~{memory}", (Int64(address), Int32(value))


@asm(Int32)
def _mapa(address, rank):
    code = "mapa.shared::cluster.u32 $0, $1, $2;"
    return code, "=r,r,r", (Int32(address), Int32(rank))


@asm()
def _tma2_stream(dst, tmap, c0, c1, mbar):
    """2-D tile load into this CTA, completing on the pair leader's barrier ``mbar``
    (shared::cluster address); L2 evict-first (weights: streamed once)."""
    code = (
        "{ .reg .b64 pol; createpolicy.fractional.L2::evict_first.b64 pol, 1.0; "
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier"
        "::complete_tx::bytes.L2::cache_hint [$0], [$1, {$2, $3}], [$4], pol; }"
    )
    return code, "r,l,r,r,r,~{memory}", (
        Int32(dst), Int64(tmap), Int32(c0), Int32(c1), Int32(mbar)
    )


@asm()
def _tma2(dst, tmap, c0, c1, mbar):
    code = (
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier"
        "::complete_tx::bytes [$0], [$1, {$2, $3}], [$4];"
    )
    return code, "r,l,r,r,r,~{memory}", (
        Int32(dst), Int64(tmap), Int32(c0), Int32(c1), Int32(mbar)
    )


@asm((Float32,) * 16)
def _tmem_ld16w(taddr):
    """16 TMEM columns and the wait in ONE asm: with separate statements the compiler
    may copy / spill the destination registers before the async load wrote them."""
    regs = ",".join(f"${i}" for i in range(16))
    code = (
        "tcgen05.ld.sync.aligned.32x32b.x16.b32 {" + regs + "}, [$16]; "
        "tcgen05.wait::ld.sync.aligned;"
    )
    return code, ",".join(["=f"] * 16) + ",r,~{memory}", (Int32(taddr),)


@asm()
def _fence_before():
    return "tcgen05.fence::before_thread_sync;", "~{memory}", ()


@asm(Float32)
def _add_lds_if(acc, address, pred):
    code = (
        "{ .reg .pred p; .reg .f32 t; setp.ne.s32 p, $3, 0; mov.f32 $0, $1; "
        "@p ld.shared.f32 t, [$2]; @p add.rn.f32 $0, $1, t; }"
    )
    return code, "=f,f,r,r,~{memory}", (Float32(acc), Int32(address), Int32(pred))


def ranges(total, slots):
    return [s * total // slots for s in range(slots + 1)]


def max_contributors(total, slots, ks):
    """The most ranges that start strictly inside one tile of ``ks`` K steps."""
    starts = ranges(total, slots)[:-1]
    worst = 0
    for t in range(total // ks):
        worst = max(worst, sum(1 for s in starts if t * ks < s < (t + 1) * ks))
    return worst


def supports(rows, n, k, pairs):
    n_mma, chunks = mma_columns(rows)
    if not (
        rows % 32 == 0 and chunks > 0 and n_mma % 16 == 0 and n % 512 == 0
        and k % K_TILE == 0 and min(pairs) > 0
    ):  # fmt: skip
        return False
    ks = k // K_TILE
    total = n // 512 * ks
    return all(max_contributors(total, p, ks) <= MAX_CONTRIBUTORS for p in pairs)


class WideSK:
    """One (rows, N, K, pairs per die) specialization; see the module docstring.
    ``debug`` bits: 1 no MMA (rings and epilogues only), 2 no activation loads,
    4 no weight loads (skeletons), 8 globaltimer stamps per CTA after the flags
    (``STAMPS`` u64: entry, slot, then per segment acc_full seen / done, end)."""

    def __init__(self, rows, n, k, pairs, ldo=None, w_issuers=3, x_issuers=2, debug=0,
                 x_stages=4):  # fmt: skip
        assert supports(rows, n, k, pairs), (rows, n, k, pairs)
        self.rows, self.n, self.k = rows, n, k
        self.p0, self.p1 = pairs
        self.pairs = self.p0 + self.p1
        self.ldo = ldo or n
        self.debug = debug
        self.n_mma, self.chunks = mma_columns(rows)
        self.mma_tiler = (256, self.n_mma, K_TILE)
        self.nw, self.nx = w_issuers, x_issuers  # issuing lanes
        assert 1 <= w_issuers <= 32 and 1 <= x_issuers <= 32
        # warps: 0-7 epilogue (two per TMEM lane quarter), 8 MMA, 9 weights, 10 x
        self.threads = 32 * 11
        self.ks = k // K_TILE
        self.total = n // 512 * self.ks  # K steps per die
        self.acc_bufs = 1  # one TMEM accumulator (288 tokens take 288 of 512 columns)
        # Separate rings: the weights stream from HBM (deep ring: the die-local rate
        # needs >= ~128-192 KiB in flight per SM), the activations from L2 (shallow).
        a_bytes = 128 * K_TILE * 2
        b_bytes = self.chunks * (self.n_mma // 2) * K_TILE * 2
        self.a_stage, self.b_stage = a_bytes, b_bytes
        smem = utils.get_smem_capacity_in_bytes("sm_100")
        self.x_stages = x_stages
        self.w_stages = min(14, (smem - 4096 - x_stages * b_bytes) // a_bytes)
        ring = self.w_stages * a_bytes + x_stages * b_bytes
        assert ring >= rows * 512  # owner partial staging
        # one issuing lane per in-flight stage at most (two lanes would arm one phase)
        assert self.nw <= self.w_stages and self.nx <= x_stages
        self.tmem_cols = 32
        while self.tmem_cols < self.acc_bufs * rows:
            self.tmem_cols *= 2

    @cute.jit
    def __call__(
        self, w0: Int64, w1: Int64, x: Int64, x1: Int64, y: cute.Tensor,
        ws: cute.Tensor, flags: cute.Tensor, table: cute.Tensor, stream: _cuda.CUstream,
    ):  # fmt: skip
        """w0 / w1: addresses of the die-0 / die-1 row halves [N / 2, K] and x of the
        activations [T, K] (x1: the same, read by the CTAs on die 1), BF16 K-major;
        y [T, N] row-major BF16 (row pitch ``ldo``); ws [pairs, 2, T, 128] FP32;
        flags [2 * pairs + 2] int32 (zeroed once); table [SMs] int32 smid -> die."""
        dtype = cutlass.BFloat16
        major = tcgen05.OperandMajorMode.K
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            dtype, major, major, Float32, tcgen05.CtaGroup.TWO, self.mma_tiler[:2]
        )
        cluster_vmnk = cute.tiled_divide(
            cute.make_layout((2, 1, 1)), (tiled_mma.thr_id.shape,)
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
        k, half, rows = self.k, self.n // 2, self.rows
        # weights: a CTA's 128 rows x one K step; activations: a CTA's T / 2 rows
        wmap0 = tensormap.encode_map(w0, bf16, (k, half), (k * 2,), (K_TILE, 128), sw128, l2)
        wmap1 = tensormap.encode_map(w1, bf16, (k, half), (k * 2,), (K_TILE, 128), sw128, l2)
        xmap0 = tensormap.encode_map(x, bf16, (k, rows), (k * 2,), (K_TILE, rows // 2),
                                     sw128, l2)  # fmt: skip
        xmap1 = tensormap.encode_map(x1, bf16, (k, rows), (k * 2,), (K_TILE, rows // 2),
                                     sw128, l2)  # fmt: skip
        self.tx_a = self.a_stage * 2
        self.tx_b = self.b_stage * 2

        @cute.struct
        class Storage:
            w_full: cute.struct.MemRange[cutlass.Int64, self.w_stages]
            w_empty: cute.struct.MemRange[cutlass.Int64, self.w_stages]
            x_full: cute.struct.MemRange[cutlass.Int64, self.x_stages]
            x_empty: cute.struct.MemRange[cutlass.Int64, self.x_stages]
            acc_full: cute.struct.MemRange[cutlass.Int64, 2]
            acc_empty: cute.struct.MemRange[cutlass.Int64, 2]
            dealloc: cutlass.Int64
            xfer: cutlass.Int64
            tmem_holder: cutlass.Int32
            slot: cutlass.Int32
            sA: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(a_layout.outer)], 1024
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(b_layout.outer)], 1024
            ]

        self.storage = Storage
        self.kernel(
            tiled_mma, wmap0, wmap1, xmap0, xmap1, a_layout, b_layout, y, ws, flags,
            table,
        ).launch(  # fmt: skip
            grid=(2 * self.pairs, 1, 1),
            block=[self.threads, 1, 1],
            cluster=(2, 1, 1),
            stream=stream,
            use_pdl=True,
            # one CTA per SM: the full register file (the default budget spilled the
            # unrolled epilogue at 72 registers per thread)
            min_blocks_per_mp=1,
            max_number_threads=[self.threads, 1, 1],
        )

    @cute.kernel
    def kernel(
        self, tiled_mma: cute.TiledMma, wmap0: tensormap.GridMap,
        wmap1: tensormap.GridMap, xmap0: tensormap.GridMap, xmap1: tensormap.GridMap, a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout, mY: cute.Tensor, mWs: cute.Tensor,
        mFlags: cute.Tensor, mTable: cute.Tensor,
    ):  # fmt: skip
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        bidx, _, _ = cute.arch.block_idx()
        t_entry = _now()
        v = bidx % 2  # rank in the CTA pair; 0 issues the MMAs
        leader = v == 0
        rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        sw, sx, chunks, nb = self.w_stages, self.x_stages, self.chunks, self.acc_bufs
        ks, total, rows = self.ks, self.total, self.rows

        smem = utils.SmemAllocator()
        st = smem.allocate(self.storage)
        w_full, w_empty = st.w_full.data_ptr(), st.w_empty.data_ptr()
        x_full, x_empty = st.x_full.data_ptr(), st.x_empty.data_ptr()
        acc_full, acc_empty = st.acc_full.data_ptr(), st.acc_empty.data_ptr()
        dealloc, xfer = st.dealloc.ptr, st.xfer.ptr
        tmem_holder = st.tmem_holder.ptr
        slot_addr = Int32(st.slot.ptr.toint())
        sA = st.sA.get_tensor(a_layout.outer, swizzle=a_layout.inner)
        sB = st.sB.get_tensor(b_layout.outer, swizzle=b_layout.inner)
        stage_addr = Int32(st.sA.data_ptr().toint())
        flags = mFlags.iterator.toint()
        claim = flags + Int64(self.pairs * 8)

        if warp == 0:
            with cute.arch.elect_one():
                for m in (wmap0, wmap1, xmap0, xmap1):
                    tensormap.prefetch_map(tensormap.map_address(m))
                for i in cutlass.range_constexpr(sw):
                    cute.arch.mbarrier_init(w_full + i, 1)
                    cute.arch.mbarrier_init(w_empty + i, 1)
                for i in cutlass.range_constexpr(sx):
                    cute.arch.mbarrier_init(x_full + i, 1)
                    cute.arch.mbarrier_init(x_empty + i, 1)
                for i in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(acc_full + i, 1)
                    cute.arch.mbarrier_init(acc_empty + i, 16)
                cute.arch.mbarrier_init(dealloc, 32)
                cute.arch.mbarrier_init(xfer, 1)
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
                    lo, hi = _claim(claim, inc)
                    mine = lo
                    if my_die != 0:
                        mine = self.pairs - 1 - hi
                    if lo + hi == self.pairs - 1:
                        _st_u64_zero(claim)
                    _sts_u32(slot_addr, mine)
                    _sts_peer(slot_addr, 1, mine)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        g = _lds_u32(slot_addr)
        stamp = flags + Int64(self.pairs * 8 + 8) + Int64(bidx) * (STAMPS * 8)
        if cutlass.const_expr(self.debug & 8):
            if warp == 0:
                if lane == 0:
                    _st_u64(stamp + 8, _now())
                    _st_u32(stamp + 14 * 8, g)
        if warp == 0:
            cute.arch.griddepcontrol_launch_dependents()
        wd = Int32(0)
        j = g
        nd = Int32(self.p0)
        if g >= self.p0:
            wd = Int32(1)
            j = g - self.p0
            nd = Int32(self.p1)
        s0 = j * total // nd
        s1 = (j + 1) * total // nd
        t0 = s0 // ks
        nseg = (s1 - 1) // ks - t0 + 1

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        acc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, chunks * nb))

        w_steps = Int32(0) if self.debug & 4 else s1 - s0
        x_steps = Int32(0) if self.debug & 2 else s1 - s0
        sa_addr = Int32(st.sA.data_ptr().toint())
        sb_addr = Int32(st.sB.data_ptr().toint())
        if warp == 9:  # weights: one issuer per lane, before the PDL wait
            wmap = tensormap.map_address(wmap0)
            if wd != 0:
                wmap = tensormap.map_address(wmap1)
            if lane < self.nw:
                for step in cutlass.range(lane, w_steps, self.nw, unroll=1):
                    s = step % sw
                    cute.arch.mbarrier_wait(w_empty + s, ((step // sw) & 1) ^ 1)
                    gk = s0 + step
                    t = gk // ks
                    bar = Int32((w_full + s).toint())
                    if leader:
                        cute.arch.mbarrier_arrive_and_expect_tx(w_full + s, self.tx_a)
                    _tma2_stream(sa_addr + s * self.a_stage, wmap, (gk - t * ks) * K_TILE,
                                 t * 256 + v * 128, _mapa(bar, 0))  # fmt: skip
        elif warp == 10:  # activations: one issuer per lane, after the PDL wait
            xmap = tensormap.map_address(xmap0)
            if mTable[_smid()] != 0:  # the copy homed on this CTA's die
                xmap = tensormap.map_address(xmap1)
            cute.arch.griddepcontrol_wait()
            if lane < self.nx:
                for step in cutlass.range(lane, x_steps, self.nx, unroll=1):
                    q = step % sx
                    cute.arch.mbarrier_wait(x_empty + q, ((step // sx) & 1) ^ 1)
                    bar = Int32((x_full + q).toint())
                    if leader:
                        cute.arch.mbarrier_arrive_and_expect_tx(x_full + q, self.tx_b)
                    _tma2(sb_addr + q * self.b_stage, xmap, ((s0 + step) % ks) * K_TILE,
                          v * (rows // 2), _mapa(bar, 0))  # fmt: skip
        elif warp == 8:
            cute.arch.barrier(barrier_id=1, number_of_threads=32 * 9)
            tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, alignment=16,
                                                   ptr_to_buffer_holding_addr=tmem_holder)  # fmt: skip
            acc = cute.make_tensor(tmem_ptr, acc_fake.layout)
            if leader:
                kblocks = cute.size(tCrA, mode=[2])
                for si in cutlass.range(nseg, unroll=1):
                    t = t0 + si
                    lo_k = t * ks
                    if lo_k < s0:
                        lo_k = s0
                    hi_k = (t + 1) * ks
                    if hi_k > s1:
                        hi_k = s1
                    b = si % nb
                    cute.arch.mbarrier_wait(acc_empty + b, ((si // nb) & 1) ^ 1)
                    _fence_after()  # the epilogue's TMEM reads before these MMAs
                    for kk in cutlass.range(lo_k, hi_k, unroll=1):
                        step = kk - s0
                        s = step % sw
                        q = step % sx
                        if cutlass.const_expr((self.debug & 4) == 0):
                            cute.arch.mbarrier_wait(w_full + s, (step // sw) & 1)
                        if cutlass.const_expr((self.debug & 2) == 0):
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
                        tcgen05.commit(acc_full + b, 3, tcgen05.CtaGroup.TWO)
        elif warp < 8:
            if warp == 0:
                cute.arch.alloc_tmem(self.tmem_cols, tmem_holder, is_two_cta=True)
            cute.arch.barrier(barrier_id=1, number_of_threads=32 * 9)
            tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, alignment=16,
                                                   ptr_to_buffer_holding_addr=tmem_holder)  # fmt: skip
            acc = cute.make_tensor(tmem_ptr, acc_fake.layout)
            if warp < 4:
                self.epilogue(acc, acc_full, acc_empty, xfer, warp, lane, v, g, wd, j, nd,
                              s0, s1, stage_addr, mY, mWs, flags, stamp, 0)  # fmt: skip
            else:
                self.epilogue(acc, acc_full, acc_empty, xfer, warp, lane, v, g, wd, j, nd,
                              s0, s1, stage_addr, mY, mWs, flags, stamp, 1)  # fmt: skip
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

    def token(self, col):
        """Token of accumulator column ``col`` (chunk-major): MMA chunk c's columns
        [0, h) are CTA 0's rows [c h, c h + h), [h, 2 h) CTA 1's (tokens T / 2 on)."""
        c, j = divmod(col, self.n_mma)
        h = self.n_mma // 2
        return c * h + j if j < h else self.rows // 2 + c * h + (j - h)

    @cute.jit
    def epilogue(
        self, acc: cute.Tensor, acc_full, acc_empty, xfer, ew: Int32, lane: Int32,
        v: Int32, g: Int32, wd: Int32, j: Int32, nd: Int32, s0: Int32, s1: Int32,
        stage_addr: Int32, mY: cute.Tensor, mWs: cute.Tensor, flags: Int64,
        stamp: Int64, half: cutlass.Constexpr[int],
    ):  # fmt: skip
        """Warp ew (0-7) reads TMEM lane quarter ew % 4 (weight row r of the CTA's 128)
        and accumulator columns [h T / 2, (h + 1) T / 2), h = ew // 4: the whole
        accumulator moves to registers and goes back to the MMA warp before any global
        store, so the stores overlap the next segment's MMAs. Per segment of the range
        [s0, s1): a later part of a tile publishes its FP32 partial ws[g][v]
        [column][row] and flags it; a first part (a whole tile, or the head of a
        shared one) adds the next range's partial (staged in the drained ring) and
        stores BF16."""
        rows, ks, total, n_mma, chunks = self.rows, self.ks, self.total, self.n_mma, self.chunks
        quarter = ew % 4
        hcols = rows // 2
        r = quarter * 32 + lane
        lane_bits = (quarter * 32) << 16
        base = [
            Int32(acc[(None, None, None, c)].iterator.toint()) + lane_bits
            for c in range(chunks)
        ]
        ws = mWs.iterator.toint()
        y = mY.iterator.toint()
        first_col = half * hcols  # accumulator column of vals[0] (chunk-major)
        lead = ew * 32 + lane == 0
        t0 = s0 // ks
        nseg = (s1 - 1) // ks - t0 + 1
        for si in cutlass.range(nseg, unroll=1):
            t = t0 + si
            lo_k = t * ks
            if lo_k < s0:
                lo_k = s0
            b = si % self.acc_bufs
            cute.arch.mbarrier_wait(acc_full + b, (si // self.acc_bufs) & 1)
            if cutlass.const_expr(self.debug & 8):
                if lead:
                    if si < 6:
                        _st_u64(stamp + (2 + 2 * si) * 8, _now())
            _fence_after()
            vals = []
            for jj in cutlass.range_constexpr(hcols // CHUNK):
                vals += list(_tmem_ld16w(self.taddr(base, b, first_col + jj * CHUNK)))
            _fence_before()
            if lane == 0:
                cute.arch.mbarrier_arrive(acc_empty + b, Int32(0))
            if lo_k != t * ks:  # a later part: publish the partial
                wsb = ws + (Int64(g * 2 + v) * rows + first_col) * 512 + r * 4
                for i in cutlass.range_constexpr(hcols):
                    _st_f32(wsb + i * 512, vals[i])
                _bar(2, 256)
                if lead:
                    _red_release_add(flags + Int64(g * 2 + v) * 4, 1)
            else:  # a whole tile or the head of a shared one
                last = (t + 1) * ks
                ncon = (last * nd + total - 1) // total - 1 - j
                if ncon < 0:
                    ncon = Int32(0)
                if ncon > 0:  # the next range's partial into the drained ring
                    if ew == 0:
                        if lane == 0:
                            flag = flags + Int64((g + 1) * 2 + v) * 4
                            while _ld_acquire(flag) == 0:
                                _nanosleep()
                            cute.arch.mbarrier_arrive_and_expect_tx(xfer, rows * 512)
                        cute.arch.sync_warp()
                        _fence_proxy_global()
                        per = rows * 512 // 32
                        src = ws + (Int64((g + 1) * 2 + v) * rows) * 512 + lane * per
                        _bulk_load(stage_addr + lane * per, src, per, xfer)
                    cute.arch.mbarrier_wait(xfer, 0)
                yrow = y + (wd * (self.n // 2) + t * 256 + v * 128 + r) * 2
                for i in cutlass.range_constexpr(hcols):
                    val = _add_lds_if(vals[i], stage_addr + ((first_col + i) * 128 + r) * 4,
                                      ncon)  # fmt: skip
                    tok = self.token(first_col + i)
                    _st_bf16(yrow + Int64(tok) * (self.ldo * 2), val)
                if ncon > 0:
                    _bar(2, 256)
                    if lead:
                        _st_relaxed(flags + Int64((g + 1) * 2 + v) * 4, 0)
            if cutlass.const_expr(self.debug & 8):
                if lead:
                    if si < 6:
                        _st_u64(stamp + (3 + 2 * si) * 8, _now())

    def taddr(self, base, b, col):
        """TMEM address of accumulator column ``col`` (chunk-major) of buffer b."""
        c, off = divmod(col, self.n_mma)
        return base[c] + b * self.rows + off


_COMPILED = {}


def die_pairs(device):
    """CTA pairs per die (two dies required)."""
    index = torch.device(device).index
    doms = dies.sm_domains(torch.cuda.current_device() if index is None else index)
    pairs = tuple(sum(1 for d in doms if d == die) // 2 for die in (0, 1))
    if min(pairs) == 0:
        raise ValueError(f"die-local GEMM needs two dies with SMs, got {pairs}")
    return doms, pairs


def place_halves(weight):
    """``weight`` [N, K] in one localized allocation: the 2 MiB chunks holding (most
    of) rows [0, N / 2) on die 0, the rest on die 1."""
    chunk = 2 << 20
    nbytes = weight.numel() * weight.element_size()
    half = nbytes // 2
    domains = [
        0 if i * chunk + chunk // 2 <= half else 1 for i in range(-(-nbytes // chunk))
    ]
    return dies.localized_like(weight.contiguous(), domains)


class PreparedWideSK:
    """``WideSK`` per prepared row count on one [N, K] BF16 weight. ``place``: copy it
    into die-homed halves (else the kernel streams it where it lies, e.g. striped).
    ``run(x, out)``: out [T, N] BF16 (row pitch ``ldo``)."""

    def __init__(self, weight, rows, ldo=None, place=True, w_issuers=3, x_issuers=2,
                 debug=0, x_stages=4):  # fmt: skip
        self.doms, self.pairs = die_pairs(weight.device)
        self.weight = place_halves(weight) if place else weight.contiguous()
        self.n, self.k = self.weight.shape
        self.rows = tuple(rows)
        self.ldo = ldo
        self.opts = (w_issuers, x_issuers, debug, x_stages)
        self.kernels, self.state = {}, {}

    def prepare(self):
        dev = self.weight.device
        half = self.n // 2
        self.w0 = self.weight[:half].data_ptr()
        self.w1 = self.weight[half:].data_ptr()
        self.table = torch.tensor(self.doms, dtype=torch.int32, device=dev)
        self.table_c = from_dlpack(self.table, assumed_align=16)
        npairs = sum(self.pairs)
        for rows in self.rows:
            gemm = WideSK(rows, self.n, self.k, self.pairs, self.ldo, *self.opts)
            x = torch.empty(rows, self.k, dtype=torch.bfloat16, device=dev)
            y = torch.empty(rows, self.ldo or self.n, dtype=torch.bfloat16, device=dev)
            ws = torch.empty(npairs, 2, rows, 128, dtype=torch.float32, device=dev)
            flags = torch.zeros(2 * npairs + 2 + 4 * npairs * STAMPS, dtype=torch.int32,
                                device=dev)  # fmt: skip
            args = (self.w0, self.w1, x.data_ptr(), x.data_ptr(), self._y(y[:, : self.n]),
                    from_dlpack(ws, assumed_align=16), from_dlpack(flags, assumed_align=16),
                    self.table_c)  # fmt: skip
            key = (rows, self.n, self.k, self.pairs, self.ldo, self.opts)
            if key not in _COMPILED:
                stream = _cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)
                _COMPILED[key] = cute.compile(gemm, *args, stream)
            self.kernels[rows] = _COMPILED[key]
            self.state[rows] = (ws, flags, args[5], args[6])
        return self

    @staticmethod
    def _k(t):
        return from_dlpack(t.unsqueeze(-1), assumed_align=16).mark_layout_dynamic(
            leading_dim=1
        )

    @staticmethod
    def _y(y):
        return from_dlpack(y, assumed_align=16).mark_layout_dynamic(leading_dim=1)

    def stamps(self, rows):
        """[2 * pairs, STAMPS] int64 stamps of the last ``debug & 8`` launch."""
        flags = self.state[rows][1]
        npairs = sum(self.pairs)
        return flags[2 * npairs + 2:].view(torch.int64).view(2 * npairs, STAMPS)

    def takes(self, x):
        return (
            x.dim() == 2 and x.shape[0] in self.kernels and x.dtype == torch.bfloat16
            and x.is_contiguous() and x.shape[1] == self.k
        )  # fmt: skip

    def run(self, x, out, x1=None):
        """x1: a copy of x homed on die 1 (x then on die 0), else the
        CTAs of both dies read x."""
        rows = x.shape[0]
        _, _, ws, flags = self.state[rows]
        stream = _cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
        xa = x.data_ptr()
        self.kernels[rows](self.w0, self.w1, xa, xa if x1 is None else x1.data_ptr(),
                           self._y(out), ws, flags, self.table_c, stream)  # fmt: skip
        return out
