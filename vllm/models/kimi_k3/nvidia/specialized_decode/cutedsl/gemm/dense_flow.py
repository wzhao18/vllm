# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The layer-0 dense MLP (gate_up + SiTU + down) as ONE persistent, die-local weight
stream (DenseFlow) for decode token counts on the SP shard.

At T <= 128 tokens the MLP streams its 1.45 GB of BF16 weights once and is bound by
the per-die HBM read rate. ``dense_stream.DenseStream`` runs it as two kernels (each
pays a launch, a prologue ramp and a stream-K tail) with the SiTU of the last gate_up
blocks after the gate_up stream. DenseFlow streams both weights in one grid:

- Phase A (gate_up + SiTU): a block is 64 SiTU outputs; its M128 weight tile holds,
  per 32-row TMEM lane quarter g, gate rows 64 b + 16 g + [0, 16) and the matching up
  rows (one 4-D TMA box over the stock [gate; up] layout), so a warp pairs gate and up
  of one output in lanes l and l ^ 16 (one shuffle). Stream-K over (block, K step)
  units of the SM's die; a block cut by a range end is finished by its head (the
  segment holding its first K step) from the other segment's FP32 partial, then
  SiTU, the BF16 h tile is stored and the block's ready flag released (epoch-tagged).
- Phase B (down): stream-K over (128-row output block, K step) units, in a per-range
  K order that takes the h blocks expected ready earliest first (host ``flow_tables``);
  a watcher warp acquires the ready flags in that order and the TMA warps load each h
  tile only after it, so the down weight streams on while the last gate_up blocks are
  finished. Heads sum the later ranges' partials (staged in the drained ring) in slot
  order: deterministic, FP32 accumulation, one BF16 rounding per GEMM, SiTU on the
  BF16-rounded gate / up as the stock op (only the FP32 summation order differs).
- Both phases share one ring (a stage: one 128 x 64 weight tile + one T x 64
  activation tile), one MMA warp (1-CTA M128 x N = T BF16) and two TMEM accumulators.
- Ranges per SM come from a table (die-local, ``flow_tables``). PDL: dependents are
  released right after setup (they take an SM only when this CTA leaves it).
"""

import functools
import math

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32, Int32, Int64
from cutlass.cute.nvgpu import cpasync, tcgen05

from ..common.dies import smid
from ..common.primitives import asm
from . import dense_stream as ds
from .dense_stream import (
    _fence_before,
    _fence_gpu,
    _ld_s32,
    _lds_s32,
    _rne_bf16,
    _st_bf16_rne,
    _st_release,
    _sts_s32,
    situ_constants,
    tanh,
)
from .wide import (
    _atom_add,
    _bar,
    _bulk_load,
    _fence_after,
    _fence_proxy_global,
    _ld_acquire,
    _ld_v4,
    _lds_v4,
    _now,
    _st_relaxed,
    _st_u64,
    _st_v4,
    _tmem_ld16,
    _tmem_wait16,
)

K_TILE = 64
CHUNK = 16
MAX_SLOTS = 512
MAX_DIE = 255
MAX_HB = 1024  # h blocks (64 outputs each)
MAX_WIN = 512  # down units of one SM (its K order lives in shared memory)
SMEM_BUDGET = 320 * 1024
RESERVE = 8 * 1024  # barriers, range bounds, K order, alignment
BULK = 32768
PRODUCERS = 3
EPI_WARPS = 24
EPI_THREADS = 32 * EPI_WARPS
EPI_GROUPS = 1  # epilogue groups finishing alternate segments (2 measured slower)
GROUP_WARPS = EPI_WARPS // EPI_GROUPS
GROUP_THREADS = 32 * GROUP_WARPS
EPI_SPLIT = GROUP_WARPS // 4  # warps of a group per TMEM lane quarter
WATCH_WARP = 4 + EPI_WARPS
STAMP_WORDS = 32
# sync words
END = 2  # CTAs done (the last one advances the epoch)
EPOCH = 3
FLAGS_A = 16
FLAGS_B = FLAGS_A + MAX_SLOTS
CLAIMS = FLAGS_B + MAX_SLOTS
READY = CLAIMS + MAX_SLOTS
TIMES = READY + MAX_HB
STAMP_BASE = TIMES + 4 * MAX_SLOTS
SYNC_WORDS = STAMP_BASE + MAX_SLOTS * STAMP_WORDS

NO_MMA = 1  # loads only (benchmarks)
NO_EPILOGUE = 2  # accumulators released unread; h never written (loads-only skeleton)
NO_SITU = 8  # timing only: h = gate * up instead of SiTU
NO_STORE = 16  # timing only: h / y never stored
FAST_TANH = 32  # timing only: approximate tanh
NO_FENCE = 64  # timing only: no fence before the h ready flag (st.release only)
H_BULK = 128  # h tile staged in shared memory, one bulk store per token row (A/B)
ROLLED = 256  # F1: SiTU finish as a rolled loop (one tanhf call site; whole-block ranges)
WHOLE = 1024  # host: whole gate_up blocks per SM (flow_tables(whole=True))
NO_EARLY_PDL = 512
NO_PREWAIT = 2048  # A/B: first ring issued after the dependency wait
OLD_FENCE = 4096  # A/B: fence.acq_rel.gpu before every ready / partial flag


@asm(Int32)
def _atom_cas(address, compare, value):
    code = "atom.relaxed.gpu.global.cas.b32 $0, [$1], $2, $3;"
    return code, "=r,l,r,r,~{memory}", (Int64(address), Int32(compare), Int32(value))


@asm(Int32)
def _ballot(flag):
    """Lanes with ``flag`` != 0 (full warp)."""
    code = "{ .reg .pred p; setp.ne.s32 p, $1, 0; vote.sync.ballot.b32 $0, p, 0xffffffff; }"
    return code, "=r,r", (Int32(flag),)


@asm(Int32)
def _low_bit(mask):
    """Index of the lowest set bit of ``mask``; 32 when none."""
    code = "{ .reg .b32 r; brev.b32 r, $1; bfind.shiftamt.u32 r, r; min.u32 $0, r, 32; }"
    return code, "=r,r", (Int32(mask),)


@asm()
def _sts_bf16_rne(saddr, value):
    """``value`` to shared memory as BF16, rounded as cvt.rn (``_rne_bf16``)."""
    code = "{ .reg .b32 u, l; .reg .b16 h; .reg .pred n; mov.b32 u, $1; shr.u32 l, u, 16; "
    code += "and.b32 l, l, 1; add.u32 u, u, l; add.u32 u, u, 0x7fff; shr.u32 u, u, 16; "
    code += "setp.nan.f32 n, $1, $1; @n mov.b32 u, 0x7fff; cvt.u16.u32 h, u; "
    code += "st.shared.b16 [$0], h; }"
    return code, "r,f,~{memory}", (Int32(saddr), Float32(value))


@asm()
def _bulk_store(gaddr, saddr, nbytes):
    code = "cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], $2;"
    return code, "l,r,r,~{memory}", (Int64(gaddr), Int32(saddr), Int32(nbytes))


@asm()
def _bulk_commit_wait():
    return "cp.async.bulk.commit_group; cp.async.bulk.wait_group 0;", "~{memory}", ()


@asm(Float32)
def _tmem_ld1(taddr):
    """One 32-bit TMEM column of this lane (tcgen05.ld 32x32b.x1) and its wait."""
    code = "tcgen05.ld.sync.aligned.32x32b.x1.b32 {$0}, [$1]; tcgen05.wait::ld.sync.aligned;"
    return code, "=f,r,~{memory}", (Int32(taddr),)


@asm()
def _sleep(ns):
    return "nanosleep.u32 $0;", "r,~{memory}", (Int32(ns),)


@asm()
def _sts_release(address, value):
    code = "st.release.cta.shared::cta.b32 [$0], $1;"
    return code, "r,r,~{memory}", (Int32(address), Int32(value))


@asm(Int64)
def _ld_u64(address):
    return "ld.global.u64 $0, [$1];", "=l,l,~{memory}", (Int64(address),)


@asm(Int32)
def _lds_acquire(address):
    return "ld.acquire.cta.shared::cta.b32 $0, [$1];", "=r,r,~{memory}", (Int32(address),)


def die_blocks(nb, s1):
    """Blocks homed on die 0 of ``nb``: half (each die streams its own HBM at the same
    rate whatever its SM count; an SM-proportional split leaves the die with more SMs
    streaming more bytes for the whole kernel)."""
    return nb // 2 if s1 else nb


# Shard rows of one DenseFlow (M128 x N = rows 1-CTA MMA, N <= 256; TMEM holds at least
# two accumulators of up to 256 columns): up to 256 since 2026-11-21 (C1152 / C1280:
# 144-176 requests per DP group = 144-176 shard rows per GPU at q 4).
MAX_ROWS = 256


def supports(rows, hidden, inter):
    t_mma = -(-rows // 16) * 16
    return (
        1 <= rows <= MAX_ROWS
        and inter % 64 == 0
        and hidden % 128 == 0
        and inter // 64 <= MAX_HB
        and hidden % K_TILE == 0
        and 2 * t_mma <= 512
    )


class DenseFlow:
    """One (rows, hidden, inter, SiTU constants) specialization; the token count of a
    call (``nrows`` <= ``rows``) is a launch argument."""

    def __init__(self, rows, hidden, inter, beta=1.0, linear_beta=-1.0, nbA0=None,
                 nbB0=None, two_dies=True, debug=0, host_stamps=0):  # fmt: skip
        assert supports(rows, hidden, inter), (rows, hidden, inter)
        self.rows, self.hidden, self.inter = rows, hidden, inter
        self.t_mma = -(-rows // 16) * 16
        self.t_pad = self.t_mma + 4
        self.consts = situ_constants(beta, linear_beta)
        self.debug = debug
        self.host_stamps = host_stamps  # diagnostics: host-mapped stamp buffer address
        self.nbA, self.nbB = inter // 64, hidden // 128
        self.ktA, self.ktB = hidden // K_TILE, inter // K_TILE
        if two_dies:
            self.nbA0 = self.nbA // 2 if nbA0 is None else nbA0
            self.nbB0 = self.nbB // 2 if nbB0 is None else nbB0
        else:
            self.nbA0, self.nbB0 = self.nbA, self.nbB
        self.mma_tiler = (128, self.t_mma, K_TILE)
        self.threads = 128 + EPI_THREADS + 32
        self.stage_bytes = 128 * K_TILE * 2 + self.t_mma * K_TILE * 2
        self.part = 128 * self.t_pad * 4
        self.hst_bytes = self.t_mma * 64 * 2 if debug & H_BULK else 0
        self.stages = min(16, (SMEM_BUDGET - RESERVE - self.hst_bytes) // self.stage_bytes)
        self.nstage_p = self.stages * self.stage_bytes // self.part
        assert self.stages >= 4, (rows, self.stages)
        # Accumulators in TMEM: as many as fit (<= 4), so the MMA runs up to three
        # segments ahead of an epilogue that waits on generic memory.
        self.nbuf = 4 if 4 * self.t_mma <= 512 else 2
        self.tmem_cols = 32
        while self.tmem_cols < self.nbuf * self.t_mma:
            self.tmem_cols *= 2

    def ws_bytes(self, slots):
        return 2 * MAX_SLOTS * self.part

    def units(self):
        """((A units of die 0, of die 1), (B units of die 0, of die 1))."""
        a = (self.nbA0 * self.ktA, (self.nbA - self.nbA0) * self.ktA)
        b = (self.nbB0 * self.ktB, (self.nbB - self.nbB0) * self.ktB)
        return a, b

    def stamp(self, sync, slot, index, value=None):
        """globaltimer (or ``value``) into stamp ``index`` of ``slot`` (debug STAMPS):
        0 claim, 1 MMA end, 2 last A stage, 3 last A epilogue, 4 watcher wait sum,
        5 watcher done, 6 end, 7 entry, 8 smid, 9 MMA full-wait sum in B, 10 in A,
        15 dependency wait returned (producer)."""
        if cutlass.const_expr(self.debug & ds.STAMPS):
            base = sync + STAMP_BASE * 4 + Int64(slot) * STAMP_WORDS * 4
            _st_u64(base + index * 8, _now() if value is None else value)

    @cute.jit
    def __call__(
        self, wa: cute.Tensor, x: cute.Tensor, wb: cute.Tensor, h: cute.Tensor,
        y: cute.Tensor, ws: cute.Tensor, sync: cute.Tensor, table: cute.Tensor,
        ranges: cute.Tensor, perm: cute.Tensor, s0: Int32, s1: Int32, nrows: Int32,
        stream: _cuda.CUstream,
    ):  # fmt: skip
        """wa gate_up (2 I, H, 1), x (T, H, 1), wb down (H, I, 1), h (Tmax, I, 1) K-major
        BF16; y [T, H] BF16; ws uint8 (``ws_bytes``); sync int32 [SYNC_WORDS] zeroed
        once; table / ranges / perm: ``flow_tables``."""
        dtype = cutlass.BFloat16
        major = tcgen05.OperandMajorMode.K
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            dtype, major, major, Float32, tcgen05.CtaGroup.ONE, self.mma_tiler[:2]
        )
        cluster_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,)
        )
        a_layout = sm100_utils.make_smem_layout_a(tiled_mma, self.mma_tiler, dtype, self.stages)
        b_layout = sm100_utils.make_smem_layout_b(tiled_mma, self.mma_tiler, dtype, self.stages)
        a_op = sm100_utils.cluster_shape_to_tma_atom_A((1, 1), tiled_mma.thr_id)
        b_op = sm100_utils.cluster_shape_to_tma_atom_B((1, 1), tiled_mma.thr_id)
        a_one = cute.slice_(a_layout, (None, None, None, 0))
        b_one = cute.slice_(b_layout, (None, None, None, 0))
        H, I = self.hidden, self.inter
        # gate_up rows of a block, per lane quarter g: gate 64 b + 16 g + i, up I + ...
        wa4 = cute.make_tensor(
            wa.iterator,
            cute.make_layout(((16, 2, 4, self.nbA), H, 1),
                             stride=((H, I * H, 16 * H, 64 * H), 1, 0)),
        )  # fmt: skip
        tma_a, tma_wa = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, wa4, a_one, self.mma_tiler, tiled_mma, cluster_vmnk.shape
        )
        tma_xa, tma_x = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, x, b_one, self.mma_tiler, tiled_mma, cluster_vmnk.shape
        )
        tma_b, tma_wb = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, wb, a_one, self.mma_tiler, tiled_mma, cluster_vmnk.shape
        )
        tma_ha, tma_h = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, h, b_one, self.mma_tiler, tiled_mma, cluster_vmnk.shape
        )
        self.tx_bytes = cute.size_in_bytes(dtype, a_one) + cute.size_in_bytes(dtype, b_one)

        @cute.struct
        class Storage:
            full: cute.struct.MemRange[cutlass.Int64, self.stages]
            empty: cute.struct.MemRange[cutlass.Int64, self.stages]
            acc_full: cute.struct.MemRange[cutlass.Int64, self.nbuf]
            acc_empty: cute.struct.MemRange[cutlass.Int64, self.nbuf]
            staged: cutlass.Int64
            permrdy: cutlass.Int64
            tmem_holder: cutlass.Int32
            info: cute.struct.MemRange[cutlass.Int32, 8]
            bnd: cute.struct.MemRange[cutlass.Int32, 2 * (MAX_DIE + 1)]
            order: cute.struct.MemRange[cutlass.Int32, MAX_WIN]
            hst: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, max(1, self.hst_bytes // 4)], 128
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(a_layout.outer)], 1024
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(b_layout.outer)], 1024
            ]

        self.storage = Storage
        self.kernel(
            tiled_mma, tma_a, tma_wa, tma_xa, tma_x, tma_b, tma_wb, tma_ha, tma_h,
            a_layout, b_layout, y, h, ws, sync, table, ranges, perm, s0, s1, nrows,
        ).launch(  # fmt: skip
            grid=(s0 + s1, 1, 1),
            block=[self.threads, 1, 1],
            smem=SMEM_BUDGET,
            stream=stream,
            use_pdl=True,
            min_blocks_per_mp=1,
            max_number_threads=[self.threads, 1, 1],
        )

    @cute.kernel
    def kernel(
        self, tiled_mma: cute.TiledMma, tma_a: cute.CopyAtom, mWa: cute.Tensor,
        tma_xa: cute.CopyAtom, mX: cute.Tensor, tma_b: cute.CopyAtom, mWb: cute.Tensor,
        tma_ha: cute.CopyAtom, mH: cute.Tensor, a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout, mY: cute.Tensor, mHp: cute.Tensor, mWs: cute.Tensor,
        mSync: cute.Tensor, mTable: cute.Tensor, mRanges: cute.Tensor, mPerm: cute.Tensor,
        s0: Int32, s1: Int32, nrows: Int32,
    ):  # fmt: skip
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        stages, ktA, ktB = self.stages, self.ktA, self.ktB

        smem = utils.SmemAllocator()
        st = smem.allocate(self.storage)
        full, empty = st.full.data_ptr(), st.empty.data_ptr()
        acc_full, acc_empty = st.acc_full.data_ptr(), st.acc_empty.data_ptr()
        staged, permrdy = st.staged.ptr, st.permrdy.ptr
        tmem_holder = st.tmem_holder.ptr
        info = Int32(st.info.data_ptr().toint())
        bnd = Int32(st.bnd.data_ptr().toint())
        order = Int32(st.order.data_ptr().toint())
        hst = Int32(st.hst.data_ptr().toint())
        sA = st.sA.get_tensor(a_layout.outer, swizzle=a_layout.inner)
        sB = st.sB.get_tensor(b_layout.outer, swizzle=b_layout.inner)
        sp = Int32(st.sA.data_ptr().toint())  # the ring: B partials after the stream
        sync = mSync.iterator.toint()

        if warp == 0:
            cpasync.prefetch_descriptor(tma_a)
            cpasync.prefetch_descriptor(tma_xa)
            cpasync.prefetch_descriptor(tma_b)
            cpasync.prefetch_descriptor(tma_ha)
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(stages):
                    cute.arch.mbarrier_init(full + i, 1)
                    cute.arch.mbarrier_init(empty + i, 1)
                for i in cutlass.range_constexpr(self.nbuf):
                    cute.arch.mbarrier_init(acc_full + i, 1)
                    cute.arch.mbarrier_init(acc_empty + i, GROUP_THREADS)
                cute.arch.mbarrier_init(staged, 1)
                cute.arch.mbarrier_init(permrdy, 1)
        if warp == 3:
            if lane == 0:
                t_entry = _now()
                entry = mTable.iterator.toint() + Int64(smid() & 1023) * 32
                pref = _ld_s32(entry)
                ua0 = _ld_s32(entry + 4)
                ua1 = _ld_s32(entry + 8)
                ub0 = _ld_s32(entry + 12)
                ub1 = _ld_s32(entry + 16)
                claims = sync + CLAIMS * 4
                slot = Int32(-1)
                if pref >= 0:
                    if _atom_cas(claims + Int64(pref) * 4, 0, 1) == 0:
                        slot = pref
                if slot < 0:  # this SM's slot is taken (or unmapped): any free slot
                    p = Int32(0)
                    while (slot < 0) & (p < s0 + s1):
                        if _atom_cas(claims + Int64(p) * 4, 0, 1) == 0:
                            slot = p
                        p += 1
                    die = Int32(cutlass.select_(slot >= s0, Int32(1), Int32(0)))
                    rr = slot - die * s0
                    rb = mRanges.iterator.toint() + Int64(die * (MAX_DIE + 1) + rr) * 4
                    ua0 = _ld_s32(rb)
                    ua1 = _ld_s32(rb + 4)
                    rb = rb + 2 * (MAX_DIE + 1) * 4
                    ub0 = _ld_s32(rb)
                    ub1 = _ld_s32(rb + 4)
                _sts_s32(info, slot)
                _sts_s32(info + 4, ua0)
                _sts_s32(info + 8, ua1)
                _sts_s32(info + 12, ub0)
                _sts_s32(info + 16, ub1)
                _sts_s32(info + 20, 0)  # h tiles of the down order confirmed ready
                self.stamp(sync, slot, 0)
                if cutlass.const_expr(self.debug & ds.STAMPS):
                    base = sync + STAMP_BASE * 4 + Int64(slot) * STAMP_WORDS * 4
                    _st_u64(base + 7 * 8, t_entry)
                    _st_u64(base + 8 * 8, Int64(smid()))
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()
        if cutlass.const_expr(not (self.debug & NO_EARLY_PDL)):
            cute.arch.griddepcontrol_launch_dependents()

        slot = _lds_s32(info)
        uA0 = _lds_s32(info + 4)
        uA1 = _lds_s32(info + 8)
        uB0 = _lds_s32(info + 12)
        uB1 = _lds_s32(info + 16)
        hready = info + 20
        d1 = slot >= s0
        r = slot
        sd = s0
        bbA = Int32(0)
        bbB = Int32(0)
        die = Int32(0)
        if d1:
            r = slot - s0
            sd = s1
            bbA = Int32(self.nbA0)
            bbB = Int32(self.nbB0)
            die = Int32(1)

        # Phase A order: the range's tail segment (published for the previous range's
        # head), then its head segment (the next range's tail is published just as
        # early), then the whole blocks: no split block waits for a range end.
        te = uA0
        if uA0 % ktA != 0:
            te = cutlass.min(uA1, (uA0 // ktA + 1) * ktA)
        hlo = uA1
        if uA1 % ktA != 0:
            hlo = cutlass.max(te, ((uA1 - 1) // ktA) * ktA)

        tile = cute.slice_(self.mma_tiler, (None, 0, None))
        gWa = cute.local_tile(mWa, tile, (None, None, None))
        gWb = cute.local_tile(mWb, tile, (None, None, None))
        btile = cute.slice_(self.mma_tiler, (0, None, None))
        gX = cute.local_tile(mX, btile, (None, None, None))
        gH = cute.local_tile(mH, btile, (None, None, None))
        thr_mma = tiled_mma.get_slice(0)
        tCgWa = thr_mma.partition_A(gWa)
        tCgWb = thr_mma.partition_A(gWb)
        tCgX = thr_mma.partition_B(gX)
        tCgH = thr_mma.partition_B(gH)
        sA3 = cute.group_modes(sA, 0, 3)
        sB3 = cute.group_modes(sB, 0, 3)
        tAsA, tAgWa = cpasync.tma_partition(tma_a, 0, cute.make_layout(1), sA3,
                                            cute.group_modes(tCgWa, 0, 3))  # fmt: skip
        _, tAgWb = cpasync.tma_partition(tma_b, 0, cute.make_layout(1), sA3,
                                         cute.group_modes(tCgWb, 0, 3))  # fmt: skip
        tBsB, tBgX = cpasync.tma_partition(tma_xa, 0, cute.make_layout(1), sB3,
                                           cute.group_modes(tCgX, 0, 3))  # fmt: skip
        _, tBgH = cpasync.tma_partition(tma_ha, 0, cute.make_layout(1), sB3,
                                        cute.group_modes(tCgH, 0, 3))  # fmt: skip
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        acc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.nbuf))

        if (warp < 4) & (warp != 2):  # TMA: stage g issued by producer g % PRODUCERS
            me = cutlass.min(warp, 2)
            # The first ring of gate_up weight tiles depends on nothing: issued before
            # the dependency wait (x tiles after it), so a CTA resident under its
            # predecessor starts with a full ring.
            npre = Int32(0)
            if cutlass.const_expr(not (self.debug & NO_PREWAIT)):
                npre = cutlass.min(Int32(stages), uA1 - uA0)
            n1 = te - uA0
            n2 = uA1 - hlo
            for g in cutlass.range(npre, unroll=1):
                if g % PRODUCERS == me:
                    up = te + g - n1 - n2
                    if g < n1:
                        up = uA0 + g
                    elif g < n1 + n2:
                        up = hlo + g - n1
                    bp_ = up // ktA
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(full + g, self.tx_bytes)
                    cute.copy(tma_a, tAgWa[(None, bbA + bp_, up - bp_ * ktA, 0)],
                              tAsA[(None, g)], tma_bar_ptr=full + g)  # fmt: skip
            cute.arch.griddepcontrol_wait()
            if cutlass.const_expr(self.debug & ds.STAMPS):
                if warp == 0:
                    with cute.arch.elect_one():
                        self.stamp(sync, slot, 15)
            for g in cutlass.range(npre, unroll=1):
                if g % PRODUCERS == me:
                    up = te + g - n1 - n2
                    if g < n1:
                        up = uA0 + g
                    elif g < n1 + n2:
                        up = hlo + g - n1
                    cute.copy(tma_xa, tBgX[(None, 0, up - (up // ktA) * ktA, 0)],
                              tBsB[(None, g)], tma_bar_ptr=full + g)  # fmt: skip
            gi = Int32(0)
            turn = Int32(0)
            s = Int32(0)
            phase = Int32(1)
            for alo, ahi in ((uA0, te), (hlo, uA1), (te, hlo)):
                u = alo
                while u < ahi:  # phase A: gate_up weight tile + x tile
                    b = u // ktA
                    kb1 = cutlass.min(ktA, ahi - b * ktA)
                    gb = bbA + b
                    for kb in cutlass.range(u - b * ktA, kb1, unroll=1):
                        if (turn == me) & (gi >= npre):
                            cute.arch.mbarrier_wait(empty + s, phase)
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive_and_expect_tx(full + s,
                                                                        self.tx_bytes)  # fmt: skip
                            cute.copy(tma_a, tAgWa[(None, gb, kb, 0)], tAsA[(None, s)],
                                      tma_bar_ptr=full + s)  # fmt: skip
                            cute.copy(tma_xa, tBgX[(None, 0, kb, 0)], tBsB[(None, s)],
                                      tma_bar_ptr=full + s)  # fmt: skip
                        gi += 1
                        turn += 1
                        if turn == PRODUCERS:
                            turn = Int32(0)
                        s += 1
                        if s == stages:
                            s = Int32(0)
                            phase = phase ^ 1
                    u = (b + 1) * ktA
            cute.arch.mbarrier_wait(permrdy, 0)
            u = uB0
            while u < uB1:  # phase B: down weight tile + h tile (once ready)
                ob = u // ktB
                kb1 = cutlass.min(ktB, uB1 - ob * ktB)
                gob = bbB + ob
                for kp in cutlass.range(u - ob * ktB, kb1, unroll=1):
                    if turn == me:
                        j = ob * ktB + kp - uB0
                        k = _lds_s32(order + j * 4)
                        cute.arch.mbarrier_wait(empty + s, phase)
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(full + s, self.tx_bytes)
                        cute.copy(tma_b, tAgWb[(None, gob, k, 0)], tAsA[(None, s)],
                                  tma_bar_ptr=full + s)  # fmt: skip
                        if cutlass.const_expr(not (self.debug & NO_EPILOGUE)):
                            while _lds_acquire(hready) <= j:
                                _sleep(256)
                            _fence_proxy_global()
                        cute.copy(tma_ha, tBgH[(None, 0, k, 0)], tBsB[(None, s)],
                                  tma_bar_ptr=full + s)  # fmt: skip
                    turn += 1
                    if turn == PRODUCERS:
                        turn = Int32(0)
                    s += 1
                    if s == stages:
                        s = Int32(0)
                        phase = phase ^ 1
                u = (ob + 1) * ktB
        elif warp == 2:  # MMA
            cute.arch.barrier(barrier_id=1, number_of_threads=32 + EPI_THREADS)
            tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, alignment=16,
                                                   ptr_to_buffer_holding_addr=tmem_holder)  # fmt: skip
            acc = cute.make_tensor(tmem_ptr, acc_fake.layout)
            kblocks = cute.size(tCrA, mode=[2])
            s = Int32(0)
            phase = Int32(0)
            seg = Int32(0)
            waits = [Int64(0), Int64(0)]
            pieces = ((uA0, te, ktA, 0), (hlo, uA1, ktA, 0), (te, hlo, ktA, 0),
                      (uB0, uB1, ktB, 1))  # fmt: skip
            for idx, (lo, hi, kt, ph) in enumerate(pieces):
                wsum = Int64(0)
                u = lo
                while u < hi:
                    b = u // kt
                    kb0 = u - b * kt
                    kb1 = cutlass.min(kt, hi - b * kt)
                    buf = seg % self.nbuf
                    cute.arch.mbarrier_wait(acc_empty + buf, ((seg // self.nbuf) & 1) ^ 1)
                    _fence_after()
                    for kk in cutlass.range(kb1 - kb0, unroll=1):
                        if cutlass.const_expr(self.debug & ds.STAMPS):
                            tw = _now()
                            cute.arch.mbarrier_wait(full + s, phase)
                            wsum = wsum + (_now() - tw)
                        else:
                            cute.arch.mbarrier_wait(full + s, phase)
                        if (seg == 0) & (kk == 0):
                            with cute.arch.elect_one():
                                _st_u64(sync + TIMES * 4 + Int64(slot) * 16, _now())
                        if cutlass.const_expr(not (self.debug & NO_MMA)):
                            acc_a = acc[(None, None, None, buf)]
                            for kb in cutlass.range_constexpr(kblocks):
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, (kk + kb) != 0)
                                cute.gemm(tiled_mma, acc_a, tCrA[(None, None, kb, s)],
                                          tCrB[(None, None, kb, s)], acc_a)  # fmt: skip
                        with cute.arch.elect_one():
                            tcgen05.commit(empty + s)
                        s += 1
                        if s == stages:
                            s = Int32(0)
                            phase = phase ^ 1
                    with cute.arch.elect_one():
                        tcgen05.commit(acc_full + buf)
                    seg += 1
                    u = (b + 1) * kt
                waits[ph] = waits[ph] + wsum
                if cutlass.const_expr(idx == 2):
                    with cute.arch.elect_one():
                        self.stamp(sync, slot, 2)
            with cute.arch.elect_one():
                self.stamp(sync, slot, 9, waits[1])
                self.stamp(sync, slot, 10, waits[0])
                _st_u64(sync + TIMES * 4 + Int64(slot) * 16 + 8, _now())
                self.stamp(sync, slot, 1)
        elif warp == WATCH_WARP:  # the h tiles of this CTA's down order, as they turn ready
            cute.arch.griddepcontrol_wait()
            target = _ld_s32(sync + EPOCH * 4) + 1
            cute.arch.mbarrier_wait(permrdy, 0)
            n = uB1 - uB0
            done = Int32(0)
            polls = Int32(0)
            while done < n:  # one acquire per lane: up to 32 entries a round trip
                j = done + lane
                late = Int32(0)
                if cutlass.const_expr(not (self.debug & NO_EPILOGUE)):
                    if j < n:
                        k = _lds_s32(order + j * 4)
                        if _ld_acquire(sync + (READY + Int64(k)) * 4) != target:
                            late = Int32(1)
                adv = cutlass.min(_low_bit(_ballot(late)), n - done)
                cute.arch.sync_warp()  # every lane's acquire before the release
                if adv > 0:
                    if lane == 0:
                        _sts_release(hready, done + adv)
                else:
                    _sleep(64)
                done = done + adv
                polls += 1
            if lane == 0:
                self.stamp(sync, slot, 4, Int64(polls))
                self.stamp(sync, slot, 5)
        elif warp >= 4:  # epilogue
            if warp == 4:
                cute.arch.alloc_tmem(self.tmem_cols, tmem_holder)
            cute.arch.barrier(barrier_id=1, number_of_threads=32 + EPI_THREADS)
            tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, alignment=16,
                                                   ptr_to_buffer_holding_addr=tmem_holder)  # fmt: skip
            acc = cute.make_tensor(tmem_ptr, acc_fake.layout)
            tid = (warp - 4) * 32 + lane
            # The die's A / B range bounds and this CTA's down K order.
            if tid <= sd:
                rd = mRanges.iterator.toint() + Int64(die * (MAX_DIE + 1) + tid) * 4
                _sts_s32(bnd + tid * 4, _ld_s32(rd))
                _sts_s32(bnd + (MAX_DIE + 1 + tid) * 4,
                         _ld_s32(rd + 2 * (MAX_DIE + 1) * 4))  # fmt: skip
            prow = mPerm.iterator.toint() + Int64(die) * mPerm.shape[1] * 4
            for j in cutlass.range(tid, uB1 - uB0, EPI_THREADS, unroll=1):
                _sts_s32(order + j * 4, _ld_s32(prow + Int64(uB0 + j) * 4))
            _bar(3, EPI_THREADS)
            if tid == 0:
                cute.arch.mbarrier_arrive(permrdy)
            cute.arch.griddepcontrol_wait()
            epoch = _ld_s32(sync + EPOCH * 4)
            self.epilogue(acc, acc_full, acc_empty, staged, warp - 4, lane, slot, r, sd,
                          d1, s0, bnd, uA0, uA1, uB0, uB1, bbA, bbB, sp, mY, mHp, mWs, sync,
                          nrows, epoch, te, hlo, hst)  # fmt: skip
            if warp == 4:
                cute.arch.relinquish_tmem_alloc_permit()
            _bar(2, EPI_THREADS)
            if warp == 4:
                cute.arch.dealloc_tmem(tmem_ptr, self.tmem_cols)
                if lane == 0:
                    self.stamp(sync, slot, 6)
                    _st_relaxed(sync + CLAIMS * 4 + Int64(slot) * 4, 0)
                    if _atom_add(sync + END * 4, 1) == s0 + s1 - 1:  # every CTA done
                        _st_relaxed(sync + END * 4, 0)
                        _st_relaxed(sync + EPOCH * 4, epoch + 1)
        if cutlass.const_expr(self.host_stamps != 0 and (self.debug & ds.STAMPS)):
            # diagnostics: this CTA's stamps + times to a host-mapped buffer at exit
            cute.arch.barrier()
            if warp == 0:
                if lane < STAMP_WORDS // 2:
                    src = sync + STAMP_BASE * 4 + Int64(slot) * STAMP_WORDS * 4 + lane * 8
                    dst = Int64(self.host_stamps) + Int64(slot) * STAMP_WORDS * 4 + lane * 8
                    _st_u64(dst, _ld_u64(src))
                elif lane < STAMP_WORDS // 2 + 2:
                    src = sync + TIMES * 4 + Int64(slot) * 16 + (lane - STAMP_WORDS // 2) * 8
                    dst = (Int64(self.host_stamps) + MAX_SLOTS * STAMP_WORDS * 4
                           + Int64(slot) * 16 + (lane - STAMP_WORDS // 2) * 8)  # fmt: skip
                    _st_u64(dst, _ld_u64(src))

    @cute.jit
    def epilogue(
        self, acc: cute.Tensor, acc_full, acc_empty, staged, ew: Int32, lane: Int32,
        slot: Int32, r: Int32, sd: Int32, d1, s0: Int32, bnd: Int32, uA0: Int32, uA1: Int32,
        uB0: Int32, uB1: Int32, bbA: Int32, bbB: Int32, sp: Int32, mY: cute.Tensor,
        mH: cute.Tensor, mWs: cute.Tensor, sync: Int64, nrows: Int32, epoch: Int32,
        te: Int32, hlo: Int32, hst: Int32,
    ):  # fmt: skip
        """Epilogue warp ew is warp gw = ew % GROUP_WARPS of group ew // GROUP_WARPS
        (the groups finish alternate segments, so one block's SiTU / stores overlap the
        next one's); it owns TMEM lane quarter gw % 4 and the 16-token chunks c with
        c % EPI_SPLIT == gw // 4."""
        grp = ew // GROUP_WARPS
        gw = ew % GROUP_WARPS
        quad = gw % 4
        half = gw // 4
        q = quad * 32 + lane
        tid = gw * 32 + lane
        lane_bits = (quad * 32) << 16
        ws = mWs.iterator.toint()
        slot_base = Int32(cutlass.select_(d1, s0, Int32(0)))
        seg = Int32(0)
        pieces = ((uA0, te, 0), (hlo, uA1, 0), (te, hlo, 0), (uB0, uB1, 1))
        for idx, (lo, hi, ph) in enumerate(pieces):
            kt = self.ktA if ph == 0 else self.ktB
            bb = bbA if ph == 0 else bbB
            u = lo
            while u < hi:
                b = u // kt
                kb0 = u - b * kt
                if seg % EPI_GROUPS == grp:
                    buf = seg % self.nbuf
                    col = Int32(acc[(None, None, None, buf)].iterator.toint()) + lane_bits
                    cute.arch.mbarrier_wait(acc_full + buf, (seg // self.nbuf) & 1)
                    _fence_after()
                    if cutlass.const_expr(not (self.debug & NO_EPILOGUE)):
                        self.segment(ph, kb0, b * kt, col, slot, r, bnd, slot_base, ws, sp,
                                     bb + b, q, quad, lane, tid, half, mY, mH, sync, nrows,
                                     epoch, staged, grp, hst)  # fmt: skip
                    _fence_before()
                    cute.arch.mbarrier_arrive(acc_empty + buf)
                seg += 1
                u = (b + 1) * kt
            if cutlass.const_expr(idx == 2):
                if tid == 0:
                    self.stamp(sync, slot, 3)

    @cute.jit
    def segment(
        self, ph: cutlass.Constexpr, kb0: Int32, bstart: Int32, col: Int32, slot: Int32,
        r: Int32, bnd: Int32, slot_base: Int32, ws: Int64, sp: Int32, gb: Int32, q: Int32,
        quad: Int32, lane: Int32, tid: Int32, half: Int32, mY: cute.Tensor,
        mH: cute.Tensor, sync: Int64, nrows: Int32, epoch: Int32, staged, grp: Int32,
        hst: Int32,
    ):  # fmt: skip
        t_mma, t_pad, part = self.t_mma, self.t_pad, self.part
        kt = self.ktA if ph == 0 else self.ktB
        flags = sync + (FLAGS_A if ph == 0 else FLAGS_B) * 4
        wsp = ws + (0 if ph == 0 else MAX_SLOTS * part)
        bp = bnd + (0 if ph == 0 else (MAX_DIE + 1) * 4)  # this phase's die bounds
        chunks = t_mma // CHUNK
        if kb0 > 0:  # not the head: publish the FP32 partial (its range's first segment)
            dst = wsp + Int64(slot) * part
            for c in cutlass.range_constexpr(chunks):
                if c % EPI_SPLIT == half:
                    v = _tmem_wait16(*_tmem_ld16(col + c * CHUNK))
                    row = dst + (q * t_pad + c * CHUNK) * 4
                    for j in cutlass.range_constexpr(CHUNK // 4):
                        _st_v4(row + j * 16, v[4 * j], v[4 * j + 1], v[4 * j + 2],
                               v[4 * j + 3])  # fmt: skip
            self.gbar(grp)  # every thread's stores before the release (cumulative)
            if tid == 0:
                if cutlass.const_expr(self.debug & OLD_FENCE):
                    _fence_gpu()
                _st_release(flags + Int64(slot) * 4, 1)
        else:  # the head: own accumulator + the later ranges' partials, in slot order
            last = r
            while _lds_s32(bp + (last + 1) * 4) < bstart + kt:  # bound[sd] = units
                last += 1
            nst = Int32(0)
            if cutlass.const_expr(ph == 1):  # partials staged in the drained ring
                for p in cutlass.range(r + 1, last + 1, unroll=1):
                    if (_lds_s32(bp + p * 4) < _lds_s32(bp + p * 4 + 4)) & (
                            nst < self.nstage_p):  # fmt: skip
                        nst += 1
            # Every partial acquired in ONE warp, one lane per later range, polled in
            # parallel: a generic round trip queues behind the weight streams (D25), so
            # per-partial (or per-thread) acquire chains cost one round trip each.
            if tid < 32:
                pl = r + 1 + tid
                late = Int32(0)
                if pl <= last:
                    if _lds_s32(bp + pl * 4) < _lds_s32(bp + pl * 4 + 4):
                        late = Int32(1)
                while _ballot(late) != 0:
                    if late != 0:
                        if _ld_acquire(flags + Int64(slot_base + pl) * 4) != 0:
                            late = Int32(0)
                if tid == 0:
                    if nst > 0:
                        _fence_proxy_global()
                        cute.arch.mbarrier_arrive_and_expect_tx(staged, nst * part)
                        j = Int32(0)
                        for p in cutlass.range(r + 1, last + 1, unroll=1):
                            if (_lds_s32(bp + p * 4) < _lds_s32(bp + p * 4 + 4)) & (j < nst):
                                src = wsp + Int64(slot_base + p) * part
                                for o in cutlass.range_constexpr(-(-part // BULK)):
                                    n_o = min(BULK, part - o * BULK)
                                    _bulk_load(sp + j * part + o * BULK, src + o * BULK,
                                               n_o, staged)  # fmt: skip
                                j += 1
            self.gbar(grp)
            if nst > 0:
                cute.arch.mbarrier_wait(staged, 0)
            tf = _now()
            self.finish(ph, col, r, last, bp, slot_base, wsp, sp, nst, gb, q, quad, lane,
                        half, mY, mH, nrows, sync, slot, hst)  # fmt: skip
            if cutlass.const_expr((self.debug & H_BULK) and ph == 0):
                # the staged h tile: one bulk store per token row, complete before
                # the ready flag
                self.gbar(grp)
                if tid < 32:
                    hp = mH.iterator.toint()
                    for i in cutlass.range_constexpr(-(-self.t_mma // 32)):
                        t = tid + i * 32
                        if t < nrows:
                            _bulk_store(hp + (Int64(t) * self.inter + gb * 64) * 2,
                                        hst + t * 128, 128)  # fmt: skip
                    _bulk_commit_wait()
                    _fence_proxy_global()
            if cutlass.const_expr((self.debug & ds.STAMPS) and ph == 0):
                if tid == 0:
                    base = sync + STAMP_BASE * 4 + Int64(slot) * STAMP_WORDS * 4 + 11 * 8
                    _st_u64(base, _ld_u64(base) + (_now() - tf))
            self.gbar(grp)
            if tid == 0:
                for p in cutlass.range(r + 1, last + 1, unroll=1):
                    _st_relaxed(flags + Int64(slot_base + p) * 4, 0)
                if cutlass.const_expr(ph == 0):  # the h block is complete
                    if cutlass.const_expr(self.debug & OLD_FENCE):
                        _fence_gpu()
                    _st_release(sync + (READY + Int64(gb)) * 4, epoch + 1)

    @cute.jit
    def gbar(self, grp: Int32):
        """Named barrier of epilogue group ``grp``."""
        if grp == 0:
            _bar(3, GROUP_THREADS)
        else:
            _bar(4, GROUP_THREADS)

    @cute.jit
    def finish(
        self, ph: cutlass.Constexpr, col: Int32, r: Int32, last: Int32, bp: Int32,
        slot_base: Int32, wsp: Int64, sp: Int32, nst: Int32, gb: Int32, q: Int32,
        quad: Int32, lane: Int32, half: Int32, mY: cute.Tensor, mH: cute.Tensor,
        nrows: Int32, sync: Int64, slot: Int32, hst: Int32 = Int32(0),
    ):  # fmt: skip
        """Own accumulator + partials of ranges r + 1 .. last in slot order. Phase A
        the partials of every chunk this thread finishes are loaded up front (one memory
        round trip per partial: generic loads queue behind the SM's weight stream) and
        summed (own + (p1 + p2 ...)), then SiTU of the BF16-rounded gate / up (lanes l,
        l ^ 16) into h. Phase B (the ring drained; the first ``nst`` partials staged
        there): BF16 y."""
        t_pad, part = self.t_pad, self.part
        chunks = self.t_mma // CHUNK
        y = mY.iterator.toint()
        hp = mH.iterator.toint()
        if cutlass.const_expr(ph == 0):
            half_beta, inv_beta, lin, inv_lin = (Float32(v) for v in self.consts)
            up_lane = lane >= 16
            nk = -(-chunks // EPI_SPLIT)
            pre = cute.make_rmem_tensor(nk * CHUNK, Float32)
            tmp = cute.make_rmem_tensor(nk * CHUNK, Float32)
            for i in cutlass.range_constexpr(nk * CHUNK):
                pre[i] = Float32(0)
                tmp[i] = Float32(0)
            has_p = last > r
            if cutlass.const_expr(self.debug & ROLLED):
                # F1: one SiTU call site; gate lane (l < 16) finishes token c*16 + j,
                # up lane token c*16 + 8 + j (whole-block ranges: no partials)
                hp_r = mH.iterator.toint()
                for kr in cutlass.range_constexpr(nk):
                    cr = kr * EPI_SPLIT + half
                    if cr < chunks:
                        for jr in cutlass.range(CHUNK // 2, unroll=1):
                            tgr = cr * CHUNK + jr
                            vg = _tmem_ld1(col + tgr)
                            vu = _tmem_ld1(col + tgr + CHUNK // 2)
                            sendr = cutlass.select_(up_lane, vg, vu)
                            recvr = cute.arch.shuffle_sync_bfly(sendr, 16)
                            gr = _rne_bf16(cutlass.select_(up_lane, recvr, vg))
                            upr = _rne_bf16(cutlass.select_(up_lane, vu, recvr))
                            ar = half_beta * tanh(gr * inv_beta)
                            ar = ar * (Float32(1) + tanh(gr * Float32(0.5)))
                            if cutlass.const_expr(self.consts[2] > 0):
                                upr = lin * tanh(upr * inv_lin)
                            tr = tgr + Int32(cutlass.select_(up_lane, Int32(CHUNK // 2), Int32(0)))
                            if tr < nrows:
                                hcolr = gb * 64 + quad * 16 + (lane & 15)
                                _st_bf16_rne(hp_r + (Int64(tr) * self.inter + hcolr) * 2, ar * upr)
                return
            for p in cutlass.range(r + 1, last + 1, unroll=1):  # one round trip each
                if _lds_s32(bp + p * 4) < _lds_s32(bp + p * 4 + 4):
                    base = wsp + Int64(slot_base + p) * part + q * t_pad * 4
                    for k in cutlass.range_constexpr(nk):
                        c = k * EPI_SPLIT + half
                        if c < chunks:
                            for v4 in cutlass.range_constexpr(CHUNK // 4):
                                vals = _ld_v4(base + (c * CHUNK + v4 * 4) * 4)
                                for e in cutlass.range_constexpr(4):
                                    tmp[k * CHUNK + v4 * 4 + e] = vals[e]
                    for i in cutlass.range_constexpr(nk * CHUNK):
                        pre[i] = pre[i] + tmp[i]
            for k in cutlass.range_constexpr(nk):
                c = k * EPI_SPLIT + half
                if c < chunks:
                    own = _tmem_wait16(*_tmem_ld16(col + c * CHUNK))
                    tot = cute.make_rmem_tensor(CHUNK, Float32)
                    for i in cutlass.range_constexpr(CHUNK):
                        tot[i] = cutlass.select_(has_p, own[i] + pre[k * CHUNK + i], own[i])
                    # lane l < 16 holds gate rows, l + 16 the matching up rows: the gate
                    # lane finishes tokens [0, 8) of the chunk, the up lane [8, 16).
                    out = []
                    for i in cutlass.range_constexpr(CHUNK // 2):
                        send = cutlass.select_(up_lane, tot[i], tot[CHUNK // 2 + i])
                        recv = cute.arch.shuffle_sync_bfly(send, 16)
                        g = _rne_bf16(cutlass.select_(up_lane, recv, tot[i]))
                        upv = _rne_bf16(cutlass.select_(up_lane, tot[CHUNK // 2 + i], recv))
                        if cutlass.const_expr(self.debug & NO_SITU):
                            out.append(g * upv)
                        else:
                            th = tanh
                            if cutlass.const_expr(self.debug & FAST_TANH):
                                th = functools.partial(cute.math.tanh, fastmath=True)
                            a = half_beta * th(g * inv_beta)
                            a = a * (Float32(1) + th(g * Float32(0.5)))
                            if cutlass.const_expr(self.consts[2] > 0):
                                upv = lin * th(upv * inv_lin)
                            out.append(a * upv)
                    t0 = c * CHUNK + Int32(cutlass.select_(up_lane, Int32(CHUNK // 2), Int32(0)))
                    hcol = gb * 64 + quad * 16 + (lane & 15)
                    lim = nrows - 1000000 if cutlass.const_expr(self.debug & NO_STORE) else nrows
                    for i in cutlass.range_constexpr(CHUNK // 2):
                        t = t0 + i
                        if cutlass.const_expr(self.debug & H_BULK):
                            _sts_bf16_rne(hst + (t * 64 + quad * 16 + (lane & 15)) * 2, out[i])
                        elif t < lim:
                            _st_bf16_rne(hp + (Int64(t) * self.inter + hcol) * 2, out[i])
        else:
            for c in cutlass.range_constexpr(chunks):
                if c % EPI_SPLIT == half:
                    tot = cute.make_rmem_tensor(CHUNK, Float32)
                    own = _tmem_wait16(*_tmem_ld16(col + c * CHUNK))
                    for i in cutlass.range_constexpr(CHUNK):
                        tot[i] = own[i]
                    off = (q * t_pad + c * CHUNK) * 4
                    j = Int32(0)
                    for p in cutlass.range(r + 1, last + 1, unroll=1):
                        if _lds_s32(bp + p * 4) < _lds_s32(bp + p * 4 + 4):
                            if j < nst:
                                src = sp + j * part + off
                                for v4 in cutlass.range_constexpr(CHUNK // 4):
                                    vals = _lds_v4(src + v4 * 16)
                                    for e in cutlass.range_constexpr(4):
                                        tot[v4 * 4 + e] = tot[v4 * 4 + e] + vals[e]
                            else:
                                src = wsp + Int64(slot_base + p) * part + off
                                for v4 in cutlass.range_constexpr(CHUNK // 4):
                                    vals = _ld_v4(src + v4 * 16)
                                    for e in cutlass.range_constexpr(4):
                                        tot[v4 * 4 + e] = tot[v4 * 4 + e] + vals[e]
                            j += 1
                    for i in cutlass.range_constexpr(CHUNK):
                        t = c * CHUNK + i
                        if t < nrows:
                            _st_bf16_rne(y + (Int64(t) * self.hidden + gb * 128 + q) * 2,
                                         tot[i])  # fmt: skip


_COMPILED = {}


def compile_dense_flow(op: DenseFlow, wa, x, wb, h, y, ws, sync, table, ranges, perm, s0,
                       s1, stream):  # fmt: skip
    """Compile once per specialization; returns (launcher, tensor args): call as
    ``fn(*args, Int32(s0), Int32(s1), Int32(nrows), stream)``."""
    as_cute = ds.as_cute
    key = (op.rows, op.hidden, op.inter, op.consts, op.debug, op.nbA0, op.nbB0,
           op.host_stamps)  # fmt: skip
    args = (as_cute(wa, k_major=True), as_cute(x, k_major=True), as_cute(wb, k_major=True),
            as_cute(h, k_major=True), as_cute(y, rows=True), as_cute(ws), as_cute(sync),
            as_cute(table), as_cute(ranges), as_cute(perm))  # fmt: skip
    fn = _COMPILED.get(key)
    if fn is None:
        fn = cute.compile(op, *args, Int32(s0), Int32(s1), Int32(x.shape[0]), stream)
        _COMPILED[key] = fn
    return fn, args


def _scatter(n):
    """A multiplier coprime to ``n`` near ``n`` / golden ratio: neighbouring SMs (one
    TPC / GPC) take far-apart slots."""
    p = max(1, int(round(n * 0.618)))
    while math.gcd(p, n) != 1:
        p += 1
    return p


def slot_sms(domains):
    """smid of each slot: die 0's slots [0, s0), then die 1's; within a die, the SM of
    smid rank i takes slot (i * _scatter(s)) % s."""
    groups = [[s for s, d in enumerate(domains) if d == die] for die in (0, 1)]
    if not groups[1]:
        groups = [[s for s, d in enumerate(domains) if d >= 0], []]
    out = []
    for group in groups:
        n = len(group)
        if not n:
            continue
        slots = [0] * n
        mult = _scatter(n)
        for i, sm in enumerate(group):
            slots[(i * mult) % n] = sm
        out += slots
    return out


def _bounds(units, weights):
    total, acc, out = sum(weights), 0.0, [0]
    for w in weights:
        acc += w
        out.append(int(round(units * acc / total)))
    out[-1] = units
    return out


def _split_work(blocks, kt, ub, w):
    """(A bounds, B bounds) of one die: every SM gets WHOLE gate_up blocks (no
    gate_up partials: a split block would wait on a generic-memory round trip that
    queues behind the SM's own weight stream) and a down range that evens out its
    total units (A + B), both in proportion to ``w``."""
    n = len(w)
    if not n:
        return [0], [0]
    tw = sum(w)
    total = blocks * kt + ub
    want = [blocks * wi / tw for wi in w]
    nb = [int(x) for x in want]
    # The extra blocks go to evenly spread slots, so short down windows (SMs with an
    # extra block) alternate with long ones: every output block spans few down ranges
    # (each later range is one partial its head stages).
    order = sorted(range(n), key=lambda i: want[i] - nb[i], reverse=True)
    extra = blocks - sum(nb)
    if extra > 0 and all(abs(x - want[0]) < 1e-9 for x in want):
        order = sorted({(i * n) // extra for i in range(extra)})
    for i in order[:extra]:
        nb[i] += 1
    share = [total * wi / tw for wi in w]
    bu = [max(0.0, share[i] - nb[i] * kt) for i in range(n)]
    tb = sum(bu) or 1.0
    a, b, acc, cum = [0], [0], 0.0, 0
    for i in range(n):
        cum += nb[i] * kt
        a.append(cum)
        acc += bu[i]
        b.append(int(round(ub * acc / tb)))
    b[-1] = ub
    return a, b


def flow_tables(op: DenseFlow, domains, weights=None, device=None, whole=False):
    """(table int32 [1024, 8], ranges int32 [4 (MAX_DIE + 1)], perm int32 [2, U]):
    slot r of die d (``dense_stream.slot_sms``) streams A units [a_d[r], a_d[r + 1])
    and B units [b_d[r], b_d[r + 1]) of its die, in proportion to ``weights[smid]``
    (default equal); table row smid = (slot, uA0, uA1, uB0, uB1, 0, 0, 0), -1 for an
    SM without a slot. perm[d][u] = the K step (h block) of die d's down unit u: per
    output block, the h blocks dealt to its ranges in proportion and, within a range,
    in the order they are expected ready (each A segment's end as a fraction of its
    SM's A + B work)."""
    import torch

    sms = slot_sms(domains)
    s0 = sum(1 for d in domains if d == 0)
    if not any(d == 1 for d in domains):
        s0 = len(sms)
    groups = (sms[:s0], sms[s0:])
    (ua0, ua1), (ub0, ub1) = op.units()
    bounds = {}
    for die, group in enumerate(groups):
        w = [1.0 if weights is None else max(float(weights[sm]), 1e-9) for sm in group]
        if whole and group:
            bounds[("a", die)], bounds[("b", die)] = _split_work(
                (ua0, ua1)[die] // op.ktA, op.ktA, (ub0, ub1)[die], w)
        else:
            bounds[("a", die)] = _bounds((ua0, ua1)[die], w) if group else [0]
            bounds[("b", die)] = _bounds((ub0, ub1)[die], w) if group else [0]
    table = torch.full((1024, 8), -1, dtype=torch.int32)
    ranges = torch.zeros(4 * (MAX_DIE + 1), dtype=torch.int32)
    for die, group in enumerate(groups):
        a, b = bounds[("a", die)], bounds[("b", die)]
        for rr, sm in enumerate(group):
            table[sm] = torch.tensor([rr + (s0 if die else 0), a[rr], a[rr + 1], b[rr],
                                      b[rr + 1], 0, 0, 0])  # fmt: skip
            assert b[rr + 1] - b[rr] <= MAX_WIN, (b[rr + 1] - b[rr], MAX_WIN)
        for bd, kt in ((a, op.ktA), (b, op.ktB)):  # one acquiring lane per later range
            for lo in range(0, bd[-1], kt):
                n = sum(1 for rr in range(len(bd) - 1) if bd[rr] < lo + kt and bd[rr + 1] > lo)
                assert n <= 33, (die, lo, n)
        ranges[die * (MAX_DIE + 1): die * (MAX_DIE + 1) + len(a)] = torch.tensor(a)
        o = (2 + die) * (MAX_DIE + 1)
        ranges[o: o + len(b)] = torch.tensor(b)
    # Expected ready time of every h block, in stream units of its SM (one unit = one
    # 16 KB weight stage; the SMs stream at about the same rate): the unit count at
    # which its last segment ends (order: tail, head, whole blocks) + a finish margin.
    margin = op.ktA // 4
    ready = [0] * op.nbA
    a_units = {}
    for die, group in enumerate(groups):
        a = bounds[("a", die)]
        base = 0 if die == 0 else op.nbA0
        for rr in range(len(group)):
            lo, hi = a[rr], a[rr + 1]
            a_units[(die, rr)] = hi - lo
            te = min(hi, (lo // op.ktA + 1) * op.ktA) if lo % op.ktA else lo
            hlo = max(te, ((hi - 1) // op.ktA) * op.ktA) if hi % op.ktA else hi
            done = 0
            for plo, phi in ((lo, te), (hlo, hi), (te, hlo)):
                u = plo
                while u < phi:
                    blk = u // op.ktA
                    end = min(phi, (blk + 1) * op.ktA)
                    done += end - u
                    g = base + blk
                    ready[g] = max(ready[g], done + margin)
                    u = end
    # Per output block, every down position's consumption time (its SM's A units +
    # its offset in the SM's down range); the h blocks sorted by ready time go to the
    # positions sorted by consumption time (earliest ready -> earliest consumed).
    by_ready = sorted(range(op.nbA), key=lambda g: (ready[g], g))
    width = max(ub0, ub1, 1)
    perm = torch.zeros((2, width), dtype=torch.int32)
    for die, group in enumerate(groups):
        b = bounds[("b", die)]
        nob = (op.nbB0, op.nbB - op.nbB0)[die]
        owner = []  # (consumption time, position) of every down unit of the die
        for rr in range(len(group)):
            for pos in range(b[rr], b[rr + 1]):
                owner.append((a_units[(die, rr)] + pos - b[rr], pos))
        owner.sort(key=lambda x: x[1])
        for ob in range(nob):
            slots = sorted(owner[ob * op.ktB: (ob + 1) * op.ktB])
            for (_, pos), g in zip(slots, by_ready):
                perm[die, pos] = g
    return table.to(device), ranges.to(device), perm.to(device)


def flow_times(sync, slots):
    import torch

    t = sync[TIMES: TIMES + 4 * slots].view(torch.int64).view(slots, 2)
    return t.cpu()
