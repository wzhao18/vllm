# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Die-local, persistent, stream-K weight-streaming BF16 GEMM for decode token counts
(the layer-0 dense MLP on the SP shard: gate_up 67584 x 7168 with SiTU, down
7168 x 33792, 32-128 tokens per GPU).

y[T, N] = x[T, K] @ W[N, K]^T with the weight rows on the 1-CTA MMA's M (128 rows a
block) and all T tokens on N (one MMA of N = T): at T <= 128 the GEMM streams its
weight once from HBM and is bound by the per-die HBM read rate, so the design is a
weight stream at the die-local rate (hwdb hw61-63: die-local placement, >= 2 TMA
issuing threads, >= 128-180 KiB in flight, lean producer loop):

- Die-local weights: block b's weight rows are homed on die ``b >= nb / 2`` (2 MiB
  chunks); a CTA streams only blocks of the die it runs on.
- One persistent wave, stream-K: one CTA per SM; each die's (block, K step) space is
  cut into equal contiguous ranges, one per SM of that die. CTAs claim ranges
  two-ended (die-0 CTAs from the bottom of the slot list, die-1 CTAs from the top), so
  every range is claimed exactly once whatever the placement (a CTA that lands on the
  other die's slot reads remote: slower, same result).
- A block cut by range ends: the segment holding its first K step (the head) sums,
  the others (each the FIRST segment of its range, so published early) store FP32
  partials. The head adds them in slot (= K) order after its own accumulator: the
  stock GEMM's FP32 accumulation and one BF16 rounding (order may differ), and
  deterministic. A head waits only on segments that wait on nothing (no cycle). A head
  with partials is its range's last segment, so it bulk-loads them into the drained
  operand ring (oversized shared memory, all of it the ring while streaming): the
  epilogue adds them from shared memory, not with dependent L2 loads.
- ``situ``: block b holds gate rows [128 b, 128 b + 128) and up rows [I + 128 b, ...)
  (the stock [gate; up] layout, two accumulators) and the epilogue applies vLLM's
  situ_and_mul to the BF16-rounded gate / up (csrc/activation_kernels.cu
  ``situ_activation``: precise tanhf, 1 / beta rounded once): the stock gate_up GEMM +
  SituAndMul, fused, bitwise for the same FP32 sums.
- Three TMA issuing warps (K step g by warp g % 3), one MMA warp, twelve epilogue warps
  (three per TMEM lane quarter, alternate 16-token chunks: the SiTU math of the last
  blocks runs after the stream and needs the issue slots); TMEM holds two
  accumulator sets so a block's epilogue overlaps the next block's MMAs.
- PDL: launched as a dependent (the slot claim and the prologue overlap the
  predecessor's tail) but never releases its own dependents early: the slot claims
  are re-armed by the last CTA, so a following instance of the same kernel must not
  claim before every CTA of this one has finished.
"""

import functools

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import numpy as np
from cutlass import Float32, Int32, Int64
from cutlass.cute.nvgpu import cpasync, tcgen05

from vllm.models.kimi_k3.nvidia.k31_spin_guard import spin_start, spin_tick

from ..common.dies import smid
from ..common.primitives import asm
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
    _round_bf16,
    _st_relaxed,
    _st_u64,
    _st_v4,
    _tmem_ld16,
    _tmem_wait16,
)

K_TILE = 64  # K per ring stage (one 128-byte swizzle atom)
CHUNK = 16  # token columns per tcgen05.ld 32x32b.x16
MAX_SLOTS = 512  # ranges (one per SM)
STAMP_WORDS = 16  # 8 u64 globaltimer stamps per slot (debug STAMPS)
SYNC_WORDS = 16 + MAX_SLOTS + MAX_SLOTS * STAMP_WORDS  # claims, flags, stamps
SMEM_BUDGET = 320 * 1024  # oversized shared memory (hwdb topic 03: 327 KiB, budget 320)
BULK = 32768  # bytes per bulk-copy op of a staged partial
PRODUCERS = 3  # TMA issuing warps (0, 1 and 3): one op in flight per issuing thread
EPI_WARPS = 12  # three per TMEM lane quarter
EPI_THREADS = 32 * EPI_WARPS
EPI_SPLIT = EPI_WARPS // 4  # warps per lane quarter (alternate 16-token chunks)

# debug bits (benchmarks only)
NO_MMA = 1  # loads only: the MMA warp releases every stage without an MMA
NO_EPILOGUE = 2  # the epilogue releases each accumulator without reading it
STAMPS = 4  # per-slot globaltimer stamps after the flags (see ``stamp``)
NO_SITU = 8  # the epilogue stores the rounded gate instead of SiTU (timing only)
NO_STORE = 16  # the epilogue computes but never stores (timing only)
CONTIG_W = 32  # with NO_MMA: w passed as (rows * K / 64, 64), each box 16 KB contiguous

tanh = functools.partial(cute.math.tanh, fastmath=False)


@asm(Int32)
def _ld_s32(address):
    return "ld.global.b32 $0, [$1];", "=r,l,~{memory}", (Int64(address),)


@asm()
def _st_release(address, value):
    code = "st.release.gpu.global.b32 [$0], $1;"
    return code, "l,r,~{memory}", (Int64(address), Int32(value))


@asm()
def _sts_s32(address, value):
    return "st.shared.b32 [$0], $1;", "r,r,~{memory}", (Int32(address), Int32(value))


@asm(Int32)
def _lds_s32(address):
    return "ld.shared.b32 $0, [$1];", "=r,r,~{memory}", (Int32(address),)


@asm()
def _fence_gpu():
    return "fence.acq_rel.gpu;", "~{memory}", ()


@asm()
def _fence_before():
    return "tcgen05.fence::before_thread_sync;", "~{memory}", ()


def _f32(value):
    return float(np.float32(value))


def situ_constants(beta, linear_beta):
    """(0.5 * beta, 1 / beta, linear_beta, 1 / linear_beta) as the stock kernel's FP32
    values (each one IEEE FP32 operation, rounded to nearest)."""
    b, lb = np.float32(beta), np.float32(linear_beta)
    inv_lb = np.float32(1) / lb if lb > 0 else np.float32(0)
    return (_f32(np.float32(0.5) * b), _f32(np.float32(1) / b), _f32(lb), _f32(inv_lb))


def split_blocks(nb, s0, s1):
    """Blocks homed on die 0 of ``nb``: proportional to die 0's SMs."""
    return max(1, min(nb - 1, round(nb * s0 / (s0 + s1)))) if s1 else nb


@asm(Float32, pure=True)
def _rne_bf16(value):
    """FP32 -> BF16 -> FP32 round to nearest even in integer ops (cvt.rn's result for
    every non-NaN input; a NaN stays a NaN): keeps the conversion pipe free."""
    code = "{ .reg .b32 u, l; .reg .pred n; mov.b32 u, $1; shr.u32 l, u, 16; "
    code += "and.b32 l, l, 1; add.u32 u, u, l; add.u32 u, u, 0x7fff; "
    code += "and.b32 u, u, 0xffff0000; setp.nan.f32 n, $1, $1; @n mov.b32 u, 0x7fff0000; "
    code += "mov.b32 $0, u; }"
    return code, "=f,f", (Float32(value),)


@asm()
def _st_bf16_rne(address, value):
    """``value`` stored as BF16, rounded as cvt.rn (``_rne_bf16``) in integer ops."""
    code = "{ .reg .b32 u, l; .reg .b16 h; .reg .pred n; mov.b32 u, $1; shr.u32 l, u, 16; "
    code += "and.b32 l, l, 1; add.u32 u, u, l; add.u32 u, u, 0x7fff; shr.u32 u, u, 16; "
    code += "setp.nan.f32 n, $1, $1; @n mov.b32 u, 0x7fff; cvt.u16.u32 h, u; "
    code += "st.global.b16 [$0], h; }"
    return code, "l,f,~{memory}", (Int64(address), Float32(value))


def supports(rows, n, k, situ_mode=False):
    """``rows`` tokens (padded to the MMA N), ``n`` output features, ``k`` inputs."""
    t_mma = -(-rows // 16) * 16
    acc = 2 if situ_mode else 1
    return (
        1 <= rows <= 256
        and 2 * acc * t_mma <= 512
        and n % 256 == 0
        and k % K_TILE == 0
    )


class DenseStream:
    """One (rows, N, K, situ) specialization; see the module docstring.

    ``n``: output features (``situ``: the hidden size I; the weight then has 2 I
    rows). ``ldo``: the output row pitch in elements (default ``n``). The token count
    of a call (``nrows`` <= ``rows``) is a launch argument."""

    def __init__(self, rows, n, k, situ_mode=False, beta=1.0, linear_beta=-1.0,
                 ldo=None, debug=0, two_dies=True, nb0=None, pairs=0):  # fmt: skip
        assert supports(rows, n, k, situ_mode), (rows, n, k, situ_mode)
        self.rows, self.n, self.k, self.situ = rows, n, k, situ_mode
        self.t_mma = -(-rows // 16) * 16
        self.t_pad = self.t_mma + 4  # FP32 partial row: 16 B pad (bank-conflict free)
        self.nacc = 2 if situ_mode else 1
        self.beta, self.linear_beta = float(beta), float(linear_beta)
        self.consts = situ_constants(beta, linear_beta)
        self.ldo = ldo or n
        self.debug = debug
        self.nb = n // 128  # blocks
        # Blocks [0, nb0) are homed on die 0 (``split_blocks``: in proportion to the
        # dies' SMs, so every SM streams the same bytes).
        # ``pairs`` (plain only, the gate_up weight unfused): blocks b and pairs + b
        # (gate and up rows) share a die; nb0 then counts die 0's gate blocks.
        self.pairs = pairs
        nbs = pairs if pairs else self.nb
        self.nb0 = (nbs // 2 if nb0 is None else nb0) if two_dies else nbs
        self.kt = k // K_TILE  # K steps a block
        self.mma_tiler = (128, self.t_mma, K_TILE)
        self.threads = 128 + EPI_THREADS  # warps 0-1 TMA, 2 MMA, 3 slot claim, 4+ epilogue
        self.stage_bytes = self.nacc * 128 * K_TILE * 2 + self.t_mma * K_TILE * 2
        self.part = self.nacc * 128 * self.t_pad * 4  # one FP32 partial (bytes)
        # Staged partials: the gate_up head has one (ranges >= a block); down's head
        # up to a few. Keep >= 4 ring stages (>= ~120 KiB of weights in flight).
        self.stages = min(16, (SMEM_BUDGET - 4096) // self.stage_bytes)
        # A head's partials staged in the drained ring; any further ones read L2.
        self.nstage_p = self.stages * self.stage_bytes // self.part
        assert self.stages >= 3, (rows, n, k, self.stages)
        self.tmem_cols = 32
        while self.tmem_cols < 2 * self.nacc * self.t_mma:
            self.tmem_cols *= 2

    def gblock(self, b, bbase, nbd):
        """Global block of the die's block b (``pairs``: its gate blocks, then their up
        blocks)."""
        if cutlass.const_expr(self.pairs):
            return bbase + b + Int32(cutlass.select_(b >= nbd, Int32(self.pairs) - nbd, Int32(0)))
        return bbase + b

    def ws_bytes(self, slots):
        return slots * self.part

    def stamp(self, sync, slot, index):
        """globaltimer into stamp ``index`` of ``slot`` (debug STAMPS only)."""
        if cutlass.const_expr(self.debug & STAMPS):
            base = sync + (16 + MAX_SLOTS) * 4 + Int64(slot) * STAMP_WORDS * 4
            _st_u64(base + index * 8, _now())

    @cute.jit
    def __call__(
        self, w: cute.Tensor, x: cute.Tensor, y: cute.Tensor, ws: cute.Tensor,
        sync: cute.Tensor, table: cute.Tensor, s0: Int32, s1: Int32, nrows: Int32,
        stream: _cuda.CUstream,
    ):  # fmt: skip
        """w (Nw, K, 1) and x (T, K, 1) K-major BF16; y [T, ldo] BF16; ws uint8
        (``ws_bytes``); sync int32 [SYNC_WORDS] zeroed once; table int32 [1024] die of
        each %smid; s0 / s1 SMs of die 0 / 1 (grid = s0 + s1); nrows tokens."""
        dtype = cutlass.BFloat16
        major = tcgen05.OperandMajorMode.K
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            dtype, major, major, Float32, tcgen05.CtaGroup.ONE, self.mma_tiler[:2]
        )
        cluster_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,)
        )
        a_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, dtype, self.stages * self.nacc
        )
        b_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, dtype, self.stages
        )
        a_op = sm100_utils.cluster_shape_to_tma_atom_A((1, 1), tiled_mma.thr_id)
        b_op = sm100_utils.cluster_shape_to_tma_atom_B((1, 1), tiled_mma.thr_id)
        a_one = cute.slice_(a_layout, (None, None, None, 0))
        b_one = cute.slice_(b_layout, (None, None, None, 0))
        tma_a, tma_w = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, w, a_one, self.mma_tiler, tiled_mma, cluster_vmnk.shape
        )
        tma_b, tma_x = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, x, b_one, self.mma_tiler, tiled_mma, cluster_vmnk.shape
        )
        a_bytes = cute.size_in_bytes(dtype, a_one)
        b_bytes = cute.size_in_bytes(dtype, b_one)
        self.tx_bytes = a_bytes * self.nacc + b_bytes

        @cute.struct
        class Storage:
            full: cute.struct.MemRange[cutlass.Int64, self.stages]
            empty: cute.struct.MemRange[cutlass.Int64, self.stages]
            acc_full: cute.struct.MemRange[cutlass.Int64, 2]
            acc_empty: cute.struct.MemRange[cutlass.Int64, 2]
            staged: cutlass.Int64
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
            tiled_mma, tma_a, tma_w, tma_b, tma_x, a_layout, b_layout, y, ws, sync,
            table, s0, s1, nrows,
        ).launch(  # fmt: skip
            grid=(s0 + s1, 1, 1),
            block=[self.threads, 1, 1],
            smem=SMEM_BUDGET,
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self, tiled_mma: cute.TiledMma, tma_a: cute.CopyAtom, mW: cute.Tensor,
        tma_b: cute.CopyAtom, mX: cute.Tensor, a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout, mY: cute.Tensor, mWs: cute.Tensor,
        mSync: cute.Tensor, mTable: cute.Tensor, s0: Int32, s1: Int32, nrows: Int32,
    ):  # fmt: skip
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        stages, nacc, kt = self.stages, self.nacc, self.kt

        smem = utils.SmemAllocator()
        st = smem.allocate(self.storage)
        full, empty = st.full.data_ptr(), st.empty.data_ptr()
        acc_full, acc_empty = st.acc_full.data_ptr(), st.acc_empty.data_ptr()
        staged = st.staged.ptr
        tmem_holder = st.tmem_holder.ptr
        slot_s = Int32(st.slot.ptr.toint())
        sA = st.sA.get_tensor(a_layout.outer, swizzle=a_layout.inner)
        sB = st.sB.get_tensor(b_layout.outer, swizzle=b_layout.inner)
        sp = Int32(st.sA.data_ptr().toint())  # the ring: staging after the stream
        sync = mSync.iterator.toint()

        if warp == 0:
            cpasync.prefetch_descriptor(tma_a)
            cpasync.prefetch_descriptor(tma_b)
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(stages):
                    cute.arch.mbarrier_init(full + i, 1)
                    cute.arch.mbarrier_init(empty + i, 1)
                for i in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(acc_full + i, 1)
                    cute.arch.mbarrier_init(acc_empty + i, EPI_THREADS)
                cute.arch.mbarrier_init(staged, 1)
        if warp == 3:
            if lane == 0:
                die = _ld_s32(mTable.iterator.toint() + Int64(smid() & 1023) * 4)
                slot = Int32(0)
                if die == 0:
                    slot = _atom_add(sync, 1)
                else:
                    slot = s0 + s1 - 1 - _atom_add(sync + 4, 1)
                _sts_s32(slot_s, slot)
                self.stamp(sync, slot, 0)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        # This CTA's range [u0, u1) of its die's (block, K step) units.
        slot = _lds_s32(slot_s)
        d1 = slot >= s0
        r = slot
        sd = s0
        bbase = Int32(0)
        nbd = Int32(self.nb0)  # the die's blocks (gate blocks with ``pairs``)
        if d1:
            r = slot - s0
            sd = s1
            bbase = Int32(self.nb0)
            nbd = Int32((self.pairs if self.pairs else self.nb) - self.nb0)
        units = nbd * kt * (2 if self.pairs else 1)
        u0 = r * units // sd
        u1 = (r + 1) * units // sd

        gW = cute.local_tile(mW, cute.slice_(self.mma_tiler, (None, 0, None)),
                             (None, None, None))  # fmt: skip
        gX = cute.local_tile(mX, cute.slice_(self.mma_tiler, (0, None, None)),
                             (None, None, None))  # fmt: skip
        thr_mma = tiled_mma.get_slice(0)
        tCgW = thr_mma.partition_A(gW)
        tCgX = thr_mma.partition_B(gX)
        tAsA, tAgW = cpasync.tma_partition(
            tma_a, 0, cute.make_layout(1), cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgW, 0, 3),
        )  # fmt: skip
        tBsB, tBgX = cpasync.tma_partition(
            tma_b, 0, cute.make_layout(1), cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgX, 0, 3),
        )  # fmt: skip
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        acc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, 2 * nacc))

        if (warp < 4) & (warp != 2):  # TMA: K step g issued by producer g % PRODUCERS
            me = cutlass.min(warp, 2)  # warps 0, 1, 3 (3 after its slot claim)
            cute.arch.griddepcontrol_wait()
            turn = Int32(0)
            s = Int32(0)
            phase = Int32(1)
            u = u0
            while u < u1:
                b = u // kt
                kb1 = cutlass.min(kt, u1 - b * kt)
                gb = self.gblock(b, bbase, nbd)
                for kb in cutlass.range(u - b * kt, kb1, unroll=1):
                    if turn == me:
                        cute.arch.mbarrier_wait(empty + s, phase)
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(full + s, self.tx_bytes)
                        if cutlass.const_expr(self.debug & CONTIG_W):
                            cute.copy(tma_a, tAgW[(None, gb * kt + kb, 0, 0)],
                                      tAsA[(None, s * nacc)], tma_bar_ptr=full + s)  # fmt: skip
                            if cutlass.const_expr(nacc == 2):
                                cute.copy(tma_a, tAgW[(None, (self.nb + gb) * kt + kb, 0, 0)],
                                          tAsA[(None, s * nacc + 1)], tma_bar_ptr=full + s)  # fmt: skip
                        else:
                            cute.copy(tma_a, tAgW[(None, gb, kb, 0)],
                                      tAsA[(None, s * nacc)], tma_bar_ptr=full + s)  # fmt: skip
                            if cutlass.const_expr(nacc == 2):
                                cute.copy(tma_a, tAgW[(None, self.nb + gb, kb, 0)],
                                          tAsA[(None, s * nacc + 1)], tma_bar_ptr=full + s)  # fmt: skip
                        cute.copy(tma_b, tBgX[(None, 0, kb, 0)], tBsB[(None, s)],
                                  tma_bar_ptr=full + s)  # fmt: skip
                    turn += 1
                    if turn == PRODUCERS:
                        turn = Int32(0)
                    s += 1
                    if s == stages:
                        s = Int32(0)
                        phase = phase ^ 1
                u = (b + 1) * kt
        elif warp == 2:  # MMA
            cute.arch.barrier(barrier_id=1, number_of_threads=32 + EPI_THREADS)
            tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, alignment=16,
                                                   ptr_to_buffer_holding_addr=tmem_holder)  # fmt: skip
            acc = cute.make_tensor(tmem_ptr, acc_fake.layout)
            kblocks = cute.size(tCrA, mode=[2])
            s = Int32(0)
            phase = Int32(0)
            seg = Int32(0)
            u = u0
            while u < u1:
                b = u // kt
                kb0 = u - b * kt
                kb1 = cutlass.min(kt, u1 - b * kt)
                buf = seg & 1
                cute.arch.mbarrier_wait(acc_empty + buf, ((seg >> 1) & 1) ^ 1)
                _fence_after()
                for kk in cutlass.range(kb1 - kb0, unroll=1):
                    cute.arch.mbarrier_wait(full + s, phase)
                    if cutlass.const_expr(not (self.debug & NO_MMA)):
                        for a in cutlass.range_constexpr(nacc):
                            acc_a = acc[(None, None, None, buf * nacc + a)]
                            for kb in cutlass.range_constexpr(kblocks):
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, (kk + kb) != 0)
                                cute.gemm(tiled_mma, acc_a, tCrA[(None, None, kb, s * nacc + a)],
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
            with cute.arch.elect_one():
                self.stamp(sync, slot, 1)
        elif warp >= 4:
            if warp == 4:
                cute.arch.alloc_tmem(self.tmem_cols, tmem_holder)
            cute.arch.barrier(barrier_id=1, number_of_threads=32 + EPI_THREADS)
            tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, alignment=16,
                                                   ptr_to_buffer_holding_addr=tmem_holder)  # fmt: skip
            acc = cute.make_tensor(tmem_ptr, acc_fake.layout)
            cute.arch.griddepcontrol_wait()
            self.epilogue(acc, acc_full, acc_empty, staged, warp - 4, lane, slot, r, sd,
                          d1, s0, bbase, units, u0, u1, sp, mY, mWs, sync, nrows,
                          nbd)  # fmt: skip
            if warp == 4:
                cute.arch.relinquish_tmem_alloc_permit()
            _bar(2, EPI_THREADS)
            if warp == 4:
                if lane == 0:
                    self.stamp(sync, slot, 6)
            if warp == 4:
                cute.arch.dealloc_tmem(tmem_ptr, self.tmem_cols)
                if lane == 0:  # the last CTA re-arms the slot claims
                    if _atom_add(sync + 8, 1) == s0 + s1 - 1:
                        _st_relaxed(sync, 0)
                        _st_relaxed(sync + 4, 0)
                        _st_relaxed(sync + 8, 0)

    @cute.jit
    def epilogue(
        self, acc: cute.Tensor, acc_full, acc_empty, staged, ew: Int32, lane: Int32,
        slot: Int32, r: Int32, sd: Int32, d1, s0: Int32, bbase: Int32, units: Int32,
        u0: Int32, u1: Int32, sp: Int32, mY: cute.Tensor, mWs: cute.Tensor,
        sync: Int64, nrows: Int32, nbd: Int32,
    ):  # fmt: skip
        """Epilogue warp ew owns TMEM lane quarter ew % 4 (weight rows / output columns
        q = 32 (ew % 4) + lane of every block) and the 16-token chunks c with
        c % EPI_SPLIT == ew // 4."""
        kt, nacc = self.kt, self.nacc
        quad = ew % 4
        half = ew // 4
        q = quad * 32 + lane
        tid = ew * 32 + lane
        lane_bits = (quad * 32) << 16
        y = mY.iterator.toint()
        ws = mWs.iterator.toint()
        flags = sync + 64
        slot_base = Int32(cutlass.select_(d1, s0, Int32(0)))  # die's first global slot
        seg = Int32(0)
        u = u0
        while u < u1:
            b = u // kt
            kb0 = u - b * kt
            gb = self.gblock(b, bbase, nbd)
            buf = seg & 1
            bstart = b * kt
            col = Int32(acc[(None, None, None, buf * nacc)].iterator.toint()) + lane_bits
            if tid == 0:
                self.stamp(sync, slot, 2)
            if cutlass.const_expr(self.debug & NO_EPILOGUE):
                cute.arch.mbarrier_wait(acc_full + buf, (seg >> 1) & 1)
            else:
                self.segment(acc_full, staged, buf, seg, kb0, bstart, col, slot, r, sd,
                             units, slot_base, ws, flags, sp, y, gb, q, tid, half,
                             nrows, sync)  # fmt: skip
            _fence_before()
            cute.arch.mbarrier_arrive(acc_empty + buf)
            seg += 1
            u = (b + 1) * kt

    @cute.jit
    def segment(
        self, acc_full, staged, buf: Int32, seg: Int32, kb0: Int32, bstart: Int32,
        col: Int32, slot: Int32, r: Int32, sd: Int32, units: Int32, slot_base: Int32,
        ws: Int64, flags: Int64, sp: Int32, y: Int64, gb: Int32, q: Int32, tid: Int32,
        half: Int32, nrows: Int32, sync: Int64,
    ):  # fmt: skip
        kt, nacc, t_mma, t_pad, part = self.kt, self.nacc, self.t_mma, self.t_pad, self.part
        chunks = t_mma // CHUNK
        cute.arch.mbarrier_wait(acc_full + buf, (seg >> 1) & 1)
        _fence_after()
        if tid == 0:
            self.stamp(sync, slot, 3)
        if kb0 > 0:  # not the head: publish the FP32 partial (its range's first segment)
            dst = ws + Int64(slot) * part
            for a in cutlass.range_constexpr(nacc):
                for c in cutlass.range_constexpr(chunks):
                    if c % EPI_SPLIT == half:
                        v = _tmem_wait16(*_tmem_ld16(col + a * t_mma + c * CHUNK))
                        row = dst + ((a * 128 + q) * t_pad + c * CHUNK) * 4
                        for j in cutlass.range_constexpr(CHUNK // 4):
                            _st_v4(row + j * 16, v[4 * j], v[4 * j + 1], v[4 * j + 2],
                                   v[4 * j + 3])  # fmt: skip
            _bar(3, EPI_THREADS)
            if tid == 0:
                _fence_gpu()
                _st_release(flags + Int64(slot) * 4, 1)
                self.stamp(sync, slot, 4)
        else:  # the head: own accumulator + the later ranges' partials
            last = r
            while (last + 1) * units // sd < bstart + kt:  # start(sd) = units: stops
                last += 1
            # Partials staged in the drained ring (a head with partials is its range's
            # last segment: every stage has been consumed).
            nst = Int32(0)
            for p in cutlass.range(r + 1, last + 1, unroll=1):
                if (p * units // sd < (p + 1) * units // sd) & (nst < self.nstage_p):
                    nst += 1
            if tid == 0:
                if nst > 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(staged, nst * part)
                j = Int32(0)
                for p in cutlass.range(r + 1, last + 1, unroll=1):
                    if (p * units // sd < (p + 1) * units // sd) & (j < nst):
                        sgn, sgt = spin_start()
                        while _ld_acquire(flags + Int64(slot_base + p) * 4) == 0:
                            pass
                            sgn, sgt = spin_tick(sgn, sgt, observed=0, target=0)
                        _fence_proxy_global()  # the bulk copy (async proxy) reads them
                        src = ws + Int64(slot_base + p) * part
                        for o in cutlass.range_constexpr(-(-part // BULK)):
                            n_o = min(BULK, part - o * BULK)
                            _bulk_load(sp + j * part + o * BULK, src + o * BULK, n_o,
                                       staged)  # fmt: skip
                        j += 1
            for p in cutlass.range(r + 1, last + 1, unroll=1):  # acquire every partial
                if p * units // sd < (p + 1) * units // sd:
                    sgn, sgt = spin_start()
                    while _ld_acquire(flags + Int64(slot_base + p) * 4) == 0:
                        pass
                        sgn, sgt = spin_tick(sgn, sgt, observed=0, target=0)
            if nst > 0:
                cute.arch.mbarrier_wait(staged, 0)
            if tid == 0:
                self.stamp(sync, slot, 4)
            self.finish(col, r, last, sd, units, slot_base, ws, sp, y, gb, q, half,
                        nrows)  # fmt: skip
            _bar(3, EPI_THREADS)
            if tid == 0:
                self.stamp(sync, slot, 5)
                for p in cutlass.range(r + 1, last + 1, unroll=1):
                    _st_relaxed(flags + Int64(slot_base + p) * 4, 0)

    @cute.jit
    def finish(
        self, col: Int32, r: Int32, last: Int32, sd: Int32, units: Int32,
        slot_base: Int32, ws: Int64, sp: Int32, y: Int64, gb: Int32, q: Int32,
        half: Int32, nrows: Int32,
    ):  # fmt: skip
        """Own accumulator + the partials of ranges r + 1 .. last (the first
        ``nstage_p`` from the staged copies in shared memory) in slot order, rounded
        once to BF16 (``situ``: SiTU of the rounded gate / up) and stored as column
        128 gb + q of this warp's tokens."""
        nacc, t_mma, t_pad, part = self.nacc, self.t_mma, self.t_pad, self.part
        half_beta, inv_beta, lin, inv_lin = (Float32(v) for v in self.consts)
        for c in cutlass.range_constexpr(t_mma // CHUNK):
            if c % EPI_SPLIT == half:
                tot = cute.make_rmem_tensor(nacc * CHUNK, Float32)
                for a in cutlass.range_constexpr(nacc):
                    own = _tmem_wait16(*_tmem_ld16(col + a * t_mma + c * CHUNK))
                    for i in cutlass.range_constexpr(CHUNK):
                        tot[a * CHUNK + i] = own[i]
                    off = ((a * 128 + q) * t_pad + c * CHUNK) * 4
                    j = Int32(0)
                    for p in cutlass.range(r + 1, last + 1, unroll=1):
                        if p * units // sd < (p + 1) * units // sd:
                            if j < self.nstage_p:
                                src = sp + j * part + off
                                for v4 in cutlass.range_constexpr(CHUNK // 4):
                                    vals = _lds_v4(src + v4 * 16)
                                    for e in cutlass.range_constexpr(4):
                                        i = a * CHUNK + v4 * 4 + e
                                        tot[i] = tot[i] + vals[e]
                            else:
                                src = ws + Int64(slot_base + p) * part + off
                                for v4 in cutlass.range_constexpr(CHUNK // 4):
                                    vals = _ld_v4(src + v4 * 16)
                                    for e in cutlass.range_constexpr(4):
                                        i = a * CHUNK + v4 * 4 + e
                                        tot[i] = tot[i] + vals[e]
                            j += 1
                # The chunk's math first (no memory clobbers: 16 independent chains),
                # then its stores.
                out = []
                for i in cutlass.range_constexpr(CHUNK):
                    if cutlass.const_expr(self.debug & NO_SITU):
                        out.append(_round_bf16(tot[i]) + tot[CHUNK * (nacc - 1) + i])
                    elif cutlass.const_expr(nacc == 2):
                        gate = _rne_bf16(tot[i])
                        up = _rne_bf16(tot[CHUNK + i])
                        g = half_beta * tanh(gate * inv_beta)
                        g = g * (Float32(1) + tanh(gate * Float32(0.5)))
                        if cutlass.const_expr(self.consts[2] > 0):
                            up = lin * tanh(up * inv_lin)
                        out.append(g * up)
                    else:
                        out.append(tot[i])
                limit = nrows - 1000000 if cutlass.const_expr(self.debug & NO_STORE) else nrows
                for i in cutlass.range_constexpr(CHUNK):
                    t = c * CHUNK + i
                    if t < limit:
                        _st_bf16_rne(y + (Int64(t) * self.ldo + gb * 128 + q) * 2, out[i])


_COMPILED = {}


def as_cute(t, *, k_major=False, rows=False):
    """``k_major``: a (rows, K) operand as (rows, K, 1); ``rows``: a [rows, cols]
    output (row count dynamic)."""
    from cutlass.cute.runtime import from_dlpack

    if k_major:  # (rows, K) -> (rows, K, 1), K contiguous
        t3 = t.unsqueeze(-1)
        return from_dlpack(t3, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    if rows:
        return from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    return from_dlpack(t, assumed_align=16)


def compile_dense_stream(op: DenseStream, w, x, y, ws, sync, table, s0, s1, stream):
    """Compile (once per specialization) and return (launcher, tensor args); call as
    ``fn(*args, Int32(s0), Int32(s1), Int32(nrows), stream)``."""
    key = (op.rows, op.n, op.k, op.situ, op.beta, op.linear_beta, op.ldo, op.debug,
           op.nb0, op.pairs, tuple(w.shape))  # fmt: skip
    fn = _COMPILED.get(key)
    args = (as_cute(w, k_major=True), as_cute(x, k_major=True), as_cute(y, rows=True),
            as_cute(ws), as_cute(sync), as_cute(table))  # fmt: skip
    if fn is None:
        fn = cute.compile(op, *args, Int32(s0), Int32(s1), Int32(x.shape[0]), stream)
        _COMPILED[key] = fn
    return fn, args
