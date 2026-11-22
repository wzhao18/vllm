# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode MXFP8 projection with the whole row block of tokens per CTA pair (VR200).

``out[R, N] = x[R, K] @ W[N, K]^T`` (W FP8 E4M3 + F8_128x4 E8M0 scales, x the MXFP8
SP gather: FP8 rows + F8_128x4 scales, out BF16), for the decode in-projections at
R = 128..384 verify rows (KDA in_proj: N 12448, K 7168).

Library GEMMs tile the output in 128-256 weight rows x 192-256 token columns over the
full K, so every CTA re-reads the activation for its weight tile: at R 384 each CTA
ingests 1.6-2.7 MB, and per-SM ingest (~200 GB/s), not HBM, bounds them. Here:

- swap-AB, ``cta_group::2`` M256: a CTA pair owns 256 weight rows (128 per CTA, the
  MMA's A halves) and ALL R tokens (N blocks of 128, each CTA holding 64 tokens of
  each block: the B halves), so W and x are each read once per pair and K split;
- K split ``ksplit`` (2) over pairs: each CTA ingests 128 x K/2 weight bytes plus
  R/2 x K/2 activation bytes (R 384: 459 + 688 KB, 40% under the best library tile);
- the N blocks reuse the A operand through the tensor core's A collector (fill /
  use / lastuse), so the shared-memory operand fetch per K64 step is 8 KB (A) +
  3 x 4 KB (B) for 3 x 64 tensor clocks: the 2x-K ``mxf8f6f4`` rate, not the
  2/3 smem-fetch bound;
- PDL: the TMA warp loads the weights (and their scales) of the first ``stages`` K
  tiles BEFORE ``griddepcontrol.wait`` (the producer of x, the SP gather, is still
  running), then the activation tiles;
- split-K reduce without a second kernel: the two CTAs of a (row block, pair half)
  pair up by an atomic ticket; the first stores its FP32 partial (L2), the second
  adds it to its own (FP32 addition commutes: bitwise independent of who is first),
  rounds once to BF16 and stores. The second only ever waits for a CTA that already
  holds its finished accumulator, so no co-scheduling is assumed.

Numerics: FP32 accumulation of the block-scaled products (stock: cuBLASLt MXFP8 with
FP32 accumulation, BF16 output); the K order differs (two K halves summed last).
"""

from typing import Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.rubin_helpers as sm107_utils
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.nvgpu.tcgen05.mma import CollectorOp
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from vllm.models.kimi_k3.nvidia.k31_spin_guard import spin_start, spin_tick

from ..common.dies import global_time, smid
from ..common.primitives import asm, copy_bulk, push_float, wait
from ..mla.qvalid import spin_ns
from .mxfp8 import TMA_MIN_ROWS

SF_VEC = 32
MMA_N = 128  # tokens per MMA (64 per CTA of the pair)
MMA_K = 64  # 2x-K mxf8f6f4
EPI_WARPS = 4
MMA_WARP = 4
TMA_WARP = 5  # weights (A, SFA): issued before the PDL wait
TMEM_COLS = 512
# bench-only debug bit: per-CTA %globaltimer stamps after the partial scratch: 0 entry,
# 1 accumulator full (epilogue), 2 partial published, 3 peer observed, 4 peer partial
# landed, 5 epilogue end, 6 activation warp past the PDL wait, 7 last stage landed
STAMP, STAMPS = 16384, 8
TMEM_BAR = 1  # named barrier ids (0 = __syncthreads)
EPI_BAR = 2


@asm()
def tmem_ld16(taddr):
    """tcgen05.ld 32x32b.x16: 16 columns of this warp's 32 TMEM lanes."""
    regs = ",".join(f"${i}" for i in range(16))
    code = "tcgen05.ld.sync.aligned.32x32b.x16.b32 {" + regs + "}, [$16];"
    return code, ",".join(["=f"] * 16) + ",r,~{memory}", (Int32(taddr),), (Float32,) * 16


@asm()
def tmem_wait_n(*vals):
    n = len(vals)
    cons = ",".join(["=f"] * n) + "," + ",".join(str(i) for i in range(n))
    return "tcgen05.wait::ld.sync.aligned;", cons, tuple(vals), (Float32,) * n


@asm(Int32)
def atom_add_gpu(address, value):
    return "atom.add.relaxed.gpu.global.s32 $0, [$1], $2;", "=r,l,r,~{memory}", (
        Int64(address), Int32(value))


@asm(Int32)
def ld_acquire_gpu(address):
    return "ld.acquire.gpu.global.b32 $0, [$1];", "=r,l,~{memory}", (Int64(address),)


@asm()
def st_release_gpu(address, value):
    return "st.release.gpu.global.b32 [$0], $1;", "l,r,~{memory}", (
        Int64(address), Int32(value))


@asm()
def fence_gpu():
    return "fence.acq_rel.gpu;", "~{memory}", ()


@asm()
def st_global_f32(address, value):
    return "st.global.f32 [$0], $1;", "l,f,~{memory}", (Int64(address), Float32(value))


@asm(Float32)
def ld_global_f32(address):
    return "ld.global.cg.f32 $0, [$1];", "=f,l,~{memory}", (Int64(address),)


@asm()
def st_global_bf16(address, value):
    code = "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; st.global.b16 [$0], t; }"
    return code, "l,f,~{memory}", (Int64(address), Float32(value))


@asm()
def st_global_v4f32(address, a, b, c, d):
    return "st.global.v4.f32 [$0], {$1,$2,$3,$4};", "l,f,f,f,f,~{memory}", (
        Int64(address), Float32(a), Float32(b), Float32(c), Float32(d))


@asm((Float32,) * 4)
def ld_global_cg_v4f32(address):
    code = "ld.global.cg.v4.f32 {$0,$1,$2,$3}, [$4];"
    return code, "=f,=f,=f,=f,l,~{memory}", (Int64(address),)


@asm()
def st_global_s64(address, value):
    return "st.global.b64 [$0], $1;", "l,l,~{memory}", (Int64(address), Int64(value))


@asm()
def fence_proxy_async_global():
    return "fence.proxy.async.global;", "~{memory}", ()


@asm(Float32)
def ld_shared_f32(address):
    return "ld.shared.f32 $0, [$1];", "=f,r,~{memory}", (Int32(address),)


@asm((Float32,) * 4)
def ld_shared_v4(address):
    return "ld.shared.v4.f32 {$0,$1,$2,$3}, [$4];", "=f,=f,=f,=f,r,~{memory}", (Int32(address),)


@asm()
def arrive_remote(barrier, peer):
    """mbarrier arrive (release, cluster scope) on the same smem barrier of cluster rank peer."""
    code = "{ .reg .b32 b; mapa.shared::cluster.u32 b,$0,$1; "
    code += "mbarrier.arrive.release.cluster.shared::cluster.b64 _,[b]; }"
    return code, "r,r,~{memory}", (Int32(barrier), Int32(peer))


@asm()
def bar_sync(barrier_id: int, count: int):
    return f"bar.sync {barrier_id},{count};", "~{memory}", ()


@asm(Int32)
def ld_shared_s32(address):
    return "ld.shared.b32 $0, [$1];", "=r,r,~{memory}", (Int32(address),)


@asm()
def st_shared_s32(address, value):
    return "st.shared.b32 [$0], $1;", "r,r,~{memory}", (Int32(address), Int32(value))


class DenseMx8:
    """One compiled shape: ``n`` x ``k`` weight, ``rows`` tokens (<= 384, any count:
    the token blocks past ``rows`` read TMA zero fill and are not stored), output row
    pitch ``ldo`` elements. ``ksplit`` 1 or 2; ``stages`` K tiles of ``kt`` in the
    shared-memory ring (all of them pre-loaded with weights before the PDL wait)."""

    def __init__(self, n: int, k: int, rows: int, ldo: int, *, ksplit: int = 2,
                 kt: int = 128, stages: int = 4, collector: bool = True,
                 mc: int = 1, kred: bool = False, pair: bool = True,
                 iw: int = 1, ix: int = 1, debug: int = 0, pre_ns: int = 0, xcopies: int = 1,
                 wdie: tuple = (), sym: bool = False, mma_n: int = MMA_N):  # fmt: skip
        assert rows <= 384 and k % (kt * ksplit) == 0 and ksplit in (1, 2)
        # iw / ix TMA issuer warps for the weights / the activation, each issuing every
        # iw-th / ix-th K tile: one issuer completes one bulk op at a time (~0.3 us
        # from L2 / HBM, hwdb hw51), so a single issuer caps the ring at one stage per
        # op latency whatever its depth.
        assert iw >= 1 and ix >= 1
        self.iw, self.ix = iw, ix
        self.x_warp = TMA_WARP + iw
        self.threads = 32 * (TMA_WARP + iw + ix)
        # kred: the two K halves of a row block share one cluster and reduce the FP32
        # partial over DSMEM (symmetric reduce-scatter: each CTA owns half of the
        # token columns), instead of the global ticket exchange.
        # pair=False: cta_group::1 M128 (128 weight rows and all tokens per CTA): with
        # kred the cluster is the 2 K halves (98 two-CTA clusters at N 12448 are
        # co-resident; 49 four-CTA clusters of M256 pairs are not).
        assert not kred or (ksplit == 2 and mc == 1)
        assert pair or mc == 1
        self.kred = kred
        self.pair = pair
        self.v = 2 if pair else 1
        self.clus = (4 if kred else 2 * mc) if pair else (2 if kred else 1)
        self.n, self.k, self.rows, self.ldo = n, k, rows, ldo
        # The activation TMA atoms span >= 2 token rows: over a 1-row tensor the
        # degenerate row mode builds a tensor map whose load faults with
        # CUDA_ERROR_ILLEGAL_INSTRUCTION on VR200. The caller's activation buffer
        # holds >= 2 rows (``TMA_MIN_ROWS``); rows past ``rows`` are never stored.
        self.tma_rows = max(rows, TMA_MIN_ROWS)
        self.ksplit, self.kt, self.stages = ksplit, kt, stages
        self.collector = collector
        # bench-only isolation: 1 no epilogue stores, 2 no MMA, 4 no TMA (barriers
        # still complete via plain arrives), 8 epilogue skips the accumulator wait
        self.debug = debug
        # bench-only: the activation warps spin pre_ns after the weight prologue, before
        # griddepcontrol.wait (a preceding op of the same resident CTA: in-grid form)
        self.pre_ns = pre_ns
        self.xcopies = xcopies
        # wdie (die-1 SM mask words, ``die_local.die_sm_mask32``): W and its scales are
        # [2] copies, copy d homed on die d; each CTA streams its own die's copy.
        self.wdie = tuple(wdie)
        # sym: the global K-half exchange is a symmetric reduce-scatter (each CTA
        # publishes the peer's token half of its FP32 partial, then adds the peer's
        # half of its own); default: the ticket scheme (first publishes all, second
        # sums all). sym waits for its K-half peer: the grid must be co-resident.
        self.sym = sym
        assert not sym or (ksplit == 2 and not kred)
        self.wcopies = 2 if self.wdie else 1
        # mc pairs per cluster share one K half and the x tiles: x is TMA-multicast
        # to the mc pairs (L2 reads of x / mc)
        self.mc = mc
        self.cs = (self.v * mc, 1)  # TMA multicast shape (the MMA pair)
        # tokens per MMA: 128, or 256 (one full-rate 2-CTA N256 MMA per K64 step for
        # 129-256 rows instead of two N128 at 2/3 rate)
        assert mma_n in (128, 256)
        self.mma_n = mma_n
        self.nb = -(-rows // mma_n)
        self.ktiles = k // kt // ksplit  # K tiles per CTA
        self.mrows = 128 * self.v
        self.mblocks = -(-(-(-n // self.mrows)) // mc) * mc
        self.tiler = (self.mrows, self.mma_n * self.nb, kt)
        self.tiler1 = (self.mrows, self.mma_n, kt)  # one N block: the scale factors of B per block
        self.tiler_sfb = (128, self.mma_n, kt)
        self.grid = self.v * self.mblocks * ksplit
        # kred receives half the FP32 partial (128 rows x cols/2) into sA..sB (contiguous)
        b_rows = self.mma_n * self.nb // self.v
        assert not kred or (self.mma_n * self.nb // 2) * 128 * 4 <= stages * kt * (128 + b_rows)
        # sym receives the peer's 128 x cols/2 FP32 partial into the drained B ring
        assert not sym or (self.mma_n * self.nb // 2) * 128 * 4 <= stages * kt * b_rows

    def scratch_words(self) -> Tuple[int, int]:
        """(FP32 partial words, int32 ticket words) the caller allocates (zeroed
        tickets, persistent across calls)."""
        slots = self.v * self.mblocks
        words = slots * 128 * self.mma_n * self.nb
        if self.debug & STAMP:  # bench: STAMPS int64 globaltimer stamps per CTA after it
            words += 2 * STAMPS * self.grid
        return words, (3 if self.sym else 2) * slots

    def _mma(self, collector):
        group = tcgen05.CtaGroup.TWO if self.pair else tcgen05.CtaGroup.ONE
        return sm107_utils.make_blockscaled_trivial_tiled_mma(
            cutlass.Float8E4M3FN, cutlass.Float8E4M3FN, OperandMajorMode.K,
            OperandMajorMode.K, cutlass.Float8E8M0FNU, SF_VEC, group,
            (self.mrows, self.mma_n, MMA_K), a_collector_op=collector,
            b_collector_op=CollectorOp.DISCARD,
        )  # fmt: skip

    @cute.jit
    def __call__(self, w_ptr: cute.Pointer, sfw_ptr: cute.Pointer, x_ptr: cute.Pointer,
                 sfx_ptr: cute.Pointer, out_ptr: cute.Pointer, part_ptr: cute.Pointer,
                 ticket_ptr: cute.Pointer, stream: cuda.CUstream):  # fmt: skip
        n, k, rows = self.n, self.k, self.rows
        fp8, sf_t = cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU
        a = cute.make_tensor(
            w_ptr, cute.make_ordered_layout((n, k, self.wcopies), order=(1, 0, 2))
        )
        # bench-only xcopies > 1: row block mb reads private activation copy mb % xcopies
        b = cute.make_tensor(
            x_ptr, cute.make_ordered_layout((self.tma_rows, k, self.xcopies), order=(1, 0, 2))
        )
        sfa = cute.make_tensor(sfw_ptr, blockscaled_utils.tile_atom_to_shape_SF(a.shape, SF_VEC))
        sfb = cute.make_tensor(sfx_ptr, blockscaled_utils.tile_atom_to_shape_SF(b.shape, SF_VEC))

        mma = self._mma(CollectorOp.DISCARD)
        mma_fill = self._mma(CollectorOp.FILL)
        mma_use = self._mma(CollectorOp.USE)
        mma_last = self._mma(CollectorOp.LASTUSE)
        # the same four with ACCUMULATE off: K tile 0's first MMAs (no ACCUMULATE
        # mutation inside the stage-unrolled MMA loop's branches)
        mma0 = (self._mma(CollectorOp.DISCARD), self._mma(CollectorOp.FILL),
                self._mma(CollectorOp.USE), self._mma(CollectorOp.LASTUSE))  # fmt: skip
        mma_sfb = sm107_utils.make_blockscaled_trivial_tiled_mma(
            fp8, fp8, OperandMajorMode.K, OperandMajorMode.K, sf_t, SF_VEC,
            tcgen05.CtaGroup.ONE, (128, self.mma_n, MMA_K),
            a_collector_op=CollectorOp.DISCARD, b_collector_op=CollectorOp.DISCARD,
        )  # fmt: skip
        # kred: the K half is the cluster K mode (rank = v + 2 kh): masks and pipelines
        # stay inside one pair; the TMA atoms are built for the pair alone.
        kc = 2 if self.kred else 1
        pair_vmnk = cute.tiled_divide(cute.make_layout((self.v * self.mc, 1, 1)), (mma.thr_id.shape,))
        cluster_vmnk = cute.tiled_divide(cute.make_layout((self.v * self.mc, 1, kc)), (mma.thr_id.shape,))
        pair_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((self.v * self.mc, 1, 1)), (mma_sfb.thr_id.shape,)
        )
        cluster_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((self.v * self.mc, 1, kc)), (mma_sfb.thr_id.shape,)
        )
        pipe_vmnk = cluster_vmnk if self.pair else pair_vmnk
        a_layout = sm100_utils.make_smem_layout_a(mma, self.tiler, fp8, self.stages)
        b_layout = sm100_utils.make_smem_layout_b(mma, self.tiler, fp8, self.stages)
        sfa_layout = blockscaled_utils.make_smem_layout_sfa(mma, self.tiler, SF_VEC, self.stages)
        sfb_layout = blockscaled_utils.make_smem_layout_sfb(mma, self.tiler1, SF_VEC, self.stages)
        tsfa_layout = blockscaled_utils.make_tmem_layout_sfa(
            mma, self.tiler, SF_VEC, cute.slice_(sfa_layout, (None, None, None, 0))
        )
        tsfb_layout = blockscaled_utils.make_tmem_layout_sfb(
            mma, self.tiler1, SF_VEC, cute.slice_(sfb_layout, (None, None, None, 0))
        )
        self.sfa_cols = cute.cosize(cute.recast_layout(32, sf_t.width, tsfa_layout)) & 0xFFFF
        self.sfb_cols = cute.cosize(cute.recast_layout(32, sf_t.width, tsfb_layout)) & 0xFFFF

        a_one = cute.slice_(a_layout, (None, None, None, 0))
        b_one = cute.slice_(b_layout, (None, None, None, 0))
        sfa_one = cute.slice_(sfa_layout, (None, None, None, 0))
        sfb_one = cute.slice_(sfb_layout, (None, None, None, 0))
        tma_a, ta = cute.nvgpu.make_tiled_tma_atom_A(
            sm100_utils.cluster_shape_to_tma_atom_A(self.cs, mma.thr_id), a, a_one,
            self.tiler, mma, pair_vmnk.shape,
        )  # fmt: skip
        tma_b, tb = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_B(self.cs, mma.thr_id), b, b_one,
            self.tiler, mma, pair_vmnk.shape,
        )  # fmt: skip
        tma_sfa, tsfa = cute.nvgpu.make_tiled_tma_atom_A(
            sm100_utils.cluster_shape_to_tma_atom_A(self.cs, mma.thr_id), sfa, sfa_one,
            self.tiler, mma, pair_vmnk.shape, internal_type=cutlass.Int16,
        )  # fmt: skip
        tma_sfb, tsfb = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_SFB(self.cs, mma.thr_id), sfb, sfb_one,
            self.tiler_sfb, mma_sfb, pair_sfb_vmnk.shape, internal_type=cutlass.Int16,
        )  # fmt: skip
        self.a_bytes = cute.size_in_bytes(fp8, a_one) + cute.size_in_bytes(sf_t, sfa_one)
        dbg = self.debug
        self.tx_bytes = self.v * (
            (0 if dbg & 512 else cute.size_in_bytes(fp8, a_one))
            + (0 if dbg & 256 else cute.size_in_bytes(fp8, b_one))
            + (0 if dbg & 128 else cute.size_in_bytes(sf_t, sfa_one))
            + (0 if dbg & 64 else self.nb * cute.size_in_bytes(sf_t, sfb_one))
        )  # fmt: skip

        @cute.struct
        class Storage:
            full: cute.struct.MemRange[cutlass.Int64, self.stages]
            empty: cute.struct.MemRange[cutlass.Int64, self.stages]
            acc_full: cute.struct.MemRange[cutlass.Int64, 1]
            acc_empty: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_dealloc: cutlass.Int64
            tmem_hold: cutlass.Int32
            ticket: cutlass.Int32
            xbar: cute.struct.MemRange[cutlass.Int64, 2]
            # scale factors first: their (version-0) smem descriptors must stay below
            # 256 KiB when the operand ring uses the oversized carveout
            sSFA: cute.struct.Align[cute.struct.MemRange[sf_t, cute.cosize(sfa_layout)], 128]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[sf_t, self.nb * cute.cosize(sfb_layout)], 128
            ]
            sA: cute.struct.Align[cute.struct.MemRange[fp8, cute.cosize(a_layout.outer)], 1024]
            sB: cute.struct.Align[cute.struct.MemRange[fp8, cute.cosize(b_layout.outer)], 1024]

        self.storage = Storage
        self.kernel(
            mma, mma_fill, mma_use, mma_last, *mma0, mma_sfb, tma_a, ta, tma_b, tb, tma_sfa,
            tsfa, tma_sfb, tsfb, cluster_vmnk, cluster_sfb_vmnk, pipe_vmnk, a_layout, b_layout,
            sfa_layout, sfb_layout, tsfa_layout, tsfb_layout, out_ptr, part_ptr,
            ticket_ptr,
        ).launch(
            grid=[self.grid, 1, 1], block=[self.threads, 1, 1], cluster=[self.clus, 1, 1],
            stream=stream, use_pdl=True,
        )  # fmt: skip

    @cute.kernel
    def kernel(self, mma: cute.TiledMma, mma_fill: cute.TiledMma, mma_use: cute.TiledMma,
               mma_last: cute.TiledMma, mma_z: cute.TiledMma, mma_fill_z: cute.TiledMma,
               mma_use_z: cute.TiledMma, mma_last_z: cute.TiledMma,
               mma_sfb: cute.TiledMma, tma_a: cute.CopyAtom,
               mA: cute.Tensor, tma_b: cute.CopyAtom, mB: cute.Tensor,
               tma_sfa: cute.CopyAtom, mSFA: cute.Tensor, tma_sfb: cute.CopyAtom,
               mSFB: cute.Tensor, cluster_vmnk: cute.Layout, cluster_sfb_vmnk: cute.Layout,
               pipe_vmnk: cute.Layout,
               a_layout: cute.ComposedLayout, b_layout: cute.ComposedLayout,
               sfa_layout: cute.Layout, sfb_layout: cute.Layout,
               tsfa_layout: cute.Layout, tsfb_layout: cute.Layout,
               out_ptr: cute.Pointer, part_ptr: cute.Pointer, ticket_ptr: cute.Pointer):  # fmt: skip
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        sf_t = cutlass.Float8E8M0FNU
        if tidx == 0:
            self._stamp(part_ptr, bidx, 0)
        if warp == TMA_WARP:
            cpasync.prefetch_descriptor(tma_a)
            cpasync.prefetch_descriptor(tma_b)
            cpasync.prefetch_descriptor(tma_sfa)
            cpasync.prefetch_descriptor(tma_sfb)
        v = bidx % self.v
        is_leader = v == 0
        rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        coord = cluster_vmnk.get_flat_coord(rank)
        coord_sfb = cluster_sfb_vmnk.get_flat_coord(rank)
        cl = bidx // (self.v * self.mc)
        mb = (cl // self.ksplit) * self.mc + (bidx % (self.v * self.mc)) // self.v
        kh = cl % self.ksplit

        smem = utils.SmemAllocator()
        st = smem.allocate(self.storage)
        ab = pipeline.PipelineTmaUmma.create(
            barrier_storage=st.full.data_ptr(), num_stages=self.stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.tx_bytes, cta_layout_vmnk=pipe_vmnk, defer_sync=True,
        )  # fmt: skip
        accp = pipeline.PipelineUmmaAsync.create(
            barrier_storage=st.acc_full.data_ptr(), num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.v * EPI_WARPS),
            cta_layout_vmnk=pipe_vmnk, defer_sync=True,
        )  # fmt: skip
        tmem = utils.TmemAllocator(
            st.tmem_hold.ptr,
            barrier_for_retrieve=pipeline.NamedBarrier(
                barrier_id=TMEM_BAR, num_threads=32 * (EPI_WARPS + 1)
            ),
            allocator_warp_id=0, is_two_cta=self.pair,
            two_cta_tmem_dealloc_mbar_ptr=st.tmem_dealloc.ptr, arch="sm_107",
        )  # fmt: skip
        if cutlass.const_expr(self.kred):
            if tidx == 0:  # DSMEM reduce: [0] partial received (tx), [1] peer ring drained
                cute.arch.mbarrier_init(st.xbar.data_ptr(), 1)
                cute.arch.mbarrier_init(st.xbar.data_ptr() + 1, 1)
                cute.arch.mbarrier_init_fence()
        pipeline_init_arrive(cluster_shape_mn=(self.clus, 1), is_relaxed=True)

        sA = st.sA.get_tensor(a_layout.outer, swizzle=a_layout.inner)
        sB = st.sB.get_tensor(b_layout.outer, swizzle=b_layout.inner)
        sSFA = st.sSFA.get_tensor(sfa_layout)
        sfb_size = cute.cosize(sfb_layout)
        sSFBs = [
            cute.make_tensor(st.sSFB.data_ptr() + j * sfb_size, sfb_layout)
            for j in range(self.nb)
        ]
        a_mask = cpasync.create_tma_multicast_mask(cluster_vmnk, coord, mcast_mode=2)
        b_mask = cpasync.create_tma_multicast_mask(cluster_vmnk, coord, mcast_mode=1)
        sfb_mask = cpasync.create_tma_multicast_mask(cluster_sfb_vmnk, coord_sfb, mcast_mode=1)

        gA = cute.local_tile(mA, cute.slice_(self.tiler, (None, 0, None)), (None, None, None))
        gB = cute.local_tile(mB, cute.slice_(self.tiler, (0, None, None)), (None, None, None))
        gSFA = cute.local_tile(mSFA, cute.slice_(self.tiler, (None, 0, None)), (None, None, None))
        gSFB = cute.local_tile(
            mSFB, cute.slice_(self.tiler_sfb, (0, None, None)), (None, None, None)
        )
        thr = mma.get_slice(v)
        thr_sfb = mma_sfb.get_slice(v)
        tCgA = thr.partition_A(gA)
        tCgB = thr.partition_B(gB)
        tCgSFA = thr.partition_A(gSFA)
        tCgSFB = thr_sfb.partition_B(gSFB)
        a_cta = cute.make_layout(cute.slice_(cluster_vmnk, (0, 0, None, 0)).shape)
        b_cta = cute.make_layout(cute.slice_(cluster_vmnk, (0, None, 0, 0)).shape)
        sfb_cta = cute.make_layout(cute.slice_(cluster_sfb_vmnk, (0, None, 0, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(
            tma_a, coord[2], a_cta, cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3)
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_b, coord[1], b_cta, cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3)
        )
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_sfa, coord[2], a_cta, cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )  # fmt: skip
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)
        sfb_parts = [
            cpasync.tma_partition(
                tma_sfb, coord_sfb[1], sfb_cta, cute.group_modes(sSFBs[j], 0, 3),
                cute.group_modes(tCgSFB, 0, 3),
            )
            for j in range(self.nb)
        ]  # fmt: skip
        tBsSFBs = [cute.filter_zeros(part[0]) for part in sfb_parts]
        tBgSFB = cute.filter_zeros(sfb_parts[0][1])
        tCrA = mma.make_fragment_A(sA)
        tCrB = mma.make_fragment_B(sB)
        acc_shape = mma.partition_shape_C(self.tiler[:2])
        acc_fake = mma.make_fragment_C(cute.append(acc_shape, 1))
        pipeline_init_wait(cluster_shape_mn=(self.clus, 1))
        cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(self.debug & 16):
            return

        k0 = kh * self.ktiles
        # K-start rotation per row block: pairs do not read the same x lines in
        # lock-step (an all-SM hot L2 region delivers a fraction of the L2 rate).
        rot = ((mb // self.mc) * 7) % self.ktiles  # one K order per cluster (multicast)
        if (warp >= TMA_WARP and warp < self.x_warp
                and not cutlass.const_expr(self.debug & 32)):  # fmt: skip
            iw = warp - TMA_WARP
            wl = self._die() if cutlass.const_expr(self.wcopies > 1) else Int32(0)
            gA_s = tAgA[(None, mb, None, wl)]
            gSFA_s = tAgSFA[(None, mb, None, wl)]
            prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.stages)
            for _ in cutlass.range(iw, unroll=1):
                prod.advance()
            if cutlass.const_expr(self.pre_ns > 0):
                # bench in-grid emulation: the previous call is done before this one's
                # weight prologue runs alongside the pre_ns predecessor-op spin
                cute.arch.griddepcontrol_wait()
                if cutlass.const_expr(self.debug & 4096):  # reference: no W pre-staging
                    spin_ns(Int64(self.pre_ns))
            # Weights do not depend on the predecessor: the ring fills with them while
            # the SP gather that writes x is still running.
            for t in cutlass.range(iw, self.ktiles, self.iw, unroll=1):
                kk = k0 + (t + rot) % self.ktiles
                ab.producer_acquire(prod)
                bar = ab.producer_get_barrier(prod)
                if cutlass.const_expr(not self.debug & 512):
                    self._tma(tma_a, gA_s[(None, kk)], tAsA[(None, prod.index)],
                              bar, a_mask)  # fmt: skip
                if cutlass.const_expr(not self.debug & 128):
                    self._tma(tma_sfa, gSFA_s[(None, kk)], tAsSFA[(None, prod.index)],
                              bar, a_mask)  # fmt: skip
                for _ in cutlass.range_constexpr(self.iw):
                    prod.advance()
            if iw == 0:
                tail = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.stages
                )
                for _ in cutlass.range(self.ktiles, unroll=1):
                    tail.advance()
                ab.producer_tail(tail)

        if (warp >= self.x_warp and warp < self.x_warp + self.ix
                and not cutlass.const_expr(self.debug & 32)):  # fmt: skip
            ix = warp - self.x_warp
            gB_s = tBgB[(None, 0, None, mb % self.xcopies)]
            xs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.stages)
            for _ in cutlass.range(ix, unroll=1):
                xs.advance()
            cute.arch.griddepcontrol_wait()
            if cutlass.const_expr(self.pre_ns > 0):
                spin_ns(Int64(self.pre_ns))
            if ix == 0:
                if lane == 0:
                    self._stamp(part_ptr, bidx, 6)
            for t in cutlass.range(ix, self.ktiles, self.ix, unroll=1):
                kk = k0 + (t + rot) % self.ktiles
                # same slot as the weight warp; only the weight warp arrives (expect_tx)
                ab.sync_object_empty.wait(xs.index, xs.phase)
                bar = ab.producer_get_barrier(xs)
                if cutlass.const_expr(not self.debug & 256):
                    self._tma(tma_b, gB_s[(None, kk)], tBsB[(None, xs.index)],
                              bar, b_mask)  # fmt: skip
                if cutlass.const_expr(not self.debug & 64):
                    for j in cutlass.range_constexpr(self.nb):
                        self._tma(
                            tma_sfb, tBgSFB[(None, j, kk, mb % self.xcopies)], tBsSFBs[j][(None, xs.index)],
                            bar, sfb_mask,
                        )  # fmt: skip
                for _ in cutlass.range_constexpr(self.ix):
                    xs.advance()

        if warp == MMA_WARP and not cutlass.const_expr(self.debug & 32):
            tmem.wait_for_alloc()
            acc_ptr = tmem.retrieve_ptr(Float32)
            tAcc = cute.make_tensor(acc_ptr, acc_fake.layout)
            acc_cols = self.mma_n * self.nb
            tSFA = cute.make_tensor(cute.recast_ptr(acc_ptr + acc_cols, dtype=sf_t), tsfa_layout)
            tSFBs = [
                cute.make_tensor(
                    cute.recast_ptr(
                        acc_ptr + acc_cols + self.sfa_cols + j * self.sfb_cols, dtype=sf_t
                    ),
                    tsfb_layout,
                )
                for j in range(self.nb)
            ]
            s2t_a = self._s2t(sSFA, tSFA)
            s2t_bs = [self._s2t(sSFBs[j], tSFBs[j]) for j in range(self.nb)]
            cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stages)
            accs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            if is_leader:
                accp.producer_acquire(accs)
            # Stage-unrolled: the ring slot of K tile t is t % stages, a compile-time
            # constant per unrolled step, so every MMA / S2T descriptor is a constant
            # offset (a runtime slot costs uniform-register conversions per MMA, which
            # left the tensor pipe idle between issues: mainloop 11.5 -> 9.2 us at R384
            # with a warp-uniform slot alone). ACCUMULATE is preset per atom outside the
            # loop: setting it inside the unrolled branches chains each atom's staged
            # value across sibling regions (IR dominance error).
            rounds = self.ktiles // self.stages
            nmma = 1 if self.debug & 65536 else self.nb  # bench: one N block only
            nkb = cute.size(tCrA, mode=[2])
            for m_ in (mma, mma_fill, mma_use, mma_last):
                m_.set(tcgen05.Field.ACCUMULATE, True)
            for m_ in (mma_z, mma_fill_z, mma_use_z, mma_last_z):
                m_.set(tcgen05.Field.ACCUMULATE, False)
            ops = (ab, s2t_a, s2t_bs, tAcc, tCrA, tCrB, tSFA, tSFBs, is_leader)
            acc_on = (mma, mma_fill, mma_use, mma_last)
            acc_off = (mma_z, mma_fill_z, mma_use_z, mma_last_z)
            rem = self.ktiles - rounds * self.stages
            # K tile 0 (ACCUMULATE off) is in the first unrolled round, else in the tail
            if cutlass.const_expr(rounds > 0):
                for _ in cutlass.range(1, unroll=1):
                    for s in cutlass.range_constexpr(self.stages):
                        z = acc_on
                        if cutlass.const_expr(s == 0):
                            z = acc_off
                        self._mma_stage(ops, cons, s, z, acc_on, nmma, nkb)
                        cons.advance()
            if cutlass.const_expr(rounds > 1):
                for _ in cutlass.range(rounds - 1, unroll=1):
                    for s in cutlass.range_constexpr(self.stages):
                        self._mma_stage(ops, cons, s, acc_on, acc_on, nmma, nkb)
                        cons.advance()
            for s in cutlass.range_constexpr(rem):  # the last ktiles % stages tiles
                z = acc_on
                if cutlass.const_expr(rounds == 0 and s == 0):
                    z = acc_off
                self._mma_stage(ops, cons, s, z, acc_on, nmma, nkb)
                cons.advance()
            if lane == 0:
                self._stamp(part_ptr, bidx, 7)
            if is_leader:
                accp.producer_commit(accs)
            accs.advance()
            accp.producer_tail(accs)

        if warp < EPI_WARPS:
            tmem.allocate(TMEM_COLS)
            tmem.wait_for_alloc()
            acc_ptr = tmem.retrieve_ptr(Float32)
            cute.arch.griddepcontrol_wait()  # (global writes after the producers)
            accc = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            if cutlass.const_expr(not self.debug & 32):
                accp.consumer_wait(accc)
            if tidx == 0:
                self._stamp(part_ptr, bidx, 1)
            if cutlass.const_expr(not self.debug & 1):
                self._epilogue(
                    acc_ptr, out_ptr, part_ptr, ticket_ptr, st.ticket.ptr.toint(), mb, v,
                    st.sA.data_ptr().toint(), st.sB.data_ptr().toint(), st.xbar.data_ptr(),
                    warp, lane, tidx, rank, kh, bidx,
                )  # fmt: skip
            if tidx == 0:
                self._stamp(part_ptr, bidx, 5)
            if cutlass.const_expr(not self.debug & 32):
                with cute.arch.elect_one():
                    accp.consumer_release(accc)
            tmem.relinquish_alloc_permit()
            tmem.free(acc_ptr)

    @cute.jit
    def _mma_stage(self, ops, cons, s, ms0, ms1, nmma, nkb):
        """One ring slot (compile-time ``s``) on the leader: wait, scale factors to
        tensor memory, the K64 x N-block MMAs with preset atoms (plain, fill, use,
        lastuse): ``ms0`` for the first K64 step (ACCUMULATE off in K tile 0), ``ms1``
        for the others, release."""
        ab, s2t_a, s2t_bs, tAcc, tCrA, tCrB, tSFA, tSFBs, is_leader = ops
        if is_leader:
            ab.consumer_wait(cons)
            if cutlass.const_expr(not self.debug & 32768):  # bench: no SF S2T
                cute.copy(s2t_a[0], s2t_a[1][(None, None, None, None, s)], s2t_a[2])
                for j in cutlass.range_constexpr(self.nb):
                    s2t_b = s2t_bs[j]
                    cute.copy(s2t_b[0], s2t_b[1][(None, None, None, None, s)], s2t_b[2])
            for kb in cutlass.range_constexpr(nkb):
                for j in cutlass.range_constexpr(0 if self.debug & 2 else nmma):
                    ms = ms1
                    if cutlass.const_expr(kb == 0):
                        ms = ms0
                    m = ms[0]
                    if cutlass.const_expr(self.collector and self.nb > 1):
                        if cutlass.const_expr(j == 0):
                            m = ms[1]
                        elif cutlass.const_expr(j == self.nb - 1):
                            m = ms[3]
                        else:
                            m = ms[2]
                    cute.gemm(
                        m, tAcc[(None, 0, j, 0)],
                        [tCrA[(None, 0, kb, s)], tSFA[(None, 0, kb)]],
                        [tCrB[(None, j, kb, s)], tSFBs[j][(None, 0, kb)]],
                        tAcc[(None, 0, j, 0)],
                    )  # fmt: skip
            ab.consumer_release(cons)

    @cute.jit
    def _stamp(self, part_ptr, bidx, i):
        if cutlass.const_expr(self.debug & STAMP):
            words = self.v * self.mblocks * 128 * self.mma_n * self.nb
            st_global_s64(
                part_ptr.toint() + Int64(words) * 4 + Int64(bidx * STAMPS + i) * 8,
                global_time(),
            )  # fmt: skip

    @cute.jit
    def _die(self) -> Int32:
        """Die of this SM from the die-1 SM mask (``wdie``)."""
        s = smid()
        word = s >> 5
        bit = s & 31
        die = Int32(0)
        for w in cutlass.range_constexpr(len(self.wdie)):
            if cutlass.const_expr(self.wdie[w] != 0):
                if word == w:
                    die = Int32((Uint32(self.wdie[w]) >> bit) & Uint32(1))
        return die

    @cute.jit
    def _tma(self, atom, src, dst, bar, mask):
        if cutlass.const_expr(self.pair):
            cute.copy(atom, src, dst, tma_bar_ptr=bar, mcast_mask=mask)
        else:
            cute.copy(atom, src, dst, tma_bar_ptr=bar)

    def _s2t(self, sSF, tSF):
        """SMEM -> TMEM scale-factor copy (tcgen05.cp 4x32dp128bit, cta_group::2)."""
        sc = cute.filter_zeros(sSF)
        tc = cute.filter_zeros(tSF)
        group = tcgen05.CtaGroup.TWO if self.pair else tcgen05.CtaGroup.ONE
        atom = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(group), cutlass.Float8E8M0FNU)
        tiled = tcgen05.make_s2t_copy(atom, tc)
        thr = tiled.get_slice(0)

        def bcast(lay):
            mn = cute.get(lay, mode=[0, 0])
            mn = cute.append(mn, cute.make_layout((4), stride=(0)))
            out = cute.append(cute.group_modes(mn, 0), cute.get(lay, mode=[0, 1]))
            out = cute.append(cute.group_modes(out, 0), cute.get(lay, mode=[1]))
            out = cute.append(out, cute.get(lay, mode=[2]))
            out = cute.append(out, cute.get(lay, mode=[3]))
            return out

        src = thr.partition_S(cute.make_tensor(sc.iterator, bcast(sc.layout)))
        src = tcgen05.get_s2t_smem_desc_tensor(tiled, src)
        dst = thr.partition_D(tc)
        return tiled, src, dst

    @cute.jit
    def _epilogue(self, acc_ptr, out_ptr, part_ptr, ticket_ptr, ticket_s, mb, v, sa_s, sb_s,
                  xbar, warp, lane, tidx, rank, kh, bidx):  # fmt: skip
        """Thread (warp w, lane l) owns weight row 128 v + 32 w + l of row block mb
        (TMEM lane 32 w + l) and all token columns."""
        cols = self.mma_n * self.nb
        r = 32 * warp + lane
        n = mb * self.mrows + 128 * v + r
        taddr = Int32(acc_ptr.toint()) + ((32 * warp) << 16)
        out = cute.make_tensor(out_ptr, cute.make_layout(self.rows * self.ldo))
        live = n < self.n
        if cutlass.const_expr(self.ksplit == 1):
            for c0 in cutlass.range_constexpr(0, cols, 16):
                vals = tmem_wait_n(*tmem_ld16(taddr + c0))
                for i in cutlass.range_constexpr(16):
                    if cutlass.const_expr(c0 + i < self.rows):
                        if live:
                            out[(c0 + i) * self.ldo + n] = vals[i].to(cutlass.BFloat16)
        elif cutlass.const_expr(self.kred):
            # Symmetric reduce-scatter with the other K half (cluster rank ^ 2, same rows):
            # this CTA owns token columns [o0, o0 + half) and pushes the other half of its
            # FP32 partial into the peer's drained operand ring (st.async, 16 B per thread
            # and column quad, quad-major [q][128 rows]) once the peer says its ring is free.
            half = cols // 2
            kh = rank // self.v
            peer = rank ^ self.v
            o0 = kh * half
            p0 = half - o0
            rbar = xbar.toint()
            fbar = (xbar + 1).toint()
            if tidx == 0:  # (debug 1024: no push, the barrier expects nothing)
                cute.arch.mbarrier_arrive_and_expect_tx(
                    xbar, 0 if self.debug & 1024 else half * 128 * 4
                )
                arrive_remote(fbar, peer)  # our ring is drained (accumulator complete)
            wait(fbar, 0, scope="cluster")
            if cutlass.const_expr(not self.debug & 1024):
                for c0 in cutlass.range_constexpr(0, half, 16):
                    vals = tmem_wait_n(*tmem_ld16(taddr + p0 + c0))
                    for qq in cutlass.range_constexpr(4):
                        q = c0 // 4 + qq
                        push_float(Int32(sa_s + (q * 128 + r) * 16), tuple(vals[4 * qq:4 * qq + 4]),
                                   Int32(rbar), peer)  # fmt: skip
            if cutlass.const_expr(not self.debug & 1024):
                wait(rbar, 0, scope="cluster")
            for c0 in cutlass.range_constexpr(0, half, 16):
                vals = tmem_wait_n(*tmem_ld16(taddr + o0 + c0))
                for qq in cutlass.range_constexpr(4):
                    p = ld_shared_v4(sa_s + ((c0 // 4 + qq) * 128 + r) * 16)
                    for i in cutlass.range_constexpr(4):
                        c = o0 + c0 + 4 * qq + i
                        if cutlass.const_expr(not self.debug & 8192):
                            if live:
                                if c < self.rows:
                                    out[c * self.ldo + n] = (vals[4 * qq + i] + p[i]).to(
                                        cutlass.BFloat16
                                    )
        elif cutlass.const_expr(self.sym):
            # Symmetric reduce-scatter with the other K half (same rows, same slot):
            # this CTA owns token columns [o0, o0 + half); it publishes its partial of
            # the peer's columns (row-major [r][half], 16 B stores), raises its flag,
            # waits for the peer's, then adds the peer's partial of its own columns. The
            # partial is quad-major [half / 4][128 rows][4]: a warp's 16 B per thread
            # accesses are 512 contiguous bytes.
            half = cols // 2
            slot = mb * self.v + v
            o0 = kh * half
            p0 = half - o0
            words = 128 * half
            pbase = part_ptr.toint() + Int64((slot * 2 + kh) * words + 4 * r) * 4
            qbase = part_ptr.toint() + Int64((slot * 2 + 1 - kh) * words + 4 * r) * 4
            tick_a = ticket_ptr.toint() + Int64(slot) * 4
            mine = ticket_ptr.toint() + Int64(self.v * self.mblocks + 2 * slot + kh) * 4
            peer = ticket_ptr.toint() + Int64(self.v * self.mblocks + 2 * slot + 1 - kh) * 4
            if tidx == 0:
                st_shared_s32(ticket_s, atom_add_gpu(tick_a, 1))
            if cutlass.const_expr(not self.debug & 1024):
                # software-pipelined: the next TMEM load is in flight during the stores
                vals = tmem_wait_n(*tmem_ld16(taddr + p0))
                for c0 in cutlass.range_constexpr(0, half, 16):
                    if cutlass.const_expr(c0 + 16 < half):
                        nxt = tmem_ld16(taddr + p0 + c0 + 16)
                    for qq in cutlass.range_constexpr(4):
                        st_global_v4f32(pbase + (c0 // 4 + qq) * 2048, *vals[4 * qq:4 * qq + 4])
                    if cutlass.const_expr(c0 + 16 < half):
                        vals = tmem_wait_n(*nxt)
            fence_gpu()
            bar_sync(EPI_BAR, 32 * EPI_WARPS)
            if tidx == 0:
                self._stamp(part_ptr, bidx, 2)
            gen = ld_shared_s32(ticket_s) // 2 + 1
            # The peer's partial of this CTA's columns arrives by two bulk copies into
            # the drained B ring (async, at the per-SM load rate), not by dependent
            # global loads per quad (one L2 round trip each).
            hb = words * 2  # bytes per copy: half of the 128 x half FP32 region
            qreg = part_ptr.toint() + Int64((slot * 2 + 1 - kh) * words) * 4
            if tidx == 0:
                cute.arch.mbarrier_init(xbar, 1)
                cute.arch.mbarrier_init(xbar + 1, 1)
                cute.arch.mbarrier_init_fence()
                st_release_gpu(mine, gen)
                if cutlass.const_expr(not self.debug & 2048):
                    sgn, sgt = spin_start()
                    while ld_acquire_gpu(peer) < gen:
                        pass
                        sgn, sgt = spin_tick(sgn, sgt, observed=0, target=0)
                self._stamp(part_ptr, bidx, 3)
                fence_proxy_async_global()
                for j in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_arrive_and_expect_tx(xbar + j, hb)
                    copy_bulk(sb_s + j * hb, qreg + j * hb, (xbar + j).toint(), hb)
            bar_sync(EPI_BAR, 32 * EPI_WARPS)
            # software-pipelined like the publish: TMEM load c0 + 16 in flight while
            # the 16 columns at c0 are summed and stored
            vals = tmem_wait_n(*tmem_ld16(taddr + o0))
            for c0 in cutlass.range_constexpr(0, half, 16):
                if cutlass.const_expr(c0 % (half // 2) == 0):
                    wait((xbar + (2 * c0 // half)).toint(), 0, scope="cta")
                    if cutlass.const_expr(c0 == 0):
                        if tidx == 0:
                            self._stamp(part_ptr, bidx, 4)
                if cutlass.const_expr(c0 + 16 < half):
                    nxt = tmem_ld16(taddr + o0 + c0 + 16)
                ps = [ld_shared_v4(sb_s + ((c0 // 4 + qq) * 128 + r) * 16) for qq in range(4)]
                for qq in cutlass.range_constexpr(4):
                    for i in cutlass.range_constexpr(4):
                        c = o0 + c0 + 4 * qq + i
                        if cutlass.const_expr(not self.debug & 8192):
                            if live:
                                if c < self.rows:
                                    out[c * self.ldo + n] = (vals[4 * qq + i] + ps[qq][i]).to(
                                        cutlass.BFloat16
                                    )
                if cutlass.const_expr(c0 + 16 < half):
                    vals = tmem_wait_n(*nxt)
        else:
            slot = mb * self.v + v
            part = cute.make_tensor(part_ptr, cute.make_layout(2 * self.mblocks * cols * 128))
            base = slot * cols * 128 + r
            tick_a = ticket_ptr.toint() + Int64(slot) * 4
            flag_a = ticket_ptr.toint() + Int64(2 * self.mblocks + slot) * 4
            if tidx == 0:
                st_shared_s32(ticket_s, atom_add_gpu(tick_a, 1))
            bar_sync(EPI_BAR, 32 * EPI_WARPS)
            ticket = ld_shared_s32(ticket_s)
            gen = ticket // 2 + 1
            if ticket % 2 == 0:  # first: publish the FP32 partial
                for c0 in cutlass.range_constexpr(0, 0 if self.debug & 1024 else cols, 16):
                    vals = tmem_wait_n(*tmem_ld16(taddr + c0))
                    for i in cutlass.range_constexpr(16):
                        part[base + (c0 + i) * 128] = vals[i]
                fence_gpu()
                bar_sync(EPI_BAR, 32 * EPI_WARPS)
                if tidx == 0:
                    st_release_gpu(flag_a, gen)
            else:  # second: add the other half's partial, round once, store
                # The partial (cols x 128 FP32, column-major) arrives by bulk copies of
                # 128-column chunks (64 KB) into the drained operand ring (sA, sB): one
                # async copy at the per-SM rate instead of latency-bound scalar loads.
                chunk = 128 * 128 * 4
                nch = self.nb
                src0 = part_ptr.toint() + Int64(slot * cols * 128) * 4
                if tidx == 0:
                    cute.arch.mbarrier_init(xbar, 1)
                    cute.arch.mbarrier_init(xbar + 1, 1)
                    cute.arch.mbarrier_init_fence()
                    if cutlass.const_expr(not self.debug & 2048):
                        sgn, sgt = spin_start()
                        while ld_acquire_gpu(flag_a) < gen:
                            pass
                            sgn, sgt = spin_tick(sgn, sgt, observed=0, target=0)
                    fence_proxy_async_global()
                    cute.arch.mbarrier_arrive_and_expect_tx(xbar, chunk)
                    copy_bulk(sa_s, src0, xbar.toint(), chunk)
                    if cutlass.const_expr(nch > 1):
                        cute.arch.mbarrier_arrive_and_expect_tx(xbar + 1, chunk)
                        copy_bulk(sb_s, src0 + chunk, (xbar + 1).toint(), chunk)
                bar_sync(EPI_BAR, 32 * EPI_WARPS)
                for j in cutlass.range_constexpr(nch):
                    sbase = sa_s if j % 2 == 0 else sb_s
                    wait((xbar + (j % 2)).toint(), (j // 2) % 2, scope="cta")
                    for c0 in cutlass.range_constexpr(0, 128, 16):
                        cc = 128 * j + c0
                        vals = tmem_wait_n(*tmem_ld16(taddr + cc))
                        p = [ld_shared_f32(sbase + ((c0 + i) * 128 + r) * 4) for i in range(16)]
                        for i in cutlass.range_constexpr(16):
                            if cutlass.const_expr(cc + i < self.rows and not self.debug & 8192):
                                if live:
                                    out[(cc + i) * self.ldo + n] = (vals[i] + p[i]).to(
                                        cutlass.BFloat16
                                    )
                    if cutlass.const_expr(j + 2 < nch):
                        bar_sync(EPI_BAR, 32 * EPI_WARPS)  # chunk j's buffer is free again
                        if tidx == 0:
                            cute.arch.mbarrier_arrive_and_expect_tx(xbar + (j % 2), chunk)
                            copy_bulk(sbase, src0 + (j + 2) * chunk, (xbar + (j % 2)).toint(),
                                      chunk)  # fmt: skip
