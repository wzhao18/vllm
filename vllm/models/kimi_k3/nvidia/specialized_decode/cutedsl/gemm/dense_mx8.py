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

from ..common.primitives import asm, copy_bulk, wait

SF_VEC = 32
MMA_N = 128  # tokens per MMA (64 per CTA of the pair)
MMA_K = 64  # 2x-K mxf8f6f4
EPI_WARPS = 4
MMA_WARP = 4
TMA_WARP = 5  # weights (A, SFA): issued before the PDL wait
X_WARP = 6  # activation (B, SFB): a second TMA issuer, after the PDL wait
THREADS = 32 * (X_WARP + 1)
TMEM_COLS = 512
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
def fence_proxy_async_global():
    return "fence.proxy.async.global;", "~{memory}", ()


@asm(Float32)
def ld_shared_f32(address):
    return "ld.shared.f32 $0, [$1];", "=f,r,~{memory}", (Int32(address),)


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
                 mc: int = 1, nsplit: int = 1, issuers: int = 1,
                 debug: int = 0):  # fmt: skip
        assert rows <= 384 and k % (kt * ksplit) == 0 and ksplit in (1, 2)
        # nsplit: the R tokens split into nsplit chunks of R / nsplit (whole 128-token
        # MMA blocks: the F8_128x4 scale tiles) over separate CTA pairs (no K split:
        # each pair owns a (row block, token chunk) output tile over the full K, the
        # weight rows are re-read per chunk out of L2, no partial exchange).
        assert nsplit == 1 or (ksplit == 1 and rows % (MMA_N * nsplit) == 0)
        self.nsplit, self.total_rows = nsplit, rows
        # issuers: TMA issuing warps per operand (weights, activations); issuer i
        # loads the K tiles t with t % issuers == i (hw51c: per-issuer ops complete
        # about one at a time, so the per-SM rate scales with the issuer count).
        assert issuers in (1, 2, 4)
        self.issuers = issuers
        self.threads = 32 * (X_WARP + 1 + 2 * (issuers - 1))
        rows //= nsplit
        self.n, self.k, self.rows, self.ldo = n, k, rows, ldo
        self.ksplit, self.kt, self.stages = ksplit, kt, stages
        self.collector = collector
        # bench-only isolation: 1 no epilogue stores, 2 no MMA, 4 no TMA (barriers
        # still complete via plain arrives), 8 epilogue skips the accumulator wait
        self.debug = debug
        # mc pairs per cluster share one K half and the x tiles: x is TMA-multicast
        # to the mc pairs (L2 reads of x / mc)
        self.mc = mc
        self.cs = (2 * mc, 1)
        self.nb = -(-rows // MMA_N)
        self.ktiles = k // kt // ksplit  # K tiles per CTA
        self.mblocks = -(-(-(-n // 256)) // mc) * mc
        self.tiler = (256, MMA_N * self.nb, kt)
        self.tiler1 = (256, MMA_N, kt)  # one N block: the scale factors of B per block
        self.tiler_sfb = (128, MMA_N, kt)
        self.chunk_clusters = self.mblocks // mc * ksplit
        self.grid = 2 * self.mblocks * ksplit * nsplit

    def scratch_words(self) -> Tuple[int, int]:
        """(FP32 partial words, int32 ticket words) the caller allocates (zeroed
        tickets, persistent across calls)."""
        slots = 2 * self.mblocks
        return slots * 128 * MMA_N * self.nb, 2 * slots

    def _mma(self, collector):
        return sm107_utils.make_blockscaled_trivial_tiled_mma(
            cutlass.Float8E4M3FN, cutlass.Float8E4M3FN, OperandMajorMode.K,
            OperandMajorMode.K, cutlass.Float8E8M0FNU, SF_VEC, tcgen05.CtaGroup.TWO,
            (256, MMA_N, MMA_K), a_collector_op=collector,
            b_collector_op=CollectorOp.DISCARD,
        )  # fmt: skip

    @cute.jit
    def __call__(self, w_ptr: cute.Pointer, sfw_ptr: cute.Pointer, x_ptr: cute.Pointer,
                 sfx_ptr: cute.Pointer, out_ptr: cute.Pointer, part_ptr: cute.Pointer,
                 ticket_ptr: cute.Pointer, stream: cuda.CUstream):  # fmt: skip
        n, k, rows = self.n, self.k, self.rows
        fp8, sf_t = cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU
        a = cute.make_tensor(w_ptr, cute.make_ordered_layout((n, k, 1), order=(1, 0, 2)))
        b = cute.make_tensor(
            x_ptr, cute.make_ordered_layout((rows, k, self.nsplit), order=(1, 0, 2))
        )
        sfa = cute.make_tensor(sfw_ptr, blockscaled_utils.tile_atom_to_shape_SF(a.shape, SF_VEC))
        sfb = cute.make_tensor(sfx_ptr, blockscaled_utils.tile_atom_to_shape_SF(b.shape, SF_VEC))

        mma = self._mma(CollectorOp.DISCARD)
        mma_fill = self._mma(CollectorOp.FILL)
        mma_use = self._mma(CollectorOp.USE)
        mma_last = self._mma(CollectorOp.LASTUSE)
        mma_sfb = sm107_utils.make_blockscaled_trivial_tiled_mma(
            fp8, fp8, OperandMajorMode.K, OperandMajorMode.K, sf_t, SF_VEC,
            tcgen05.CtaGroup.ONE, (128, MMA_N, MMA_K),
            a_collector_op=CollectorOp.DISCARD, b_collector_op=CollectorOp.DISCARD,
        )  # fmt: skip
        cluster_vmnk = cute.tiled_divide(cute.make_layout((2 * self.mc, 1, 1)), (mma.thr_id.shape,))
        cluster_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((2 * self.mc, 1, 1)), (mma_sfb.thr_id.shape,)
        )
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
            self.tiler, mma, cluster_vmnk.shape,
        )  # fmt: skip
        tma_b, tb = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_B(self.cs, mma.thr_id), b, b_one,
            self.tiler, mma, cluster_vmnk.shape,
        )  # fmt: skip
        tma_sfa, tsfa = cute.nvgpu.make_tiled_tma_atom_A(
            sm100_utils.cluster_shape_to_tma_atom_A(self.cs, mma.thr_id), sfa, sfa_one,
            self.tiler, mma, cluster_vmnk.shape, internal_type=cutlass.Int16,
        )  # fmt: skip
        tma_sfb, tsfb = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_SFB(self.cs, mma.thr_id), sfb, sfb_one,
            self.tiler_sfb, mma_sfb, cluster_sfb_vmnk.shape, internal_type=cutlass.Int16,
        )  # fmt: skip
        self.a_bytes = cute.size_in_bytes(fp8, a_one) + cute.size_in_bytes(sf_t, sfa_one)
        dbg = self.debug
        self.tx_bytes = 2 * (
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
            sA: cute.struct.Align[cute.struct.MemRange[fp8, cute.cosize(a_layout.outer)], 1024]
            sB: cute.struct.Align[cute.struct.MemRange[fp8, cute.cosize(b_layout.outer)], 1024]
            sSFA: cute.struct.Align[cute.struct.MemRange[sf_t, cute.cosize(sfa_layout)], 128]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[sf_t, self.nb * cute.cosize(sfb_layout)], 128
            ]

        self.storage = Storage
        self.kernel(
            mma, mma_fill, mma_use, mma_last, mma_sfb, tma_a, ta, tma_b, tb, tma_sfa,
            tsfa, tma_sfb, tsfb, cluster_vmnk, cluster_sfb_vmnk, a_layout, b_layout,
            sfa_layout, sfb_layout, tsfa_layout, tsfb_layout, out_ptr, part_ptr,
            ticket_ptr,
        ).launch(
            grid=[self.grid, 1, 1], block=[self.threads, 1, 1], cluster=[2 * self.mc, 1, 1],
            stream=stream, use_pdl=True,
        )  # fmt: skip

    @cute.kernel
    def kernel(self, mma: cute.TiledMma, mma_fill: cute.TiledMma, mma_use: cute.TiledMma,
               mma_last: cute.TiledMma, mma_sfb: cute.TiledMma, tma_a: cute.CopyAtom,
               mA: cute.Tensor, tma_b: cute.CopyAtom, mB: cute.Tensor,
               tma_sfa: cute.CopyAtom, mSFA: cute.Tensor, tma_sfb: cute.CopyAtom,
               mSFB: cute.Tensor, cluster_vmnk: cute.Layout, cluster_sfb_vmnk: cute.Layout,
               a_layout: cute.ComposedLayout, b_layout: cute.ComposedLayout,
               sfa_layout: cute.Layout, sfb_layout: cute.Layout,
               tsfa_layout: cute.Layout, tsfb_layout: cute.Layout,
               out_ptr: cute.Pointer, part_ptr: cute.Pointer, ticket_ptr: cute.Pointer):  # fmt: skip
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        sf_t = cutlass.Float8E8M0FNU
        if warp == TMA_WARP:
            cpasync.prefetch_descriptor(tma_a)
            cpasync.prefetch_descriptor(tma_b)
            cpasync.prefetch_descriptor(tma_sfa)
            cpasync.prefetch_descriptor(tma_sfb)
        v = bidx % 2
        is_leader = v == 0
        rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        coord = cluster_vmnk.get_flat_coord(rank)
        coord_sfb = cluster_sfb_vmnk.get_flat_coord(rank)
        cl = bidx // (2 * self.mc)
        chunk = cl // self.chunk_clusters  # token chunk (the B / SFB L coordinate)
        cl = cl % self.chunk_clusters
        mb = (cl // self.ksplit) * self.mc + (bidx % (2 * self.mc)) // 2
        kh = cl % self.ksplit

        smem = utils.SmemAllocator()
        st = smem.allocate(self.storage)
        ab = pipeline.PipelineTmaUmma.create(
            barrier_storage=st.full.data_ptr(), num_stages=self.stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.tx_bytes, cta_layout_vmnk=cluster_vmnk, defer_sync=True,
        )  # fmt: skip
        accp = pipeline.PipelineUmmaAsync.create(
            barrier_storage=st.acc_full.data_ptr(), num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 2 * EPI_WARPS),
            cta_layout_vmnk=cluster_vmnk, defer_sync=True,
        )  # fmt: skip
        tmem = utils.TmemAllocator(
            st.tmem_hold.ptr,
            barrier_for_retrieve=pipeline.NamedBarrier(
                barrier_id=TMEM_BAR, num_threads=32 * (EPI_WARPS + 1)
            ),
            allocator_warp_id=0, is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=st.tmem_dealloc.ptr, arch="sm_107",
        )  # fmt: skip
        pipeline_init_arrive(cluster_shape_mn=self.cs, is_relaxed=True)

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
        pipeline_init_wait(cluster_shape_mn=self.cs)
        cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(self.debug & 16):
            return

        k0 = kh * self.ktiles
        # K-start rotation per row block: pairs do not read the same x lines in
        # lock-step (an all-SM hot L2 region delivers a fraction of the L2 rate).
        rot = ((mb // self.mc) * 7) % self.ktiles  # one K order per cluster (multicast)
        nis = self.issuers
        a_issuer = Int32(-1)
        if warp == TMA_WARP:
            a_issuer = Int32(0)
        if cutlass.const_expr(nis > 1):
            if warp > X_WARP and (warp - X_WARP) % 2 == 1:
                a_issuer = (warp - X_WARP + 1) // 2
        if a_issuer >= 0 and not cutlass.const_expr(self.debug & 32):
            gA_s = tAgA[(None, mb, None, 0)]
            gSFA_s = tAgSFA[(None, mb, None, 0)]
            prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.stages)
            for _ in cutlass.range(a_issuer, unroll=1):
                prod.advance()
            # Weights do not depend on the predecessor: the ring fills with them while
            # the SP gather that writes x is still running.
            for t in cutlass.range(a_issuer, self.ktiles, nis, unroll=1):
                kk = k0 + (t + rot) % self.ktiles
                ab.producer_acquire(prod)
                bar = ab.producer_get_barrier(prod)
                if cutlass.const_expr(not self.debug & 512):
                    cute.copy(tma_a, gA_s[(None, kk)], tAsA[(None, prod.index)],
                              tma_bar_ptr=bar, mcast_mask=a_mask)  # fmt: skip
                if cutlass.const_expr(not self.debug & 128):
                    cute.copy(tma_sfa, gSFA_s[(None, kk)], tAsSFA[(None, prod.index)],
                              tma_bar_ptr=bar, mcast_mask=a_mask)  # fmt: skip
                for _ in cutlass.range_constexpr(nis):
                    prod.advance()
            if a_issuer == 0:
                end = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.stages
                )
                for _ in cutlass.range(self.ktiles, unroll=1):
                    end.advance()
                ab.producer_tail(end)

        x_issuer = Int32(-1)
        if warp == X_WARP:
            x_issuer = Int32(0)
        if cutlass.const_expr(nis > 1):
            if warp > X_WARP and (warp - X_WARP) % 2 == 0:
                x_issuer = (warp - X_WARP) // 2
        if x_issuer >= 0 and not cutlass.const_expr(self.debug & 32):
            gB_s = tBgB[(None, 0, None, chunk)]
            xs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.stages)
            for _ in cutlass.range(x_issuer, unroll=1):
                xs.advance()
            cute.arch.griddepcontrol_wait()
            for t in cutlass.range(x_issuer, self.ktiles, nis, unroll=1):
                kk = k0 + (t + rot) % self.ktiles
                # same slot as the weight warp; only the weight warp arrives (expect_tx)
                ab.sync_object_empty.wait(xs.index, xs.phase)
                bar = ab.producer_get_barrier(xs)
                if cutlass.const_expr(not self.debug & 256):
                    cute.copy(tma_b, gB_s[(None, kk)], tBsB[(None, xs.index)],
                              tma_bar_ptr=bar, mcast_mask=b_mask)  # fmt: skip
                if cutlass.const_expr(not self.debug & 64):
                    for j in cutlass.range_constexpr(self.nb):
                        cute.copy(
                            tma_sfb, tBgSFB[(None, j, kk, chunk)], tBsSFBs[j][(None, xs.index)],
                            tma_bar_ptr=bar, mcast_mask=sfb_mask,
                        )  # fmt: skip
                for _ in cutlass.range_constexpr(nis):
                    xs.advance()

        if warp == MMA_WARP and not cutlass.const_expr(self.debug & 32):
            tmem.wait_for_alloc()
            acc_ptr = tmem.retrieve_ptr(Float32)
            tAcc = cute.make_tensor(acc_ptr, acc_fake.layout)
            acc_cols = MMA_N * self.nb
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
            for t in cutlass.range(self.ktiles, unroll=1):
                if is_leader:
                    ab.consumer_wait(cons)
                    s = cons.index
                    cute.copy(s2t_a[0], s2t_a[1][(None, None, None, None, s)], s2t_a[2])
                    for j in cutlass.range_constexpr(self.nb):
                        s2t_b = s2t_bs[j]
                        cute.copy(s2t_b[0], s2t_b[1][(None, None, None, None, s)], s2t_b[2])
                    for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                        for j in cutlass.range_constexpr(0 if self.debug & 2 else self.nb):
                            m = mma
                            if cutlass.const_expr(self.collector and self.nb > 1):
                                if cutlass.const_expr(j == 0):
                                    m = mma_fill
                                elif cutlass.const_expr(j == self.nb - 1):
                                    m = mma_last
                                else:
                                    m = mma_use
                            m.set(tcgen05.Field.ACCUMULATE, t != 0 or kb != 0)
                            cute.gemm(
                                m, tAcc[(None, 0, j, 0)],
                                [tCrA[(None, 0, kb, s)], tSFA[(None, 0, kb)]],
                                [tCrB[(None, j, kb, s)], tSFBs[j][(None, 0, kb)]],
                                tAcc[(None, 0, j, 0)],
                            )  # fmt: skip
                    ab.consumer_release(cons)
                cons.advance()
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
            if cutlass.const_expr(not self.debug & 1):
                self._epilogue(
                    acc_ptr, out_ptr, part_ptr, ticket_ptr, st.ticket.ptr.toint(), mb, v,
                    chunk, st.sA.data_ptr().toint(), st.sB.data_ptr().toint(), st.xbar.data_ptr(),
                    warp, lane, tidx,
                )  # fmt: skip
            if cutlass.const_expr(not self.debug & 32):
                with cute.arch.elect_one():
                    accp.consumer_release(accc)
            tmem.relinquish_alloc_permit()
            tmem.free(acc_ptr)

    def _s2t(self, sSF, tSF):
        """SMEM -> TMEM scale-factor copy (tcgen05.cp 4x32dp128bit, cta_group::2)."""
        sc = cute.filter_zeros(sSF)
        tc = cute.filter_zeros(tSF)
        atom = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.TWO), cutlass.Float8E8M0FNU)
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
    def _epilogue(self, acc_ptr, out_ptr, part_ptr, ticket_ptr, ticket_s, mb, v, chunk,
                  sa_s, sb_s,
                  xbar, warp, lane, tidx):  # fmt: skip
        """Thread (warp w, lane l) owns weight row 128 v + 32 w + l of row block mb
        (TMEM lane 32 w + l) and all token columns."""
        cols = MMA_N * self.nb
        r = 32 * warp + lane
        n = mb * 256 + 128 * v + r
        taddr = Int32(acc_ptr.toint()) + ((32 * warp) << 16)
        out = cute.make_tensor(out_ptr, cute.make_layout(self.total_rows * self.ldo))
        n0 = chunk * (self.rows * self.ldo)  # this pair's token chunk
        live = n < self.n
        if cutlass.const_expr(self.ksplit == 1):
            for c0 in cutlass.range_constexpr(0, cols, 16):
                vals = tmem_wait_n(*tmem_ld16(taddr + c0))
                for i in cutlass.range_constexpr(16):
                    if cutlass.const_expr(c0 + i < self.rows):
                        if live:
                            out[n0 + (c0 + i) * self.ldo + n] = vals[i].to(cutlass.BFloat16)
        else:
            slot = mb * 2 + v
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
                for c0 in cutlass.range_constexpr(0, cols, 16):
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
                    while ld_acquire_gpu(flag_a) < gen:
                        pass
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
                            if cutlass.const_expr(cc + i < self.rows):
                                if live:
                                    out[n0 + (cc + i) * self.ldo + n] = (vals[i] + p[i]).to(
                                        cutlass.BFloat16
                                    )
                    if cutlass.const_expr(j + 2 < nch):
                        bar_sync(EPI_BAR, 32 * EPI_WARPS)  # chunk j's buffer is free again
                        if tidx == 0:
                            cute.arch.mbarrier_arrive_and_expect_tx(xbar + (j % 2), chunk)
                            copy_bulk(sbase, src0 + (j + 2) * chunk, (xbar + (j % 2)).toint(),
                                      chunk)  # fmt: skip
