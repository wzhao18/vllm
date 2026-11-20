# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 SP o-projection fused with its reduce-scatter push (VR200, KDA layers of the
NVFP4 checkpoint): ``OprojPushMx``.

out[R, 7168] = x[R, K] @ W[7168, K]^T (W: this rank's row-parallel K shard, FP8 E4M3
rows + F8_128x4 E8M0 scales; x: the MXFP8 attention output, the same formats), FP32
accumulation of the block-scaled products, one BF16 rounding per element (stock: the
online-MXFP8 linear, cuBLASLt / FlashInfer with FP32 accumulation; the K summation
order differs). Instead of writing the partial to ``SpComm.partial`` for
``SpReduceAttnRes(push=True)`` to read and push, the epilogue pushes each output row
straight into its owner's receive slot (``SpComm.recv``, Lamport words: stock's
``sanitize_lamport_payload`` via ``positive_zero32``); the owner kernel then runs
with ``push=False``, exactly as for ``oproj.sp.SpOprojWarp``.

Mainloop (``gemm.dense_mx8`` form): swap-AB ``cta_group::2`` M256 block-scaled MMAs
(``mxf8f6f4`` with the sm_107a 2x-K K64 steps): a CTA pair owns 256 output features
(128 per CTA: the A halves) and one block of 128 tokens (64 per CTA: the B halves)
over the full K, so a pair's operand bytes are (128 + 64) x K per CTA; the weight
warp TMAs the first ``stages`` weight K tiles before ``griddepcontrol.wait``. Grid:
28 feature blocks x ceil(R / 128) token blocks of pairs (R 384: 84 pairs).

Epilogue: TMEM -> FP32 -> BF16 (RN) into a token-major shared tile (the drained A
ring), then 16-byte chunks (8 features of one token) sanitized and stored relaxed at
system scope into ``peers[owner] + (rank, token - owner n, feature)`` of ``recv``.
"""

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

from ..common.primitives import asm, prefetch_bytes, push_float, store_relaxed, wait
from ..common.sp_comm import positive_zero32

WIDTH = 7168  # output features (the hidden size)
SF_VEC = 32
MMA_N = 128  # tokens per pair tile (64 per CTA of the pair)
MMA_K = 64  # 2x-K mxf8f6f4
FEATURES = 256  # output features per pair (128 per CTA)
EPI_WARPS = 4
MMA_WARP = 4
TMA_WARP = 5  # weights (A, SFA): the first stages before the PDL wait
TMEM_COLS = 512
TMEM_BAR = 1  # named barrier ids (0 = __syncthreads)
EPI_BAR = 2
TMA_MIN_ROWS = 2  # activation rows the TMA atoms span (a 1-row map faults on VR200)
PF_BYTES = 32768  # L2 prefetch op size (weights)


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


@asm()
def st_shared_bf16(address, value):
    """BF16 (RN) of ``value`` at shared ``address``."""
    code = "{ .reg .b16 h; cvt.rn.bf16.f32 h,$1; st.shared.b16 [$0],h; }"
    return code, "r,f,~{memory}", (Int32(address), Float32(value))


@asm((Uint32,) * 4)
def ld_shared_u32x4(address):
    code = "ld.shared.v4.b32 {$0,$1,$2,$3}, [$4];"
    return code, "=r,=r,=r,=r,r,~{memory}", (Int32(address),)


@asm((Float32,) * 4)
def ld_shared_f32x4(address):
    code = "ld.shared.v4.f32 {$0,$1,$2,$3}, [$4];"
    return code, "=f,=f,=f,=f,r,~{memory}", (Int32(address),)


@asm()
def arrive_remote(barrier, peer):
    """mbarrier arrive (release, cluster scope) on the same barrier of cluster CTA peer."""
    code = "{ .reg .b32 b; mapa.shared::cluster.u32 b,$0,$1; "
    code += "mbarrier.arrive.release.cluster.shared::cluster.b64 _,[b]; }"
    return code, "r,r,~{memory}", (Int32(barrier), Int32(peer))


@asm()
def bar_sync(barrier_id: int, count: int):
    return f"bar.sync {barrier_id},{count};", "~{memory}", ()


class OprojPushMx:
    """One compiled shape: K (heads x 128), ``rows`` tokens (R = TP x n), ``tp`` row
    owners, ``cap`` rows per sender slot of ``recv`` (``SpComm.rows_cap / TP``).
    ``stages`` K tiles of ``kt`` in the operand ring; ``iw`` / ``ix`` TMA issuer warps
    for the weights / the activation (one issuer completes one bulk op at a time)."""

    def __init__(self, k: int, rows_max: int, tp: int, cap: int, *, kt: int = 128,
                 stages: int = 8, iw: int = 1, ix: int = 1, prefetch: bool = False,
                 ksplit: int = 1, debug: int = 0):  # fmt: skip
        ktiles = k // kt // ksplit
        tblocks = -(-rows_max // MMA_N)
        # ksplit 2: the two K halves of a (feature, token) block are two pairs of one
        # 4-CTA cluster that reduce over DSMEM; 4-CTA clusters co-reside only up to
        # ~46 at one CTA per SM (hwdb hw56C), so one token block (28 clusters).
        if (k % (kt * ksplit) or ktiles % stages or rows_max > min(512, tp * cap)
                or ksplit not in (1, 2) or (ksplit == 2 and tblocks > 1)):
            raise ValueError(f"OprojPushMx k {k} rows {rows_max} tp {tp} cap {cap} "
                             f"kt {kt} stages {stages} ksplit {ksplit}")  # fmt: skip
        self.k, self.tp, self.cap = k, tp, cap
        self.kt, self.stages, self.ktiles, self.ksplit = kt, stages, ktiles, ksplit
        self.iw, self.ix, self.prefetch = iw, ix, prefetch
        # bench only: 1 no MMAs, 2 no activation (B) loads, 4 no weight (A) loads
        self.debug = debug
        self.x_warp = TMA_WARP + iw
        self.threads = 32 * (TMA_WARP + iw + ix)
        self.mblocks = WIDTH // FEATURES
        self.tblocks = tblocks
        self.clus = 2 * ksplit
        self.grid = self.clus * self.mblocks * self.tblocks
        self.tiler = (FEATURES, MMA_N, kt)
        self.tiler_sfb = (128, MMA_N, kt)

    def _mma(self):
        return sm107_utils.make_blockscaled_trivial_tiled_mma(
            cutlass.Float8E4M3FN, cutlass.Float8E4M3FN, OperandMajorMode.K,
            OperandMajorMode.K, cutlass.Float8E8M0FNU, SF_VEC, tcgen05.CtaGroup.TWO,
            (FEATURES, MMA_N, MMA_K), a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
        )  # fmt: skip

    @cute.jit
    def __call__(self, w_ptr: cute.Pointer, sfw_ptr: cute.Pointer, x_ptr: cute.Pointer,
                 sfx_ptr: cute.Pointer, peers: cute.Tensor, rank: Int32, rows: Int32,
                 stream: cuda.CUstream):  # fmt: skip
        k = self.k
        fp8, sf_t = cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU
        a = cute.make_tensor(w_ptr, cute.make_ordered_layout((WIDTH, k, 1), order=(1, 0, 2)))
        b = cute.make_tensor(
            x_ptr, cute.make_ordered_layout((rows, k, 1), order=(1, 0, 2))
        )
        sfa = cute.make_tensor(sfw_ptr, blockscaled_utils.tile_atom_to_shape_SF(a.shape, SF_VEC))
        sfb = cute.make_tensor(sfx_ptr, blockscaled_utils.tile_atom_to_shape_SF(b.shape, SF_VEC))

        mma = self._mma()
        mma_sfb = sm107_utils.make_blockscaled_trivial_tiled_mma(
            fp8, fp8, OperandMajorMode.K, OperandMajorMode.K, sf_t, SF_VEC,
            tcgen05.CtaGroup.ONE, (128, MMA_N, MMA_K),
            a_collector_op=CollectorOp.DISCARD, b_collector_op=CollectorOp.DISCARD,
        )  # fmt: skip
        pair_vmnk = cute.tiled_divide(cute.make_layout((2, 1, 1)), (mma.thr_id.shape,))
        pair_sfb_vmnk = cute.tiled_divide(cute.make_layout((2, 1, 1)), (mma_sfb.thr_id.shape,))
        # the K half is the cluster K mode (rank = v + 2 kh): masks and pipelines stay
        # inside one pair; the TMA atoms are built for the pair alone
        kc = self.ksplit
        cl_vmnk = cute.tiled_divide(cute.make_layout((2, 1, kc)), (mma.thr_id.shape,))
        cl_sfb_vmnk = cute.tiled_divide(cute.make_layout((2, 1, kc)), (mma_sfb.thr_id.shape,))
        a_layout = sm100_utils.make_smem_layout_a(mma, self.tiler, fp8, self.stages)
        b_layout = sm100_utils.make_smem_layout_b(mma, self.tiler, fp8, self.stages)
        sfa_layout = blockscaled_utils.make_smem_layout_sfa(mma, self.tiler, SF_VEC, self.stages)
        sfb_layout = blockscaled_utils.make_smem_layout_sfb(mma, self.tiler, SF_VEC, self.stages)
        tsfa_layout = blockscaled_utils.make_tmem_layout_sfa(
            mma, self.tiler, SF_VEC, cute.slice_(sfa_layout, (None, None, None, 0))
        )
        tsfb_layout = blockscaled_utils.make_tmem_layout_sfb(
            mma, self.tiler, SF_VEC, cute.slice_(sfb_layout, (None, None, None, 0))
        )
        self.sfa_cols = cute.cosize(cute.recast_layout(32, sf_t.width, tsfa_layout)) & 0xFFFF

        a_one = cute.slice_(a_layout, (None, None, None, 0))
        b_one = cute.slice_(b_layout, (None, None, None, 0))
        sfa_one = cute.slice_(sfa_layout, (None, None, None, 0))
        sfb_one = cute.slice_(sfb_layout, (None, None, None, 0))
        cs = (2, 1)
        tma_a, ta = cute.nvgpu.make_tiled_tma_atom_A(
            sm100_utils.cluster_shape_to_tma_atom_A(cs, mma.thr_id), a, a_one,
            self.tiler, mma, pair_vmnk.shape,
        )  # fmt: skip
        tma_b, tb = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_B(cs, mma.thr_id), b, b_one,
            self.tiler, mma, pair_vmnk.shape,
        )  # fmt: skip
        tma_sfa, tsfa = cute.nvgpu.make_tiled_tma_atom_A(
            sm100_utils.cluster_shape_to_tma_atom_A(cs, mma.thr_id), sfa, sfa_one,
            self.tiler, mma, pair_vmnk.shape, internal_type=cutlass.Int16,
        )  # fmt: skip
        tma_sfb, tsfb = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_SFB(cs, mma.thr_id), sfb, sfb_one,
            self.tiler_sfb, mma_sfb, pair_sfb_vmnk.shape, internal_type=cutlass.Int16,
        )  # fmt: skip
        dbg = self.debug
        self.tx_bytes = 2 * (
            (0 if dbg & 4 else cute.size_in_bytes(fp8, a_one) + cute.size_in_bytes(sf_t, sfa_one))
            + (0 if dbg & 2 else cute.size_in_bytes(fp8, b_one) + cute.size_in_bytes(sf_t, sfb_one))
        )  # fmt: skip

        @cute.struct
        class Storage:
            full: cute.struct.MemRange[cutlass.Int64, self.stages]
            empty: cute.struct.MemRange[cutlass.Int64, self.stages]
            acc_full: cute.struct.MemRange[cutlass.Int64, 1]
            acc_empty: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_dealloc: cutlass.Int64
            tmem_hold: cutlass.Int32
            xbar: cute.struct.MemRange[cutlass.Int64, 2]
            # scale factors first: their smem descriptors must stay below 256 KiB
            sSFA: cute.struct.Align[cute.struct.MemRange[sf_t, cute.cosize(sfa_layout)], 128]
            sSFB: cute.struct.Align[cute.struct.MemRange[sf_t, cute.cosize(sfb_layout)], 128]
            sA: cute.struct.Align[cute.struct.MemRange[fp8, cute.cosize(a_layout.outer)], 1024]
            sB: cute.struct.Align[cute.struct.MemRange[fp8, cute.cosize(b_layout.outer)], 1024]

        self.storage = Storage
        self.kernel(
            mma, mma_sfb, tma_a, ta, tma_b, tb, tma_sfa, tsfa, tma_sfb, tsfb, cl_vmnk,
            cl_sfb_vmnk, a_layout, b_layout, sfa_layout, sfb_layout, tsfa_layout,
            tsfb_layout, peers, rank, rows, w_ptr.toint(), sfw_ptr.toint(),
        ).launch(
            grid=[self.grid, 1, 1], block=[self.threads, 1, 1], cluster=[self.clus, 1, 1],
            stream=stream, use_pdl=True,
        )  # fmt: skip

    @cute.kernel
    def kernel(self, mma: cute.TiledMma, mma_sfb: cute.TiledMma, tma_a: cute.CopyAtom,
               mA: cute.Tensor, tma_b: cute.CopyAtom, mB: cute.Tensor,
               tma_sfa: cute.CopyAtom, mSFA: cute.Tensor, tma_sfb: cute.CopyAtom,
               mSFB: cute.Tensor, pair_vmnk: cute.Layout, pair_sfb_vmnk: cute.Layout,
               a_layout: cute.ComposedLayout, b_layout: cute.ComposedLayout,
               sfa_layout: cute.Layout, sfb_layout: cute.Layout,
               tsfa_layout: cute.Layout, tsfb_layout: cute.Layout, peers: cute.Tensor,
               rank: Int32, rows: Int32, w_addr: Int64, sfw_addr: Int64):  # fmt: skip
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        sf_t = cutlass.Float8E8M0FNU
        if warp == TMA_WARP:
            cpasync.prefetch_descriptor(tma_a)
            cpasync.prefetch_descriptor(tma_b)
            cpasync.prefetch_descriptor(tma_sfa)
            cpasync.prefetch_descriptor(tma_sfb)
        v = bidx % 2
        is_leader = v == 0
        crank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        coord = pair_vmnk.get_flat_coord(crank)
        coord_sfb = pair_sfb_vmnk.get_flat_coord(crank)
        pair = bidx // 2
        kh = pair % self.ksplit
        blk = pair // self.ksplit
        mb = blk % self.mblocks
        tb = blk // self.mblocks
        k0 = kh * self.ktiles

        smem = utils.SmemAllocator()
        st = smem.allocate(self.storage)
        ab = pipeline.PipelineTmaUmma.create(
            barrier_storage=st.full.data_ptr(), num_stages=self.stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.tx_bytes, cta_layout_vmnk=pair_vmnk, defer_sync=True,
        )  # fmt: skip
        accp = pipeline.PipelineUmmaAsync.create(
            barrier_storage=st.acc_full.data_ptr(), num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 2 * EPI_WARPS),
            cta_layout_vmnk=pair_vmnk, defer_sync=True,
        )  # fmt: skip
        tmem = utils.TmemAllocator(
            st.tmem_hold.ptr,
            barrier_for_retrieve=pipeline.NamedBarrier(
                barrier_id=TMEM_BAR, num_threads=32 * (EPI_WARPS + 1)
            ),
            allocator_warp_id=0, is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=st.tmem_dealloc.ptr, arch="sm_107",
        )  # fmt: skip
        if cutlass.const_expr(self.ksplit == 2):
            if tidx == 0:  # DSMEM reduce: [0] partial received (tx), [1] peer ring drained
                cute.arch.mbarrier_init(st.xbar.data_ptr(), 1)
                cute.arch.mbarrier_init(st.xbar.data_ptr() + 1, 1)
                cute.arch.mbarrier_init_fence()
        pipeline_init_arrive(cluster_shape_mn=(self.clus, 1), is_relaxed=True)

        sA = st.sA.get_tensor(a_layout.outer, swizzle=a_layout.inner)
        sB = st.sB.get_tensor(b_layout.outer, swizzle=b_layout.inner)
        sSFA = st.sSFA.get_tensor(sfa_layout)
        sSFB = st.sSFB.get_tensor(sfb_layout)
        a_mask = cpasync.create_tma_multicast_mask(pair_vmnk, coord, mcast_mode=2)
        b_mask = cpasync.create_tma_multicast_mask(pair_vmnk, coord, mcast_mode=1)
        sfb_mask = cpasync.create_tma_multicast_mask(pair_sfb_vmnk, coord_sfb, mcast_mode=1)

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
        a_cta = cute.make_layout(cute.slice_(pair_vmnk, (0, 0, None, 0)).shape)
        b_cta = cute.make_layout(cute.slice_(pair_vmnk, (0, None, 0, 0)).shape)
        sfb_cta = cute.make_layout(cute.slice_(pair_sfb_vmnk, (0, None, 0, 0)).shape)
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
        tBsSFB, tBgSFB = cpasync.tma_partition(
            tma_sfb, coord_sfb[1], sfb_cta, cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )  # fmt: skip
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)
        tCrA = mma.make_fragment_A(sA)
        tCrB = mma.make_fragment_B(sB)
        acc_shape = mma.partition_shape_C(self.tiler[:2])
        acc_fake = mma.make_fragment_C(cute.append(acc_shape, 1))
        pipeline_init_wait(cluster_shape_mn=(self.clus, 1))
        cute.arch.griddepcontrol_launch_dependents()

        # K-start rotation per feature block: pairs do not read the same x lines in
        # lock-step (an all-SM hot L2 region delivers a fraction of the L2 rate).
        rot = (mb * 7) % self.ktiles
        if warp >= TMA_WARP and warp < self.x_warp:
            iw = warp - TMA_WARP
            gA_s = tAgA[(None, mb, None, 0)]
            gSFA_s = tAgSFA[(None, mb, None, 0)]
            prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.stages)
            for _ in cutlass.range(iw, unroll=1):
                prod.advance()
            # Weights do not depend on the predecessor: the ring fills with them while
            # the attention core that writes x is still running.
            for t in cutlass.range(iw, self.ktiles, self.iw, unroll=1):
                kk = k0 + (t + rot) % self.ktiles
                ab.producer_acquire(prod)
                bar = ab.producer_get_barrier(prod)
                if cutlass.const_expr(not self.debug & 4):
                    cute.copy(tma_a, gA_s[(None, kk)], tAsA[(None, prod.index)],
                              tma_bar_ptr=bar, mcast_mask=a_mask)  # fmt: skip
                    cute.copy(tma_sfa, gSFA_s[(None, kk)], tAsSFA[(None, prod.index)],
                              tma_bar_ptr=bar, mcast_mask=a_mask)  # fmt: skip
                if cutlass.const_expr(self.prefetch):
                    if t == self.stages - 1:
                        self._prefetch(w_addr, sfw_addr, mb, v)
                for _ in cutlass.range_constexpr(self.iw):
                    prod.advance()
            if iw == 0:
                tail = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.stages
                )
                for _ in cutlass.range(self.ktiles, unroll=1):
                    tail.advance()
                ab.producer_tail(tail)

        if warp >= self.x_warp and warp < self.x_warp + self.ix:
            ix = warp - self.x_warp
            gB_s = tBgB[(None, tb, None, 0)]
            xs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.stages)
            for _ in cutlass.range(ix, unroll=1):
                xs.advance()
            cute.arch.griddepcontrol_wait()
            for t in cutlass.range(ix, self.ktiles, self.ix, unroll=1):
                kk = k0 + (t + rot) % self.ktiles
                # same slot as the weight warp; only the weight warp arrives (expect_tx)
                ab.sync_object_empty.wait(xs.index, xs.phase)
                bar = ab.producer_get_barrier(xs)
                if cutlass.const_expr(not self.debug & 2):
                    cute.copy(tma_b, gB_s[(None, kk)], tBsB[(None, xs.index)],
                              tma_bar_ptr=bar, mcast_mask=b_mask)  # fmt: skip
                    cute.copy(tma_sfb, tBgSFB[(None, tb, kk, 0)], tBsSFB[(None, xs.index)],
                              tma_bar_ptr=bar, mcast_mask=sfb_mask)  # fmt: skip
                for _ in cutlass.range_constexpr(self.ix):
                    xs.advance()

        if warp == MMA_WARP:
            tmem.wait_for_alloc()
            acc_ptr = tmem.retrieve_ptr(Float32)
            tAcc = cute.make_tensor(acc_ptr, acc_fake.layout)
            tSFA = cute.make_tensor(cute.recast_ptr(acc_ptr + MMA_N, dtype=sf_t), tsfa_layout)
            tSFB = cute.make_tensor(
                cute.recast_ptr(acc_ptr + MMA_N + self.sfa_cols, dtype=sf_t), tsfb_layout
            )
            s2t_a = self._s2t(sSFA, tSFA)
            s2t_b = self._s2t(sSFB, tSFB)
            cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stages)
            accs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            if is_leader:
                accp.producer_acquire(accs)
            # Stage-unrolled (constant ring slot per unrolled step: constant MMA / S2T
            # descriptor offsets); only a round's first MMA has a runtime ACCUMULATE.
            rounds = self.ktiles // self.stages
            nkb = cute.size(tCrA, mode=[2])
            for tt in cutlass.range(rounds, unroll=1):
                for s in cutlass.range_constexpr(self.stages):
                    if is_leader:
                        ab.consumer_wait(cons)
                        cute.copy(s2t_a[0], s2t_a[1][(None, None, None, None, s)], s2t_a[2])
                        cute.copy(s2t_b[0], s2t_b[1][(None, None, None, None, s)], s2t_b[2])
                        for kb in cutlass.range_constexpr(0 if self.debug & 1 else nkb):
                            if cutlass.const_expr(s == 0 and kb == 0):
                                mma.set(tcgen05.Field.ACCUMULATE, tt != 0)
                            else:
                                mma.set(tcgen05.Field.ACCUMULATE, True)
                            cute.gemm(
                                mma, tAcc[(None, 0, 0, 0)],
                                [tCrA[(None, 0, kb, s)], tSFA[(None, 0, kb)]],
                                [tCrB[(None, 0, kb, s)], tSFB[(None, 0, kb)]],
                                tAcc[(None, 0, 0, 0)],
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
            cute.arch.griddepcontrol_wait()  # (owner slots written after the producers)
            accc = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            accp.consumer_wait(accc)
            self._push(
                acc_ptr, peers, rank, rows, mb, tb, v, kh, st.sA.data_ptr().toint(),
                st.sB.data_ptr().toint(), st.xbar.data_ptr(), crank, warp, tidx,
            )  # fmt: skip
            with cute.arch.elect_one():
                accp.consumer_release(accc)
            tmem.relinquish_alloc_permit()
            tmem.free(acc_ptr)

    @cute.jit
    def _prefetch(self, w_addr, sfw_addr, mb, v):
        """Once the ring's first fill is issued: the CTA's 128 weight rows (contiguous;
        both K halves) and their scales (one 128-row F8_128x4 block) into L2, so the
        later ring loads hit L2 instead of HBM."""
        lane = cute.arch.lane_idx()
        row0 = mb * FEATURES + 128 * v
        ops = 128 * self.k // PF_BYTES
        if lane < ops:
            prefetch_bytes(w_addr + Int64(row0 * self.k + lane * PF_BYTES), PF_BYTES)
        if lane == ops:
            span = 128 * (self.k // SF_VEC)
            prefetch_bytes(sfw_addr + Int64((row0 // 128) * span), span)

    def _s2t(self, sSF, tSF):
        """SMEM -> TMEM scale-factor copy (tcgen05.cp 4x32dp128bit, cta_group::2)."""
        sc = cute.filter_zeros(sSF)
        tc = cute.filter_zeros(tSF)
        atom = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.TWO), cutlass.Float8E8M0FNU
        )
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
    def _push(self, acc_ptr, peers, rank, rows, mb, tb, v, kh, sa_s, sb_s, xbar, crank, warp,
              tidx):  # fmt: skip
        """Thread (warp w, lane l) holds feature 128 v + 32 w + l of feature block mb
        (TMEM lane 32 w + l) for the pair's 128 tokens. ksplit 1: BF16 into the
        token-major tile T[128 tokens][128 features] (the drained A ring). ksplit 2:
        symmetric reduce-scatter with the other K half (cluster CTA crank ^ 2, the
        same features): this CTA owns tokens [64 kh, 64 kh + 64), pushes its FP32
        partial of the other 64 into the peer's drained A ring (st.async, quad-major
        [q][128 features]) once the peer's ring is free, adds the peer's partial of
        its own (FP32 addition commutes: either CTA's sum is the same bits), rounds
        once into T[64][128] (the drained B ring). Then 16 B chunks of T go to the
        token's row owner."""
        r = tidx  # 32 w + l
        taddr = Int32(acc_ptr.toint()) + ((32 * warp) << 16)
        cols = MMA_N // self.ksplit
        tile = sb_s if cutlass.const_expr(self.ksplit == 2) else sa_s
        if cutlass.const_expr(self.ksplit == 1):
            for c0 in cutlass.range_constexpr(0, MMA_N, 16):
                vals = tmem_wait_n(*tmem_ld16(taddr + c0))
                for i in cutlass.range_constexpr(16):
                    st_shared_bf16(sa_s + ((c0 + i) * 128 + r) * 2, vals[i])
        else:
            o0 = kh * cols
            p0 = cols - o0
            peer = crank ^ 2
            rbar = xbar.toint()
            fbar = (xbar + 1).toint()
            if tidx == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(xbar, cols * 128 * 4)
                arrive_remote(fbar, peer)  # our ring is drained (accumulator complete)
            wait(fbar, 0, scope="cluster")
            for c0 in cutlass.range_constexpr(0, cols, 16):
                vals = tmem_wait_n(*tmem_ld16(taddr + p0 + c0))
                for qq in cutlass.range_constexpr(4):
                    q = c0 // 4 + qq
                    push_float(Int32(sa_s + (q * 128 + r) * 16),
                               tuple(vals[4 * qq:4 * qq + 4]), Int32(rbar), peer)  # fmt: skip
            wait(rbar, 0, scope="cluster")
            for c0 in cutlass.range_constexpr(0, cols, 16):
                vals = tmem_wait_n(*tmem_ld16(taddr + o0 + c0))
                for qq in cutlass.range_constexpr(4):
                    p = ld_shared_f32x4(sa_s + ((c0 // 4 + qq) * 128 + r) * 16)
                    for i in cutlass.range_constexpr(4):
                        c = c0 + 4 * qq + i
                        st_shared_bf16(sb_s + (c * 128 + r) * 2, vals[4 * qq + i] + p[i])
        bar_sync(EPI_BAR, 32 * EPI_WARPS)
        column = mb * FEATURES + 128 * v
        t0 = tb * MMA_N + kh * cols
        n = rows // self.tp
        for j in cutlass.range_constexpr(cols * 16 // (32 * EPI_WARPS)):
            q = j * 32 * EPI_WARPS + tidx
            token = t0 + q // 16
            chunk = q % 16
            if token < rows:
                w = ld_shared_u32x4(tile + ((q // 16) * 128 + chunk * 8) * 2)
                words = tuple(positive_zero32(w[e]) for e in range(4))
                owner = token // n
                offset = (rank * self.cap + token - owner * n) * WIDTH + column + chunk * 8
                store_relaxed(Int64(peers[owner]) + Int64(offset) * 2, words)
