# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 SP o-projection that pushes its partial rows to their owners block by block
while it still computes (VR200, KDA layers of the NVFP4 checkpoint): ``OprojKsPush``.

out[R, 7168] = x[R, K] @ W[7168, K]^T for this rank's row-parallel K shard (FP8 E4M3
rows + F8_128x4 E8M0 scales, x the MXFP8 attention output), FP32 accumulation of the
block-scaled products, one BF16 rounding per element; the rounded rows go straight
into their owners' Lamport receive slots (``SpComm`` recv, ``positive_zero32`` words)
and the owner kernel ``SpReduceAttnRes`` runs with ``push=False``.

Why: a one-wave GEMM finishes every tile at once, so its 3 n x 14336 B push drains
after the mainloop (``gemm.sm107.push_epilogue``). Here the pushes of the early token
blocks run under the MMAs of the later ones. A CTA can only hold its weight slice for
all token blocks if it owns a K slice (W 128 x 3072 B does not fit shared memory), so:

- grid: 28 feature blocks of 256 (``cta_group::2`` M256 pairs, swap-AB: weights on M)
  x KS = 3 K slices of 1024 = 168 CTAs, one per SM, cluster = the pair;
- the pair's W slice (8 K tiles of 128, + scales) is loaded once, before
  griddepcontrol.wait; x arrives per token block of 128 rows (8 K tiles, one 2-slot
  ring), blocks in an order that puts this rank's own rows last (``block_order``);
- each block's FP32 accumulator (TMEM, one buffer per block up to 3) is reduce-
  scattered over the K slices through L2: the CTA of K slice s owns token chunk s
  (48 / 48 / 32 rows), writes its partials of the two other chunks to their owners'
  scratch slots, bumps their counters (release), and once both peers' partials of its
  chunk are in (acquire) sums the three in K-slice order in FP32, rounds once to BF16
  and pushes the chunk rows (16 B, consecutive lanes along a row) to the row owners.

Precision: FP32 products of the block-scaled MMA accumulated over each K slice, the
three slice sums added in slice order, one BF16 rounding: the GEMM's FP32 summation
in another order (as every MXFP8 kind). Deterministic (fixed order, no atomics on
data). Every pushed word is sanitized like ``SpReduceAttnRes(push=True)``'s."""

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

from ..common.primitives import asm, bf16x2, store_relaxed
from ..common.sp_comm import positive_zero32

WIDTH = 7168  # output features (the hidden size)
SF_VEC = 32
KT = 128  # K tile (bytes of FP8 per row)
MMA_K = 64  # 2x-K mxf8f6f4
FEATURES = 256  # output features per pair (128 per CTA)
TB = 128  # tokens per block (MMA N; 64 per CTA of the pair)
KS = 3  # K slices
CHUNK = (48, 48, 32)  # token rows of a block reduced and pushed by K slice s
START = (0, 48, 96)
CMAX = 48
EPI_WARPS = 4
MMA_WARP = 4
W_WARP = 5
X_WARP = 6
TMEM_COLS = 512
GATHER_WARP = 7  # warps 7..10: the gather / push group (TMEM lane quarter = warp % 4)
TMEM_BAR = 1
EPI_BAR = 2  # the scatter group (warps 0..3)
GATHER_BAR = 3
DONE_BAR = 4  # both groups, before the TMEM is freed
WRITERS = 2 * 32 * EPI_WARPS  # counter target: both writer CTAs' scatter threads
MAX_BLOCKS = 4  # R <= 512
STAMPS = 32  # bench probe (dbg & 32): int64 globaltimer stamps per CTA after the scratch
STAMP_FLOATS = 2 * STAMPS * 2 * (WIDTH // FEATURES) * KS


def block_order(rows: int, n: int, rank: int) -> list[int]:
    """Token blocks of an R-row call, those holding fewer of ``rank``'s own rows
    first (stable): the last block's pushes are mostly local."""
    blocks = -(-rows // TB)

    def own(b):
        lo, hi = max(b * TB, rank * n), min((b + 1) * TB, (rank + 1) * n, rows)
        return max(0, hi - lo)

    return sorted(range(blocks), key=lambda b: (own(b), b))


def scratch_floats() -> int:
    """FP32 words of the reduce-scatter scratch: [56 CTA features][block][chunk]
    [2 writer slots][128 features][48 tokens]."""
    return (WIDTH // 128) * MAX_BLOCKS * KS * 2 * 128 * CMAX + STAMP_FLOATS


def counter_words() -> int:
    return (WIDTH // 128) * MAX_BLOCKS * KS


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


@asm()
def st_f32x4(address, a, b, c, d):
    code = "st.global.v4.f32 [$0], {$1,$2,$3,$4};"
    return code, "l,f,f,f,f,~{memory}", (Int64(address), *(Float32(v) for v in (a, b, c, d)))


@asm((Float32,) * 4)
def ld_f32x4_cg(address):
    code = "ld.global.cg.v4.f32 {$0,$1,$2,$3}, [$4];"
    return code, "=f,=f,=f,=f,l,~{memory}", (Int64(address),)


@asm()
def red_release_add(address, value):
    code = "red.release.gpu.global.add.s32 [$0], $1;"
    return code, "l,r,~{memory}", (Int64(address), Int32(value))


@asm(Int32)
def ld_acquire(address):
    return "ld.acquire.gpu.global.b32 $0, [$1];", "=r,l,~{memory}", (Int64(address),)


@asm()
def st_relaxed_gpu(address, value):
    code = "st.relaxed.gpu.global.b32 [$0], $1;"
    return code, "l,r,~{memory}", (Int64(address), Int32(value))


@asm(Int64)
def globaltimer():
    return "mov.u64 $0, %globaltimer;", "=l,~{memory}", ()


@asm()
def st_i64(address, value):
    return "st.global.u64 [$0], $1;", "l,l,~{memory}", (Int64(address), Int64(value))


@asm(Float32)
def shfl_xor1(value):
    return "shfl.sync.bfly.b32 $0, $1, 1, 0x1f, 0xffffffff;", "=f,f", (Float32(value),)


@asm()
def st_shared_bf16_pz(address, value):
    """BF16 (RN) of ``value``, -0 stored as +0, at shared ``address``."""
    code = (
        "{ .reg .b16 h; .reg .pred p; cvt.rn.bf16.f32 h,$1; setp.eq.b16 p,h,0x8000; "
        "@p mov.b16 h,0; st.shared.b16 [$0],h; }"
    )
    return code, "r,f,~{memory}", (Int32(address), Float32(value))


@asm()
def st_shared_u32(address, word):
    return "st.shared.b32 [$0], $1;", "r,r,~{memory}", (Int32(address), Uint32(word))


@asm()
def bulk_s2g(dst, src, size):
    code = "cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], $2;"
    return code, "l,r,r,~{memory}", (Int64(dst), Int32(src), Int32(size))


@asm()
def bulk_commit_wait_read():
    code = "cp.async.bulk.commit_group; cp.async.bulk.wait_group.read 0;"
    return code, "~{memory}", ()


@asm()
def fence_proxy_async_smem():
    return "fence.proxy.async.shared::cta;", "~{memory}", ()


@asm()
def bar_sync(barrier_id: int, count: int):
    return f"bar.sync {barrier_id},{count};", "~{memory}", ()


class OprojKsPush:
    """One compiled token-block count (``blocks`` = ceil(R_max / 128) <= 4; R is a
    runtime argument): K (heads x 128), ``tp`` row owners, ``cap`` rows per sender
    slot of ``SpComm.recv``."""

    def __init__(self, k: int, blocks: int, tp: int, cap: int, dbg: int = 0):
        if k % (KS * KT) or not 1 <= blocks <= MAX_BLOCKS:
            raise ValueError(f"OprojKsPush k {k} blocks {blocks}")
        self.k, self.tp, self.cap, self.blocks = k, tp, cap, blocks
        # bench probes (wrong results): 1 no counter wait, 2 no scratch stores, 4 no
        # owner pushes, 8 no MMAs
        self.dbg = dbg
        self.ktiles = k // KT
        self.kslice = self.ktiles // KS
        self.nbuf = min(blocks, 3)
        self.mblocks = WIDTH // FEATURES
        self.grid = 2 * self.mblocks * KS
        self.threads = 32 * (GATHER_WARP + EPI_WARPS)
        self.tiler = (FEATURES, TB, KT)
        self.tiler_sfb = (128, TB, KT)

    def _mma(self):
        return sm107_utils.make_blockscaled_trivial_tiled_mma(
            cutlass.Float8E4M3FN, cutlass.Float8E4M3FN, OperandMajorMode.K,
            OperandMajorMode.K, cutlass.Float8E8M0FNU, SF_VEC, tcgen05.CtaGroup.TWO,
            (FEATURES, TB, MMA_K), a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
        )  # fmt: skip

    @cute.jit
    def __call__(self, w_ptr: cute.Pointer, sfw_ptr: cute.Pointer, x_ptr: cute.Pointer,
                 sfx_ptr: cute.Pointer, peers: cute.Tensor, scratch: cute.Tensor,
                 counters: cute.Tensor, order: cute.Tensor, rank: Int32, rows: Int32,
                 stream: cuda.CUstream):  # fmt: skip
        k = self.k
        fp8, sf_t = cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU
        a = cute.make_tensor(w_ptr, cute.make_ordered_layout((WIDTH, k, 1), order=(1, 0, 2)))
        b = cute.make_tensor(x_ptr, cute.make_ordered_layout((rows, k, 1), order=(1, 0, 2)))
        sfa = cute.make_tensor(sfw_ptr, blockscaled_utils.tile_atom_to_shape_SF(a.shape, SF_VEC))
        sfb = cute.make_tensor(sfx_ptr, blockscaled_utils.tile_atom_to_shape_SF(b.shape, SF_VEC))
        mma = self._mma()
        mma_sfb = sm107_utils.make_blockscaled_trivial_tiled_mma(
            fp8, fp8, OperandMajorMode.K, OperandMajorMode.K, sf_t, SF_VEC,
            tcgen05.CtaGroup.ONE, (128, TB, MMA_K),
            a_collector_op=CollectorOp.DISCARD, b_collector_op=CollectorOp.DISCARD,
        )  # fmt: skip
        pair_vmnk = cute.tiled_divide(cute.make_layout((2, 1, 1)), (mma.thr_id.shape,))
        pair_sfb_vmnk = cute.tiled_divide(cute.make_layout((2, 1, 1)), (mma_sfb.thr_id.shape,))
        kslice, bstages = self.kslice, 2 * self.kslice
        a_layout = sm100_utils.make_smem_layout_a(mma, self.tiler, fp8, kslice)
        b_layout = sm100_utils.make_smem_layout_b(mma, self.tiler, fp8, bstages)
        sfa_layout = blockscaled_utils.make_smem_layout_sfa(mma, self.tiler, SF_VEC, kslice)
        sfb_layout = blockscaled_utils.make_smem_layout_sfb(mma, self.tiler, SF_VEC, bstages)
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
        self.tx_a = 2 * kslice * (
            cute.size_in_bytes(fp8, a_one) + cute.size_in_bytes(sf_t, sfa_one)
        )
        self.tx_b = 2 * kslice * (
            cute.size_in_bytes(fp8, b_one) + cute.size_in_bytes(sf_t, sfb_one)
        )
        nbuf = self.nbuf

        @cute.struct
        class Storage:
            a_full: cute.struct.MemRange[cutlass.Int64, 1]
            a_empty: cute.struct.MemRange[cutlass.Int64, 1]
            b_full: cute.struct.MemRange[cutlass.Int64, 2]
            b_empty: cute.struct.MemRange[cutlass.Int64, 2]
            acc_full: cute.struct.MemRange[cutlass.Int64, nbuf]
            acc_empty: cute.struct.MemRange[cutlass.Int64, nbuf]
            tmem_dealloc: cutlass.Int64
            tmem_hold: cutlass.Int32
            # scale factors first: their smem descriptors must stay below 256 KiB
            sSFA: cute.struct.Align[cute.struct.MemRange[sf_t, cute.cosize(sfa_layout)], 128]
            sSFB: cute.struct.Align[cute.struct.MemRange[sf_t, cute.cosize(sfb_layout)], 128]
            sStage: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, CMAX * 128], 128]
            sA: cute.struct.Align[cute.struct.MemRange[fp8, cute.cosize(a_layout.outer)], 1024]
            sB: cute.struct.Align[cute.struct.MemRange[fp8, cute.cosize(b_layout.outer)], 1024]

        self.storage = Storage
        self.kernel(
            mma, mma_sfb, tma_a, ta, tma_b, tb, tma_sfa, tsfa, tma_sfb, tsfb, pair_vmnk,
            pair_sfb_vmnk, a_layout, b_layout, sfa_layout, sfb_layout, tsfa_layout,
            tsfb_layout, peers, scratch, counters, order, rank, rows,
        ).launch(
            grid=[self.grid, 1, 1], block=[self.threads, 1, 1], cluster=[2, 1, 1],
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
               scratch: cute.Tensor, counters: cute.Tensor, order: cute.Tensor,
               rank: Int32, rows: Int32):  # fmt: skip
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        sf_t = cutlass.Float8E8M0FNU
        kslice, nbuf = self.kslice, self.nbuf
        if warp == W_WARP:
            cpasync.prefetch_descriptor(tma_a)
            cpasync.prefetch_descriptor(tma_sfa)
        if warp == X_WARP:
            cpasync.prefetch_descriptor(tma_b)
            cpasync.prefetch_descriptor(tma_sfb)
        v = bidx % 2
        is_leader = v == 0
        crank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        coord = pair_vmnk.get_flat_coord(crank)
        coord_sfb = pair_sfb_vmnk.get_flat_coord(crank)
        pair = bidx // 2
        mb = pair % self.mblocks
        ks = pair // self.mblocks

        smem = utils.SmemAllocator()
        st = smem.allocate(self.storage)
        thread = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        a_pipe = pipeline.PipelineTmaUmma.create(
            barrier_storage=st.a_full.data_ptr(), num_stages=1, producer_group=thread,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.tx_a, cta_layout_vmnk=pair_vmnk, defer_sync=True,
        )  # fmt: skip
        b_pipe = pipeline.PipelineTmaUmma.create(
            barrier_storage=st.b_full.data_ptr(), num_stages=2, producer_group=thread,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.tx_b, cta_layout_vmnk=pair_vmnk, defer_sync=True,
        )  # fmt: skip
        acc_pipe = pipeline.PipelineUmmaAsync.create(
            barrier_storage=st.acc_full.data_ptr(), num_stages=nbuf, producer_group=thread,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 4 * EPI_WARPS),
            cta_layout_vmnk=pair_vmnk, defer_sync=True,
        )  # fmt: skip
        tmem = utils.TmemAllocator(
            st.tmem_hold.ptr,
            barrier_for_retrieve=pipeline.NamedBarrier(
                barrier_id=TMEM_BAR, num_threads=32 * (2 * EPI_WARPS + 1)
            ),
            allocator_warp_id=0, is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=st.tmem_dealloc.ptr, arch="sm_107",
        )  # fmt: skip
        pipeline_init_arrive(cluster_shape_mn=(2, 1), is_relaxed=True)

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
        pipeline_init_wait(cluster_shape_mn=(2, 1))
        cute.arch.griddepcontrol_launch_dependents()

        stamps = scratch.iterator.toint() + Int64(
            (scratch_floats() - STAMP_FLOATS + 2 * STAMPS * bidx) * 4
        )
        self._stamps = stamps
        if warp == W_WARP:
            if cutlass.const_expr(self.dbg & 32):
                if cute.arch.lane_idx() == 0:
                    st_i64(stamps, globaltimer())
            # The weight slice does not depend on the predecessor: issued at once.
            gA_s = tAgA[(None, mb, None, 0)]
            gSFA_s = tAgSFA[(None, mb, None, 0)]
            prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            a_pipe.producer_acquire(prod)
            bar = a_pipe.producer_get_barrier(prod)
            for j in cutlass.range_constexpr(kslice):
                kk = ks * kslice + j
                cute.copy(tma_a, gA_s[(None, kk)], tAsA[(None, j)], tma_bar_ptr=bar,
                          mcast_mask=a_mask)  # fmt: skip
                cute.copy(tma_sfa, gSFA_s[(None, kk)], tAsSFA[(None, j)], tma_bar_ptr=bar,
                          mcast_mask=a_mask)  # fmt: skip
            prod.advance()
            a_pipe.producer_tail(prod)

        if warp == X_WARP:
            xs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 2)
            cute.arch.griddepcontrol_wait()
            if cutlass.const_expr(self.dbg & 32):
                if cute.arch.lane_idx() == 0:
                    st_i64(stamps + 15 * 8, globaltimer())
            for blk in cutlass.range_constexpr(self.blocks):
                tb = order[blk]
                gB_s = tBgB[(None, tb, None, 0)]
                b_pipe.producer_acquire(xs)
                bar = b_pipe.producer_get_barrier(xs)
                for j in cutlass.range_constexpr(kslice):
                    kk = ks * kslice + j
                    stage = (blk % 2) * kslice + j
                    cute.copy(tma_b, gB_s[(None, kk)], tBsB[(None, stage)], tma_bar_ptr=bar,
                              mcast_mask=b_mask)  # fmt: skip
                    cute.copy(tma_sfb, tBgSFB[(None, tb, kk, 0)], tBsSFB[(None, stage)],
                              tma_bar_ptr=bar, mcast_mask=sfb_mask)  # fmt: skip
                xs.advance()
            b_pipe.producer_tail(xs)

        if warp == MMA_WARP:
            tmem.wait_for_alloc()
            acc_ptr = tmem.retrieve_ptr(Float32)
            sf_col = nbuf * TB
            tSFA = cute.make_tensor(cute.recast_ptr(acc_ptr + sf_col, dtype=sf_t), tsfa_layout)
            tSFB = cute.make_tensor(
                cute.recast_ptr(acc_ptr + sf_col + self.sfa_cols, dtype=sf_t), tsfb_layout
            )
            s2t_a = self._s2t(sSFA, tSFA)
            s2t_b = self._s2t(sSFB, tSFB)
            a_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            b_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 2)
            accs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, nbuf)
            nkb = cute.size(tCrA, mode=[2])
            if is_leader:
                a_pipe.consumer_wait(a_cons)
                if cutlass.const_expr(self.dbg & 32):
                    if cute.arch.lane_idx() == 0:
                        st_i64(stamps + 1 * 8, globaltimer())
            for blk in cutlass.range_constexpr(self.blocks):
                tAcc = cute.make_tensor(acc_ptr + (blk % nbuf) * TB, acc_fake.layout)
                if is_leader:
                    acc_pipe.producer_acquire(accs)
                    b_pipe.consumer_wait(b_cons)
                    if cutlass.const_expr(self.dbg & 32):
                        if cute.arch.lane_idx() == 0:
                            st_i64(stamps + (2 + blk) * 8, globaltimer())
                    for j in cutlass.range_constexpr(kslice):
                        stage = (blk % 2) * kslice + j
                        cute.copy(s2t_a[0], s2t_a[1][(None, None, None, None, j)], s2t_a[2])
                        cute.copy(s2t_b[0], s2t_b[1][(None, None, None, None, stage)], s2t_b[2])
                        for kb in cutlass.range_constexpr(nkb if not self.dbg & 8 else 0):
                            mma.set(tcgen05.Field.ACCUMULATE, j != 0 or kb != 0)
                            cute.gemm(
                                mma, tAcc[(None, 0, 0, 0)],
                                [tCrA[(None, 0, kb, j)], tSFA[(None, 0, kb)]],
                                [tCrB[(None, 0, kb, stage)], tSFB[(None, 0, kb)]],
                                tAcc[(None, 0, 0, 0)],
                            )  # fmt: skip
                    b_pipe.consumer_release(b_cons)
                    acc_pipe.producer_commit(accs)
                    if cutlass.const_expr(self.dbg & 32):
                        if cute.arch.lane_idx() == 0:
                            st_i64(stamps + (5 + blk) * 8, globaltimer())
                b_cons.advance()
                accs.advance()
            if is_leader:
                a_pipe.consumer_release(a_cons)
            a_cons.advance()
            acc_pipe.producer_tail(accs)

        if warp < EPI_WARPS:  # scatter group: the other slices' chunks -> scratch
            tmem.allocate(TMEM_COLS)
            tmem.wait_for_alloc()
            acc_ptr = tmem.retrieve_ptr(Float32)
            cute.arch.griddepcontrol_wait()
            waits = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, nbuf)
            taddr = Int32(acc_ptr.toint()) + ((32 * warp) << 16)
            fv = mb * 2 + v  # this CTA's 128 output features: [128 fv, 128 fv + 128)
            for blk in cutlass.range_constexpr(self.blocks):
                acc_pipe.consumer_wait(waits)
                if cutlass.const_expr(self.dbg & 32):
                    if tidx == 0:
                        st_i64(stamps + (8 + blk) * 8, globaltimer())
                self._scatter(taddr + (blk % nbuf) * TB, scratch, counters, fv, blk, ks, tidx)
                with cute.arch.elect_one():
                    acc_pipe.consumer_release(waits)
                waits.advance()
            if cutlass.const_expr(self.dbg & 32):
                if tidx == 0:
                    st_i64(stamps + 14 * 8, globaltimer())
            bar_sync(DONE_BAR, 64 * EPI_WARPS)  # the gather group read its last chunk
            tmem.relinquish_alloc_permit()
            tmem.free(acc_ptr)

        if warp >= GATHER_WARP:  # gather group: this slice's chunk summed and pushed
            tmem.wait_for_alloc()
            acc_ptr = tmem.retrieve_ptr(Float32)
            cute.arch.griddepcontrol_wait()  # (owner slots written after the producers)
            waits = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, nbuf)
            gid = tidx - 32 * GATHER_WARP  # 0..127
            feature = 32 * (warp % 4) + cute.arch.lane_idx()
            taddr = Int32(acc_ptr.toint()) + ((32 * (warp % 4)) << 16)
            fv = mb * 2 + v
            stage = Int32(st.sStage.data_ptr().toint())
            for blk in cutlass.range_constexpr(self.blocks):
                acc_pipe.consumer_wait(waits)
                self._gather_push(
                    taddr + (blk % nbuf) * TB, scratch, counters, peers, order, stage, fv,
                    blk, ks, rank, rows, gid, feature, acc_pipe, waits,
                )  # fmt: skip
                waits.advance()
                if cutlass.const_expr(self.dbg & 32):
                    if gid == 0:
                        st_i64(stamps + (11 + blk) * 8, globaltimer())
            bar_sync(DONE_BAR, 64 * EPI_WARPS)

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
    def _slot_base(self, scratch, fv, blk, chunk, slot, feature):
        """Address of feature ``feature``'s first token quad in the [fv][blk][chunk][slot]
        slab of the FP32 scratch: [token / 4][128 features][4 tokens], so the 32 lanes
        of a warp (32 features) move 512 contiguous bytes per v4 access."""
        index = (((fv * MAX_BLOCKS + blk) * KS + chunk) * 2 + slot) * 128 * CMAX
        return scratch.iterator.toint() + Int64(index + feature * 4) * 4

    @cute.jit
    def _stage_sums(self, own, s0, s1, where, size, stage, feature):
        """FP32 P0 + P1 + P2 of this thread's feature for the chunk's tokens (``where``:
        the K-slice positions of (own, slot 0, slot 1)), rounded once to BF16 (-0 made
        +0: no pushed word is the Lamport sentinel; the owner's sum is unchanged) into
        the token-major staging tile."""
        for g in cutlass.range_constexpr(CMAX // 16):
            t0 = 16 * g
            if t0 < size:
                for q in cutlass.range_constexpr(4):
                    for e in cutlass.range_constexpr(4):
                        parts = [None, None, None]
                        parts[where[0]] = own[g][4 * q + e]
                        parts[where[1]] = s0[4 * g + q][e]
                        parts[where[2]] = s1[4 * g + q][e]
                        total = (parts[0] + parts[1]) + parts[2]
                        st_shared_bf16_pz(stage + ((t0 + 4 * q + e) * 128 + feature) * 2,
                                          total)  # fmt: skip

    @cute.jit
    def _scatter(self, taddr, scratch, counters, fv, blk, ks, tidx):
        """This K slice's FP32 partials of the two token chunks other slices own:
        thread = TMEM lane = feature, 32 tokens per TMEM wait, written as token quads
        into the owner slice's slab, then the owners' counters bumped by every thread
        (release: its own stores)."""
        for c in cutlass.range_constexpr(KS):
            if ks != c:
                slot = (ks - c + 2) % 3
                base = self._slot_base(scratch, fv, blk, c, slot, tidx)
                for t0 in cutlass.range_constexpr(0, CHUNK[c], 32):
                    if cutlass.const_expr(t0 + 16 < CHUNK[c]):
                        a = tmem_ld16(taddr + START[c] + t0)
                        b = tmem_ld16(taddr + START[c] + t0 + 16)
                        vals = tmem_wait_n(*a, *b)
                    else:
                        vals = tmem_wait_n(*tmem_ld16(taddr + START[c] + t0))
                    for q in cutlass.range_constexpr(len(vals) // 4 if not self.dbg & 2 else 0):
                        quad = (t0 + 4 * q) // 4  # token quad: 128 x 16 B apart
                        st_f32x4(base + quad * 2048, *vals[4 * q : 4 * q + 4])
                if cutlass.const_expr(self.dbg & 32):
                    if tidx == 0:
                        st_i64(self._stamps + (16 + 2 * blk) * 8, globaltimer())
                where = (fv * MAX_BLOCKS + blk) * KS + c
                red_release_add(counters.iterator.toint() + Int64(where) * 4, 1)

    @cute.jit
    def _gather_push(self, taddr, scratch, counters, peers, order, stage, fv, blk, ks,
                     rank, rows, gid, feature, acc_pipe, waits):  # fmt: skip
        """This slice's own chunk read from TMEM (then the accumulator is released),
        both peers' partials once their counter is full, the sum in K-slice order (FP32),
        one BF16 rounding, sanitized pairs staged token-major, one bulk copy per row to
        the row owner."""
        start = Int32(START[0]) + ks * Int32(START[1])  # 0 / 48 / 96
        size = Int32(cutlass.select_(ks == 2, Int32(CHUNK[2]), Int32(CMAX)))
        own = []
        for t0 in cutlass.range_constexpr(0, CMAX, 16):
            vals = tmem_wait_n(*tmem_ld16(taddr + start + t0))  # (beyond size: unused)
            own.append(vals)
        with cute.arch.elect_one():
            acc_pipe.consumer_release(waits)
        where = counters.iterator.toint() + Int64((fv * MAX_BLOCKS + blk) * KS + ks) * 4
        if gid == 0:
            if cutlass.const_expr(not self.dbg & 1):
                while ld_acquire(where) < WRITERS:
                    pass
            st_relaxed_gpu(where, 0)  # (the next call's writers start after this one)
        bar_sync(GATHER_BAR, 32 * EPI_WARPS)
        if cutlass.const_expr(self.dbg & 32):
            if gid == 0:
                st_i64(self._stamps + (22 + blk) * 8, globaltimer())
        base0 = self._slot_base(scratch, fv, blk, ks, 0, feature)
        base1 = self._slot_base(scratch, fv, blk, ks, 1, feature)
        # every slab load of the chunk (2 slots x 12 token quads) in flight before any
        # use: one L2 round trip per block (the quads past a 32-row chunk are unused)
        s0 = [ld_f32x4_cg(base0 + quad * 2048) for quad in range(CMAX // 4)]
        s1 = [ld_f32x4_cg(base1 + quad * 2048) for quad in range(CMAX // 4)]
        # slice order P0 + P1 + P2 (slot (k - ks - 1) mod 3 holds P_k), one uniform
        # branch per slice: (own, slot 0, slot 1) -> (P0, P1, P2) positions
        if ks == 0:
            self._stage_sums(own, s0, s1, (0, 1, 2), size, stage, feature)
        elif ks == 1:
            self._stage_sums(own, s0, s1, (1, 2, 0), size, stage, feature)
        else:
            self._stage_sums(own, s0, s1, (2, 0, 1), size, stage, feature)
        fence_proxy_async_smem()
        bar_sync(GATHER_BAR, 32 * EPI_WARPS)
        if cutlass.const_expr(self.dbg & 32):
            if gid == 0:
                st_i64(self._stamps + (25 + blk) * 8, globaltimer())
        n = rows // self.tp
        token = order[blk] * TB + start + gid
        if (gid < size) & (token < rows):
            owner = token // n
            offset = (rank * self.cap + token - owner * n) * WIDTH + fv * 128
            if cutlass.const_expr(not self.dbg & 4):
                bulk_s2g(Int64(peers[owner]) + Int64(offset) * 2, stage + gid * 256, 256)
        bulk_commit_wait_read()
        bar_sync(GATHER_BAR, 32 * EPI_WARPS)  # the staging tile is free again
