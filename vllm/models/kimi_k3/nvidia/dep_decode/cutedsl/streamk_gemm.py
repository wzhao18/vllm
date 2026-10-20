# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Built on FlashInfer v0.7.2 (BSD-3-Clause):
# flashinfer/gemm/kernels/dense_blockscaled_gemm_sm100.py
"""Stream-K MXFP8 skinny GEMM for the Kimi K3 DEP decode: y[M, N] = x[M, K] @ W[N, K]^T.

Swap-AB: the weight is the MMA A operand (128-row tiles) and the M <= 192 decode
tokens are one MMA N tile, so the kernel streams W exactly once. CTA ``c`` of ``G``
owns the global K-block range ``[c * T // G, (c + 1) * T // G)`` of the
``T = row_tiles * k_blocks`` blocks, so every SM moves the same number of weight
bytes regardless of N. A row tile whose K range spans several CTAs is reduced in FP32
in global memory: every part adds its accumulator with fire-and-forget ``red`` and
bumps the tile's counter; the last arriver reads the sum, stores it, and re-zeroes
the accumulator and counter for the next launch (graph replays need no reset).
The FP32 sum order across parts is not fixed, so split tiles are not bitwise
reproducible run to run.

Weights and activations have separate producer warps and pipelines. The weight
stream (as deep as shared memory allows) never waits on the preceding kernel under
PDL; activations, read from L2, follow the PDL wait through a few stages.
The epilogue stages BF16 tiles in shared memory for TMA stores, and optionally
adds a BF16 residual of the output's shape, TMA-loaded whole when a tile is stored.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass import Float32, Int32
from cutlass.cute.arch import griddepcontrol_launch_dependents, griddepcontrol_wait
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import PipelineTmaUmma, PipelineUmmaAsync
from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm100 import (
    Sm100BlockScaledPersistentDenseGemmKernel,
    _l2_cache_policy,
)

ROW_TILE = 128
SUPPORTED_TILE_N = (64, 80, 96, 112, 128, 144, 160, 192)


@cute.jit
def _cta_of(block: Int32, total: Int32, grid: Int32) -> Int32:
    """The CTA whose K-block range contains global block ``block``."""
    return ((block + 1) * grid - 1) // total


@cute.jit
def _cta_start(cta: Int32, total: Int32, grid: Int32) -> Int32:
    return cta * total // grid


class DepMxfp8StreamKGemm(Sm100BlockScaledPersistentDenseGemmKernel):
    """See the module docstring. ``tile_n`` covers every token of a launch."""

    def __init__(
        self,
        tile_n: int,
        residual: bool,
        b_stages: int = 4,
        ablate: int = 0,
        k_tile: int = 128,
    ):
        if tile_n not in SUPPORTED_TILE_N:
            raise ValueError(f"tile_n must be one of {SUPPORTED_TILE_N}, got {tile_n}")
        super().__init__(
            sf_vec_size=32,
            mma_tiler_mn=(ROW_TILE, tile_n),
            cluster_shape_mn=(1, 1),
            use_prefetch=False,
            enable_pdl=True,
            a_l2_evict_first=True,
            b_l2_evict_last=True,
            # FP8 MMA K is 32: 4 instructions per 128-wide K block, 8 per 256.
            mma_inst_tile_k=k_tile // 32,
        )
        self.residual = residual
        self.b_stages = b_stages
        # Weights and activations have separate producer warps and pipelines: the
        # weight stream runs as deep as shared memory allows and never waits on the
        # previous kernel, while activations (L2 resident) need only a few stages.
        self.tma_b_warp_id = 6
        self.threads_per_cta = 32 * 7
        # Profiling only: 1 skips split-tile reduction, 2 also skips stores.
        self.ablate = ablate
        if residual:
            # The residual tile is prefetched whole into shared memory.
            self.smem_capacity -= ROW_TILE * tile_n * 2 + 1024

    def _setup_attributes(self):
        super()._setup_attributes()
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mnk[:2],
        )

        def staged(stages):
            return (
                sm100_utils.make_smem_layout_a(
                    tiled_mma, self.mma_tiler, self.a_dtype, stages
                ),
                blockscaled_utils.make_smem_layout_sfa(
                    tiled_mma, self.mma_tiler, self.sf_vec_size, stages
                ),
                sm100_utils.make_smem_layout_b(
                    tiled_mma, self.mma_tiler, self.b_dtype, stages
                ),
                blockscaled_utils.make_smem_layout_sfb(
                    tiled_mma, self.mma_tiler, self.sf_vec_size, stages
                ),
            )

        a1, sfa1, b1, sfb1 = staged(1)
        self.a_tx_bytes = cute.size_in_bytes(self.a_dtype, a1) + cute.size_in_bytes(
            self.sf_dtype, sfa1
        )
        self.b_tx_bytes = cute.size_in_bytes(self.b_dtype, b1) + cute.size_in_bytes(
            self.sf_dtype, sfb1
        )
        self.num_c_stage = 2
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage
        )
        c_bytes = cute.size_in_bytes(self.c_dtype, self.c_smem_layout_staged)
        budget = (
            self.smem_capacity
            - 4096
            - c_bytes
            - self.sfb_smem_pad_bytes
            - self.b_stages * self.b_tx_bytes
        )
        self.num_a_stage = budget // self.a_tx_bytes
        a_layouts = staged(self.num_a_stage)
        b_layouts = staged(self.b_stages)
        self.a_smem_layout_staged, self.sfa_smem_layout_staged = (
            a_layouts[0],
            a_layouts[1],
        )
        self.b_smem_layout_staged, self.sfb_smem_layout_staged = (
            b_layouts[2],
            b_layouts[3],
        )

    @cute.jit
    def wrapper(
        self,
        mA: cute.Tensor,  # weight [N, K] fp8
        mB: cute.Tensor,  # activation [M, K] fp8
        mC: cute.Tensor,  # output [M, N] bf16 (row-major)
        mR: cute.Tensor,  # residual [M, N] bf16, read only when self.residual
        workspace: cute.Tensor,  # fp32 [grid * ROW_TILE * tile_n], zero at rest
        counters: cute.Tensor,  # int32 [grid], zero between launches
        sf_m: cutlass.Int64,
        sf_n: cutlass.Int64,
        sf_k: cutlass.Int64,
        a_sf_ptr: cute.Pointer,
        b_sf_ptr: cute.Pointer,
        grid_ctas: Int32,
        stream: cuda.CUstream,
    ):
        n = cute.size(mA, mode=[0])
        k = cute.size(mA, mode=[1])
        m = cute.size(mB, mode=[0])
        a_tensor = cute.make_tensor(
            mA.iterator, cute.make_ordered_layout((n, k, 1), order=(1, 0, 2))
        )
        b_tensor = cute.make_tensor(
            mB.iterator, cute.make_ordered_layout((m, k, 1), order=(1, 0, 2))
        )
        # Kernel (M, N) = (weight rows, tokens); the [tokens, rows] output is N-major.
        c_tensor = cute.make_tensor(
            mC.iterator, cute.make_ordered_layout((n, m, 1), order=(0, 1, 2))
        )
        r_tensor = cute.make_tensor(
            mR.iterator, cute.make_ordered_layout((n, m, 1), order=(0, 1, 2))
        )
        sfa_tensor = cute.make_tensor(
            a_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, sf_m, 4, sf_k, 1), order=(2, 1, 4, 0, 3, 5)
            ),
        )
        sfb_tensor = cute.make_tensor(
            b_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, sf_n, 4, sf_k, 1), order=(2, 1, 4, 0, 3, 5)
            ),
        )
        (
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
        ) = self._prepare_tma(a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_tensor)
        assert self.num_acc_stage == 2 and not self.overlapping_accum
        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        self.num_epi_tiles = (ROW_TILE // cute.size(self.epi_tile[0])) * (
            self.cta_tile_shape_mnk[1] // cute.size(self.epi_tile[1])
        )
        # One residual buffer per epilogue tile, all loaded behind one barrier.
        self.r_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, self.num_epi_tiles
        )
        tma_atom_r, tma_tensor_r = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), r_tensor, epi_smem_layout, self.epi_tile
        )
        self.r_epi_tile_bytes = cute.size_in_bytes(self.c_dtype, epi_smem_layout)
        r_elems = cute.cosize(self.r_smem_layout_staged.outer) if self.residual else 128

        self.buffer_align_bytes = 1024

        @cute.struct
        class SharedStorage:
            a_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_a_stage]
            a_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_a_stage]
            b_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.b_stages]
            b_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.b_stages]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_holding_buf: cutlass.Int32
            last_flag: cutlass.Int32
            r_full_mbar_ptr: cutlass.Int64
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype, cute.cosize(self.c_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sR: cute.struct.Align[
                cute.struct.MemRange[self.c_dtype, r_elems],
                self.buffer_align_bytes,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype,
                    cute.cosize(self.sfb_smem_layout_staged) + self.sfb_smem_pad_bytes,
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            c_tensor,
            tma_atom_c,
            tma_tensor_c,
            tma_atom_r,
            tma_tensor_r,
            workspace,
            counters,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.c_smem_layout_staged,
            self.r_smem_layout_staged,
            self.epi_tile,
        ).launch(
            grid=[grid_ctas, 1, 1],
            block=[self.threads_per_cta, 1, 1],
            cluster=(1, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            min_blocks_per_mp=1,
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        mC_mnl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_tma: cute.Tensor,
        tma_atom_r: cute.CopyAtom,
        mR_tma: cute.Tensor,
        workspace: cute.Tensor,
        counters: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: cute.ComposedLayout,
        r_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        cta = Int32(bidx)
        grid = Int32(gdim)

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_sfa)
        if warp_idx == self.tma_b_warp_id:
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfb)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        tmem_holding_buf = storage.tmem_holding_buf.ptr
        last_flag = storage.last_flag.ptr

        a_pipeline = PipelineTmaUmma.create(
            barrier_storage=storage.a_full_mbar_ptr.data_ptr(),
            num_stages=self.num_a_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.a_tx_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        b_pipeline = PipelineTmaUmma.create(
            barrier_storage=storage.b_full_mbar_ptr.data_ptr(),
            num_stages=self.b_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.b_tx_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        acc_pipeline = PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len(self.epilog_warp_id)
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        cute.arch.mbarrier_init_fence()

        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )
        gC_mnl = cute.local_tile(
            mC_tma, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        gR_mnl = cute.local_tile(
            mR_tma, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_blocks = Int32(cute.size(gA_mkl, mode=[3]))
        row_tiles = Int32(cute.size(gA_mkl, mode=[2]))
        total = row_tiles * k_blocks
        seg_begin = _cta_start(cta, total, grid)
        seg_end = _cta_start(cta + 1, total, grid)

        thr_mma = tiled_mma.get_slice(0)
        thr_mma_sfb = tiled_mma_sfb.get_slice(0)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)
        tCgR = thr_mma.partition_C(gR_mnl)

        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            0,
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            0,
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa,
            0,
            a_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        tBsSFB, tBgSFB = cpasync.tma_partition(
            tma_atom_sfb,
            0,
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        cute.arch.barrier(
            barrier_id=self.cta_sync_bar_id, number_of_threads=self.threads_per_cta
        )

        #
        # Weight producer: independent of the previous kernel, so no PDL wait.
        #
        if warp_idx == self.tma_warp_id:
            a_policy = _l2_cache_policy("evict_first")
            producer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_a_stage
            )
            for block in cutlass.range(seg_begin, seg_end, unroll=1):
                tile = block // k_blocks
                kb = block - tile * k_blocks
                a_pipeline.producer_acquire(producer)
                bar = a_pipeline.producer_get_barrier(producer)
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, tile, kb, 0)],
                    tAsA[(None, producer.index)],
                    tma_bar_ptr=bar,
                    cache_policy=a_policy,
                )
                cute.copy(
                    tma_atom_sfa,
                    tAgSFA[(None, tile, kb, 0)],
                    tAsSFA[(None, producer.index)],
                    tma_bar_ptr=bar,
                    cache_policy=a_policy,
                )
                producer.advance()
            griddepcontrol_launch_dependents()
            a_pipeline.producer_tail(producer)

        #
        # Activation producer: activations come from the previous kernel.
        #
        if warp_idx == self.tma_b_warp_id:
            b_policy = _l2_cache_policy("evict_last")
            producer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.b_stages
            )
            griddepcontrol_wait()
            for block in cutlass.range(seg_begin, seg_end, unroll=1):
                kb = block % k_blocks
                b_pipeline.producer_acquire(producer)
                bar = b_pipeline.producer_get_barrier(producer)
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, 0, kb, 0)],
                    tBsB[(None, producer.index)],
                    tma_bar_ptr=bar,
                    cache_policy=b_policy,
                )
                cute.copy(
                    tma_atom_sfb,
                    tBgSFB[(None, 0, kb, 0)],
                    tBsSFB[(None, producer.index)],
                    tma_bar_ptr=bar,
                    cache_policy=b_policy,
                )
                producer.advance()
            b_pipeline.producer_tail(producer)

        #
        # MMA warp
        #
        if warp_idx == self.mma_warp_id:
            cute.arch.barrier(
                barrier_id=self.tmem_ptr_sync_bar_id,
                number_of_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)),
            )
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols, dtype=self.sf_dtype
            )
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
            copy_sfa, tCsSFA_s2t, tCtSFA_s2t = self.mainloop_s2t_copy_and_partition(
                sSFA, tCtSFA
            )
            copy_sfb, tCsSFB_s2t, tCtSFB_s2t = self.mainloop_s2t_copy_and_partition(
                sSFB, tCtSFB
            )

            a_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_a_stage
            )
            b_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.b_stages
            )
            acc_producer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            block = seg_begin
            while block < seg_end:
                tile = block // k_blocks
                seg_stop = cutlass.min(seg_end, (tile + 1) * k_blocks)
                tCtAcc = tCtAcc_base[(None, None, None, acc_producer.index)]
                acc_pipeline.producer_acquire(acc_producer)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for _ in cutlass.range(block, seg_stop, unroll=1):
                    a_pipeline.consumer_wait(a_consumer)
                    b_pipeline.consumer_wait(b_consumer)
                    cute.copy(
                        copy_sfa,
                        tCsSFA_s2t[(None, None, None, None, a_consumer.index)],
                        tCtSFA_s2t,
                    )
                    cute.copy(
                        copy_sfb,
                        tCsSFB_s2t[(None, None, None, None, b_consumer.index)],
                        tCtSFB_s2t,
                    )
                    for kphase in cutlass.range(
                        cute.size(tCrA, mode=[2]), unroll_full=True
                    ):
                        tiled_mma.set(
                            tcgen05.Field.SFA, tCtSFA[(None, None, kphase)].iterator
                        )
                        tiled_mma.set(
                            tcgen05.Field.SFB, tCtSFB[(None, None, kphase)].iterator
                        )
                        cute.gemm(
                            tiled_mma,
                            tCtAcc,
                            tCrA[(None, None, kphase, a_consumer.index)],
                            tCrB[(None, None, kphase, b_consumer.index)],
                            tCtAcc,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    a_pipeline.consumer_release(a_consumer)
                    b_pipeline.consumer_release(b_consumer)
                    a_consumer.advance()
                    b_consumer.advance()
                acc_pipeline.producer_commit(acc_producer)
                acc_producer.advance()
                block = seg_stop
            acc_pipeline.producer_tail(acc_producer)

        #
        # Epilogue warps
        #
        if warp_idx < self.mma_warp_id:
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.alloc_tmem(self.num_tmem_alloc_cols, tmem_holding_buf)
            cute.arch.barrier(
                barrier_id=self.tmem_ptr_sync_bar_id,
                number_of_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)),
            )
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = (
                self.epilog_tmem_copy_and_partition(
                    tidx, tCtAcc_base, tCgC, epi_tile, False
                )
            )
            sC = storage.sC.get_tensor(
                c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
            )
            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r, tTR_rC, tidx, sC
            )
            tma_atom_c, bSG_sC, bSG_gC_partitioned = (
                self.epilog_gmem_copy_and_partition(
                    tidx, tma_atom_c, tCgC, epi_tile, sC
                )
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, 32 * len(self.epilog_warp_id)
                ),
            )
            r_full = storage.r_full_mbar_ptr.ptr
            if cutlass.const_expr(self.residual):
                sR = storage.sR.get_tensor(
                    r_smem_layout_staged.outer, swizzle=r_smem_layout_staged.inner
                )
                # A plain copy tiled like the TMEM load, so residual elements line up
                # with the accumulator fragment (the r2s stmatrix copy reshuffles
                # lanes).
                tiled_copy_s2r = cute.make_tiled_copy_D(
                    cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.c_dtype),
                    tiled_copy_t2r,
                )
                tTR_sR = tiled_copy_s2r.get_slice(tidx).partition_D(sR)
                gR_epi = cute.flat_divide(
                    tCgR[((None, None), 0, 0, None, None, None)], epi_tile
                )
                bSG_sR, bSG_gR_partitioned = cpasync.tma_partition(
                    tma_atom_r,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sR, 0, 2),
                    cute.group_modes(gR_epi, 0, 2),
                )
                if warp_idx == self.epilog_warp_id[0]:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_init(r_full, 1)
                cute.arch.mbarrier_init_fence()
            r_phase = Int32(0)
            stores = Int32(0)
            frag = cute.size(tTR_rAcc)
            num_epi = cute.size(tTR_tAcc_base, mode=[3]) * cute.size(
                tTR_tAcc_base, mode=[4]
            )
            slot_elems = num_epi * frag * len(self.epilog_warp_id) * 32
            epi_threads = 32 * len(self.epilog_warp_id)

            acc_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            griddepcontrol_wait()
            block = seg_begin
            while block < seg_end:
                tile = block // k_blocks
                tile_first = tile * k_blocks
                seg_stop = cutlass.min(seg_end, tile_first + k_blocks)
                partial = (block != tile_first) | (seg_stop != tile_first + k_blocks)
                tTR_tAcc = tTR_tAcc_base[
                    (None, None, None, None, None, acc_consumer.index)
                ]
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                acc_pipeline.consumer_wait(acc_consumer)
                is_last = Int32(1)
                # A split tile's accumulator and counter belong to its first CTA, which
                # no other split tile shares.
                split_id = _cta_of(tile_first, total, grid)
                parts = _cta_of(tile_first + k_blocks - 1, total, grid) - split_id + 1
                acc_base = split_id * slot_elems
                if partial & (self.ablate == 0):
                    # Fire-and-forget FP32 reductions into the tile accumulator,
                    # thread-major per element so every warp instruction is coalesced.
                    for e in cutlass.range(num_epi, unroll=1):
                        cute.copy(
                            tiled_copy_t2r, tTR_tAcc[(None, None, None, e)], tTR_rAcc
                        )
                        for i in cutlass.range_constexpr(frag):
                            # Result unused: ptxas emits red.global.add.f32.
                            cute.arch.atomic_add(
                                workspace.iterator
                                + (acc_base + (e * frag + i) * epi_threads + tidx),
                                tTR_rAcc[i],
                                scope="gpu",
                            )
                    cute.arch.fence_acq_rel_gpu()
                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id,
                        number_of_threads=epi_threads,
                    )
                    if tidx == 0:
                        old = cute.arch.atomic_add(
                            counters.iterator + split_id, Int32(1), scope="gpu"
                        )
                        flag = Int32(0)
                        if old == parts - 1:
                            flag = Int32(1)
                        cute.arch.store(last_flag, flag)
                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id,
                        number_of_threads=epi_threads,
                    )
                    is_last = cute.arch.load(last_flag, Int32)
                    cute.arch.fence_acq_rel_gpu()
                if (is_last == 1) & (self.ablate < 2):
                    if cutlass.const_expr(self.residual):
                        cute.arch.barrier(
                            barrier_id=self.epilog_sync_bar_id,
                            number_of_threads=epi_threads,
                        )
                        if warp_idx == self.epilog_warp_id[0]:
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive_and_expect_tx(
                                    r_full, self.r_epi_tile_bytes * self.num_epi_tiles
                                )
                            bSG_gR = cute.group_modes(
                                bSG_gR_partitioned[(None, None, None, tile, 0, 0)], 1, 3
                            )
                            for e in cutlass.range_constexpr(self.num_epi_tiles):
                                cute.copy(
                                    tma_atom_r,
                                    bSG_gR[(None, e)],
                                    bSG_sR[(None, e)],
                                    tma_bar_ptr=r_full,
                                )
                        cute.arch.mbarrier_wait(r_full, r_phase)
                        r_phase = r_phase ^ 1
                    bSG_gC = cute.group_modes(
                        bSG_gC_partitioned[(None, None, None, tile, 0, 0)], 1, 3
                    )
                    for e in cutlass.range(num_epi, unroll=1):
                        if partial:
                            for i in cutlass.range_constexpr(frag):
                                # .cg: other CTAs reduced into these during this launch.
                                tTR_rAcc[i] = cute.arch.load(
                                    workspace.iterator
                                    + (acc_base + (e * frag + i) * epi_threads + tidx),
                                    Float32,
                                    cop="cg",
                                )
                            for i in cutlass.range_constexpr(frag):
                                workspace[
                                    acc_base + (e * frag + i) * epi_threads + tidx
                                ] = Float32(0.0)
                        else:
                            cute.copy(
                                tiled_copy_t2r,
                                tTR_tAcc[(None, None, None, e)],
                                tTR_rAcc,
                            )
                        if cutlass.const_expr(self.residual):
                            tTR_sR_e = tTR_sR[(None, None, None, e)]
                            for i in cutlass.range_constexpr(frag):
                                tTR_rAcc[i] = tTR_rAcc[i] + tTR_sR_e[i].to(Float32)
                        acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                        tRS_rC.store(acc_vec.to(self.c_dtype))
                        c_buffer = stores % self.num_c_stage
                        cute.copy(
                            tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)]
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        cute.arch.barrier(
                            barrier_id=self.epilog_sync_bar_id,
                            number_of_threads=epi_threads,
                        )
                        if warp_idx == self.epilog_warp_id[0]:
                            cute.copy(
                                tma_atom_c, bSG_sC[(None, c_buffer)], bSG_gC[(None, e)]
                            )
                            c_pipeline.producer_commit()
                            c_pipeline.producer_acquire()
                        cute.arch.barrier(
                            barrier_id=self.epilog_sync_bar_id,
                            number_of_threads=epi_threads,
                        )
                        stores = stores + 1
                    if partial:  # noqa: SIM102 (dynamic conditions cannot use `and`)
                        if tidx == 0:
                            counters[split_id] = Int32(0)
                cute.arch.fence_view_async_tmem_load()
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer)
                acc_consumer.advance()
                block = seg_stop

            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.relinquish_tmem_alloc_permit()
            c_pipeline.producer_tail()
            cute.arch.barrier(
                barrier_id=self.epilog_sync_bar_id, number_of_threads=epi_threads
            )
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.dealloc_tmem(acc_tmem_ptr, self.num_tmem_alloc_cols)
