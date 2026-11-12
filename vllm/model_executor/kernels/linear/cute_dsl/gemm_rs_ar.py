# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SM100 BF16/MXFP8 GEMM with fused tensor-parallel reduce-scatter/all-reduce.

Model-neutral: any row-parallel projection whose output is reduced across TP
(Kimi-K3 ``o_proj``/``down_proj``, DeepSeek-V4.1 ``wo_b``) can bind to the
process-wide workspace through ``maybe_init_gemm_rs_ar`` + ``GemmRsAr.apply``.
"""

# Based on CUTLASS's Blackwell distributed GEMM-RS example at dcf215a.
# See https://github.com/NVIDIA/cutlass/issues/3117 for memory semantics.
# The MXFP8 path follows the block-scaled SM100 GEMM template from
# https://github.com/gau-nernst/gn-kernels (sm100_mm_mxfp8.py).

from functools import cache
from typing import TYPE_CHECKING

import cutlass
import torch
import torch.distributed._symmetric_memory as symm_mem
from cuda.bindings.driver import CUstream
from cutlass import (
    BFloat16,
    Float8E4M3FN,
    Float8E8M0FNU,
    Int32,
    Int64,
    Uint16,
    Uint64,
    cute,
    utils,
)
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, nvvm, vector
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor, make_ptr, nullptr
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.utils import get_smem_capacity_in_bytes

from vllm.cute_utils import _tcgen05, mbarrier, simple_tma_copy, to_cta0_smem
from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


@dsl_user_op
def nanosleep(ns: int, *, loc=None, ip=None) -> None:
    nvvm.nanosleep(Int32(ns).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)


@dsl_user_op
def multimem_ld_reduce_16B(x: cute.Tensor, *, loc=None, ip=None) -> cute.Tensor:
    # The vector instruction assumes x is contiguous and 16-byte aligned.
    assert x.element_type == BFloat16
    vec_type = ".v4.bf16x2"

    ptr = x.iterator.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    asm = (
        "multimem.ld_reduce.relaxed.gpu.global.add.acc::f32"
        f"{vec_type} {{$0, $1, $2, $3}}, [$4];"
    )
    struct = llvm.inline_asm(
        llvm.StructType.get_literal([Int32.mlir_type] * 4),
        [ptr],
        asm,
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )
    vec = vector.from_elements(
        ir.VectorType.get([4], Int32.mlir_type, loc=loc),
        [
            llvm.extractvalue(Int32.mlir_type, struct, [i], loc=loc, ip=ip)
            for i in range(4)
        ],
        loc=loc,
        ip=ip,
    )
    ssa = cute.TensorSSA(vec, 4, Int32)

    y = cute.make_rmem_tensor(4, Int32)
    y.store(ssa)
    return cute.recast_tensor(y, x.element_type)


@dsl_user_op
def bulk_copy_g2s(
    dst_smem: Int32,
    src_gmem: Int64,
    size: Int32,
    mbar_smem: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    # Bulk async copy (TMA engine, no tensor map) of `size` bytes from global
    # (local or NVLink peer) memory into this CTA's shared memory, completing
    # `size` transaction bytes on the mbarrier.
    llvm.inline_asm(
        None,
        [
            Int32(dst_smem).ir_value(loc=loc, ip=ip),
            Int64(src_gmem).ir_value(loc=loc, ip=ip),
            Int32(size).ir_value(loc=loc, ip=ip),
            Int32(mbar_smem).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[$0], [$1], $2, [$3];",
        "r,l,r,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_acquire_sys(ptr: cute.Pointer, *, loc=None, ip=None) -> Int32:
    # Acquire load of a flag the producers (peer GPUs) raise with red.release:
    # system scope. Unlike fence.acq_rel.sys it does not wait for this thread's
    # earlier stores to become system-visible, it only orders the accesses
    # that follow it.
    addr = ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    return Int32(
        llvm.inline_asm(
            Int32.mlir_type,
            [addr],
            "ld.acquire.sys.global.b32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def fence_proxy_async_global(*, loc=None, ip=None) -> None:
    # Orders this thread's generic-proxy view of global memory (the acquired
    # producer partials) before its subsequent async-proxy accesses
    # (cp.async.bulk reads of those partials).
    llvm.inline_asm(
        Int32.mlir_type,
        [],
        "mov.u32 $0, 0;\n\tfence.proxy.async.global;",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def multimem_st_16B(dst: cute.Tensor, value: cute.Tensor, *, loc=None, ip=None) -> None:
    # The vector instruction assumes dst and value are contiguous BF16x8 vectors
    # and dst is 16-byte aligned.
    assert dst.element_type == BFloat16
    assert value.element_type == BFloat16
    ptr = dst.iterator.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    regs = cute.recast_tensor(value, Int32, loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [ptr] + [regs[i].ir_value(loc=loc, ip=ip) for i in range(4)],
        "multimem.st.relaxed.sys.global.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


# Tile schedule shared by the GEMM roles and the communication warps.
#
# GEMM tiles are visited in bands of ``_BAND_PAIRS`` M-tile pairs (2-CTA) or
# M-tiles (1-CTA) x every N tile, M fastest inside a band, so a wave of CTAs
# reuses one band of A and a slab of B from L2 instead of streaming all of A
# from HBM once per N column. The M tiles of a band are drawn evenly from the
# TP ranks' row shards (shard-interleaved), so every rank always has freshly
# produced rows to reduce and the NVLink load stays balanced across ranks.
# Grids of at most two waves keep the plain M-fastest order.
_BAND_PAIRS = 8
# Rows per communication unit: one unit is ``_COMM_ROWS`` rows x BN columns of
# this rank's rows of one GEMM tile; CTAs take units from a per-rank queue.
_COMM_ROWS = 64
# RS staging ring in shared memory: _COMM_STAGES chunks of _COMM_CHUNK_BYTES,
# each holding every rank's partial of a few unit rows (64 KB in flight per
# CTA; TP4 n17k comm-only time saturates at 48-64 KB).
_COMM_STAGES = 4
_COMM_CHUNK_BYTES = 16 * 1024


class Sm100GemmRsAr:
    def __init__(
        self,
        rank: int,
        num_ranks: int,
        cta_group: int = 1,
        all_reduce: bool = False,
        dtype: str = "bf16",
        residual: bool = False,
        BN: int = 128,
    ) -> None:
        assert dtype in ("bf16", "mxfp8")
        # MXFP8 keeps 16 TMEM columns for scale factors: BN=256 would leave a
        # single accumulator stage, so it is BF16-only.
        assert BN == 128 or (BN == 256 and dtype == "bf16")
        # residual: add a BF16 [R, N] tensor to GEMM rows [row0, row0 + R) in the
        # epilogue (FP32, before the BF16 partial is written and reduced).
        self.residual = residual
        self.rank = rank
        self.num_ranks = num_ranks
        BM = 128
        BK = {"bf16": 64, "mxfp8": 128}[dtype]  # 128B
        self.cta_tile = (BM, BN, BK)
        self.cta_group = cta_group
        self.all_reduce = all_reduce
        self.dtype = dtype

        smem_bytes = get_smem_capacity_in_bytes()
        if dtype == "bf16":
            self.stage_size = (BM + (BN // cta_group)) * BK * 2
        elif dtype == "mxfp8":
            self.stage_size = (BM + (BN // cta_group)) * BK + (BM + BN) * (BK // 32)
        # Leave 1 KB for the barriers and the scheduler slot, and room for the
        # communication staging ring.
        # (AR reduces through NVLS and only keeps a minimal ring allocation.)
        comm_bytes = (1 if all_reduce else _COMM_STAGES) * _COMM_CHUNK_BYTES
        self.num_stages = (smem_bytes - 1024 - comm_bytes) // self.stage_size
        if all_reduce:
            # The AR reduction is NVLS loads issued from the LSU, whose
            # in-flight capacity lives in L1: a smaller GEMM ring leaves a
            # larger L1 carve-out (AR GEMMs are short-K and comm-bound).
            # TP4 M=16384: 441 -> 375 us.
            self.num_stages = min(self.num_stages, 4)

    @staticmethod
    def shard_pair(r, local_M, pair_rows):
        # First M-tile pair (2-CTA) or M tile (1-CTA) holding rank r's rows.
        return (r * local_M) // pair_rows

    @cute.jit
    def tile_coords(self, bid, grid_m, grid_n, local_M, num_slots_m, banded):
        """Map a schedule index to (bid_m, bid_n, valid); see _BAND_PAIRS.

        Group r = the M-tile pairs from the one holding rank r's first row up to
        the next rank's (the last group runs to the end). Slot s of a band takes
        pair s // R of group s % R; slots past the end of a shorter group are
        holes. The communication warps enumerate their rows in the same order.
        """
        BM = self.cta_tile[0]
        cg = self.cta_group
        R = self.num_ranks
        pair_rows = BM * cg
        num_pairs = grid_m // cg
        band = R * max(1, _BAND_PAIRS // R)

        pair_slot = bid // cg
        sub = bid % cg
        band_idx = pair_slot // (band * grid_n)
        band_w = min(band, num_slots_m - band_idx * band)
        offset = pair_slot - band_idx * band * grid_n
        slot_m = band_idx * band + offset % band_w
        bid_n = offset // band_w
        r = slot_m % R
        pair = self.shard_pair(r, local_M, pair_rows) + slot_m // R
        group_end = self.shard_pair(r + 1, local_M, pair_rows)
        if r == R - 1:
            group_end = num_pairs
        valid = pair < group_end
        bid_m = pair * cg + sub
        if not banded:
            # One or two waves: no L2 reuse to win, and the band holes would
            # fall on a CTA stride that is a multiple of R. Plain M-fastest.
            bid_m = bid % grid_m
            bid_n = bid // grid_m
            valid = bid < grid_m * grid_n
        return bid_m, bid_n, valid

    @cute.jit
    def num_slots_m(self, grid_m, local_M):
        # Pair slots per band row: the largest group, times the TP size.
        BM = self.cta_tile[0]
        cg = self.cta_group
        R = self.num_ranks
        pair_rows = BM * cg
        num_pairs = grid_m // cg
        q = num_pairs - self.shard_pair(R - 1, local_M, pair_rows)
        for r in cutlass.range_constexpr(R - 1):
            q = max(
                q,
                self.shard_pair(r + 1, local_M, pair_rows)
                - self.shard_pair(r, local_M, pair_rows),
            )
        return q * R

    @cute.jit
    def nvls_unit(
        self,
        tid_: Int32,
        mc_vecs: cute.Tensor,
        unit_row0: Int32,
        row_lo: Int32,
        row_hi: Int32,
        bid_n: Int32,
    ) -> None:
        """All-reduce one communication unit through NVLS: multimem.ld_reduce
        (FP32 accumulation in the switch) and a multimem.st broadcast. Every
        load is unconditional (rows clamped into the unit), so a batch of
        loads is in flight together; only the stores are predicated."""
        BN = self.cta_tile[1]
        vec_width = 8
        vec_cols = BN // vec_width
        vecs_per_thread = _COMM_ROWS * vec_cols // 128
        # 4 vectors per thread per batch (TP4: 4-16 per batch measured equal);
        # a batch whose rows all lie outside the unit's rows is skipped.
        batch_vecs = min(vecs_per_thread, 4)
        batch_rows = batch_vecs * 128 // vec_cols
        for batch in cutlass.range_constexpr(vecs_per_thread // batch_vecs):
            batch_row0 = unit_row0 + batch * batch_rows
            if batch_row0 < row_hi and batch_row0 + batch_rows > row_lo:
                vals = []
                for v in cutlass.range_constexpr(batch_vecs):
                    vec_idx = tid_ + (batch * batch_vecs + v) * 128
                    row = unit_row0 + vec_idx // vec_cols
                    row = min(max(row, row_lo), row_hi - 1)
                    col = bid_n * vec_cols + vec_idx % vec_cols
                    vals.append(multimem_ld_reduce_16B(mc_vecs[None, (row, col)]))
                for v in cutlass.range_constexpr(batch_vecs):
                    vec_idx = tid_ + (batch * batch_vecs + v) * 128
                    row = unit_row0 + vec_idx // vec_cols
                    col = bid_n * vec_cols + vec_idx % vec_cols
                    if row >= row_lo and row < row_hi:
                        multimem_st_16B(mc_vecs[None, (row, col)], vals[v])

    @cute.jit
    def next_unit(
        self,
        tid_: Int32,
        unit_ctr: cute.Pointer,
        comm_slot: cute.Tensor,
        bar_id: cutlass.Constexpr,
    ) -> Int32:
        # One thread takes the next communication unit of this rank from the
        # queue and hands it to the other communication threads.
        cute.arch.barrier(barrier_id=bar_id, number_of_threads=128)
        if tid_ == 0:
            comm_slot[0] = cute.arch.atomic_add(
                unit_ctr, Int32(1), sem="relaxed", scope="gpu"
            )
        cute.arch.barrier(barrier_id=bar_id, number_of_threads=128)
        return comm_slot[0]

    @cute.jit
    def prepare_tma(
        self, tensor: cute.Tensor, BM: cutlass.Constexpr, BK: cutlass.Constexpr
    ) -> cpasync.TmaInfo:
        tma_group = (
            tcgen05.CtaGroup.TWO if self.cta_group == 2 else tcgen05.CtaGroup.ONE
        )
        tma_op = cpasync.CopyBulkTensorTileG2SOp(cta_group=tma_group)
        swizzle_128b = cute.make_swizzle(3, 4, 3)
        layout = cute.make_layout(
            (BM, BK, self.num_stages),
            stride=(BK, 1, BM * BK),
        )
        layout = cute.make_composed_layout(swizzle_128b, 0, layout)
        return cpasync.make_tiled_tma_atom(tma_op, tensor, layout, (BM, BK))

    @cute.jit
    def prepare_sf_tma(
        self,
        SF: cute.Tensor,
        R: Int32,
        Ksf: Int32,
        BR: cutlass.Constexpr,
        BKsf: cutlass.Constexpr,
    ) -> cpasync.TmaInfo:
        tma_group = (
            tcgen05.CtaGroup.TWO if self.cta_group == 2 else tcgen05.CtaGroup.ONE
        )
        tma_op = cpasync.CopyBulkTensorTileG2SOp(cta_group=tma_group)
        # NVIDIA SF gmem layout: an [R, Ksf] scale grid padded to R % 128 == 0
        # and permuted as [R/128, Ksf/4, 32, 4, 4]. The trailing [32, 4, 4]
        # atom is the opaque 512B block tcgen05 expects, so treat it as bytes.
        # The Int64 recast keeps the TMA boxDim <= 256.
        g_layout = cute.make_layout((Ksf * 128, cute.ceil_div(R, 128)))
        SF_i64 = cute.recast_tensor(cute.make_tensor(SF.iterator, g_layout), Int64)
        tiler = (128 * BKsf // 8, BR // 128)
        layout = cute.make_layout((*tiler, self.num_stages))
        return cpasync.make_tiled_tma_atom(tma_op, SF_i64, layout, tiler)

    @cute.jit
    def __call__(
        self,
        A: cute.Tensor,
        B: cute.Tensor,
        SFA: cute.Tensor,
        SFB: cute.Tensor,
        partial_uc: cute.Tensor,
        partial_mc_ptr: cute.Pointer,
        output: cute.Tensor,
        flags_uc: cute.Tensor,
        flags_mc_ptr: cute.Pointer,
        peer_flag_ptr: cute.Pointer,
        peer_partial_ptr: cute.Pointer,
        residual: cute.Tensor,
        residual_row0: Int32,
        sched: cute.Tensor,
        grid_size: Int32,
        stream: CUstream,
    ) -> None:
        M, K = A.shape
        N = B.shape[0]
        BM, BN, BK = self.cta_tile
        A_tma = self.prepare_tma(A, BM, BK)
        B_tma = self.prepare_tma(B, BN // self.cta_group, BK)
        if cutlass.const_expr(self.dtype == "mxfp8"):
            SFA_tma = self.prepare_sf_tma(SFA, M, K // 32, BM, BK // 32)
            SFB_tma = self.prepare_sf_tma(SFB, N, K // 32, BN, BK // 32)
        else:
            SFA_tma = None
            SFB_tma = None
        padded_M = partial_uc.shape[0]
        partial_mc = cute.make_tensor(
            partial_mc_ptr,
            cute.make_layout((padded_M, N), stride=(N, 1)),
        )
        peer_flags = cute.make_tensor(
            peer_flag_ptr,
            cute.make_layout(self.num_ranks),
        )
        peer_partials = cute.make_tensor(
            peer_partial_ptr,
            cute.make_layout(self.num_ranks),
        )

        grid = (grid_size, 1, 1)
        block = (10 * 32, 1, 1)
        cluster = (self.cta_group, 1, 1)
        self.kernel(
            A_tma,
            B_tma,
            SFA_tma,
            SFB_tma,
            partial_uc,
            partial_mc,
            output,
            flags_uc.iterator,
            flags_mc_ptr,
            peer_flags,
            peer_partials,
            residual,
            residual_row0,
            sched.iterator,
        ).launch(grid=grid, block=block, cluster=cluster, stream=stream)

    @cute.kernel
    def kernel(
        self,
        A_tma: cpasync.TmaInfo,
        B_tma: cpasync.TmaInfo,
        SFA_tma: cpasync.TmaInfo,
        SFB_tma: cpasync.TmaInfo,
        partial_uc: cute.Tensor,
        partial_mc: cute.Tensor,
        output: cute.Tensor,
        flags_uc_ptr: cute.Pointer,
        flags_mc_ptr: cute.Pointer,
        peer_flags: cute.Tensor,
        peer_partials: cute.Tensor,
        residual: cute.Tensor,
        residual_row0: Int32,
        sched_ptr: cute.Pointer,
    ) -> None:
        tid, _, _ = cute.arch.thread_idx()
        raw_bid, _, _ = cute.arch.block_idx()
        num_bids, _, _ = cute.arch.grid_dim()
        warp_id = cute.arch.make_warp_uniform(tid // 32)

        BM, BN, BK = self.cta_tile
        cta_group = self.cta_group
        num_stages = self.num_stages
        num_ranks = self.num_ranks

        is_2cta = cta_group == 2
        cta_rank = raw_bid % self.cta_group

        # MXFP8 reserves the last 16 tmem columns (496-512) for the A/B scale
        # factors.
        num_tmem_stages = (496 if self.dtype == "mxfp8" else 512) // BN

        smem = utils.SmemAllocator()
        sA = smem.allocate_tensor(
            A_tma.atom.value_type,
            A_tma.smem_layout.outer,
            byte_alignment=128,
            swizzle=A_tma.smem_layout.inner,
        )
        sB = smem.allocate_tensor(
            B_tma.atom.value_type,
            B_tma.smem_layout.outer,
            byte_alignment=128,
            swizzle=B_tma.smem_layout.inner,
        )
        if cutlass.const_expr(self.dtype == "mxfp8"):
            sSFA = smem.allocate_tensor(Int64, SFA_tma.smem_layout, byte_alignment=128)
            sSFB = smem.allocate_tensor(Int64, SFB_tma.smem_layout, byte_alignment=128)

        tma_full_mbar = smem.allocate_array(Int64, num_stages)
        tma_empty_mbar = smem.allocate_array(Int64, num_stages)
        tmem_full_mbar = smem.allocate_array(Int64, num_tmem_stages)
        tmem_empty_mbar = smem.allocate_array(Int64, num_tmem_stages)

        taddr = smem.allocate(Int32, 4)
        # Communication unit handed from the scheduling thread to its warps.
        comm_slot = smem.allocate_tensor(Int32, cute.make_layout(4), byte_alignment=16)
        comm_full_mbar = smem.allocate_array(Int64, _COMM_STAGES)
        comm_empty_mbar = smem.allocate_array(Int64, _COMM_STAGES)
        sComm = smem.allocate_tensor(
            BFloat16,
            cute.make_layout(
                (1 if self.all_reduce else _COMM_STAGES) * _COMM_CHUNK_BYTES // 2
            ),
            byte_alignment=128,
        )

        # Named barriers
        BAR_TMEM_ALLOC = 1
        BAR_EPILOGUE = 2
        BAR_COMM = 3

        M, K = A_tma.tma_tensor.shape
        N, _ = B_tma.tma_tensor.shape
        local_M = (
            partial_uc.shape[0] // num_ranks if self.all_reduce else output.shape[0]
        )
        grid_m = cute.ceil_div(M, BM)
        # Keep 2-CTA clusters within a single N tile.
        grid_m = cute.ceil_div(grid_m, cta_group) * cta_group
        grid_n = cute.ceil_div(N, BN)

        if warp_id == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(num_stages):
                    cute.arch.mbarrier_init(tma_full_mbar + i, cta_group)
                    cute.arch.mbarrier_init(tma_empty_mbar + i, 1)
                for i in cutlass.range_constexpr(num_tmem_stages):
                    cute.arch.mbarrier_init(tmem_full_mbar + i, 1)
                    cute.arch.mbarrier_init(tmem_empty_mbar + i, 128 * cta_group)
                for i in cutlass.range_constexpr(_COMM_STAGES):
                    cute.arch.mbarrier_init(comm_full_mbar + i, 1)
                    cute.arch.mbarrier_init(comm_empty_mbar + i, 128)
                cute.arch.mbarrier_init_fence()
        elif warp_id == 1:
            cpasync.prefetch_descriptor(A_tma.atom)
            cpasync.prefetch_descriptor(B_tma.atom)
            if cutlass.const_expr(self.dtype == "mxfp8"):
                cpasync.prefetch_descriptor(SFA_tma.atom)
                cpasync.prefetch_descriptor(SFB_tma.atom)

        if cutlass.const_expr(is_2cta):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            cute.arch.sync_threads()

        total_tiles = grid_m * grid_n
        num_slots_m = self.num_slots_m(grid_m, local_M)
        # Schedule length in CTA tiles; slots past a short rank group are holes.
        banded = total_tiles > 2 * num_bids
        total_slots = total_tiles
        if banded:
            total_slots = num_slots_m * cta_group * grid_n

        if warp_id == 9:
            # TMA warp
            tma_stage = 0
            parity = 1

            if cutlass.const_expr(is_2cta):
                tma_full_mbar_ = to_cta0_smem(tma_full_mbar)
            else:
                tma_full_mbar_ = tma_full_mbar

            # Select global-memory tiles.
            # [(BM, BK), (M/BM, K/BK)]
            gA_tiles = cute.zipped_divide(A_tma.tma_tensor, (BM, BK))
            gB_tiles = cute.zipped_divide(B_tma.tma_tensor, (BN // cta_group, BK))
            if cutlass.const_expr(self.dtype == "mxfp8"):
                BKsf = BK // 32
                gSFA_tiles = cute.zipped_divide(
                    SFA_tma.tma_tensor, (128 * BKsf // 8, BM // 128)
                )
                gSFB_tiles = cute.zipped_divide(
                    SFB_tma.tma_tensor, (128 * BKsf // 8, BN // 128)
                )

            for bid in range(raw_bid, total_slots, num_bids):
                bid_m, bid_n, valid = self.tile_coords(
                    bid, grid_m, grid_n, local_M, num_slots_m, banded
                )
                if valid:
                    if cutlass.const_expr(cta_group == 2):
                        bid_n_b = bid_n * cta_group + cta_rank
                    else:
                        bid_n_b = bid_n

                    num_k = cute.ceil_div(K, BK)
                    for iter_k in cutlass.range(num_k, unroll=1):
                        mbar = tma_full_mbar_ + tma_stage
                        cute.arch.mbarrier_wait(tma_empty_mbar + tma_stage, parity)

                        with cute.arch.elect_one():
                            mbarrier.arrive_expect_tx(mbar, self.stage_size, "cluster")
                        simple_tma_copy(
                            A_tma.atom,
                            gA_tiles[None, (bid_m, iter_k)],
                            sA[None, None, tma_stage],
                            mbar,
                        )
                        simple_tma_copy(
                            B_tma.atom,
                            gB_tiles[None, (bid_n_b, iter_k)],
                            sB[None, None, tma_stage],
                            mbar,
                        )
                        if cutlass.const_expr(self.dtype == "mxfp8"):
                            simple_tma_copy(
                                SFA_tma.atom,
                                gSFA_tiles[None, (iter_k, bid_m)],
                                sSFA[None, None, tma_stage],
                                mbar,
                            )
                            simple_tma_copy(
                                SFB_tma.atom,
                                gSFB_tiles[None, (iter_k, bid_n)],
                                sSFB[None, None, tma_stage],
                                mbar,
                            )

                        tma_stage = (tma_stage + 1) % num_stages
                        if tma_stage == 0:
                            parity ^= 1

        elif warp_id == 8:
            # MMA warp
            cute.arch.barrier(barrier_id=BAR_TMEM_ALLOC, number_of_threads=5 * 32)

            if cta_rank == 0:
                tma_stage = 0
                tma_full_parity = 0
                tmem_stage = 0
                tmem_empty_parity = 1

                MMA_M = BM * cta_group
                MMA_N = BN
                sdesc = _tcgen05.make_sdesc_128B_swizzle(0)
                if cutlass.const_expr(self.dtype == "mxfp8"):
                    # SF smem descriptor: no swizzle, SBO = 8 * 16 bytes.
                    sdesc_sf = Uint64((8 << 32) | (1 << 46))
                    sfa_tmem = 496
                    sfb_tmem = sfa_tmem + 4
                    idesc = _tcgen05.make_mxfp8_idesc(MMA_M, MMA_N)
                else:
                    idesc = _tcgen05.make_bf16_idesc(MMA_M, MMA_N)
                multicast_mask = Uint16((1 << self.cta_group) - 1)

                for bid in range(raw_bid, total_slots, num_bids):
                    _, _, valid = self.tile_coords(
                        bid, grid_m, grid_n, local_M, num_slots_m, banded
                    )
                    if valid:
                        cute.arch.mbarrier_wait(
                            tmem_empty_mbar + tmem_stage, tmem_empty_parity
                        )
                        _tcgen05.fence_after_thread_sync()

                        for iter_k in cutlass.range(cute.ceil_div(K, BK), unroll=1):
                            d_tmem = BN * tmem_stage
                            a_addr = sA[None, None, tma_stage].iterator.toint()
                            b_addr = sB[None, None, tma_stage].iterator.toint()
                            a_desc = sdesc | (a_addr >> 4)
                            b_desc = sdesc | (b_addr >> 4)
                            if cutlass.const_expr(self.dtype == "mxfp8"):
                                sfa_desc = sdesc_sf | (
                                    sSFA[None, None, tma_stage].iterator.toint() >> 4
                                )
                                sfb_desc = sdesc_sf | (
                                    sSFB[None, None, tma_stage].iterator.toint() >> 4
                                )

                            cute.arch.mbarrier_wait(
                                tma_full_mbar + tma_stage, tma_full_parity
                            )
                            _tcgen05.fence_after_thread_sync()

                            # load SF from smem->tmem
                            if cutlass.const_expr(self.dtype == "mxfp8"):
                                _tcgen05.cp(
                                    sfa_tmem, sfa_desc, "32x128b", "warpx4", cta_group
                                )
                                for j in cutlass.range_constexpr(BN // 128):
                                    _tcgen05.cp(
                                        sfb_tmem + j * 4,
                                        sfb_desc,
                                        "32x128b",
                                        "warpx4",
                                        cta_group,
                                    )
                                    sfb_desc += (128 * 4) >> 4

                            # 4 = 128B / 32B
                            for mma_k in cutlass.range_constexpr(4):
                                enable_d = iter_k > 0 or mma_k > 0
                                if cutlass.const_expr(self.dtype == "mxfp8"):
                                    # select SF IDs
                                    idesc_k = idesc + ((mma_k << 4) | (mma_k << 29))
                                    _tcgen05.mma_mxfp8(
                                        d_tmem,
                                        a_desc,
                                        b_desc,
                                        idesc_k,
                                        sfa_tmem,
                                        sfb_tmem,
                                        enable_d,
                                        cta_group,
                                    )
                                else:
                                    _tcgen05.mma_f16(
                                        d_tmem,
                                        a_desc,
                                        b_desc,
                                        idesc,
                                        enable_d,
                                        cta_group,
                                    )
                                a_desc += 32 >> 4
                                b_desc += 32 >> 4
                            _tcgen05.commit(
                                tma_empty_mbar + tma_stage, multicast_mask, cta_group
                            )

                            tma_stage = (tma_stage + 1) % num_stages
                            if tma_stage == 0:
                                tma_full_parity ^= 1

                        _tcgen05.commit(
                            tmem_full_mbar + tmem_stage, multicast_mask, cta_group
                        )

                        tmem_stage = (tmem_stage + 1) % num_tmem_stages
                        if tmem_stage == 0:
                            tmem_empty_parity ^= 1

        elif warp_id >= 4:
            # Communication warps
            # Keep epilogue in warps 0-3 and communication in warps 4-7.
            # Swapping the warpgroups can hang due to warp scheduling.
            tid_ = tid % 128

            # Offset in the [M, N] GEMM result.
            rank_start = self.rank * local_M
            rank_end = min(rank_start + local_M, M)

            st_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128
            )

            # A communication unit is _COMM_ROWS rows x BN columns of this rank's
            # rows of one GEMM tile. Its rows of every rank's BF16 partial are
            # fetched with bulk async copies (one row segment each, local or over
            # NVLink) into a shared-memory ring of _COMM_STAGES chunks, then
            # summed in FP32 in a fixed rank order and rounded once to BF16 (as
            # multimem.ld_reduce.acc::f32 does in the switch). Measured on VR200
            # NVLink at TP4: the NVLS reduction moves ~0.45 TB/s per GPU, unicast
            # pulls ~0.9 TB/s; issuing them as bulk copies keeps them off the
            # L1 path, whose in-flight capacity this kernel's shared-memory
            # carve-out leaves too small for NVLink latency.
            vec_width = 8
            vec_cols = BN // vec_width
            unit_rows = _COMM_ROWS
            units_per_tile = BM // unit_rows
            seg_bytes = BN * 2
            chunk_rows = max(1, _COMM_CHUNK_BYTES // (num_ranks * seg_bytes))
            chunks_per_unit = unit_rows // chunk_rows
            chunk_bytes = num_ranks * chunk_rows * seg_bytes
            vecs_per_chunk = chunk_rows * vec_cols
            if cutlass.const_expr(self.all_reduce):
                mc_vecs = cute.zipped_divide(partial_mc, (1, vec_width))
            else:
                output_vecs = cute.zipped_divide(output, (1, vec_width))
            ld_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128
            )
            partial_bases = cute.make_rmem_tensor(num_ranks, Int64)
            for src in cutlass.range_constexpr(num_ranks):
                partial_bases[src] = cute.arch.load(
                    (peer_partials.iterator + src).llvm_ptr, Int64
                )
            row_bytes = Int64(N) * 2
            comm_base = sComm.iterator.toint()
            issued = Int32(0)
            consumed = Int32(0)

            # This rank's units, in the order the band schedule produces their
            # GEMM tiles: band by band (pairs of this rank's row group), N
            # slowest inside a band. CTAs pull units from a per-rank counter,
            # so the communication load is balanced whatever the GEMM order.
            pair_rows = BM * cta_group
            first_pair = self.shard_pair(self.rank, local_M, pair_rows)
            num_my_pairs = Int32(0)
            if rank_end > rank_start:
                num_my_pairs = (rank_end - 1) // pair_rows - first_pair + 1
            band_pairs = max(1, _BAND_PAIRS // num_ranks)
            band_units = band_pairs * cta_group * grid_n * units_per_tile
            total_units = num_my_pairs * cta_group * grid_n * units_per_tile

            unit_ctr = sched_ptr
            done_ctr = sched_ptr + 1

            unit = self.next_unit(tid_, unit_ctr, comm_slot, BAR_COMM)
            while unit < total_units:
                band_idx = unit // band_units
                band_w = min(band_pairs, num_my_pairs - band_idx * band_pairs)
                offset = unit - band_idx * band_units
                unit_in_tile = offset % units_per_tile
                tile_in_band = offset // units_per_tile
                m_in_band = tile_in_band % (band_w * cta_group)
                bid_n = tile_in_band // (band_w * cta_group)
                bid_m = (first_pair + band_idx * band_pairs) * cta_group + m_in_band

                tile_row0 = bid_m * BM
                unit_row0 = tile_row0 + unit_in_tile * unit_rows
                row_lo = max(unit_row0, rank_start)
                row_hi = min(unit_row0 + unit_rows, rank_end)

                if row_lo < row_hi:
                    flag_ptr = flags_uc_ptr + bid_n * grid_m + bid_m
                    # Poll local L2 with relaxed GPU-scope loads. RS: once every
                    # rank has arrived, one acquire load of the flag (it only
                    # grows until this unit's own consumer count below) pairs
                    # with the producers' red.release, and the CTA barrier
                    # carries that order to the issuing warp. The partials are
                    # then read through the async proxy (cp.async.bulk), which
                    # a generic-proxy acquire does not order: the issuing warp
                    # adds fence.proxy.async.global below. AR reads every
                    # rank's L2 through multimem.ld_reduce (generic proxy, no
                    # L1), as before, and needs neither.
                    if tid_ == 0:
                        arrivals = cute.arch.load(
                            flag_ptr, Int32, sem="relaxed", scope="gpu"
                        )
                        while arrivals < num_ranks:
                            nanosleep(64)
                            arrivals = cute.arch.load(
                                flag_ptr, Int32, sem="relaxed", scope="gpu"
                            )
                        if cutlass.const_expr(not self.all_reduce):
                            arrivals = ld_acquire_sys(flag_ptr)
                    cute.arch.barrier(barrier_id=BAR_COMM, number_of_threads=128)

                    if cutlass.const_expr(self.all_reduce):
                        # AR: NVLS reduce + broadcast (the switch reads every
                        # rank and replicates the result), so each rank moves
                        # only ~1x its rows over NVLink instead of 2(R-1)/R.
                        self.nvls_unit(tid_, mc_vecs, unit_row0, row_lo, row_hi, bid_n)
                    else:
                        col_byte = Int64(bid_n) * seg_bytes
                        if tid_ < 32:
                            # The issuing warp: generic -> async proxy order for
                            # the acquired partials (see the poll above).
                            fence_proxy_async_global()
                        for c in cutlass.range_constexpr(
                            chunks_per_unit + _COMM_STAGES - 1
                        ):
                            # Issue chunk c (the ring runs _COMM_STAGES - 1 ahead).
                            if cutlass.const_expr(c < chunks_per_unit):
                                # Chunks wholly outside the unit's rows are skipped.
                                live_row0 = unit_row0 + c * chunk_rows
                                if (
                                    live_row0 < row_hi
                                    and live_row0 + chunk_rows > row_lo
                                ):
                                    # The first communication warp issues the
                                    # chunk: lane 0 claims the stage, then each lane
                                    # copies one row segment per rank.
                                    if tid_ < 32:
                                        stage = issued % _COMM_STAGES
                                        phase = (issued // _COMM_STAGES) % 2
                                        if tid_ == 0:
                                            cute.arch.mbarrier_wait(
                                                comm_empty_mbar + stage, phase ^ 1
                                            )
                                            mbarrier.arrive_expect_tx(
                                                comm_full_mbar + stage, chunk_bytes
                                            )
                                        cute.arch.sync_warp()
                                        mbar_addr = (comm_full_mbar + stage).toint()
                                        dst0 = comm_base + stage * _COMM_CHUNK_BYTES
                                        for j in cutlass.range_constexpr(
                                            (chunk_rows + 31) // 32
                                        ):
                                            r = tid_ + j * 32
                                            if r < chunk_rows:
                                                row = unit_row0 + c * chunk_rows + r
                                                row = min(max(row, row_lo), row_hi - 1)
                                                gaddr = (
                                                    Int64(row) * row_bytes + col_byte
                                                )
                                                for src in cutlass.range_constexpr(
                                                    num_ranks
                                                ):
                                                    bulk_copy_g2s(
                                                        dst0
                                                        + (src * chunk_rows + r)
                                                        * seg_bytes,
                                                        partial_bases[src] + gaddr,
                                                        seg_bytes,
                                                        mbar_addr,
                                                    )
                                    issued += 1
                            # Consume chunk c - (_COMM_STAGES - 1).
                            if cutlass.const_expr(c >= _COMM_STAGES - 1):
                                cc = c - (_COMM_STAGES - 1)
                                cons_row0 = unit_row0 + cc * chunk_rows
                                if (
                                    cons_row0 < row_hi
                                    and cons_row0 + chunk_rows > row_lo
                                ):
                                    stage = consumed % _COMM_STAGES
                                    phase = (consumed // _COMM_STAGES) % 2
                                    cute.arch.mbarrier_wait(
                                        comm_full_mbar + stage, phase
                                    )
                                    for it in cutlass.range_constexpr(
                                        (vecs_per_chunk + 127) // 128
                                    ):
                                        v = tid_ + it * 128
                                        if v < vecs_per_chunk:
                                            vrow = v // vec_cols
                                            vcol = v % vec_cols
                                            acc = cute.make_rmem_tensor(
                                                vec_width, cutlass.Float32
                                            )
                                            acc.fill(0.0)
                                            for src in cutlass.range_constexpr(
                                                num_ranks
                                            ):
                                                sv = cute.make_tensor(
                                                    cute.make_ptr(
                                                        BFloat16,
                                                        comm_base
                                                        + stage * _COMM_CHUNK_BYTES
                                                        + (src * chunk_rows + vrow)
                                                        * seg_bytes
                                                        + vcol * 16,
                                                        cute.AddressSpace.smem,
                                                        assumed_align=16,
                                                    ),
                                                    cute.make_layout(vec_width),
                                                )
                                                val = cute.make_rmem_tensor(
                                                    vec_width, BFloat16
                                                )
                                                cute.copy(ld_atom, sv, val)
                                                acc.store(
                                                    acc.load()
                                                    + val.load().to(cutlass.Float32)
                                                )
                                            reduced = cute.make_rmem_tensor(
                                                vec_width, BFloat16
                                            )
                                            reduced.store(acc.load().to(BFloat16))
                                            row = unit_row0 + cc * chunk_rows + vrow
                                            col = bid_n * vec_cols + vcol
                                            if row >= row_lo and row < row_hi:
                                                if cutlass.const_expr(self.all_reduce):
                                                    # Broadcast the result to all ranks.
                                                    multimem_st_16B(
                                                        mc_vecs[None, (row, col)],
                                                        reduced,
                                                    )
                                                else:
                                                    cute.copy(
                                                        st_atom,
                                                        reduced,
                                                        output_vecs[
                                                            None,
                                                            (row - rank_start, col),
                                                        ],
                                                    )
                                    mbarrier.arrive(comm_empty_mbar + stage)
                                    consumed += 1
                    cute.arch.barrier(barrier_id=BAR_COMM, number_of_threads=128)

                    # Release the GEMM tile. Its consumers on this rank are the
                    # units overlapping this rank's rows; the last one resets
                    # the producer flag for the next launch.
                    if tid_ == 0:
                        first_unit = (max(rank_start, tile_row0) - tile_row0) // (
                            unit_rows
                        )
                        last_unit = (
                            min(rank_end, tile_row0 + BM) - 1 - tile_row0
                        ) // unit_rows
                        num_consumers = last_unit - first_unit + 1
                        old_count = cute.arch.atomic_add(
                            flag_ptr, Int32(1), sem="relaxed", scope="gpu"
                        )
                        if old_count == num_ranks + num_consumers - 1:
                            cute.arch.store(
                                flag_ptr, Int32(0), sem="relaxed", scope="gpu"
                            )

                unit = self.next_unit(tid_, unit_ctr, comm_slot, BAR_COMM)

            # Every CTA has taken its final (failing) unit before it counts
            # itself done, so the last CTA can reset both counters for the
            # next launch.
            if tid_ == 0:
                done = cute.arch.atomic_add(
                    done_ctr, Int32(1), sem="relaxed", scope="gpu"
                )
                if done == num_bids - 1:
                    cute.arch.store(unit_ctr, Int32(0), sem="relaxed", scope="gpu")
                    cute.arch.store(done_ctr, Int32(0), sem="relaxed", scope="gpu")

            # Exit barrier. GPU scope is sufficient for RS because the kernel
            # writes local global memory and only needs to flush it to local L2.
            # AR uses system scope to flush the multimem stores to remote L2.
            cute.arch.barrier(barrier_id=BAR_COMM, number_of_threads=128)
            if tid_ == 0:
                exit_flag = total_tiles + raw_bid
                scope = "sys" if self.all_reduce else "gpu"
                utils.distributed.multimem_red_add1(
                    flags_mc_ptr + exit_flag, order="release", scope=scope
                )
                utils.distributed.spin_lock_atom_cas_acquire_wait(
                    flags_uc_ptr + exit_flag,
                    expected_val=num_ranks,
                    reset_val=0,
                    scope=scope,
                )

        else:
            # Epilogue warps
            warp_id_ = warp_id % 4
            tid_ = tid % 128

            peer_flag_bases = cute.make_rmem_tensor(num_ranks, Int64)
            if tid_ == 0:
                for rank in cutlass.range_constexpr(num_ranks):
                    peer_flag_bases[rank] = cute.arch.load(
                        (peer_flags.iterator + rank).llvm_ptr,
                        Int64,
                    )

            if warp_id_ == 0:
                _tcgen05.alloc(taddr, cta_group)
            cute.arch.barrier(barrier_id=BAR_TMEM_ALLOC, number_of_threads=5 * 32)

            WIDTH = cutlass.const_expr(16)
            partial_vecs = cute.zipped_divide(partial_uc, (1, WIDTH))

            bf16x16_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                BFloat16,
                num_bits_per_copy=256,
                l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
            )

            if cutlass.const_expr(self.residual):
                residual_vecs = cute.zipped_divide(residual, (1, WIDTH))
                residual_row1 = residual_row0 + residual.shape[0]
                bf16x16_ld_atom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=256
                )

            tmem_stage = 0
            parity = 0

            if cutlass.const_expr(is_2cta):
                tmem_empty_mbar_ = to_cta0_smem(tmem_empty_mbar)
            else:
                tmem_empty_mbar_ = tmem_empty_mbar

            for bid in range(raw_bid, total_slots, num_bids):
                bid_m, bid_n, valid = self.tile_coords(
                    bid, grid_m, grid_n, local_M, num_slots_m, banded
                )
                if valid:
                    if warp_id_ == 0:
                        cute.arch.mbarrier_wait(tmem_full_mbar + tmem_stage, parity)
                    cute.arch.barrier(barrier_id=BAR_EPILOGUE, number_of_threads=128)
                    _tcgen05.fence_after_thread_sync()

                    for i in cutlass.range_constexpr(BN // WIDTH):
                        tcol = tmem_stage * BN + i * WIDTH
                        regs = _tcgen05.ld(warp_id_ * 32, tcol, "32x32b", WIDTH)
                        _tcgen05.wait_ld()

                        if cutlass.const_expr(i == BN // WIDTH - 1):
                            _tcgen05.fence_before_thread_sync()
                            mbarrier.arrive(tmem_empty_mbar_ + tmem_stage, "cluster")

                        global_row = bid_m * BM + tid_
                        tmp = cute.make_rmem_tensor(WIDTH, BFloat16)
                        if cutlass.const_expr(self.residual):  # noqa: SIM102 (compile-time)
                            if (
                                global_row >= residual_row0
                                and global_row < residual_row1
                            ):
                                res = cute.make_rmem_tensor(WIDTH, BFloat16)
                                cute.copy(
                                    bf16x16_ld_atom,
                                    residual_vecs[
                                        None,
                                        (
                                            global_row - residual_row0,
                                            bid_n * (BN // WIDTH) + i,
                                        ),
                                    ],
                                    res,
                                )
                                regs = regs + res.load().to(cutlass.Float32)
                        tmp.store(regs.to(BFloat16))

                        if global_row < M:
                            coord = (global_row, bid_n * (BN // WIDTH) + i)
                            cute.copy(bf16x16_atom, tmp, partial_vecs[None, coord])

                    cute.arch.barrier(barrier_id=BAR_EPILOGUE, number_of_threads=128)

                    # Signal GEMM completion. GPU scope is sufficient because the
                    # partial output only needs to be flushed to local L2.
                    if tid_ == 0 and bid_m * BM < M:
                        gemm_start = bid_m * BM
                        gemm_end = min(gemm_start + BM, M)
                        first_owner = gemm_start // local_M
                        last_owner = (gemm_end - 1) // local_M

                        # signal to all consuming ranks (can be more than 1)
                        for rank in cutlass.range_constexpr(num_ranks):
                            if first_owner <= rank and rank <= last_owner:
                                ptr = cute.make_ptr(
                                    Int32,
                                    peer_flag_bases[rank],
                                    cute.AddressSpace.gmem,
                                    assumed_align=16,
                                )
                                ptr += bid_m + bid_n * grid_m
                                utils.distributed.red_add1(
                                    ptr, order="release", scope="gpu"
                                )

                    tmem_stage = (tmem_stage + 1) % num_tmem_stages
                    if tmem_stage == 0:
                        parity ^= 1

            if cutlass.const_expr(is_2cta):
                cute.arch.cluster_arrive_relaxed()
                cute.arch.cluster_wait()
            else:
                cute.arch.barrier(barrier_id=BAR_EPILOGUE, number_of_threads=128)
            if warp_id_ == 0:
                _tcgen05.dealloc(cta_group)

    @cache
    @staticmethod
    def compile(
        rank: int,
        num_ranks: int,
        cta_group: int,
        all_reduce: bool = False,
        dtype: str = "bf16",
        residual: bool = False,
        BN: int = 128,
    ):
        M = cute.sym_int()
        padded_M = cute.sym_int()
        N = cute.sym_int()
        K = cute.sym_int()
        local_M = cute.sym_int()
        num_flags = cute.sym_int()

        ab_dtype = Float8E4M3FN if dtype == "mxfp8" else BFloat16
        # Row stride must keep rows 16B-aligned: 8 BF16 or 16 FP8 elements.
        row_div = 16 if dtype == "mxfp8" else 8
        A = make_fake_tensor(
            ab_dtype,
            (M, K),
            (cute.sym_int64(divisibility=row_div), 1),
            assumed_align=16,
        )
        B = make_fake_tensor(
            ab_dtype,
            (N, K),
            (cute.sym_int64(divisibility=row_div), 1),
            assumed_align=16,
        )
        if dtype == "mxfp8":
            SFA = make_fake_tensor(
                Float8E8M0FNU,
                (cute.sym_int(divisibility=512),),
                (1,),
                assumed_align=16,
            )
            SFB = make_fake_tensor(
                Float8E8M0FNU,
                (cute.sym_int(divisibility=512),),
                (1,),
                assumed_align=16,
            )
        else:
            SFA = None
            SFB = None
        partial = make_fake_tensor(
            BFloat16,
            (padded_M, N),
            (cute.sym_int64(divisibility=16), 1),
            assumed_align=32,
        )
        partial_mc_ptr = nullptr(BFloat16, cute.AddressSpace.gmem, assumed_align=32)
        output = (
            None
            if all_reduce
            else make_fake_tensor(
                BFloat16,
                (local_M, N),
                (cute.sym_int64(divisibility=16), 1),
                assumed_align=32,
            )
        )
        flags = make_fake_tensor(Int32, (num_flags,), (1,), assumed_align=16)
        sched = make_fake_tensor(Int32, (4,), (1,), assumed_align=16)
        flags_mc_ptr = nullptr(Int32, cute.AddressSpace.gmem, assumed_align=16)
        peer_flag_ptr = nullptr(Int64, cute.AddressSpace.gmem, assumed_align=8)
        peer_partial_ptr = nullptr(Int64, cute.AddressSpace.gmem, assumed_align=8)

        residual_t = (
            make_fake_tensor(
                BFloat16,
                (cute.sym_int(), N),
                (cute.sym_int64(divisibility=16), 1),
                assumed_align=32,
            )
            if residual
            else None
        )

        stream = make_fake_stream(use_tvm_ffi_env_stream=True)
        kernel = Sm100GemmRsAr(
            rank,
            num_ranks,
            cta_group,
            all_reduce=all_reduce,
            dtype=dtype,
            residual=residual,
            BN=BN,
        )
        return cute.compile(
            kernel,
            A,
            B,
            SFA,
            SFB,
            partial,
            partial_mc_ptr,
            output,
            flags,
            flags_mc_ptr,
            peer_flag_ptr,
            peer_partial_ptr,
            residual_t,
            0,
            sched,
            128,
            stream,
            options="--enable-tvm-ffi",
        )


class GemmRsAr:
    """Own the symmetric workspace for GEMM-RS/AR launches.

    All TP ranks must belong to one NVLink domain for multimem instructions.

    Each instance is bound to either RS or AR. A vLLM worker has one static
    sequence-parallel topology, so the process-wide singleton only needs one
    mode. Two independent mode-specific singletons would lift that restriction
    but duplicate the large symmetric workspace. A future mixed-mode design
    should instead use lightweight RS/AR frontends over one shared multicast
    workspace; that is outside this integration's current scope.
    """

    def __init__(self, *, max_M: int, N: int, all_reduce: bool = False) -> None:
        tp_group = get_tp_group()
        group = tp_group.device_group
        rank = tp_group.rank_in_group
        world_size = tp_group.world_size
        device = torch.device("cuda", torch.accelerator.current_device_index())

        assert 1 < world_size <= 16
        assert 128 % world_size == 0
        assert max_M >= 128 and N % 128 == 0

        max_M = (max_M + world_size - 1) // world_size * world_size
        self.rank = rank
        self.world_size = world_size
        self.max_M = max_M
        self.N = N
        self.device = device
        self.all_reduce = all_reduce
        self.dtypes: set[str] = set()
        self.residual_used = False

        self.partial = symm_mem.empty((max_M, N), dtype=torch.bfloat16, device=device)
        self.partial_handle = symm_mem.rendezvous(self.partial, group)
        if self.partial_handle.multicast_ptr == 0:
            raise RuntimeError("GEMM-RS/AR requires NVLink multicast memory")
        self.partial_mc_ptr = make_ptr(
            BFloat16,
            self.partial_handle.multicast_ptr,
            cute.AddressSpace.gmem,
            assumed_align=32,
        )

        grid_m = (max_M + 127) // 128
        cta_group = 2 if max_M >= 1024 or grid_m % 2 == 0 else 1
        grid_m = (grid_m + cta_group - 1) // cta_group * cta_group
        self.num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        max_flags = grid_m * (N // 128) + self.num_sms
        self.flags = symm_mem.empty(max_flags, dtype=torch.int32, device=device)
        self.flags_handle = symm_mem.rendezvous(self.flags, group)
        if self.flags_handle.multicast_ptr == 0:
            raise RuntimeError("GEMM-RS/AR requires NVLink multicast memory")
        self.flags.zero_()
        # Per-rank communication-unit counters (next unit, finished CTAs); the
        # last CTA of every launch resets them.
        self.sched = torch.zeros(4, dtype=torch.int32, device=device)
        self.flags_mc_ptr = make_ptr(
            Int32,
            self.flags_handle.multicast_ptr,
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        self.peer_partial_ptr = make_ptr(
            Int64,
            self.partial_handle.buffer_ptrs_dev,
            cute.AddressSpace.gmem,
            assumed_align=8,
        )
        self.peer_flag_ptr = make_ptr(
            Int64,
            self.flags_handle.buffer_ptrs_dev,
            cute.AddressSpace.gmem,
            assumed_align=8,
        )

        torch.accelerator.synchronize(device)
        tp_group.barrier()

    def can_run(self, linear: LinearBase) -> bool:
        from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
            FlashInferCutedslMxfp8LinearKernel,
            FlashInferCutlassMxfp8LinearKernel,
        )

        # Called before weight loading; online quantization starts on meta.
        if isinstance(linear.quant_method, UnquantizedLinearMethod):
            dtype, k_alignment = "bf16", 64
            if linear.weight.dtype != torch.bfloat16:
                return False
        else:
            method = getattr(linear, "scheme", linear.quant_method)
            kernel = getattr(method, "kernel", None)
            if not isinstance(
                kernel,
                (
                    FlashInferCutedslMxfp8LinearKernel,
                    FlashInferCutlassMxfp8LinearKernel,
                ),
            ):
                return False
            dtype, k_alignment = "mxfp8", 128

        w = linear.weight
        if (
            w.ndim != 2
            or w.shape[0] != self.N
            or w.shape[1] % k_alignment != 0
            or (w.device != self.device and not w.is_meta)
            or not w.is_contiguous()
        ):
            return False
        self.dtypes.add(dtype)
        return True

    def enable_residual(self) -> None:
        """Declare that some caller passes ``residual`` to :meth:`apply`, so the
        startup warmup also compiles the residual-epilogue variants."""
        self.residual_used = True

    def warn_incompatible_projection(self) -> None:
        logger.warning_once(
            "Some projections are incompatible with GEMM-RS/AR; using the "
            "unfused path instead.",
            scope="global",
        )

    def owns(self, t: torch.Tensor) -> bool:
        """Whether ``t`` is a view of this op's output workspace (see
        ``apply(copy_out=False)``)."""
        return (
            t.device == self.partial.device
            and t.untyped_storage().data_ptr()
            == self.partial.untyped_storage().data_ptr()
        )

    def should_run(self, x: torch.Tensor) -> bool:
        # Use the same threshold for RS and AR for now. Small-M shapes are
        # supported but faster on the existing LL path.
        return x.shape[0] >= 128

    def apply(
        self,
        x: torch.Tensor,
        linear: LinearBase,
        residual: torch.Tensor | None = None,
        residual_row0: int = 0,
        copy_out: bool = True,
    ) -> torch.Tensor:
        """``x @ linear.weight.T``, reduce-scattered or all-reduced over TP.

        ``copy_out=False`` (AR only): return a view of the shared workspace
        instead of a copy. The caller must finish reading it before the next
        GEMM-RS/AR launch on this stream (the next launch rewrites it), and copy
        anything it keeps longer; ``owns()`` tells whether a tensor is such a view.

        ``residual`` (optional, BF16 ``[R, N]``, contiguous): added in the GEMM
        epilogue to rows ``[residual_row0, residual_row0 + R)`` of this rank's
        partial before the reduction, i.e. contributed exactly once to the sum.
        """
        from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
            FlashInferCutedslMxfp8LinearKernel,
        )

        method = getattr(linear, "scheme", linear.quant_method)
        w = linear.weight
        if isinstance(
            getattr(method, "kernel", None), FlashInferCutedslMxfp8LinearKernel
        ):
            # This backend stores a column-major [K, N] view after loading.
            w = w.t()
        w_sf = getattr(linear, "weight_scale", None)

        assert x.ndim == 2
        M, K = x.shape
        assert 0 < M <= self.max_M
        assert x.dtype == torch.bfloat16
        assert x.device == self.device
        assert x.is_contiguous()
        assert w.device == self.device
        N = self.N
        dtype = "mxfp8" if w.dtype == torch.float8_e4m3fn else "bf16"
        x_sf = None
        if dtype == "mxfp8":
            from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
                mxfp8_e4m3_quantize,
            )

            assert w_sf is not None
            assert w_sf.dtype == torch.uint8 and w_sf.device == self.device
            assert w_sf.ndim == 1 and w_sf.is_contiguous()
            assert w_sf.numel() == N * (K // 32)
            assert w.shape == (N, K) and K % 128 == 0
            assert w.is_contiguous()
            x, x_sf = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
            # The kernel declares the scales as E8M0; the stored/quantized
            # tensors carry the same bits as uint8.
            x_sf = x_sf.view(torch.float8_e8m0fnu)
            w_sf = w_sf.view(torch.float8_e8m0fnu)
        else:
            assert w_sf is None
            assert w.shape == (N, K) and K % 64 == 0
            assert w.dtype == torch.bfloat16
            assert w.is_contiguous()
        cta_group, BN = self.select_tiles(M, dtype)
        return self._launch(
            x, w, x_sf, w_sf, residual, residual_row0, copy_out, dtype, cta_group, BN
        )

    def select_tiles(self, M: int, dtype: str) -> tuple[int, int]:
        """Per-shape (cta_group, BN).

        2-CTA (256-row MMA) unless a small odd M-tile grid would need padding.
        BF16 2-CTA RS uses 256-column tiles (a 256x256 MMA) once they still
        give the SMs >= 6 waves (TP4 K=3072: M=4096 128-col 94 vs 98 us, M=8192
        256-col 144 vs 154 us): it halves the A traffic from L2 and the
        number of tile handoffs to the communication warps (TP4 M=17400:
        330 -> 257 us). AR (comm-bound, NVLS) and MXFP8 (its scale factors
        leave room for one 256-column TMEM stage only) stay at 128 columns.
        """
        grid_m = (M + 127) // 128
        cta_group = 2 if M >= 1024 or grid_m % 2 == 0 else 1
        BN = 128
        if (
            dtype == "bf16"
            and not self.all_reduce
            and cta_group == 2
            and self.N % 256 == 0
            and (grid_m + 1) // 2 * (self.N // 256) * 2 >= 6 * self.num_sms
        ):
            BN = 256
        return cta_group, BN

    def _launch(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        x_sf: torch.Tensor | None,
        w_sf: torch.Tensor | None,
        residual: torch.Tensor | None,
        residual_row0: int,
        copy_out: bool,
        dtype: str,
        cta_group: int,
        BN: int,
    ) -> torch.Tensor:
        M = x.shape[0]
        N = self.N
        padded_M = (M + self.world_size - 1) // self.world_size
        padded_M *= self.world_size
        local_M = padded_M // self.world_size
        grid_m = (M + 127) // 128
        grid_m = (grid_m + cta_group - 1) // cta_group * cta_group

        num_tiles = grid_m * (N // BN)
        num_ctas = min(num_tiles, self.num_sms)
        num_ctas = num_ctas // cta_group * cta_group
        assert self.flags.numel() >= num_tiles + num_ctas

        output = None
        if not self.all_reduce:
            output = torch.empty((local_M, N), dtype=torch.bfloat16, device=self.device)
            # The kernel never touches rows at or beyond M. Callers that hand
            # in an unpadded M (DeepSeek-V4.1 slices the SP all-gather back to
            # the real token count) expect the same zero padding rows that
            # ``sp_reduce_scatter`` produces, so clear them here. This is a
            # no-op whenever M is already a multiple of the TP size.
            valid_rows = min(max(M - self.rank * local_M, 0), local_M)
            if valid_rows < local_M:
                output[valid_rows:].zero_()
        if residual is not None:
            assert residual.dtype == torch.bfloat16 and residual.is_contiguous()
            assert residual.ndim == 2 and residual.shape[1] == N
            assert residual.device == self.device
            assert residual_row0 >= 0 and residual_row0 + residual.shape[0] <= M
        compiled = Sm100GemmRsAr.compile(
            self.rank,
            self.world_size,
            cta_group,
            self.all_reduce,
            dtype=dtype,
            residual=residual is not None,
            BN=BN,
        )
        compiled(
            x,
            w,
            x_sf,
            w_sf,
            self.partial[:padded_M],
            self.partial_mc_ptr,
            output,
            self.flags,
            self.flags_mc_ptr,
            self.peer_flag_ptr,
            self.peer_partial_ptr,
            residual,
            residual_row0,
            self.sched,
            num_ctas,
        )
        if self.all_reduce:
            if not copy_out:
                return self.partial[:M]
            # AttnRes may retain output past the next workspace reuse.
            # A future kernel could overlap this copy using an extra warp or
            # the communication warp.
            return self.partial[:M].clone()
        assert output is not None
        return output


_gemm_rs_ar: GemmRsAr | None = None


def init_gemm_rs_ar(max_M: int, N: int, *, all_reduce: bool = False) -> None:
    """Collectively initialize the process-wide, mode-bound GEMM-RS/AR state."""
    global _gemm_rs_ar
    if _gemm_rs_ar is not None:
        if _gemm_rs_ar.all_reduce != all_reduce:
            current = "AR" if _gemm_rs_ar.all_reduce else "RS"
            requested = "AR" if all_reduce else "RS"
            raise RuntimeError(
                f"GEMM-RS/AR is already initialized for {current}; "
                f"a worker cannot reinitialize it for {requested}"
            )
        assert _gemm_rs_ar.max_M >= max_M and _gemm_rs_ar.N == N
        return
    _gemm_rs_ar = GemmRsAr(max_M=max_M, N=N, all_reduce=all_reduce)


def maybe_init_gemm_rs_ar(
    vllm_config: "VllmConfig", *, N: int, all_reduce: bool
) -> bool:
    """Collectively initialize the mode-bound GEMM-RS/AR state if supported.

    Callers gate on their own feature flag first; this checks the worker
    topology (SM100-family CUDA, BF16 model dtype, no ubatching, a TP size in
    2-16 that divides 128) and the NVLink multicast rendezvous. Returns whether
    projections may bind to the kernel through ``get_gemm_rs_ar().can_run``.
    """
    mode = "GEMM-AR" if all_reduce else "GEMM-RS"
    parallel_config = vllm_config.parallel_config
    tp_size = parallel_config.tensor_parallel_size
    if parallel_config.use_ubatching:
        reason = "ubatching is enabled"
    elif vllm_config.model_config.dtype != torch.bfloat16:
        reason = "the model dtype is not BF16"
    elif not current_platform.is_cuda():
        reason = "the device is not CUDA"
    elif not current_platform.is_device_capability_family(100):
        reason = "the device is not SM100-family"
    elif not 1 < tp_size <= 16:
        reason = "TP size is not in the supported range 2-16"
    elif 128 % tp_size != 0:
        reason = "TP size does not divide 128"
    elif N % 128 != 0:
        reason = f"the output width {N} is not a multiple of 128"
    else:
        reason = None

    if reason is not None:
        logger.warning_once("%s is disabled because %s.", mode, reason)
        return False

    try:
        init_gemm_rs_ar(
            max_M=vllm_config.scheduler_config.max_num_batched_tokens,
            N=N,
            all_reduce=all_reduce,
        )
    except RuntimeError as e:
        logger.warning_once(
            "%s is disabled because initialization failed: %s. This may mean "
            "the TP ranks do not share one NVLink domain.",
            mode,
            e,
        )
        return False
    logger.info_once("%s is enabled.", mode)
    return True


def warmup_gemm_rs_ar() -> int:
    """Compile every reachable dispatch for the initialized GEMM-RS/AR mode."""
    # Initialization can be disabled or fail when multicast is unavailable.
    if _gemm_rs_ar is None:
        return 0

    variants = (False, True) if _gemm_rs_ar.residual_used else (False,)
    count = 0
    for dtype in sorted(_gemm_rs_ar.dtypes):
        # Every (cta_group, BN) that select_tiles can return for this dtype.
        tiles = [(1, 128), (2, 128)]
        if dtype == "bf16" and _gemm_rs_ar.N % 256 == 0:
            tiles.append((2, 256))
        for cta_group, BN in tiles:
            for residual in variants:
                Sm100GemmRsAr.compile(
                    _gemm_rs_ar.rank,
                    _gemm_rs_ar.world_size,
                    cta_group,
                    _gemm_rs_ar.all_reduce,
                    dtype=dtype,
                    residual=residual,
                    BN=BN,
                )
                count += 1
    return count


def get_gemm_rs_ar() -> GemmRsAr:
    assert _gemm_rs_ar is not None, "GEMM-RS/AR is not initialized"
    return _gemm_rs_ar
