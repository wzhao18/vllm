# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Combine reduce for the dedup path: sums the ``rank_slots[token][rank]`` rows named by ``valid_rank_mask`` in
ascending rank order into ``reduced_output[token]``; top-k weights are already folded in. An optional ``shared_term``
adds one more unweighted BF16 row after the rank fold. vLLM-local E2a-dedup (VENDOR.md): with ``rmsnorm_weight`` the
reduce also applies the consumer's stock RMSNorm to the BF16-rounded row (one CTA per token row) and writes only the
normalized row."""

from typing import ClassVar, Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass.cutlass_dsl import Float32, Int32

from ......common.dsl_helpers import mark_alignment
from .....helpers.topk_reduce import add_shared_term, load_shared_term, shared_term_view
from .....quant_def import CombineFormat


class TopkReduceDedup:
    _threads: ClassVar[int] = 128
    _hidden_per_thread: ClassVar[int] = 16  # one 32-byte BF16 load per rank slot

    def __init__(
        self, hidden: int, world_size: int, combine_format: CombineFormat, norm_hidden_per_thread: int = 8
    ) -> None:
        if combine_format.is_quantized or combine_format.act_dtype is not cutlass.BFloat16:
            raise ValueError("TopkReduceDedup supports the BF16 combine format only.")
        if world_size > 32:
            raise ValueError("TopkReduceDedup keeps the rank mask in an Int32; world_size must be <= 32.")
        self.hidden = int(hidden)
        self.world_size = int(world_size)
        self.combine_format = combine_format
        if self.hidden % self._hidden_per_thread != 0:
            raise ValueError(f"hidden ({self.hidden}) must be divisible by {self._hidden_per_thread}.")
        self.hidden_tiles = self.hidden // self._hidden_per_thread
        self.require_predicate = self.hidden_tiles % self._threads != 0
        # Fused-RMSNorm form: one token row per CTA, ``norm_hidden_per_thread`` elements per thread.
        self.norm_hpt = int(norm_hidden_per_thread)
        if self.norm_hpt not in (8, 16) or self.hidden % self.norm_hpt != 0:
            raise ValueError(f"norm_hidden_per_thread {self.norm_hpt} must be 8 or 16 and divide hidden.")
        self.norm_tiles = self.hidden // self.norm_hpt

    @cute.jit
    def __call__(
        self,
        rank_slots: cute.Tensor,  # (token, world_size, hidden) bf16
        valid_rank_mask: cute.Tensor,  # (token,) i32 bitmask over ranks
        reduced_output: cute.Tensor,  # (token, hidden)
        stream: cuda.CUstream,
        shared_term: Optional[cute.Tensor] = None,  # (token, hidden) BF16, added after the rank fold
        rmsnorm_weight: Optional[cute.Tensor] = None,  # (hidden,) BF16: output = rms_norm(reduce)
        rmsnorm_eps: Optional[Float32] = None,
    ):
        threads = self._threads
        total_workers = reduced_output.shape[0] * self.hidden_tiles
        grid = [(total_workers + threads - 1) // threads, 1, 1]
        block = [threads, 1, 1]
        rank_slots = cute.make_tensor(
            rank_slots.iterator,
            cute.make_layout((rank_slots.shape[0], self.world_size, self.hidden), stride=rank_slots.stride),
        )
        reduced_output = cute.make_tensor(
            reduced_output.iterator,
            cute.make_layout((reduced_output.shape[0], self.hidden), stride=reduced_output.stride),
        )
        if cutlass.const_expr(shared_term is not None):
            shared_term = shared_term_view(shared_term, self.hidden)
        if cutlass.const_expr(rmsnorm_weight is not None):
            if cutlass.const_expr(self.norm_tiles > 1024):
                raise ValueError("the fused RMSNorm reduce keeps one token row per CTA (<= 1024 threads).")
            rmsnorm_weight = cute.make_tensor(rmsnorm_weight.iterator, cute.make_layout((self.hidden,)))
            self._reduce_bf16_rmsnorm(
                rank_slots, valid_rank_mask, reduced_output, rmsnorm_weight, rmsnorm_eps, shared_term
            ).launch(
                grid=[reduced_output.shape[0], 1, 1],
                block=[(self.norm_tiles + 31) // 32 * 32, 1, 1],
                stream=stream,
            )
        else:
            self._reduce_bf16(rank_slots, valid_rank_mask, reduced_output, shared_term).launch(
                grid=grid, block=block, stream=stream
            )

    @cute.jit
    def _fold(
        self,
        rank_slots: cute.Tensor,
        valid_rank_mask: cute.Tensor,
        token_idx: Int32,
        hidden_tile_idx: Int32,
        shared_term: Optional[cute.Tensor],
        hidden_per_thread: cutlass.Constexpr[int],
    ) -> cute.Tensor:
        """FP32 fold of one worker's ``hidden_per_thread``-wide hidden tile: rank slots ascending, then the shared
        term."""
        world_size: cutlass.Constexpr[int] = self.world_size
        # (token, rank, hidden) -> (rank, hidden_per_thread) slice owned by this worker
        slots = cute.zipped_divide(rank_slots[token_idx, None, None], (world_size, hidden_per_thread))[
            (None, None), (0, hidden_tile_idx)
        ]
        rank_mask = valid_rank_mask[token_idx]

        access_bits: cutlass.Constexpr[int] = hidden_per_thread * cutlass.BFloat16.width
        load_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=access_bits)
        acc = cute.make_rmem_tensor((hidden_per_thread,), cutlass.Float32)
        for i in cutlass.range_constexpr(hidden_per_thread):
            acc[i] = Float32(0.0)
        one_pair = (Float32(1.0), Float32(1.0))

        # Issue every rank slot's load first (predicated, one register buffer per rank) so all of them are in
        # flight together, then accumulate ranks ascending; fma by 1.0 is an exact fp32 add.
        terms = [cute.make_rmem_tensor((hidden_per_thread,), cutlass.BFloat16) for _ in range(world_size)]
        # The unpredicated shared-term load goes out with the rank loads; it is still added after the rank fold.
        shared_reg = None
        if cutlass.const_expr(shared_term is not None):
            shared_reg = load_shared_term(shared_term, token_idx, hidden_tile_idx, hidden_per_thread)
        for rank in cutlass.range_constexpr(world_size):
            if ((rank_mask >> Int32(rank)) & Int32(1)) != Int32(0):
                cute.copy(
                    load_atom,
                    mark_alignment(slots[rank, None], hidden_per_thread * cutlass.BFloat16.width // 8),
                    terms[rank],
                )
        for rank in cutlass.range_constexpr(world_size):
            if ((rank_mask >> Int32(rank)) & Int32(1)) != Int32(0):
                for i in cutlass.range_constexpr(0, hidden_per_thread, 2):
                    value_pair = (Float32(terms[rank][i]), Float32(terms[rank][i + 1]))
                    acc[i], acc[i + 1] = cute.arch.fma_packed_f32x2(value_pair, one_pair, (acc[i], acc[i + 1]))

        if cutlass.const_expr(shared_term is not None):
            add_shared_term(acc, shared_reg, hidden_per_thread)
        return acc


    @cute.kernel
    def _reduce_bf16_rmsnorm(
        self,
        rank_slots: cute.Tensor,
        valid_rank_mask: cute.Tensor,
        reduced_output: cute.Tensor,
        rmsnorm_weight: cute.Tensor,
        rmsnorm_eps: Float32,
        shared_term: Optional[cute.Tensor],
    ):
        """One token row per CTA (thread = one ``norm_hpt``-wide hidden tile): the rank fold of ``_reduce_bf16``, then
        the stock vLLM ``rms_norm`` of the BF16-rounded row: FP32 sum of squares of the BF16 values,
        ``s = rsqrt(sum / hidden + eps)``, ``out = bf16(x * s * w)`` (as E2a on the GenPhase reduce). The weight tile
        is loaded with the rank slots, so the only dependent step after the block reduction is the store."""
        hidden_per_thread: cutlass.Constexpr[int] = self.norm_hpt
        hidden_tiles = self.norm_tiles
        warps: cutlass.Constexpr[int] = (hidden_tiles + 31) // 32
        out_dtype = reduced_output.element_type
        token_idx = cute.arch.block_idx()[0]
        thread_idx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.make_warp_uniform(thread_idx // Int32(32))
        lane_idx = cute.arch.lane_idx()
        hidden_tile_idx = thread_idx
        active = thread_idx < Int32(hidden_tiles)
        tile_bytes = hidden_per_thread * cutlass.BFloat16.width // 8

        smem = cutlass_utils.SmemAllocator()
        partial = smem.allocate_tensor(cutlass.Float32, cute.make_layout((32,)), 16)
        x = cute.make_rmem_tensor((hidden_per_thread,), out_dtype)
        weight = cute.make_rmem_tensor((hidden_per_thread,), rmsnorm_weight.element_type)
        sumsq = Float32(0)
        if active:
            wsrc = cute.zipped_divide(rmsnorm_weight, (hidden_per_thread,))[(None,), (hidden_tile_idx,)]
            cute.copy(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128),
                mark_alignment(wsrc, tile_bytes),
                weight,
            )
            acc = self._fold(rank_slots, valid_rank_mask, token_idx, hidden_tile_idx, shared_term, hidden_per_thread)
            x.store(acc.load().to(out_dtype))
            for i in cutlass.range_constexpr(hidden_per_thread):
                xf = Float32(x[i])
                sumsq = sumsq + xf * xf
        for shift in cutlass.range_constexpr(5):
            sumsq = sumsq + cute.arch.shuffle_sync_bfly(sumsq, 16 >> shift)
        if lane_idx == Int32(0):
            partial[warp_idx] = sumsq
        cute.arch.sync_threads()
        total = Float32(0)
        for w in cutlass.range_constexpr(warps):
            total = total + partial[w]
        scale = cute.math.rsqrt(total / Float32(self.hidden) + rmsnorm_eps, fastmath=True)
        if active:
            out = cute.make_rmem_tensor((hidden_per_thread,), out_dtype)
            for i in cutlass.range_constexpr(hidden_per_thread):
                out[i] = out_dtype(Float32(x[i]) * scale * Float32(weight[i]))
            dst = cute.zipped_divide(reduced_output[token_idx, None], (hidden_per_thread,))[(None,), (hidden_tile_idx,)]
            cute.copy(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), out_dtype, num_bits_per_copy=128),
                out,
                mark_alignment(dst, 16),
            )

    @cute.kernel
    def _reduce_bf16(
        self,
        rank_slots: cute.Tensor,
        valid_rank_mask: cute.Tensor,
        reduced_output: cute.Tensor,
        shared_term: Optional[cute.Tensor],
    ):
        threads = self._threads
        hidden_per_thread = self._hidden_per_thread
        hidden_tiles = self.hidden_tiles
        needs_guard = self.require_predicate
        out_dtype = reduced_output.element_type

        worker_idx = cute.arch.block_idx()[0] * Int32(threads) + cute.arch.thread_idx()[0]
        token_idx = worker_idx // hidden_tiles
        hidden_tile_idx = worker_idx % hidden_tiles

        if (not needs_guard) or token_idx < reduced_output.shape[0]:
            dst = cute.zipped_divide(reduced_output[token_idx, None], (hidden_per_thread,))[(None,), (hidden_tile_idx,)]
            acc = self._fold(rank_slots, valid_rank_mask, token_idx, hidden_tile_idx, shared_term, hidden_per_thread)

            out = cute.make_rmem_tensor((hidden_per_thread,), out_dtype)
            out.store(acc.load().to(out_dtype))
            cute.copy(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), out_dtype, num_bits_per_copy=128),
                out,
                mark_alignment(dst, 16),
            )


__all__ = ["TopkReduceDedup"]
