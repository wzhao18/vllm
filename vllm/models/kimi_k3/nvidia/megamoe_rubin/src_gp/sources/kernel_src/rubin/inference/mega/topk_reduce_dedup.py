# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Combine reduce for the dedup path: sums the ``rank_slots[token][rank]`` rows named by ``valid_rank_mask`` in
ascending rank order into ``reduced_output[token]``; top-k weights are already folded in. An optional ``shared_term``
adds one more unweighted BF16 row after the rank fold. With ``rmsnorm_weight`` the row is then normalized in the
same kernel (vLLM-local E2a, one CTA per token): the stock vLLM ``rms_norm`` of the BF16-rounded row (FP32 sum of
squares, ``s = rsqrt(sum / hidden + eps)``, ``out = bf16(x * s * w)``); only the normalized row is written."""

from typing import ClassVar, Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass.cutlass_dsl import Float32, Int32

from ......common.dsl_helpers import mark_alignment
from .....helpers.topk_reduce import add_shared_term, load_shared_term, shared_term_view
from .....quant_def import CombineFormat


@cute.jit
def _popc31(v: Int32) -> Int32:
    """Population count of a non-negative 31-bit mask (SWAR)."""
    v = v - ((v >> Int32(1)) & Int32(0x55555555))
    v = (v & Int32(0x33333333)) + ((v >> Int32(2)) & Int32(0x33333333))
    v = (v + (v >> Int32(4))) & Int32(0x0F0F0F0F)
    v = v + (v >> Int32(8))
    return (v + (v >> Int32(16))) & Int32(0x3F)


class TopkReduceDedup:
    _threads: ClassVar[int] = 128
    _hidden_per_thread: ClassVar[int] = 16  # one 32-byte BF16 load per rank slot

    # Shared-memory rank slots of the fused-RMSNorm form: the first 8 valid ranks of a row, then the rest (2 phases),
    # so 4 rows fit per SM (57 KiB each) instead of 2 with a slot per rank.
    _stage_slots: ClassVar[int] = 8

    def _staged_smem_bytes(self) -> int:
        """Fused-RMSNorm form: every rank slot of the row staged in shared memory (cp.async), plus the warp sums."""
        return self._stage_slots * self.hidden * 2 + 32 * 4 + 256

    def __init__(self, hidden: int, world_size: int, combine_format: CombineFormat) -> None:
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

    @cute.jit
    def __call__(
        self,
        rank_slots: cute.Tensor,  # (token, world_size, hidden) bf16
        valid_rank_mask: cute.Tensor,  # (token,) i32 bitmask over ranks
        reduced_output: cute.Tensor,  # (token, hidden)
        stream: cuda.CUstream,
        shared_term: Optional[cute.Tensor] = None,  # (token, hidden) BF16, added after the rank fold
        rmsnorm_weight: Optional[cute.Tensor] = None,  # (hidden,) BF16: output = rms_norm(reduced row)
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
        if cutlass.const_expr(rmsnorm_weight is not None and self.hidden_tiles <= 1024):
            rmsnorm_weight = cute.make_tensor(rmsnorm_weight.iterator, cute.make_layout((self.hidden,)))
            self._reduce_bf16_rmsnorm_staged(
                rank_slots, valid_rank_mask, reduced_output, shared_term, rmsnorm_weight, rmsnorm_eps
            ).launch(
                grid=[reduced_output.shape[0], 1, 1],
                block=[(self.hidden_tiles + 31) // 32 * 32, 1, 1],
                smem=self._staged_smem_bytes(),
                stream=stream,
            )
        elif cutlass.const_expr(rmsnorm_weight is not None):
            norm_threads = (self.hidden_tiles + 31) // 32 * 32
            if cutlass.const_expr(norm_threads > 1024):
                raise ValueError(f"fused RMSNorm reduce: hidden {self.hidden} needs more than 1024 threads.")
            rmsnorm_weight = cute.make_tensor(rmsnorm_weight.iterator, cute.make_layout((self.hidden,)))
            self._reduce_bf16_rmsnorm(
                rank_slots, valid_rank_mask, reduced_output, shared_term, rmsnorm_weight, rmsnorm_eps
            ).launch(grid=[reduced_output.shape[0], 1, 1], block=[norm_threads, 1, 1], stream=stream)
        else:
            self._reduce_bf16(rank_slots, valid_rank_mask, reduced_output, shared_term).launch(
                grid=grid, block=block, stream=stream
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
        world_size: cutlass.Constexpr[int] = self.world_size
        needs_guard = self.require_predicate
        out_dtype = reduced_output.element_type

        worker_idx = cute.arch.block_idx()[0] * Int32(threads) + cute.arch.thread_idx()[0]
        token_idx = worker_idx // hidden_tiles
        hidden_tile_idx = worker_idx % hidden_tiles

        if (not needs_guard) or token_idx < reduced_output.shape[0]:
            # (token, rank, hidden) -> (rank, hidden_per_thread) slice owned by this worker
            slots = cute.zipped_divide(rank_slots[token_idx, None, None], (world_size, hidden_per_thread))[
                (None, None), (0, hidden_tile_idx)
            ]
            dst = cute.zipped_divide(reduced_output[token_idx, None], (hidden_per_thread,))[(None,), (hidden_tile_idx,)]
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

            out = cute.make_rmem_tensor((hidden_per_thread,), out_dtype)
            out.store(acc.load().to(out_dtype))
            cute.copy(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), out_dtype, num_bits_per_copy=128),
                out,
                mark_alignment(dst, 16),
            )

    @cute.kernel
    def _reduce_bf16_rmsnorm(
        self,
        rank_slots: cute.Tensor,
        valid_rank_mask: cute.Tensor,
        reduced_output: cute.Tensor,
        shared_term: Optional[cute.Tensor],
        rmsnorm_weight: cute.Tensor,
        rmsnorm_eps: Float32,
    ):
        """``_reduce_bf16`` of one token row per CTA (thread = one hidden tile; same loads, rank order and FP32 add
        form), then the stock ``rms_norm`` of the BF16-rounded row."""
        hidden_per_thread = self._hidden_per_thread
        hidden_tiles = self.hidden_tiles
        world_size: cutlass.Constexpr[int] = self.world_size
        warps: cutlass.Constexpr[int] = (self.hidden_tiles + 31) // 32
        out_dtype = reduced_output.element_type
        thread_idx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.make_warp_uniform(thread_idx // Int32(32))
        lane_idx = cute.arch.lane_idx()
        token_idx = cute.arch.block_idx()[0]
        hidden_tile_idx = thread_idx
        active = thread_idx < Int32(hidden_tiles)

        smem = cutlass_utils.SmemAllocator()
        partial = smem.allocate_tensor(cutlass.Float32, cute.make_layout((32,)), 16)
        tile_bits: cutlass.Constexpr[int] = hidden_per_thread * cutlass.BFloat16.width
        load_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=tile_bits)
        x = cute.make_rmem_tensor((hidden_per_thread,), out_dtype)
        weight = cute.make_rmem_tensor((hidden_per_thread,), rmsnorm_weight.element_type)
        sumsq = Float32(0)
        if active:
            slots = cute.zipped_divide(rank_slots[token_idx, None, None], (world_size, hidden_per_thread))[
                (None, None), (0, hidden_tile_idx)
            ]
            rank_mask = valid_rank_mask[token_idx]
            terms = [cute.make_rmem_tensor((hidden_per_thread,), cutlass.BFloat16) for _ in range(world_size)]
            shared_reg = None
            if cutlass.const_expr(shared_term is not None):
                shared_reg = load_shared_term(shared_term, token_idx, hidden_tile_idx, hidden_per_thread)
            for rank in cutlass.range_constexpr(world_size):
                if ((rank_mask >> Int32(rank)) & Int32(1)) != Int32(0):
                    cute.copy(load_atom, mark_alignment(slots[rank, None], tile_bits // 8), terms[rank])
            wsrc = cute.zipped_divide(rmsnorm_weight, (hidden_per_thread,))[(None,), (hidden_tile_idx,)]
            w_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128)
            cute.copy(w_atom, mark_alignment(wsrc, 16), weight)
            acc = cute.make_rmem_tensor((hidden_per_thread,), cutlass.Float32)
            for i in cutlass.range_constexpr(hidden_per_thread):
                acc[i] = Float32(0.0)
            one_pair = (Float32(1.0), Float32(1.0))
            for rank in cutlass.range_constexpr(world_size):
                if ((rank_mask >> Int32(rank)) & Int32(1)) != Int32(0):
                    for i in cutlass.range_constexpr(0, hidden_per_thread, 2):
                        value_pair = (Float32(terms[rank][i]), Float32(terms[rank][i + 1]))
                        acc[i], acc[i + 1] = cute.arch.fma_packed_f32x2(value_pair, one_pair, (acc[i], acc[i + 1]))
            if cutlass.const_expr(shared_term is not None):
                add_shared_term(acc, shared_reg, hidden_per_thread)
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
    def _reduce_bf16_rmsnorm_staged(
        self,
        rank_slots: cute.Tensor,
        valid_rank_mask: cute.Tensor,
        reduced_output: cute.Tensor,
        shared_term: Optional[cute.Tensor],
        rmsnorm_weight: cute.Tensor,
        rmsnorm_eps: Float32,
    ):
        """``_reduce_bf16`` of one token row per CTA, then the stock ``rms_norm`` of the BF16 row. The valid rank
        slots go global -> shared by cp.async (no registers held while the loads fly, so several rows are in flight
        per SM); each thread reads back only the columns it copied, adds the ranks ascending in FP32 (same form as
        ``_reduce_bf16``), rounds to BF16, and the block sums the squares (``s = rsqrt(sum / hidden + eps)``,
        ``out = bf16(x * s * w)``)."""
        hidden_per_thread = self._hidden_per_thread
        vec: cutlass.Constexpr[int] = 8
        parts: cutlass.Constexpr[int] = hidden_per_thread // vec
        hidden_tiles = self.hidden_tiles
        world_size: cutlass.Constexpr[int] = self.world_size
        warps: cutlass.Constexpr[int] = (self.hidden_tiles + 31) // 32
        out_dtype = reduced_output.element_type
        thread_idx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.make_warp_uniform(thread_idx // Int32(32))
        lane_idx = cute.arch.lane_idx()
        token_idx = cute.arch.block_idx()[0]
        active = thread_idx < Int32(hidden_tiles)

        slots: cutlass.Constexpr[int] = self._stage_slots
        smem = cutlass_utils.SmemAllocator()
        stage = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((slots, self.hidden), stride=(self.hidden, 1)), 128)
        partial = smem.allocate_tensor(cutlass.Float32, cute.make_layout((32,)), 16)
        g2s = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL),
            cutlass.BFloat16,
            num_bits_per_copy=128,
        )
        rank_mask = valid_rank_mask[token_idx]
        x = cute.make_rmem_tensor((hidden_per_thread,), out_dtype)
        sumsq = Float32(0)
        acc = cute.make_rmem_tensor((hidden_per_thread,), cutlass.Float32)
        for i in cutlass.range_constexpr(hidden_per_thread):
            acc[i] = Float32(0.0)
        one_pair = (Float32(1.0), Float32(1.0))
        shared_reg = None
        if cutlass.const_expr(shared_term is not None):
            shared_reg = load_shared_term(shared_term, token_idx, thread_idx, hidden_per_thread)
        row_slots = cute.zipped_divide(rank_slots[token_idx, None, None], (world_size, vec))
        # Phase n stages the valid ranks of slot index 8n..8n+7 (ascending rank) into the same slots; each
        # thread reads back only what it copied, so a cp.async wait (no barrier) orders a phase.
        for phase in cutlass.range_constexpr((world_size + slots - 1) // slots):
            if active:
                for rank in cutlass.range_constexpr(world_size):
                    slot = _popc31(rank_mask & Int32((1 << rank) - 1)) - Int32(phase * slots)
                    if ((rank_mask >> Int32(rank)) & Int32(1)) != Int32(0):
                        if (slot >= Int32(0)) & (slot < Int32(slots)):
                            for p in cutlass.range_constexpr(parts):
                                tile = thread_idx * Int32(parts) + Int32(p)
                                src = row_slots[(None, None), (0, tile)][rank, None]
                                dst = cute.zipped_divide(stage[slot, None], (vec,))[(None,), (tile,)]
                                cute.copy(g2s, mark_alignment(src, 16), mark_alignment(dst, 16))
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            if active:
                for rank in cutlass.range_constexpr(world_size):
                    slot = _popc31(rank_mask & Int32((1 << rank) - 1)) - Int32(phase * slots)
                    if ((rank_mask >> Int32(rank)) & Int32(1)) != Int32(0):
                        if (slot >= Int32(0)) & (slot < Int32(slots)):
                            for p in cutlass.range_constexpr(parts):
                                tile = thread_idx * Int32(parts) + Int32(p)
                                term = cute.make_rmem_tensor((vec,), cutlass.BFloat16)
                                term.store(cute.zipped_divide(stage[slot, None], (vec,))[(None,), (tile,)].load())
                                for i in cutlass.range_constexpr(0, vec, 2):
                                    j = p * vec + i
                                    value_pair = (Float32(term[i]), Float32(term[i + 1]))
                                    acc[j], acc[j + 1] = cute.arch.fma_packed_f32x2(
                                        value_pair, one_pair, (acc[j], acc[j + 1])
                                    )
        if active:
            if cutlass.const_expr(shared_term is not None):
                add_shared_term(acc, shared_reg, hidden_per_thread)
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
            wsrc = cute.zipped_divide(rmsnorm_weight, (vec,))
            dst_row = cute.zipped_divide(reduced_output[token_idx, None], (vec,))
            w_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), rmsnorm_weight.element_type, num_bits_per_copy=128)
            st_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), out_dtype, num_bits_per_copy=128)
            for p in cutlass.range_constexpr(parts):
                tile = thread_idx * Int32(parts) + Int32(p)
                weight = cute.make_rmem_tensor((vec,), rmsnorm_weight.element_type)
                cute.copy(w_atom, mark_alignment(wsrc[(None,), (tile,)], 16), weight)
                out = cute.make_rmem_tensor((vec,), out_dtype)
                for i in cutlass.range_constexpr(vec):
                    out[i] = out_dtype(Float32(x[p * vec + i]) * scale * Float32(weight[i]))
                cute.copy(st_atom, out, mark_alignment(dst_row[(None,), (tile,)], 16))

__all__ = ["TopkReduceDedup"]
