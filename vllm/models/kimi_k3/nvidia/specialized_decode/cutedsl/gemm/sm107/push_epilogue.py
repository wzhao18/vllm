# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reduce-scatter push epilogue for the vendored sm_107 MXFP8 GEMM (``prewait`` kind
"pre", ``Sm107BlockScaledPersistentDenseGemmKernel(push=True)``): the SP
o-projection's partial rows go straight into their owners' Lamport receive slots
(``common.sp_comm.SpComm`` recv ring) instead of a local output tile, so the owner
kernel ``attn_res.sp_fused.SpReduceAttnRes`` runs with ``push=False`` (as after
``oproj.sp.SpOprojWarp``).

Per tile (one tile per CTA: ``prewait.Mxfp8PreGemm`` checks it): TMEM -> FP32 -> BF16
(RN, as the TMA-store epilogue; alpha is 1) -> sanitized words (``positive_zero32``,
as ``SpReduceAttnRes(push=True)`` pushes them) into a token-major staging tile in the
drained operand ring, then 16-byte chunks, consecutive lanes along one row, stored
relaxed at system scope into ``peers[o] + (rank cap + row - o n, feature)``: the words
the owner receives are bit for bit those of the GEMM + push pair it replaces.

The push arguments ride in the GEMM's alpha buffer (float32 [ARGS]): [0] alpha 1.0;
int32 [1] rank, [2] n (rows per owner), [3] cap (rows per sender slot), [4] R (valid
rows); int64 [4 + o] owner o's receive base (``SpComm.views(parity)`` peers)."""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass import Int32, Int64
from cutlass.utils.gemm.sm100 import (
    epilogue_tmem_copy_and_partition,
    transform_partitioned_tensor_layout,
)

from ...common.primitives import bf16x2, load_words, store_relaxed, store_words
from ...common.sp_comm import positive_zero32

ARGS = 16  # float32 words of the alpha / push-argument buffer
WIDTH = 7168  # output features (the hidden size): the receive slot's row pitch
ROWS = 128  # tile rows per CTA (the pre tactics: M128, or M256 over a CTA pair)
PAD = 16  # staging row padding (bytes)


def staging_bytes(tile_n: int) -> int:
    return ROWS * (2 * tile_n + PAD)


@cute.jit
def epilogue_push(gemm_kernel, epi_tidx: Int32, tCtAcc_base: cute.Tensor,
                  tCgC_base: cute.Tensor, epi_tile: cute.Tile, mma_tile_coord_mnl,
                  acc_consumer_state: pipeline.PipelineState,
                  acc_pipeline: pipeline.PipelineAsync, aux: cute.Tensor,
                  stage: Int32) -> pipeline.PipelineState:  # fmt: skip
    """One tile's push (module docstring); ``tCgC_base`` is the TMA coordinate
    tensor of C (its elements are (row, feature, l)), ``stage`` the shared address of
    the drained A ring."""
    tile_n = cutlass.const_expr(gemm_kernel.mma_tiler[1])
    stride = cutlass.const_expr(2 * tile_n + PAD)
    tCgC = transform_partitioned_tensor_layout(tCgC_base)
    tCtAcc = transform_partitioned_tensor_layout(tCtAcc_base)
    tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = epilogue_tmem_copy_and_partition(
        gemm_kernel, epi_tidx, tCtAcc, tCgC, epi_tile, gemm_kernel.use_2cta_instrs
    )
    gC_epi = cute.flat_divide(tCgC, epi_tile)
    thr = tiled_copy_t2r.get_slice(epi_tidx)
    tTR_gC = thr.partition_D(gC_epi)[(None, None, None, None, None, *mma_tile_coord_mnl)]
    tTR_gC = cute.group_modes(tTR_gC, 3, cute.rank(tTR_gC))
    tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]
    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
    words = cute.make_tensor(cute.recast_ptr(aux.iterator, dtype=Int32), cute.make_layout(ARGS))
    bases = cute.make_tensor(
        cute.recast_ptr(aux.iterator, dtype=Int64), cute.make_layout(ARGS // 2)
    )
    vec = cutlass.const_expr(cute.size(tTR_rAcc, mode=[0]))
    assert vec % 8 == 0, "push epilogue: a thread's vector is not 16-byte chunks"

    acc_pipeline.consumer_wait(acc_consumer_state)
    subtiles = cutlass.const_expr(cute.size(tTR_tAcc.shape, mode=[3]))
    row0, col0 = Int32(0), Int32(0)
    for s in cutlass.range_constexpr(subtiles):
        cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, s)], tTR_rAcc)
        if s == subtiles - 1:
            with cute.arch.elect_one():
                acc_pipeline.consumer_release(acc_consumer_state)
            acc_consumer_state.advance()
        coords = tTR_gC[(None, None, None, s)]
        for mi in cutlass.range_constexpr(cute.size(coords, mode=[1])):
            for ni in cutlass.range_constexpr(cute.size(coords, mode=[2])):
                for c in cutlass.range_constexpr(vec // 8):
                    # each 8-element run is 8 consecutive features (16-B chunk)
                    first = coords[(8 * c, mi, ni)]
                    row, col = first[0], first[1]
                    lr, lc = row % ROWS, col % tile_n
                    row0, col0 = row - lr, col - lc
                    packed = tuple(
                        positive_zero32(
                            bf16x2(
                                tTR_rAcc[(8 * c + 2 * j, mi, ni)],
                                tTR_rAcc[(8 * c + 2 * j + 1, mi, ni)],
                            )
                        )
                        for j in range(4)
                    )
                    store_words(stage + lr * stride + lc * 2, packed, shared=True)
    threads = cutlass.const_expr(32 * len(gemm_kernel.epilogue_warp_id))
    pipeline.NamedBarrier(
        barrier_id=gemm_kernel.epilog_sync_bar_id, num_threads=threads
    ).arrive_and_wait()

    rank, n, cap, rows = words[1], words[2], words[3], words[4]
    chunks = cutlass.const_expr(tile_n // 8)  # 16-byte chunks per staged row
    for k in cutlass.range_constexpr(chunks * ROWS // threads):
        index = k * threads + epi_tidx
        lr, c = index // chunks, index % chunks
        row = row0 + lr
        if row < rows:
            owner = row // n
            offset = Int64((rank * cap + row - owner * n) * WIDTH + col0 + c * 8) * 2
            data = load_words(stage + lr * stride + c * 16, count=4, shared=True)
            store_relaxed(bases[4 + owner] + offset, data)
    return acc_consumer_state
