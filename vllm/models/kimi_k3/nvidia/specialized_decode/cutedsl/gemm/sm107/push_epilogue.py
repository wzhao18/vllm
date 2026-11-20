# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reduce-scatter push epilogue for the vendored sm_107 MXFP8 GEMM (``prewait`` kind
"pre", ``Sm107BlockScaledPersistentDenseGemmKernel(push=True)``): the SP
o-projection's partial rows go straight into their owners' Lamport receive slots
(``common.sp_comm.SpComm`` recv ring) instead of a local output tile, so the owner
kernel ``attn_res.sp_fused.SpReduceAttnRes`` runs with ``push=False`` (as after
``oproj.sp.SpOprojWarp``).

Per tile (one tile per CTA: ``prewait.Mxfp8PreGemm`` checks it): TMEM -> FP32 -> BF16
(RN, as the TMA-store epilogue; alpha is 1) staged at each value's own coordinate in a
token-major tile in the drained operand ring, then 16-byte chunks sanitized
(``positive_zero32``, as ``SpReduceAttnRes(push=True)``), consecutive lanes along
one row, stored relaxed at system scope into ``peers[o] + (rank cap + row - o n,
feature)``: the words
the owner receives are bit for bit those of the GEMM + push pair it replaces.

The push arguments ride in the GEMM's alpha buffer (float32 [ARGS]): [0] alpha 1.0;
int32 [1] rank, [2] n (rows per owner), [3] cap (rows per sender slot), [4] R (valid
rows); int64 [4 + o] owner o's receive base (``SpComm.views(parity)`` peers)."""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass import Float32, Int32, Int64

from ...common.primitives import asm, bf16x2, load_words, store_relaxed, store_words
from ...common.sp_comm import positive_zero32

ARGS = 16  # float32 words of the alpha / push-argument buffer
WIDTH = 7168  # output features (the hidden size): the receive slot's row pitch
ROWS = 128  # tile rows per CTA (the pre tactics: M128, or M256 over a CTA pair)
PAD = 16  # staging row padding (bytes)


def staging_bytes(tile_n: int) -> int:
    return ROWS * (2 * tile_n + PAD)


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


@cute.jit
def epilogue_push(gemm_kernel, epi_tidx: Int32, acc_tmem_ptr, mma_tile_coord_mnl,
                  cta_v: Int32, acc_consumer_state: pipeline.PipelineState,
                  acc_pipeline: pipeline.PipelineAsync, aux: cute.Tensor,
                  stage: Int32) -> pipeline.PipelineState:  # fmt: skip
    """One tile's push (module docstring). Thread (warp w, lane l) = TMEM lane 32 w + l
    = tile row 32 w + l of this CTA (M128 per CTA, as every pre tactic); its 16-column
    TMEM loads are BF16-packed into its row of the token-major staging tile, then the
    sanitized 16-byte chunks leave along rows (``acc_tmem_ptr``: the accumulator stage
    0 column; stage s is ``tile_n`` columns further, as the kernel allocates them)."""
    tile_n = cutlass.const_expr(gemm_kernel.mma_tiler[1])
    stride = cutlass.const_expr(2 * tile_n + PAD)
    words = cute.make_tensor(cute.recast_ptr(aux.iterator, dtype=Int32), cute.make_layout(ARGS))
    bases = cute.make_tensor(
        cute.recast_ptr(aux.iterator, dtype=Int64), cute.make_layout(ARGS // 2)
    )
    warp = epi_tidx // 32
    taddr = (
        Int32(acc_tmem_ptr.toint()) + acc_consumer_state.index * tile_n + ((32 * warp) << 16)
    )
    acc_pipeline.consumer_wait(acc_consumer_state)
    row_smem = stage + epi_tidx * stride
    for c0 in cutlass.range_constexpr(0, tile_n, 16):
        vals = tmem_wait_n(*tmem_ld16(taddr + c0))
        packed = tuple(bf16x2(vals[2 * j], vals[2 * j + 1]) for j in range(8))
        if cutlass.const_expr(gemm_kernel.push == 3):  # the bulk copies send smem as is
            packed = tuple(positive_zero32(w) for w in packed)
        store_words(row_smem + c0 * 2, packed[0:4], shared=True)
        store_words(row_smem + c0 * 2 + 16, packed[4:8], shared=True)
    with cute.arch.elect_one():
        acc_pipeline.consumer_release(acc_consumer_state)
    acc_consumer_state.advance()
    if cutlass.const_expr(gemm_kernel.push == 3):
        fence_proxy_async_smem()
    threads = cutlass.const_expr(32 * len(gemm_kernel.epilogue_warp_id))
    pipeline.NamedBarrier(
        barrier_id=gemm_kernel.epilog_sync_bar_id, num_threads=threads
    ).arrive_and_wait()

    rank, n, cap, rows = words[1], words[2], words[3], words[4]
    row0 = mma_tile_coord_mnl[0] * gemm_kernel.mma_tiler[0] + cta_v * ROWS
    col0 = mma_tile_coord_mnl[1] * tile_n
    chunks = cutlass.const_expr(tile_n // 8)  # 16-byte chunks per staged row
    if cutlass.const_expr(gemm_kernel.push == 3):
        # one bulk copy of this thread's staged row (2 tile_n bytes) to its owner slot
        row = row0 + epi_tidx
        if row < rows:
            owner = row // n
            offset = Int64((rank * cap + row - owner * n) * WIDTH + col0) * 2
            bulk_s2g(bases[4 + owner] + offset, stage + epi_tidx * stride, 2 * tile_n)
        bulk_commit_wait_read()
    for k in cutlass.range_constexpr(chunks * ROWS // threads if gemm_kernel.push < 3 else 0):
        index = k * threads + epi_tidx
        lr, c = index // chunks, index % chunks
        row = row0 + lr
        if row < rows:
            owner = row // n
            offset = Int64((rank * cap + row - owner * n) * WIDTH + col0 + c * 8) * 2
            data = load_words(stage + lr * stride + c * 16, count=4, shared=True)
            clean = tuple(positive_zero32(w) for w in data)
            if cutlass.const_expr(gemm_kernel.push == 2):
                store_words(bases[4 + owner] + offset, clean)  # weak: self-validating words
            else:
                store_relaxed(bases[4 + owner] + offset, clean)
    return acc_consumer_state

