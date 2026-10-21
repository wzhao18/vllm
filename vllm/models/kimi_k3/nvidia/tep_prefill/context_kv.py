# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused MLA prefill context packing: paged FP8 latent -> per-head FP8 K and V.

The stock chunked-context path of a plain FP8 cache runs, per context chunk,
``cp_gather_cache`` (paged latent -> workspace), a cast of the FP8 latent to BF16,
the BF16 ``kv_b_proj`` GEMM (a BF16 [tokens, heads x 256] intermediate) and
``fused_mla_kv_concat_quant_fp8`` (K = [k_nope | k_pe] and V cast to FP8).

Here a gather pre-pass writes the BF16 latent (exact) and every head's rope part of
K, then two tcgen05 GEMMs (``cutedsl/gemm.py``, the K and the V rows of
``kv_b_proj``) write FP8 K and V straight from the accumulator: rounded to BF16 as
the stock GEMM output, then cast to FP8, so K and V match the stock path bit for
bit. No BF16 intermediate of the up-projection is materialized.
"""

from functools import cache

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

LATENT = 512
ROPE = 64
NOPE = 128
VDIM = 128
LATENT_C = tl.constexpr(LATENT)
ROPE_C = tl.constexpr(ROPE)
NOPE_C = tl.constexpr(NOPE)
# GEMM tiling: 2-CTA 256 x 256 tiles, rasterized along N so each latent tile is
# reused from L2 by every head.
MMA_TILER = (256, 256)
CLUSTER = (2, 1)


@triton.jit
def _gather_kernel(
    cache_ptr,
    block_table_ptr,
    cu_seq_lens_ptr,
    starts_ptr,
    token_to_seq_ptr,
    latent_ptr,
    k_ptr,
    num_tokens,
    num_heads,
    block_table_stride,
    cache_block_stride,
    cache_entry_stride,
    PAGE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Rows [m * BLOCK_M, +BLOCK_M): the BF16 latent, and k_pe into every head's K."""
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < num_tokens
    seq = tl.load(token_to_seq_ptr + rows, mask=row_mask, other=0)
    pos = tl.load(starts_ptr + seq, mask=row_mask, other=0) + (
        rows - tl.load(cu_seq_lens_ptr + seq, mask=row_mask, other=0)
    )
    block = tl.load(
        block_table_ptr + seq.to(tl.int64) * block_table_stride + pos // PAGE,
        mask=row_mask,
        other=0,
    )
    entry = (
        block.to(tl.int64) * cache_block_stride
        + (pos % PAGE).to(tl.int64) * cache_entry_stride
    )
    rows64 = rows.to(tl.int64)
    cols = tl.arange(0, LATENT_C)
    latent = tl.load(
        cache_ptr + entry[:, None] + cols[None, :], mask=row_mask[:, None], other=0.0
    )
    tl.store(
        latent_ptr + rows64[:, None] * LATENT_C + cols[None, :],
        latent.to(tl.bfloat16),
        mask=row_mask[:, None],
    )
    rope = tl.arange(0, ROPE_C)
    pe = tl.load(
        cache_ptr + entry[:, None] + LATENT_C + rope[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )
    for head in range(num_heads):
        k_row = (rows64 * num_heads + head) * (NOPE_C + ROPE_C) + NOPE_C
        tl.store(k_ptr + k_row[:, None] + rope[None, :], pe, mask=row_mask[:, None])


def _bf16_rounding(x):
    """The stock GEMM's BF16 output: round the Float32 accumulator to BF16."""
    return x.to(cutlass.BFloat16).to(cutlass.Float32)


@cute.jit
def _up_project(
    gemm: cutlass.Constexpr,
    latent: cute.Tensor,
    w_k: cute.Tensor,
    w_v: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    scale: cute.Tensor,
    max_active_clusters: cutlass.Constexpr,
    stream,
):
    """K[:, h, :128] = latent @ W_k[h]^T and V[:, h, :] = latent @ W_v[h]^T for all
    heads h, as two GEMMs over N = heads x 128 (MNL / NKL views for the kernel)."""
    tokens = latent.shape[0]
    heads = k.shape[1]
    a = cute.make_tensor(
        latent.iterator, cute.make_layout((tokens, LATENT, 1), stride=(LATENT, 1, 0))
    )
    b_k = cute.make_tensor(
        w_k.iterator,
        cute.make_layout((heads * NOPE, LATENT, 1), stride=(LATENT, 1, 0)),
    )
    b_v = cute.make_tensor(
        w_v.iterator,
        cute.make_layout((heads * VDIM, LATENT, 1), stride=(LATENT, 1, 0)),
    )
    # K is [tokens, heads, 128 + 64]: the GEMM's N mode is (128 columns, heads).
    c_k = cute.make_tensor(
        k.iterator,
        cute.make_layout(
            (tokens, (NOPE, heads), 1),
            stride=(heads * (NOPE + ROPE), (1, NOPE + ROPE), 0),
        ),
    )
    c_v = cute.make_tensor(
        v.iterator,
        cute.make_layout((tokens, heads * VDIM, 1), stride=(heads * VDIM, 1, 0)),
    )
    gemm(a, b_k, c_k, max_active_clusters, stream, scale, _bf16_rounding)
    gemm(a, b_v, c_v, max_active_clusters, stream, scale, _bf16_rounding)


@cache
def _compile(heads: int):
    from .cutedsl.gemm import PersistentDenseGemmKernel

    gemm = PersistentDenseGemmKernel(
        cutlass.Float32, True, MMA_TILER, CLUSTER, True, raster_along="n"
    )
    clusters = utils.HardwareInfo().get_max_active_clusters(CLUSTER[0] * CLUSTER[1])
    tokens = cute.sym_int()

    def fake(dtype, shape):
        order = tuple(reversed(range(len(shape))))  # row-major
        return cute.runtime.make_fake_compact_tensor(
            dtype, shape, stride_order=order, assumed_align=16
        )

    return cute.compile(
        _up_project,
        gemm,
        fake(cutlass.BFloat16, (tokens, LATENT)),
        fake(cutlass.BFloat16, (heads * NOPE, LATENT)),
        fake(cutlass.BFloat16, (heads * VDIM, LATENT)),
        fake(cutlass.Float8E4M3FN, (tokens, heads, NOPE + ROPE)),
        fake(cutlass.Float8E4M3FN, (tokens, heads * VDIM)),
        cute.runtime.make_fake_compact_tensor(cutlass.Float32, (1,), assumed_align=4),
        clusters,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


class ContextKv:
    """The fused context pack of one MLA layer (plain FP8 cache, FP8 prefill)."""

    def __init__(self, module):
        weight = module.kv_b_proj.weight
        heads = module.num_local_heads
        assert module.kv_lora_rank == LATENT and module.qk_rope_head_dim == ROPE
        assert module.qk_nope_head_dim == NOPE and module.v_head_dim == VDIM
        assert weight.dtype == torch.bfloat16 and weight.shape == (
            heads * (NOPE + VDIM),
            LATENT,
        ), (weight.dtype, weight.shape)
        rows = weight.detach().view(heads, NOPE + VDIM, LATENT)
        self.w_k = rows[:, :NOPE].reshape(heads * NOPE, LATENT).contiguous()
        self.w_v = rows[:, NOPE:].reshape(heads * VDIM, LATENT).contiguous()
        self.heads = heads
        self.one = torch.ones(1, dtype=torch.float32, device=weight.device)
        self.kernel = _compile(heads)

    def __call__(self, chunk, kv_cache: torch.Tensor, block_table: torch.Tensor):
        """FP8 (k [T, H, 192], v [T, H, 128]) of ``chunk``'s context tokens, as
        ``cp_gather_cache`` + ``kv_b_proj`` + ``fused_mla_kv_concat_quant_fp8``."""
        logger.info_once("Kimi K3 TEP prefill packs context with the fused kernel")
        tokens = chunk.num_context_tokens
        fp8 = torch.float8_e4m3fn
        device = kv_cache.device
        k = torch.empty((tokens, self.heads, NOPE + ROPE), dtype=fp8, device=device)
        v = torch.empty((tokens, self.heads, VDIM), dtype=fp8, device=device)
        if tokens == 0:
            return k, v
        cache = kv_cache.view(fp8)
        latent = torch.empty((tokens, LATENT), dtype=torch.bfloat16, device=device)
        block_m = 64
        _gather_kernel[(triton.cdiv(tokens, block_m),)](
            cache,
            block_table,
            chunk.cu_seq_lens,
            chunk.starts,
            chunk.token_to_seq,
            latent,
            k,
            tokens,
            self.heads,
            block_table.stride(0),
            cache.stride(0),
            cache.stride(1),
            PAGE=cache.shape[1],
            BLOCK_M=block_m,
            num_warps=4,
        )
        self.kernel(latent, self.w_k, self.w_v, k, v.view(tokens, -1), self.one)
        return k, v
