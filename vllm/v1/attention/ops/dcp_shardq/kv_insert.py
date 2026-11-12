# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MLA rows of a DCP ShardQ step's new tokens, into both of their caches.

A DCP ShardQ step writes each new token's MLA row (``kv_c`` then ``k_pe``,
quantized to the per-tensor FP8 cache) twice: into the DCP-local paged cache,
where only the tokens this rank owns have a slot, and into the gathered-context
workspace, where every token has one. ``concat_and_cache_mla`` does each write
as its own launch of one CTA per token that converts one element per thread
(~110 us per DSA layer at 17K tokens for ~10 MB of rows). This kernel reads
each token's row once, converts it once and stores it to both slots with
16-byte stores: the arithmetic is ``concat_and_cache_mla``'s (bf16 -> fp32,
IEEE division by the scale, saturating round-to-nearest E4M3), so the bytes
are identical.
"""

from dataclasses import dataclass
from typing import Any

import torch

from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    LaunchSpec,
    TritonWarmupTensor,
    VllmTritonJitKernel,
)
from vllm.triton_utils import tl, triton


class ShardQKvInsertKernel(VllmTritonJitKernel["ShardQKvInsertKernel.CompileKey"]):
    """``kv_c`` [T, NOPE] and ``k_pe`` [T, ROPE] (16-bit) -> the FP8 rows of
    their local slots (``local_slots``, -1: not this rank's) and workspace
    slots (``ws_slots`` for tokens from ``ws_first`` on)."""

    TOKENS = 4

    @dataclass(frozen=True)
    class CompileKey:
        src_dtype: torch.dtype
        nope: int
        rope: int

    @staticmethod
    # The strides keep Triton's divisibility specialization (16-byte vector
    # accesses); the token counts do not specialize.
    @triton.jit(do_not_specialize=["num_tokens", "ws_first"])
    def shardq_kv_insert_kernel(
        kv_c,
        k_pe,
        scale,
        local_cache,
        local_slots,
        ws_cache,
        ws_slots,
        num_tokens,
        ws_first,
        kv_c_stride,
        k_pe_stride,
        local_block_size,
        local_block_stride,
        local_entry_stride,
        NOPE: tl.constexpr,
        ROPE: tl.constexpr,
        TOKENS: tl.constexpr,
    ):
        row = NOPE + ROPE
        tokens = tl.program_id(0) * TOKENS + tl.arange(0, TOKENS)
        live = tokens < num_tokens
        tok64 = tokens.to(tl.int64)
        s = tl.load(scale).to(tl.float32)
        local = tl.load(local_slots + tok64, mask=live, other=-1)
        has_local = live & (local >= 0)
        local_row = (local // local_block_size) * local_block_stride + (
            local % local_block_size
        ) * local_entry_stride
        has_ws = live & (tokens >= ws_first)
        ws = tl.load(ws_slots + (tok64 - ws_first), mask=has_ws, other=0)
        ws_row = ws * row

        cols = tl.arange(0, NOPE)[None, :]
        x = tl.load(
            kv_c + tok64[:, None] * kv_c_stride + cols, mask=live[:, None], other=0.0
        ).to(tl.float32)
        y = tl.fdiv(x, s, ieee_rounding=True).to(local_cache.dtype.element_ty)
        tl.store(local_cache + local_row[:, None] + cols, y, mask=has_local[:, None])
        tl.store(ws_cache + ws_row[:, None] + cols, y, mask=has_ws[:, None])

        cols = tl.arange(0, ROPE)[None, :]
        x = tl.load(
            k_pe + tok64[:, None] * k_pe_stride + cols, mask=live[:, None], other=0.0
        ).to(tl.float32)
        y = tl.fdiv(x, s, ieee_rounding=True).to(local_cache.dtype.element_ty)
        tl.store(
            local_cache + local_row[:, None] + NOPE + cols, y, mask=has_local[:, None]
        )
        tl.store(ws_cache + ws_row[:, None] + NOPE + cols, y, mask=has_ws[:, None])

    # A descriptive CUDA symbol for traces.
    kernel = shardq_kv_insert_kernel

    def dispatch(  # type: ignore[override]
        self, *, src_dtype: torch.dtype, nope: int, rope: int
    ) -> CompileKey:
        return self.CompileKey(src_dtype=src_dtype, nope=nope, rope=rope)

    def get_warmup_keys(  # type: ignore[override]
        self, *, src_dtype: torch.dtype, nope: int, rope: int
    ) -> list[CompileKey]:
        return [self.dispatch(src_dtype=src_dtype, nope=nope, rope=rope)]

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        nope, rope = compile_key.nope, compile_key.rope
        return dict(
            kv_c=TritonWarmupTensor(compile_key.src_dtype, shape=(2, nope)),
            k_pe=TritonWarmupTensor(compile_key.src_dtype, shape=(2, rope)),
            scale=TritonWarmupTensor(torch.float32),
            local_cache=TritonWarmupTensor(torch.float8_e4m3fn),
            local_slots=TritonWarmupTensor(torch.int64),
            ws_cache=TritonWarmupTensor(torch.float8_e4m3fn),
            ws_slots=TritonWarmupTensor(torch.int64),
            num_tokens=2,
            ws_first=0,
            # The deployed strides' specialization: multiples of 16.
            kv_c_stride=2112,
            k_pe_stride=2112,
            local_block_size=64,
            local_block_stride=64 * (nope + rope),
            local_entry_stride=nope + rope,
        )

    @kernel_launcher
    def __call__(
        self,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        scale: torch.Tensor,
        local_cache: torch.Tensor,
        local_slots: torch.Tensor,
        ws_cache: torch.Tensor,
        ws_slots: torch.Tensor,
        num_tokens: int,
        ws_first: int,
        kv_c_stride: int,
        k_pe_stride: int,
        local_block_size: int,
        local_block_stride: int,
        local_entry_stride: int,
    ) -> LaunchSpec:
        nope = kv_c.shape[-1]
        rope = k_pe.shape[-1]
        return (triton.cdiv(num_tokens, self.TOKENS),), dict(
            NOPE=nope, ROPE=rope, TOKENS=self.TOKENS, num_warps=4
        )


SHARDQ_KV_INSERT_KERNEL = ShardQKvInsertKernel()


def shardq_kv_insert_supported(
    kv_c: torch.Tensor, k_pe: torch.Tensor, kv_cache: torch.Tensor
) -> bool:
    """Whether ``shardq_kv_insert`` takes these rows and this cache: 16-bit
    rows with unit inner stride and power-of-two widths, a one-byte E4M3 cache
    of rows of exactly their width."""
    nope = kv_c.shape[-1]
    rope = k_pe.shape[-1]
    return (
        kv_c.dtype in (torch.bfloat16, torch.float16)
        and k_pe.dtype == kv_c.dtype
        and kv_c.stride(-1) == 1
        and k_pe.stride(-1) == 1
        and nope > 0
        and rope > 0
        and not nope & (nope - 1)
        and not rope & (rope - 1)
        and kv_cache.dtype == torch.float8_e4m3fn
        and kv_cache.dim() == 3
        and kv_cache.shape[-1] == nope + rope
        and kv_cache.stride(-1) == 1
    )


def shardq_kv_insert(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    scale: torch.Tensor,
    local_cache: torch.Tensor,
    local_slots: torch.Tensor,
    ws_cache: torch.Tensor,
    ws_slots: torch.Tensor,
    ws_first: int,
) -> None:
    """Write the FP8 MLA rows of tokens ``[0, local_slots.numel())`` into
    ``local_cache`` ([blocks, block_size, row], slot -1 skipped) and those of
    tokens from ``ws_first`` on into the dense workspace ``ws_cache``
    ([pages, 64, row], at ``ws_slots``) in one launch, bitwise as two
    ``concat_and_cache_mla`` calls (``k_pe`` may be [T, 1, R])."""
    num_tokens = local_slots.numel()
    if num_tokens == 0:
        return
    k_pe = k_pe.reshape(k_pe.shape[0], k_pe.shape[-1]) if k_pe.dim() == 3 else k_pe
    assert ws_slots.numel() == num_tokens - ws_first
    assert ws_cache.is_contiguous() or (
        ws_cache.stride(1) == ws_cache.shape[2]
        and ws_cache.stride(0) == ws_cache.shape[1] * ws_cache.shape[2]
    )
    SHARDQ_KV_INSERT_KERNEL(
        kv_c,
        k_pe,
        scale,
        local_cache,
        local_slots,
        ws_cache,
        ws_slots,
        num_tokens,
        ws_first,
        kv_c.stride(0),
        k_pe.stride(0),
        local_cache.shape[1],
        local_cache.stride(0),
        local_cache.stride(1),
    )
