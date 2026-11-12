# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The FP8 sparse-MLA query of a DCP ShardQ token slice, in one kernel.

``forward_mqa_shardq_proj`` runs W_UK on the token slice with every head and
then needs cat(ql_nope, q_pe) quantized to FP8 with the layer's static
per-tensor scale, zero-padded to the MQA kernel's heads. This is the MLA
layer's cat + ``scaled_fp8_quant`` (x * (1 / scale) with an IEEE reciprocal,
clamped to the E4M3 range, round-to-nearest) and the head pad, bit for bit,
as FlashInfer's decode query (``decode_query_quant_pad``) does it, warmed with
the ShardQ kernels.
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


class ShardQQueryQuantKernel(VllmTritonJitKernel["ShardQQueryQuantKernel.CompileKey"]):
    HEADS_PER_PROGRAM = 16

    @dataclass(frozen=True)
    class CompileKey:
        src_dtype: torch.dtype
        nope: int
        rope: int

    @staticmethod
    # The strides keep Triton's divisibility specialization (16-byte vector
    # accesses); the head counts do not specialize.
    @triton.jit(do_not_specialize=["num_heads", "padded_heads"])
    def shardq_query_quant_kernel(
        nope,
        pe,
        scale,
        out,
        nope_stride0,
        nope_stride1,
        pe_stride0,
        pe_stride1,
        num_heads,
        padded_heads,
        NOPE: tl.constexpr,
        ROPE: tl.constexpr,
        HEADS_PER_PROGRAM: tl.constexpr,
    ):
        token = tl.program_id(0).to(tl.int64)
        heads = tl.program_id(1) * HEADS_PER_PROGRAM + tl.arange(0, HEADS_PER_PROGRAM)
        live = (heads < num_heads)[:, None]
        stored = (heads < padded_heads)[:, None]
        inv_scale = tl.fdiv(1.0, tl.load(scale).to(tl.float32), ieee_rounding=True)
        fp8_max: tl.constexpr = 448.0
        row = out + (token * padded_heads + heads[:, None]) * (NOPE + ROPE)

        cols = tl.arange(0, NOPE)[None, :]
        x = tl.load(
            nope + token * nope_stride0 + heads[:, None] * nope_stride1 + cols,
            mask=live,
            other=0.0,
        ).to(tl.float32)
        x = tl.minimum(tl.maximum(x * inv_scale, -fp8_max), fp8_max)
        tl.store(row + cols, x.to(out.dtype.element_ty), mask=stored)

        cols = tl.arange(0, ROPE)[None, :]
        x = tl.load(
            pe + token * pe_stride0 + heads[:, None] * pe_stride1 + cols,
            mask=live,
            other=0.0,
        ).to(tl.float32)
        x = tl.minimum(tl.maximum(x * inv_scale, -fp8_max), fp8_max)
        tl.store(row + NOPE + cols, x.to(out.dtype.element_ty), mask=stored)

    # A descriptive CUDA symbol for traces.
    kernel = shardq_query_quant_kernel

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
            # The token slice's strides: [T, H, L] contiguous and the rope part of
            # the exchanged [T, H, P + R] query (multiples of 16).
            ql_nope=TritonWarmupTensor(compile_key.src_dtype, shape=(1, 16, nope)),
            q_pe=TritonWarmupTensor(
                compile_key.src_dtype, shape=(1, 16, rope), strides=(16 * 192, 192, 1)
            ),
            scale=TritonWarmupTensor(torch.float32),
            out=TritonWarmupTensor(torch.float8_e4m3fn, shape=(1, 1, nope + rope)),
        )

    @kernel_launcher
    def __call__(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        scale: torch.Tensor,
        out: torch.Tensor,
    ) -> LaunchSpec:
        num_tokens, num_heads, nope = ql_nope.shape
        padded_heads = out.shape[1]
        grid = (max(num_tokens, 1), triton.cdiv(padded_heads, self.HEADS_PER_PROGRAM))
        return grid, dict(
            nope=ql_nope,
            pe=q_pe,
            nope_stride0=ql_nope.stride(0),
            nope_stride1=ql_nope.stride(1),
            pe_stride0=q_pe.stride(0),
            pe_stride1=q_pe.stride(1),
            num_heads=num_heads,
            padded_heads=padded_heads,
            NOPE=nope,
            ROPE=q_pe.shape[-1],
            HEADS_PER_PROGRAM=self.HEADS_PER_PROGRAM,
            num_warps=4,
        )


SHARDQ_QUERY_QUANT_KERNEL = ShardQQueryQuantKernel()


def shardq_query_quant_pad(
    ql_nope: torch.Tensor, q_pe: torch.Tensor, scale: torch.Tensor, padded_heads: int
) -> torch.Tensor | None:
    """cat(``ql_nope`` [T, H, L], ``q_pe`` [T, H, R]) quantized to E4M3 with the
    static per-tensor ``scale`` into a new [T, ``padded_heads``, L + R] buffer
    whose extra heads are zero; None outside the kernel's shapes (16-bit
    inputs, power-of-two L and R, unit inner stride, one scale)."""
    num_tokens, num_heads, nope = ql_nope.shape
    rope = q_pe.shape[-1]
    if (
        ql_nope.dtype not in (torch.bfloat16, torch.float16)
        or q_pe.dtype != ql_nope.dtype
        or q_pe.shape[:2] != ql_nope.shape[:2]
        or ql_nope.stride(2) != 1
        or q_pe.stride(2) != 1
        or nope & (nope - 1)
        or rope <= 0
        or rope & (rope - 1)
        or padded_heads < num_heads
        or scale.numel() != 1
    ):
        return None
    out = ql_nope.new_empty(
        (num_tokens, padded_heads, nope + rope), dtype=torch.float8_e4m3fn
    )
    if num_tokens:
        SHARDQ_QUERY_QUANT_KERNEL(ql_nope, q_pe, scale, out)
    return out
