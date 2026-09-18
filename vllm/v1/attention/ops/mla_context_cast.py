# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 context KV preparation while preserving the projected-KV V view."""

import torch

from vllm.triton_utils import tl, triton


def can_fuse_context_cast(
    x: torch.Tensor, rope: torch.Tensor, output_dtype: torch.dtype | None
) -> bool:
    # Row-major non-overlapping inputs make the original .to() output contiguous.
    # Restrict dimensions and dtype to the explicitly validated conversion.
    return (
        output_dtype == torch.float8_e4m3fn
        and x.dtype == rope.dtype == torch.bfloat16
        and x.is_cuda
        and rope.is_cuda
        and x.device == rope.device
        and x.ndim == rope.ndim == 3
        and x.shape[0] > 0
        and x.shape[1] > 0
        and x.shape[2] == 256
        and rope.shape == (x.shape[0], 1, 64)
        and x.stride(2) > 0
        and x.stride(1) >= 256 * x.stride(2)
        and x.stride(0) >= x.shape[1] * x.stride(1)
        and rope.stride(2) > 0
        and rope.stride(0) >= 64 * rope.stride(2)
    )


@triton.jit
def fused_kernel(
    X,
    R,
    Y,
    K,
    N,
    H: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    X2: tl.constexpr,
    R0: tl.constexpr,
    R1: tl.constexpr,
    R2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    d = i % 256
    h = i // 256 % H
    t = i // (256 * H)
    valid = i < N
    x = tl.load(X + t * X0 + h * X1 + d * X2, valid, other=0)
    value = x.to(tl.float32).to(tl.float8e4nv)
    tl.store(Y + i, value, valid)
    tl.store(K + (t * H + h) * 192 + d, value, valid & (d < 128))
    r = tl.load(R + t * R0 + h * R1 + d * R2, valid & (d < 64), other=0)
    rope = r.to(tl.float32).to(tl.float8e4nv)
    tl.store(K + (t * H + h) * 192 + 128 + d, rope, valid & (d < 64))


def fused_context_cast_concat(
    x: torch.Tensor, rope: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert and concatenate, retaining the original V strides and alias."""
    t, h, _ = x.shape
    kv = torch.empty((t, h, 256), dtype=torch.float8_e4m3fn, device=x.device)
    k = torch.empty((t, h, 192), dtype=kv.dtype, device=x.device)
    expanded_rope = rope.expand(t, h, 64)
    fused_kernel[(triton.cdiv(x.numel(), 1024),)](
        x,
        expanded_rope,
        kv,
        k,
        x.numel(),
        h,
        *x.stride(),
        *expanded_rope.stride(),
        BLOCK=1024,
    )
    return k, kv[..., 128:]
