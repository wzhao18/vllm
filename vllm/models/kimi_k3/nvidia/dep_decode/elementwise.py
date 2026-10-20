# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused producers of the DEP decode MXFP8 GEMM inputs.

Each kernel computes an activation the stock path materializes in BF16 and
quantizes it to MXFP8 in the same pass, writing the UE8M0 scales straight into
the F8_128x4 swizzled layout the GEMMs read (``swizzle_mxfp8_scale``). Numerics
follow the stock path exactly: FP32 math, the BF16 rounding of the stock output,
then ``ceil(log2(amax / 448))`` block scales.
"""

import torch

from vllm.triton_utils import tl, triton

FP8_MAX = tl.constexpr(448.0)
BLOCK = 32  # MX block


def swizzled_scale_numel(rows: int, cols: int) -> int:
    k_blocks = cols // BLOCK
    return triton.cdiv(rows, 128) * 128 * triton.cdiv(k_blocks, 4) * 4


@triton.jit
def _mx_block(y):
    """MX blocks of y (rows of 32): UE8M0 exponents ``ceil(log2(amax / 448))``
    (biased) and the E4M3 values. Scales by the reciprocal power of two, which
    stays a normal float for every exponent (dividing by a denormal 2^-127 scale
    flushes to 0/0 on all-zero blocks), and saturates at the E4M3 range."""
    amax = tl.maximum(tl.max(tl.abs(y), axis=1), 1.1754943508222875e-38)
    exp = tl.ceil(tl.log2(amax / 448.0))
    exp = tl.minimum(tl.maximum(exp + 127.0, 0.0), 254.0)
    q = y * tl.exp2(127.0 - tl.maximum(exp, 1.0))[:, None]
    q = tl.where(exp[:, None] == 0.0, q * 2.0, q)
    q = tl.minimum(tl.maximum(q, -FP8_MAX), FP8_MAX).to(tl.float8e4nv)
    return q, exp


@triton.jit
def _quantize_store(y, offs_m, pid_b, m_mask, q_ptr, s_ptr, stride_qm, k_tiles):
    """Quantize a [BLOCK_M, 32] FP32 tile (already BF16-rounded) as one MX block."""
    q, exp = _mx_block(y)
    offs_k = pid_b * 32 + tl.arange(0, 32)
    tl.store(
        q_ptr + offs_m[:, None] * stride_qm + offs_k[None, :], q, mask=m_mask[:, None]
    )
    # F8_128x4: (m_tile, k_tile, row % 32, (row // 32) % 4, block % 4).
    m_tile = offs_m // 128
    i32 = offs_m % 32
    i4 = (offs_m // 32) % 4
    k_tile = pid_b // 4
    c4 = pid_b % 4
    s_off = (((m_tile * k_tiles + k_tile) * 32 + i32) * 4 + i4) * 4 + c4
    tl.store(s_ptr + s_off, exp.to(tl.uint8), mask=m_mask)


@triton.jit
def _situ_and_mul_mxfp8_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    M,
    D,
    stride_xm,
    stride_qm,
    k_tiles,
    beta,
    linear_beta,
    HAS_LINEAR_BETA: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_b * 32 + tl.arange(0, 32)
    m_mask = offs_m < M
    row = x_ptr + offs_m[:, None] * stride_xm
    gate = tl.load(row + offs_k[None, :], mask=m_mask[:, None], other=0.0).to(
        tl.float32
    )
    up = tl.load(row + D + offs_k[None, :], mask=m_mask[:, None], other=0.0).to(
        tl.float32
    )
    # beta * tanh(gate / beta) * sigmoid(gate), tanh(z) = 2 * sigmoid(2z) - 1.
    gate = beta * (2.0 * tl.sigmoid(2.0 * gate / beta) - 1.0) * tl.sigmoid(gate)
    if HAS_LINEAR_BETA:
        up = linear_beta * (2.0 * tl.sigmoid(2.0 * up / linear_beta) - 1.0)
    y = (gate * up).to(tl.bfloat16).to(tl.float32)
    _quantize_store(y, offs_m, pid_b, m_mask, q_ptr, s_ptr, stride_qm, k_tiles)


@triton.jit
def _rms_norm_mxfp8_kernel(
    x_ptr,
    w_ptr,
    q_ptr,
    s_ptr,
    M,
    H,
    stride_xm,
    stride_qm,
    k_tiles,
    eps,
    BLOCK_H: tl.constexpr,
):
    # One program per token: the RMS needs the whole row.
    m = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H
    x = tl.load(x_ptr + m * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / H + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = ((x * rstd).to(tl.bfloat16).to(tl.float32) * w).to(tl.bfloat16).to(tl.float32)
    y = tl.reshape(y, (BLOCK_H // 32, 32))
    q, exp = _mx_block(y)
    q = tl.reshape(q, (BLOCK_H,))
    tl.store(q_ptr + m * stride_qm + offs, q, mask=mask)
    blocks = tl.arange(0, BLOCK_H // 32)
    s_off = (
        (((m // 128) * k_tiles + blocks // 4) * 32 + m % 32) * 4 + (m // 32) % 4
    ) * 4 + blocks % 4
    tl.store(s_ptr + s_off, exp.to(tl.uint8), mask=blocks < H // 32)


@triton.jit
def _gated_rms_norm_mxfp8_kernel(
    x_ptr,
    g_ptr,
    w_ptr,
    q_ptr,
    s_ptr,
    stride_xm,
    stride_gm,
    stride_qm,
    k_tiles,
    eps,
    D: tl.constexpr,
    HEADS: tl.constexpr,
):
    # One program per (token, HEADS heads): RMS per head, then D / 32 MX blocks each.
    m = tl.program_id(0)
    heads = tl.program_id(1) * HEADS + tl.arange(0, HEADS)
    offs = heads[:, None] * D + tl.arange(0, D)[None, :]
    x = tl.load(x_ptr + m * stride_xm + offs).to(tl.float32)
    rstd = 1 / tl.sqrt(tl.sum(x * x, axis=1) / D + eps)
    w = tl.load(w_ptr + tl.arange(0, D)).to(tl.float32)
    g = tl.load(g_ptr + m * stride_gm + offs).to(tl.float32)
    y = x * rstd[:, None] * w[None, :] * tl.sigmoid(g)
    y = tl.reshape(y.to(tl.bfloat16).to(tl.float32), (HEADS * D // 32, 32))
    q, exp = _mx_block(y)
    tl.store(q_ptr + m * stride_qm + offs, tl.reshape(q, (HEADS, D)))
    blocks = tl.program_id(1) * (HEADS * D // 32) + tl.arange(0, HEADS * D // 32)
    s_off = (
        (((m // 128) * k_tiles + blocks // 4) * 32 + m % 32) * 4 + (m // 32) % 4
    ) * 4 + blocks % 4
    tl.store(s_ptr + s_off, exp.to(tl.uint8))


def gated_rms_norm_mxfp8(
    x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """KDA's per-head ``FusedRMSNormGated`` (sigmoid gate; the stock kernel's
    order and BF16 output rounding) then MXFP8 quantization. ``x`` and ``gate``
    are [T, heads * D] (``gate`` may be row strided), ``weight`` is [D]."""
    m, h = x.shape
    d = weight.numel()
    assert x.stride(1) == 1 and gate.stride(1) == 1 and h % d == 0 and d % 32 == 0
    q = torch.empty(m, h, dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty(swizzled_scale_numel(m, h), dtype=torch.uint8, device=x.device)
    heads = 8 if (h // d) % 8 == 0 else 1
    _gated_rms_norm_mxfp8_kernel[(m, h // d // heads)](
        x,
        gate,
        weight,
        q,
        s,
        x.stride(0),
        gate.stride(0),
        q.stride(0),
        triton.cdiv(h // BLOCK, 4),
        float(eps),
        D=d,
        HEADS=heads,
        num_warps=4,
    )
    return q, s


@triton.jit
def _sigmoid_gate_mxfp8_kernel(
    x_ptr,
    g_ptr,
    q_ptr,
    s_ptr,
    stride_xm,
    stride_gm,
    stride_qm,
    k_tiles,
    BLOCKS: tl.constexpr,
):
    # One program per (token, BLOCKS MX blocks).
    m = tl.program_id(0)
    offs = tl.program_id(1) * BLOCKS * 32 + tl.arange(0, BLOCKS * 32)
    x = tl.load(x_ptr + m * stride_xm + offs).to(tl.float32)
    g = tl.load(g_ptr + m * stride_gm + offs).to(tl.float32)
    y = (x * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
    y = tl.reshape(y, (BLOCKS, 32))
    q, exp = _mx_block(y)
    tl.store(q_ptr + m * stride_qm + offs, tl.reshape(q, (BLOCKS * 32,)))
    blocks = tl.program_id(1) * BLOCKS + tl.arange(0, BLOCKS)
    s_off = (
        (((m // 128) * k_tiles + blocks // 4) * 32 + m % 32) * 4 + (m // 32) % 4
    ) * 4 + blocks % 4
    tl.store(s_ptr + s_off, exp.to(tl.uint8))


def sigmoid_gate_mxfp8(
    x: torch.Tensor, gate: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """MLA's sigmoid output gate ``x * sigmoid(gate)`` (FP32 math, one BF16
    rounding) then MXFP8 quantization; ``gate`` may be row strided."""
    m, h = x.shape
    blocks = 32
    assert x.stride(1) == 1 and gate.stride(1) == 1 and h % (blocks * 32) == 0
    q = torch.empty(m, h, dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty(swizzled_scale_numel(m, h), dtype=torch.uint8, device=x.device)
    _sigmoid_gate_mxfp8_kernel[(m, h // (blocks * 32))](
        x,
        gate,
        q,
        s,
        x.stride(0),
        gate.stride(0),
        q.stride(0),
        triton.cdiv(h // BLOCK, 4),
        BLOCKS=blocks,
        num_warps=4,
    )
    return q, s


def situ_and_mul_mxfp8(
    gate_up: torch.Tensor, beta: float, linear_beta: float | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """``SituAndMul`` then MXFP8 quantization; ``gate_up`` may be a row-strided view."""
    m, two_d = gate_up.shape
    d = two_d // 2
    assert gate_up.stride(1) == 1 and d % 128 == 0
    q = torch.empty(m, d, dtype=torch.float8_e4m3fn, device=gate_up.device)
    s = torch.empty(
        swizzled_scale_numel(m, d), dtype=torch.uint8, device=gate_up.device
    )
    block_m = 16
    grid = (triton.cdiv(m, block_m), d // BLOCK)
    _situ_and_mul_mxfp8_kernel[grid](
        gate_up,
        q,
        s,
        m,
        d,
        gate_up.stride(0),
        q.stride(0),
        triton.cdiv(d // BLOCK, 4),
        float(beta),
        float(linear_beta or 0.0),
        HAS_LINEAR_BETA=linear_beta is not None,
        BLOCK_M=block_m,
    )
    return q, s


def rms_norm_mxfp8(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """``RMSNorm`` (vLLM's BF16 rounding order) then MXFP8 quantization."""
    m, h = x.shape
    assert x.stride(1) == 1 and h % 128 == 0
    q = torch.empty(m, h, dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty(swizzled_scale_numel(m, h), dtype=torch.uint8, device=x.device)
    _rms_norm_mxfp8_kernel[(m,)](
        x,
        weight,
        q,
        s,
        m,
        h,
        x.stride(0),
        q.stride(0),
        triton.cdiv(h // BLOCK, 4),
        float(eps),
        BLOCK_H=triton.next_power_of_2(h),
    )
    return q, s
