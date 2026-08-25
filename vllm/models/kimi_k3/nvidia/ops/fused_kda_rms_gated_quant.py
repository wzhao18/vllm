# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

_NUM_HEADS = 96
_HEAD_DIM = 128
_BLOCK_ROWS = 16


@triton.jit
def _fused_kda_rms_gated_quant_kernel(
    x_ptr,
    gate_ptr,
    weight_ptr,
    output_ptr,
    scale_ptr,
    eps,
    tma_aligned_m,
    gate_stride_m,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_rows: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    row_start = tl.program_id(0) * block_rows
    rows = row_start + tl.arange(0, block_rows)
    cols = tl.arange(0, head_dim)
    offsets = rows[:, None] * head_dim + cols[None, :]
    gate_offsets = (
        (rows // num_heads)[:, None] * gate_stride_m
        + (rows % num_heads)[:, None] * head_dim
        + cols[None, :]
    )

    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    x = tl.load(x_ptr + offsets).to(tl.float32)
    gate = tl.load(gate_ptr + gate_offsets).to(tl.float32)
    weight = tl.load(weight_ptr + cols).to(tl.float32)
    variance = tl.sum(x * x, axis=1) / head_dim
    output = x * (1.0 / tl.sqrt(variance[:, None] + eps)) * weight[None, :]
    output *= tl.sigmoid(gate)

    # Match the existing norm -> quant path's intermediate BF16 rounding.
    output = output.to(tl.bfloat16).to(tl.float32)
    absmax = tl.max(tl.abs(output), axis=1)
    raw_scale = tl.maximum(absmax / 448.0, 1e-10)
    scale_exp = tl.ceil(tl.log2(raw_scale)).to(tl.int32) + 127
    scale = tl.exp2((scale_exp - 127).to(tl.float32))
    quantized = tl.clamp(output / scale[:, None], -448.0, 448.0)
    tl.store(output_ptr + offsets, quantized)

    scale_exp = tl.reshape(scale_exp, (block_rows // 4, 4))
    shifts = tl.arange(0, 4) * 8
    packed_scale = tl.sum(scale_exp << shifts[None, :], axis=1)
    token = row_start // num_heads
    scale_group_start = (row_start % num_heads) // 4
    scale_groups = scale_group_start + tl.arange(0, block_rows // 4)
    tl.store(scale_ptr + scale_groups * tma_aligned_m + token, packed_scale)


def _allocate_outputs(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m = x.numel() // (_NUM_HEADS * _HEAD_DIM)
    tma_aligned_m = triton.cdiv(m, 4) * 4
    output = torch.empty(
        (m, _NUM_HEADS * _HEAD_DIM),
        dtype=current_platform.fp8_dtype(),
        device=x.device,
    )
    scale = torch.empty_strided(
        (m, _NUM_HEADS // 4),
        (1, tma_aligned_m),
        dtype=torch.int32,
        device=x.device,
    )
    return output, scale


def _fused_kda_rms_gated_quant(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    output, scale = _allocate_outputs(x)
    m = output.shape[0]
    if m == 0:
        return output, scale
    use_small_tile = m <= 42 or 51 <= m <= 63
    block_rows = 4 if use_small_tile else _BLOCK_ROWS
    num_warps = 2 if use_small_tile else 8
    _fused_kda_rms_gated_quant_kernel[(m * _NUM_HEADS // block_rows,)](
        x,
        gate,
        weight,
        output,
        scale,
        eps,
        scale.stride(1),
        gate.stride(-3),
        _NUM_HEADS,
        _HEAD_DIM,
        block_rows,
        launch_pdl=current_platform.is_arch_support_pdl(),
        num_warps=num_warps,
        num_stages=1,
    )
    return output, scale


def _fused_kda_rms_gated_quant_fake(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _allocate_outputs(x)


direct_register_custom_op(
    "fused_kimi_k3_kda_rms_gated_quant",
    _fused_kda_rms_gated_quant,
    fake_impl=_fused_kda_rms_gated_quant_fake,
)


def fused_kda_rms_gated_quant(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.shape[-2:] != (_NUM_HEADS, _HEAD_DIM):
        raise ValueError(f"Expected x[...,96,128], got {tuple(x.shape)}")
    if gate.shape[-2:] != (_NUM_HEADS, _HEAD_DIM):
        raise ValueError(f"Expected gate[...,96,128], got {tuple(gate.shape)}")
    if weight.shape != (_HEAD_DIM,):
        raise ValueError(f"Expected weight[128], got {tuple(weight.shape)}")
    if not x.is_contiguous() or gate.stride(-1) != 1:
        raise ValueError("KDA output and gate must be contiguous by head")
    return torch.ops.vllm.fused_kimi_k3_kda_rms_gated_quant(
        x,
        gate,
        weight,
        eps,
    )
