# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _fused_k128_fp8_gemm_kernel(
    a_ptr,
    b_ptr,
    b_scale_ptr,
    output_ptr,
    m,
    n: tl.constexpr,
    stride_am,
    stride_bk,
    stride_bn,
    stride_bsn,
    stride_om,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
) -> None:
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offsets_m = pid_m * block_m + tl.arange(0, block_m)
    offsets_n = pid_n * block_n + tl.arange(0, block_n)
    offsets_k = tl.arange(0, 128)
    mask_m = offsets_m < m
    mask_n = offsets_n < n

    a = tl.load(
        a_ptr + offsets_m[:, None] * stride_am + offsets_k[None, :],
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(a), axis=1) / 448.0, 1e-10)
    scale = tl.exp2(tl.ceil(tl.log2(scale)))
    a_q = tl.clamp(a / scale[:, None], -448.0, 448.0).to(b_ptr.dtype.element_ty)
    b = tl.load(
        b_ptr + offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn,
        mask=mask_n[None, :],
        other=0.0,
    )
    b_scale = tl.load(
        b_scale_ptr + (offsets_n // 128) * stride_bsn,
        mask=mask_n,
        other=0.0,
    )
    output = tl.dot(a_q, b).to(tl.float32)
    output *= scale[:, None] * b_scale[None, :]
    tl.store(
        output_ptr + offsets_m[:, None] * stride_om + offsets_n[None, :],
        output,
        mask=mask_m[:, None] & mask_n[None, :],
    )


def _fused_k128_fp8_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    m = x.shape[0]
    block_m = 4 if m <= 2 else 16
    num_warps = 2 if m == 64 else 4
    block_n = 64
    output = torch.empty(
        (m, weight.shape[0]),
        dtype=x.dtype,
        device=x.device,
    )
    grid = (triton.cdiv(m, block_m), triton.cdiv(weight.shape[0], block_n))
    _fused_k128_fp8_gemm_kernel[grid](
        x,
        weight,
        weight_scale,
        output,
        m,
        weight.shape[0],
        x.stride(0),
        weight.stride(1),
        weight.stride(0),
        weight_scale.stride(0),
        output.stride(0),
        block_m,
        block_n,
        num_warps=num_warps,
        num_stages=1,
    )
    return output


def _fused_k128_fp8_gemm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        (x.shape[0], weight.shape[0]),
        dtype=x.dtype,
        device=x.device,
    )


direct_register_custom_op(
    "fused_kimi_k3_k128_fp8_gemm",
    _fused_k128_fp8_gemm,
    fake_impl=_fused_k128_fp8_gemm_fake,
)


def fused_k128_fp8_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.vllm.fused_kimi_k3_k128_fp8_gemm(x, weight, weight_scale)
