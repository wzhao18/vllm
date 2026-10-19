# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Raw latent conversion before Kimi-K3's context KV projection."""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_tokens"])
def _fp8_context_to_bf16(source, output, num_tokens, row_stride):
    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    cols = tl.arange(0, 512)
    values = tl.load(
        source + rows[:, None].to(tl.int64) * row_stride + cols[None, :],
        rows[:, None] < num_tokens,
        other=0.0,
    )
    tl.store(
        output + rows[:, None].to(tl.int64) * 512 + cols[None, :],
        values.to(tl.bfloat16),
        rows[:, None] < num_tokens,
    )


def cast_mla_context(source: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Cast the gathered latent without dequantization or extra output storage."""
    if (
        not source.is_cuda
        or source.dtype != torch.float8_e4m3fn
        or dtype != torch.bfloat16
        or source.ndim != 2
        or source.shape[1] != 512
        or source.stride(1) != 1
        or not current_platform.is_cuda()
        or not current_platform.supports_fp8()
    ):
        return source.to(dtype)

    # ATen's generic FP8-source conversion is costly for large context chunks.
    output = torch.empty(source.shape, dtype=dtype, device=source.device)
    num_tokens = source.shape[0]
    if num_tokens:
        _fp8_context_to_bf16[(triton.cdiv(num_tokens, 16),)](
            source, output, num_tokens, source.stride(0), num_warps=4
        )
    return output
