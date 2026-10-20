# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode-sized batches of an online-MXFP8 linear through the stream-K GEMM."""

import torch
from torch import nn

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
)

from .gemm import MAX_TILE_N, streamk_mxfp8_gemm


def fp8_rows(linear: nn.Module) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``(weight [N, K], swizzled scales)`` of an online-MXFP8 FlashInfer CuTe-DSL
    linear (weight stored as a ``[K, N]`` view of ``[N, K]``), else None."""
    weight = getattr(linear, "weight", None)
    scale = getattr(linear, "weight_scale", None)
    if weight is None or scale is None or weight.dtype != torch.float8_e4m3fn:
        return None
    weight = weight.t()
    if not weight.is_contiguous() or scale.dim() != 1 or weight.shape[0] % 128:
        return None
    return weight, scale


def install_streamk_forward(linear: nn.Module) -> bool:
    """Run ``linear`` on up to ``MAX_TILE_N`` tokens through the stream-K GEMM.

    The stock forward still serves larger batches. Returns False (and changes
    nothing) for a linear in another format.
    """
    rows = fp8_rows(linear)
    if rows is None or getattr(linear, "bias", None) is not None:
        return False
    weight, scale = rows
    stock_forward = linear.forward

    def forward(x: torch.Tensor):
        if x.dim() == 2 and 0 < x.shape[0] <= MAX_TILE_N:
            xq, xs = mxfp8_e4m3_quantize(x.contiguous(), is_sf_swizzled_layout=True)
            return streamk_mxfp8_gemm(xq, xs, weight, scale), None
        return stock_forward(x)

    linear.forward = forward
    return True
