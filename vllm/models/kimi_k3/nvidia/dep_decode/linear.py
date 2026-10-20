# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode-sized batches of an online-MXFP8 linear through the stream-K GEMM."""

import torch
from torch import nn


def fp8_rows(linear: nn.Module) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``(weight [N, K], swizzled scales)`` of an online-MXFP8 FlashInfer CuTe-DSL
    linear (weight stored as a ``[K, N]`` view of ``[N, K]``), else None."""
    weight = getattr(linear, "weight", None)
    scale = getattr(linear, "weight_scale", None)
    if weight is None or scale is None or weight.dtype != torch.float8_e4m3fn:
        return None
    weight = weight.t()
    n, k = weight.shape
    # Scales cover whole 128-row tiles (zero-padded past N).
    padded = -(-n // 128) * 128 * (k // 32)
    if not weight.is_contiguous() or scale.dim() != 1 or scale.numel() != padded:
        return None
    return weight, scale
