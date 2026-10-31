# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row packing of online-MXFP8 linears that read the same input.

MXFP8 weights are quantized per row in blocks of 32 along K, so stacking the
rows of several MXFP8 linears (and their E8M0 scale rows) gives exactly the
MXFP8 weight of the concatenated linear: one GEMM of the packed weight equals
the per-linear GEMMs bit for bit (same activation quantization, same rows).
Kimi K3.1's fused DSA forward packs ``[q_a | kv_a | wk | weights_proj]`` and
``[q_b | wq_index]`` this way when they are online MXFP8.
"""

import torch
from torch import nn

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
)


def unswizzle_mxfp8_scale(sf: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """Inverse of ``swizzle_mxfp8_scale``: F8_128x4 layout -> row-major
    ``[n, k // 32]`` uint8."""
    m_tiles, k_tiles = -(-n // 128), -(-k // (MXFP8_BLOCK_SIZE * 4))
    tiles = sf.view(torch.uint8).reshape(m_tiles, k_tiles, 32, 4, 4)
    rows = tiles.transpose(1, 3).reshape(m_tiles * 128, k_tiles * 4)
    return rows[:n, : k // MXFP8_BLOCK_SIZE].contiguous()


def _kernel_layout(linear: nn.Module) -> str | None:
    """How the linear's MXFP8 kernel stores its weight: "nk" ([N, K] row-major),
    "kn" (its transpose view), or None (a repacked layout this module can't
    unpack)."""
    from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
        FlashInferCutedslMxfp8LinearKernel,
        FlashInferCutlassMxfp8LinearKernel,
    )
    from vllm.model_executor.kernels.linear.mxfp8.pytorch import (
        TorchMxFp8LinearKernel,
    )
    from vllm.platforms import current_platform

    kernel = getattr(linear.quant_method, "kernel", None)
    if type(kernel) is FlashInferCutedslMxfp8LinearKernel:
        return "kn"
    if type(kernel) is FlashInferCutlassMxfp8LinearKernel or (
        type(kernel) is TorchMxFp8LinearKernel and current_platform.is_cuda()
    ):
        return "nk"
    return None


def is_online_mxfp8(linear: nn.Module) -> bool:
    from vllm.model_executor.layers.quantization.online.mxfp8 import (
        Mxfp8OnlineLinearMethod,
    )

    return isinstance(getattr(linear, "quant_method", None), Mxfp8OnlineLinearMethod)


def can_pack_mxfp8(linears) -> bool:
    if not all(is_online_mxfp8(linear) for linear in linears):
        return False
    if any(_kernel_layout(linear) is None for linear in linears):
        return False
    return len({linear.weight.shape[0 if _kernel_layout(linear) == "kn" else 1]
                for linear in linears}) == 1  # fmt: skip


def _rows(linear: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """The linear's MXFP8 weight as row-major ``[N, K]`` FP8 plus ``[N, K/32]``
    uint8 scales."""
    weight = linear.weight.data
    if _kernel_layout(linear) == "kn":
        weight = weight.t()
    n, k = weight.shape
    return weight, unswizzle_mxfp8_scale(linear.weight_scale.data, n, k)


class _PackedHolder(nn.Module):
    pass


class PackedMxfp8Linear:
    """One MXFP8 GEMM over the stacked rows of ``linears`` (all online MXFP8 on
    the same input). The linears' weights become views of the packed weight;
    their own (small) scale tensors stay so they keep working on their own."""

    def __init__(self, linears) -> None:
        from vllm.model_executor.kernels.linear import init_mxfp8_linear_kernel

        assert can_pack_mxfp8(linears)
        parts = [_rows(linear) for linear in linears]
        weight = torch.cat([w for w, _ in parts])
        scale = torch.cat([s for _, s in parts])
        n, k = weight.shape
        self.kernel = init_mxfp8_linear_kernel(weight_shape=(n, k))
        holder = _PackedHolder()
        holder.register_parameter("weight", nn.Parameter(weight, requires_grad=False))
        holder.register_parameter(
            "weight_scale", nn.Parameter(scale, requires_grad=False)
        )
        holder.input_scale = None
        holder.params_dtype = linears[0].params_dtype
        self.kernel.process_weights_after_loading(holder)
        self.holder = holder
        self.rows = [w.shape[0] for w, _ in parts]
        # Share the weight storage when the packed and the per-linear layouts
        # agree (same kernel class, row-major or transposed rows).
        packed_rows = holder.weight.data
        layout = (
            "kn"
            if packed_rows.shape == (k, n) and packed_rows.stride() == (1, k)
            else "nk"
            if packed_rows.shape == (n, k) and packed_rows.is_contiguous()
            else None
        )
        if layout == "kn":
            packed_rows = packed_rows.t()
        if layout is not None:
            row = 0
            for linear, rows in zip(linears, self.rows):
                if _kernel_layout(linear) == layout:
                    view = packed_rows[row : row + rows]
                    linear.weight.data = view.t() if layout == "kn" else view
                row += rows

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.kernel.apply_weights(self.holder, x)
