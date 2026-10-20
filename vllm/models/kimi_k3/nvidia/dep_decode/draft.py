# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 projections for the BF16 DSpark draft model on DEP decode.

The draft's dense projections and its LM head (shared with the target) are
weight-bandwidth bound at decode batch sizes, so the draft runs them on MXFP8
weights (1x32 E8M0 block scales, the online-MXFP8 layout): its own projections are
replaced, the shared LM head gets a draft-only copy. Decode-sized batches run the
stream-K GEMM; larger ones run FlashInfer's MXFP8 GEMM (projections) or the BF16
head. Only the drafted tokens can change, never the verified output.
"""

import torch
from torch import nn

from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
)
from vllm.utils import flashinfer as vllm_flashinfer

from .gemm import MAX_TILE_N, streamk_mxfp8_gemm

# Projections that run once per drafted token. The cross-layer context KV and
# Q/KV-A projections (row counts not a multiple of 128) and q_b_proj (K = 1536:
# no faster in MXFP8) stay in BF16.
_TARGETS = (
    "context_proj",
    "self_attn.o_proj",
    "mlp.gate_up_proj",
    "mlp.down_proj",
)


def _mxfp8_forward(linear: nn.Module, weight: torch.Tensor, scale: torch.Tensor):
    """``linear``'s forward over an MXFP8 ``weight`` [N, K] with swizzled scales."""
    weight_kn = weight.t()
    return_bias = getattr(linear, "return_bias", True)

    def forward(x: torch.Tensor):
        shape = x.shape
        x = x.reshape(-1, shape[-1]).contiguous()
        xq, xs = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
        if 0 < x.shape[0] <= MAX_TILE_N:
            out = streamk_mxfp8_gemm(xq, xs, weight, scale)
        else:
            out = vllm_flashinfer.mm_mxfp8(
                xq, weight_kn, xs, scale, out_dtype=x.dtype, backend="cute-dsl"
            )
        out = out.view(*shape[:-1], weight.shape[0])
        return (out, None) if return_bias else out

    return forward


def quantize_draft(model: nn.Module) -> list[str]:
    """Swap the draft's per-token BF16 projections for MXFP8; returns their names."""
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, LinearBase) or not name.endswith(_TARGETS):
            continue
        weight = getattr(module, "weight", None)
        if (
            weight is None
            or weight.dtype != torch.bfloat16
            or getattr(module, "bias", None) is not None
            or weight.shape[0] % 128
            or weight.shape[1] % 128
        ):
            continue
        wq, ws = mxfp8_e4m3_quantize(
            weight.data.contiguous(), is_sf_swizzled_layout=True
        )
        del module.weight
        module.weight = nn.Parameter(wq.t(), requires_grad=False)
        module.weight_scale = nn.Parameter(ws.reshape(-1), requires_grad=False)
        module.forward = _mxfp8_forward(module, wq, module.weight_scale)
        names.append(name)
    torch.cuda.empty_cache()
    return names


class _Mxfp8HeadMethod:
    def __init__(self, head: "Mxfp8LogitsHead"):
        self.head = head

    def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None):
        assert bias is None
        head = self.head
        if 0 < x.shape[0] <= MAX_TILE_N:
            xq, xs = mxfp8_e4m3_quantize(x.contiguous(), is_sf_swizzled_layout=True)
            return streamk_mxfp8_gemm(xq, xs, head.weight_fp8, head.scale)
        stock = head.stock
        return stock.quant_method.apply(stock, x, bias=bias)


class Mxfp8LogitsHead(nn.Module):
    """A draft-only MXFP8 copy of the target's LM head, in the interface
    ``LogitsProcessor`` reads (``quant_method.apply``, ``tp_size``)."""

    def __init__(self, stock: nn.Module):
        super().__init__()
        assert getattr(stock, "tp_size", 1) == 1
        weight = stock.weight.data
        rows = -(-weight.shape[0] // 128) * 128
        if rows != weight.shape[0]:
            weight = torch.nn.functional.pad(weight, (0, 0, 0, rows - weight.shape[0]))
        self.weight_fp8, scale = mxfp8_e4m3_quantize(
            weight.contiguous(), is_sf_swizzled_layout=True
        )
        self.scale = scale.reshape(-1)
        self.vocab = stock.weight.shape[0]
        self.tp_size = 1
        self.quant_method = _Mxfp8HeadMethod(self)
        # Not a submodule: the head's weights belong to the target.
        object.__setattr__(self, "stock", stock)
