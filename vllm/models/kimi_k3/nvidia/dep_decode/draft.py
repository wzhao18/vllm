# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 projections for the BF16 DSpark draft model on DEP decode.

The draft's dense projections are weight-bandwidth bound at decode batch sizes, so
their BF16 weights are replaced by MXFP8 copies (1x32 E8M0 block scales, the
online-MXFP8 layout). Decode-sized batches run the stream-K GEMM; larger ones run
FlashInfer's MXFP8 GEMM. Only the drafted tokens can change, never the verified
output.
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
