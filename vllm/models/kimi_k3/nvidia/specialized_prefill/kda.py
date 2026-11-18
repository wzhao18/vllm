# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KDA prefill: the gated o_norm + MXFP8 o_proj input at the HBM rate.

Stock (ops/kda_norm.py ``rmsnorm_gated_mxfp8``, Triton) runs the 96-head DEP16
prefill at ~5 TB/s (204 us per layer at 16K tokens on VR200). This kernel
(ops/flashkda_split/csrc/smxx/o_norm_mx.cu) gives each (token, head) row 8
lanes x 16 columns, keeps two 4-row steps of loads in flight and does 32-bit
row arithmetic. Per element it is the Triton kernel's expressions and
instructions (approximate sqrt, full-range division, ex2 sigmoid, BF16 round,
UE8M0 per 32, RN-satfinite E4M3); the row sum of squares runs as a lane
butterfly, another order than Triton's tl.sum (stock tolerance).
"""

from __future__ import annotations

import torch

from vllm.models.kimi_k3.nvidia.ops import kda_segscan


def o_norm_mxfp8(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out,
) -> None:
    """Gated RMSNorm (sigmoid) of x (..., H, 128) with gate rows g (T, >= H 128)
    into ``out`` (an ``mxfp8_act.Mxfp8Buffer`` of T rows)."""
    kda_segscan.load_extension()
    torch.ops.k3_flashkda_split.o_norm_mxfp8(
        x, g, weight.float().contiguous(), eps, out.data, out.scale
    )
