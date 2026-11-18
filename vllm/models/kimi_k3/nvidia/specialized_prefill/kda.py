# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KDA prefill chain with the Q|K|V short convolution fused into FlashKDA.

Stock runs the fused Q|K|V conv (ops/kda_qkv_conv: reads the raw in_proj rows,
writes dense conv'd Q, K and V) and then FlashKDA (prepare K1 reads Q, K, g;
recurrence K2 reads V). Here K1 reads the raw Q / K rows itself and applies the
convolution in its load path (ops/flashkda_split/csrc/smxx/conv_fused.cuh, the
conv kernel's arithmetic bit for bit) and also writes the V plane's conv output
for K2, so the dense Q / K round trip and the separate conv launch go away.
The conv state update is the conv kernel's (the first tile of each sequence is
its only reader and writes the new state). Outputs, final recurrent state and
conv state are bitwise those of qkv_conv followed by the same FlashKDA build.
"""

from __future__ import annotations

import torch

from vllm.models.kimi_k3.nvidia.ops import kda_segscan


def fused_conv_supported(
    mixed_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    head_dim: int,
    checkpoint: bool,
) -> bool:
    """Whether fused_conv_prefill covers this call: FP32 conv weights (the KDA
    layer's), bf16 or FP32 conv state, width 4, D = 128, 16 B aligned
    unit-stride raw rows, and no prefill checkpoint export (the exporter keeps
    the stock path)."""
    return (
        not checkpoint
        and head_dim == 128
        and mixed_qkv.dtype == torch.bfloat16
        and mixed_qkv.dim() == 2
        and mixed_qkv.stride(1) == 1
        and mixed_qkv.stride(0) % 8 == 0
        and mixed_qkv.data_ptr() % 16 == 0
        and conv_weight.dtype == torch.float32
        and conv_weight.dim() == 2
        and conv_weight.size(-1) == 4
        and conv_weight.is_contiguous()
        and conv_state.dtype in (torch.bfloat16, torch.float32)
    )


def fused_conv_prefill(
    *,
    mixed_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    null_block_id: int,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor,
    final_state: torch.Tensor,
    workspace: torch.Tensor,
    v_buf: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Short conv + FlashKDA prefill. mixed_qkv (T, 3 H 128) raw rows;
    conv_state (lines, 3 H 128, >= 3) updated in place; v_buf (1, T, H, 128)
    receives the conv'd V; out (1, T, H, 128) and final_state (N, H, 128, 128)
    are written. Returns (out, final_state)."""
    kda_segscan.load_extension()
    D = out.shape[-1]
    torch.ops.k3_flashkda_split.fwd_conv(
        mixed_qkv,
        conv_weight,
        conv_state,
        cache_indices,
        has_initial_state,
        null_block_id,
        v_buf,
        g.contiguous(),
        beta,
        D**-0.5,
        out,
        workspace,
        A_log.contiguous(),
        dt_bias.view(-1, D).contiguous(),
        lower_bound,
        initial_state.contiguous(),
        final_state,
        cu_seqlens.contiguous(),
        None,
        None,
        3,
    )
    return out, final_state
