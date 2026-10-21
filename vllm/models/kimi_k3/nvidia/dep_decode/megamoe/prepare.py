# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MegaMoE input staging for the DEP decode, launched with programmatic dependent
launch.

The stock staging kernel (``prepare_megamoe_inputs``: MXFP8 hidden states with UE8M0
group scales and the int64 / fp32 top-k repack into the symmetric buffer) is reused
through its Triton helpers, so the staged bytes are identical. This copy releases its
dependents first and waits for its own producers before reading, so the DEP MegaMoE
kernel (which waits on this grid before reading its inputs) launches and runs its
prologue while the staging runs.
"""

import torch
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

from vllm.model_executor.layers.fused_moe.prepare_megamoe import (
    _pack_bytes,
    _quantize_fp8_groups,
    _resolve_hidden_quant,
    _store_topk,
)
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_tokens"])
def _prepare_pdl_kernel(
    hidden_states,
    x_out,
    x_sf,
    topk_ids,
    topk_weights,
    is_padding,
    topk_idx_out,
    topk_weights_out,
    hidden_stride_m: tl.constexpr,
    x_stride_m: tl.constexpr,
    x_sf_stride_m: tl.constexpr,
    x_sf_stride_k: tl.constexpr,
    topk_ids_stride_m: tl.constexpr,
    topk_ids_stride_k: tl.constexpr,
    topk_weights_stride_m: tl.constexpr,
    topk_weights_stride_k: tl.constexpr,
    is_padding_stride_m: tl.constexpr,
    topk_idx_stride_m: tl.constexpr,
    topk_idx_stride_k: tl.constexpr,
    topk_weights_out_stride_m: tl.constexpr,
    topk_weights_out_stride_k: tl.constexpr,
    num_tokens,
    hidden_size: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    gdc_launch_dependents()
    gdc_wait()
    token_id = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    token_mask = token_id < num_tokens
    k_block_id = tl.program_id(1)
    k_offsets = k_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = token_mask[:, None] & (k_offsets[None, :] < hidden_size)
    hidden = tl.load(
        hidden_states + token_id[:, None] * hidden_stride_m + k_offsets[None, :],
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)
    num_groups: tl.constexpr = BLOCK_K // GROUP_K
    fp8, _, scale_exp = _quantize_fp8_groups(hidden, BLOCK_M, BLOCK_K, GROUP_K, True)
    tl.store(
        x_out + token_id[:, None] * x_stride_m + k_offsets[None, :], fp8, mask=k_mask
    )
    tl.store(
        x_sf + token_id * x_sf_stride_m + k_block_id * x_sf_stride_k,
        _pack_bytes(scale_exp, num_groups),
        mask=token_mask,
    )
    if k_block_id == 0:
        _store_topk(
            token_id,
            token_mask,
            topk_ids,
            topk_weights,
            is_padding,
            topk_idx_out,
            topk_weights_out,
            topk_ids_stride_m,
            topk_ids_stride_k,
            topk_weights_stride_m,
            topk_weights_stride_k,
            is_padding_stride_m,
            topk_idx_stride_m,
            topk_idx_stride_k,
            topk_weights_out_stride_m,
            topk_weights_out_stride_k,
            top_k,
            BLOCK_M,
            BLOCK_TOPK,
        )


def prepare_megamoe_inputs_pdl(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
    is_padding: torch.Tensor | None,
    hidden_quant,
) -> None:
    """``prepare_megamoe_inputs`` for MXFP8 (UE8M0) hidden states, without shared
    experts, launched with PDL."""
    num_tokens, hidden_size = hidden_states.shape
    if num_tokens == 0:
        return
    block_k = 128
    group_k, use_ue8m0 = _resolve_hidden_quant(hidden_quant, block_k)
    assert use_ue8m0 and hidden_size % block_k == 0
    assert hidden_states.stride(1) == 1 and x.stride(1) == 1
    top_k = topk_ids.shape[1]
    block_m = 8 if num_tokens >= 64 else 1
    grid = (triton.cdiv(num_tokens, block_m), hidden_size // block_k)
    _prepare_pdl_kernel[grid](
        hidden_states,
        x,
        x_sf,
        topk_ids,
        topk_weights,
        is_padding,
        topk_idx_out,
        topk_weights_out,
        hidden_states.stride(0),
        x.stride(0),
        x_sf.stride(0),
        x_sf.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        is_padding.stride(0) if is_padding is not None else 0,
        topk_idx_out.stride(0),
        topk_idx_out.stride(1),
        topk_weights_out.stride(0),
        topk_weights_out.stride(1),
        num_tokens,
        hidden_size,
        top_k,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        GROUP_K=group_k,
        BLOCK_TOPK=triton.next_power_of_2(top_k),
        num_warps=4,
        launch_pdl=True,
    )
