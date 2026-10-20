# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Kimi K3 MoE block for DEP decode batches.

Stock MegaMoE layer (one rank, M tokens):

    router(x) ‖ down(x) -> MegaMoE -> up(norm(.)) + shared_down(act(shared_gate_up(x)))

with every projection a separate quantize + MXFP8 GEMM. Here the two projections
that read ``x`` run as one stream-K GEMM over a concatenated weight
(``[down | shared_gate_up]``, N = 3584 + 12288), the shared expert finishes before
MegaMoE, and the up projection adds the shared output in its epilogue:

    q(x) -> front GEMM -> act+q -> shared_down  ‖  router (aux stream)
         -> MegaMoE(front[:, :3584]) -> norm+q -> up GEMM (+ shared)

The concatenated weight replaces the two stock weights: those modules keep views
into it, so the stock path (prefill, large batches) is unchanged and no weight is
duplicated.
"""

import torch
from torch import nn

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
)

from .elementwise import rms_norm_mxfp8, situ_and_mul_mxfp8
from .gemm import MAX_TILE_N, streamk_mxfp8_gemm
from .linear import fp8_rows


def _fp8_rows(linear: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    rows = fp8_rows(linear)
    assert rows is not None, (
        f"{linear} is not an online-MXFP8 FlashInfer CuTe-DSL linear"
    )
    return rows


def _concat_rows(linears: list[nn.Module]) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack weights along N into one buffer and re-point each linear at its slice.

    Scales in the 128x4 swizzled layout are row-tile major, so stacking row tiles
    is a flat concatenation.
    """
    parts = [_fp8_rows(linear) for linear in linears]
    weight = torch.cat([w for w, _ in parts])
    scale = torch.cat([s for _, s in parts])
    row = sf = 0
    for linear, (w, s) in zip(linears, parts):
        rows, sf_len = w.shape[0], s.numel()
        linear.weight = nn.Parameter(weight[row : row + rows].t(), requires_grad=False)
        linear.weight_scale = nn.Parameter(scale[sf : sf + sf_len], requires_grad=False)
        row, sf = row + rows, sf + sf_len
    return weight, scale


class DepMoE:
    """Decode-time forward for one ``KimiMoE`` layer (see the module docstring)."""

    def __init__(self, moe: nn.Module):
        shared = moe.shared_experts
        assert moe.use_mega_moe and moe.use_latent_moe and shared is not None
        self.moe = moe
        self.latent = moe.moe_hidden_size
        self.front_weight, self.front_scale = _concat_rows(
            [moe.routed_expert_down_proj, shared.gate_up_proj]
        )
        self.shared_down = _fp8_rows(shared.down_proj)
        self.up = _fp8_rows(moe.routed_expert_up_proj)
        self.beta = shared.act_fn.beta
        self.linear_beta = shared.act_fn.linear_beta
        norm = moe.routed_expert_norm
        self.norm_weight, self.norm_eps = norm.weight, norm.variance_epsilon
        self.events = (torch.cuda.Event(), torch.cuda.Event())

    @staticmethod
    def supports(num_tokens: int) -> bool:
        return 0 < num_tokens <= MAX_TILE_N

    def _router(self, x: torch.Tensor):
        moe = self.moe
        from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
            fused_grouped_topk,
        )

        logits, _ = moe.gate(x)
        return fused_grouped_topk(
            hidden_states=x,
            gating_output=logits,
            topk=moe.experts.top_k,
            renormalize=moe.moe_renormalize,
            e_score_correction_bias=moe.gate.e_score_correction_bias.data,
            num_expert_group=moe.num_expert_group,
            topk_group=moe.topk_group,
            scoring_func=moe.moe_router_activation_func,
            routed_scaling_factor=moe.routed_scaling_factor,
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        # The router only feeds MegaMoE: on the aux stream it overlaps the whole
        # front GEMM -> activation -> shared down chain, not just the front GEMM.
        main, aux = torch.cuda.current_stream(), self.moe._down_proj_stream
        self.events[0].record(main)
        with torch.cuda.stream(aux):
            aux.wait_event(self.events[0])
            topk_weights, topk_ids = self._router(x)
            self.events[1].record(aux)
        xq, xs = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
        front = streamk_mxfp8_gemm(xq, xs, self.front_weight, self.front_scale)
        aq, a_s = situ_and_mul_mxfp8(
            front[:, self.latent :], self.beta, self.linear_beta
        )
        shared = streamk_mxfp8_gemm(aq, a_s, *self.shared_down)
        main.wait_event(self.events[1])
        routed = self.moe.experts(front[:, : self.latent], topk_weights, topk_ids)
        rq, r_s = rms_norm_mxfp8(routed, self.norm_weight, self.norm_eps)
        return streamk_mxfp8_gemm(rq, r_s, *self.up, residual=shared)
