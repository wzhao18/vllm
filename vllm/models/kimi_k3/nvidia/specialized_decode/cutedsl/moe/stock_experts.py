# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stock routed experts on routes computed elsewhere (seam S5).

The stock Kimi K3 MoE layer runs the monolithic TRT-LLM-Gen MXFP4 x MXFP8 kernel, which
routes inside the kernel. A specialized MoE whose front and routing run its own kernels
(wide decode shapes, before a specialized experts kernel covers them) still needs the
stock experts' math on its routes: ``StockExperts`` builds the modular form of the same
kernel (FlashInfer ``trtllm_fp4_block_scale_routed_moe``, top-k given) once per layer,
over the layer's own weights and scales (``place``d weights included: same layout).

``run`` returns this rank's routed partial [M, hidden] BF16: the top-k weighted sum of
every routed expert's output over the rank's share of the intermediate dimension,
before the TP reduction, the routed-expert norm and the up-projection. The routes must
carry the stock router's semantics (``router.select_experts`` of the layer's runner:
bias-corrected selection, renormalized weights with the routed scaling factor folded
in). Not bitwise equal to the stock layer, which finalizes inside the monolithic kernel
or defers the top-k reduction to its tail. The kernel launches with PDL and may release
its dependents before its output is stored: consumers read the partial after their
grid-dependency wait.
"""

import copy

import torch


class StockExperts:
    """``routed``: the MoE layer's ``RoutedExperts`` (``module.experts.routed_experts``)
    with the FlashInfer TRT-LLM MXFP4 x MXFP8 backend. Build outside CUDA graph
    captures."""

    def __init__(self, routed):
        from vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe import (
            TrtLlmMxfp4ExpertsModular,
        )
        from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
            Mxfp4MoeBackend,
            make_mxfp4_moe_kernel,
        )

        method = routed.quant_method
        backend = Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8
        if getattr(method, "mxfp4_backend", None) != backend:
            raise ValueError(
                "StockExperts needs the FlashInfer TRT-LLM MXFP4 x MXFP8 experts, "
                f"not {getattr(method, 'mxfp4_backend', None)}"
            )
        if method.moe_quant_config is None:
            raise ValueError("StockExperts: the experts' quant config is not built")
        # A private copy of the layer's MoE config: the modular experts lower its
        # deferred-finalize cap, and these calls always finalize into their output.
        config = copy.copy(method.moe)
        config._defer_moe_finalize = False
        self.kernel = make_mxfp4_moe_kernel(
            moe_quant_config=method.moe_quant_config,
            moe_config=config,
            experts_cls=TrtLlmMxfp4ExpertsModular,
            mxfp4_backend=backend,
            routing_tables=routed._expert_routing_tables(),
        )
        self.routed = routed

    def run(self, latent, topk_weights, topk_ids, out=None):
        """``latent`` [M, hidden] BF16 (the routed experts' input), ``topk_weights``
        [M, top_k] and ``topk_ids`` [M, top_k] from the stock router; returns the
        partial (copied into ``out`` [M, hidden] when given)."""
        routed = self.routed
        partial = self.kernel.apply(
            hidden_states=latent,
            w1=routed.w13_weight,
            w2=routed.w2_weight,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            activation=routed.activation,
            global_num_experts=routed.global_num_experts,
            expert_map=routed.expert_map,
            apply_router_weight_on_input=routed.apply_router_weight_on_input,
        )
        if not isinstance(partial, torch.Tensor):
            raise RuntimeError("StockExperts: the experts deferred their top-k sum")
        if out is None:
            return partial
        return out.copy_(partial)
