# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 adapter for the FlashInfer ``moe_ep`` MegaMoE backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from vllm.models.deepseek_v4.nvidia.fi_moe import DeepseekV4MegaMoEExpertsFI

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.models.kimi_k3.nvidia.model import KimiMLP


class KimiK3MegaMoEExpertsFI(DeepseekV4MegaMoEExpertsFI):
    """Use Kimi's SiTU activation and fused BF16 shared expert with FI."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        activation: str,
        activation_beta: float | None,
        activation_linear_beta: float | None,
        **kwargs: Any,
    ) -> None:
        super().__init__(vllm_config, **kwargs)
        if not self._nvfp4_prequant:
            raise ValueError(
                "Kimi K3 FlashInfer MegaMoE requires NVFP4 routed experts."
            )
        self._fi_activation = activation
        self._fi_situ_beta = activation_beta
        self._fi_situ_linear_beta = activation_linear_beta

    def _prepare_shared_expert_weights(
        self, shared_experts: KimiMLP | None
    ) -> tuple[Any | None, int | None, int | None]:
        if shared_experts is None:
            return None, None, None

        from flashinfer.moe_ep import (
            MoEWeightPack,
            preprocess_nvfp4_cutedsl_mega_weights,
        )

        gate_up = shared_experts.gate_up_proj.weight
        down = shared_experts.down_proj.weight
        if gate_up.dtype != torch.bfloat16 or down.dtype != torch.bfloat16:
            raise ValueError(
                "Kimi K3 FlashInfer MegaMoE requires BF16 shared-expert weights."
            )
        shared_hidden_size = gate_up.shape[1]
        shared_intermediate_size = gate_up.shape[0] // 2
        if gate_up.shape != (2 * shared_intermediate_size, shared_hidden_size):
            raise ValueError(f"Invalid shared gate/up weight shape: {gate_up.shape}.")
        if down.shape != (shared_hidden_size, shared_intermediate_size):
            raise ValueError(f"Invalid shared down weight shape: {down.shape}.")

        transformed = preprocess_nvfp4_cutedsl_mega_weights(
            MoEWeightPack(w13=gate_up.data.unsqueeze(0), w2=down.data.unsqueeze(0)),
            intermediate_size=shared_intermediate_size,
            hidden_size=shared_hidden_size,
        )
        shared_experts.gate_up_proj.weight = None
        shared_experts.down_proj.weight = None
        return transformed, shared_hidden_size, shared_intermediate_size


KimiK3MegaMoEExpertsFI.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]
