# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from the source branch's tests/models/kimi_k3/test_mxfp8_projections.py
# (test_specialized_decode_rejects_online_linear_quant) for the SM107 specialized
# decode (vllm/models/kimi_k3/nvidia/specialized_decode/sm_107). The rest of that
# file covers the online-MXFP8 projection kernels (vllm/models/kimi_k3/nvidia/
# mxfp8.py, quant_presets.py), which this tree does not have.
"""The SM107 specialized decode reads the BF16 projection weights directly: it
refuses online linear quantization (a linear weight spec or per-layer targets) and
accepts an activation-only MoE override."""

import pytest

from vllm.config.quantization import QuantizationConfigArgs
from vllm.platforms import current_platform

# The SM107 specialized decode (specialized_decode/sm_107) serves VR200 only.
pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability((10, 7)),
    reason="the SM107 specialized decode runs on compute capability 10.7 only",
)

# The source branch's quant_presets.K31_DEP_MXFP8_TARGETS (every BF16 non-expert
# projection of the target model), inlined: this tree has no quant_presets module.
DEP_MXFP8_TARGETS: dict[str, str] = {
    r"re:.*\.self_attn\.(in_proj_qkvgfab|fused_qkv_a_proj|q_b_proj|g_proj|o_proj)$": (
        "mxfp8"
    ),
    r"re:.*\.self_attn\.dsa_indexer\.(wq_index|wkv_index)$": "mxfp8",
    r"re:.*\.block_sparse_moe\.shared_experts\.(gate_up_proj|down_proj)$": "mxfp8",
    r"re:.*\.block_sparse_moe\.routed_expert_(up|down)_proj$": "mxfp8",
    r"re:.*\.layers\.\d+\.mlp\.(gate_up_proj|down_proj)$": "mxfp8",
}


def test_specialized_decode_rejects_online_linear_quant():
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.runtime import (
        online_linear_quantization,
    )

    assert not online_linear_quantization(None)
    assert not online_linear_quantization(
        QuantizationConfigArgs(moe={"activation": "mxfp8"})
    )
    assert online_linear_quantization(QuantizationConfigArgs(linear="mxfp8"))
    # The source branch let per-layer targets compose with an activation-only
    # ``moe`` override; this tree's QuantizationConfigArgs makes them mutually
    # exclusive, so the targets alone (the decision only reads ``targets``).
    assert online_linear_quantization(
        QuantizationConfigArgs(targets=DEP_MXFP8_TARGETS)
    )
