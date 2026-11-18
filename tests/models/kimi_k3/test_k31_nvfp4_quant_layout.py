# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 NVFP4 checkpoint (nvidia/Kimi-K3.1-Pro-NVFP4, ModelOpt MIXED_PRECISION):
every Kimi K3 module prefix resolves to the precision the export stores.

The export's ``quantization_config`` (config.json) lists each quantized module under
three spellings (``language_model.model.layers.N``, ``model.layers.N``, ``layers.N``)
and both ``block_sparse_moe`` / ``mlp`` aliases; its ``ignore`` list is reproduced
verbatim. Routed experts are NVFP4 (group 16), the shared experts and the KDA
projections (q/k/v/g/b/f_a and their fused in-projection, o) MXFP8, everything
else BF16 (MLA/DSA projections, f_b_proj, router, latent projections, dense MLP,
lm_head). CPU only.
"""

import pytest

from vllm.model_executor.layers.fused_moe import deep_gemm_mega_moe as dg
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
)
from vllm.models.kimi_k3.nvidia import model as kimi_model

IGNORE = [
    "*block_sparse_moe.gate*",
    "*block_sparse_moe.routed_expert_up_proj*",
    "*mlp.routed_expert_up_proj*",
    "*block_sparse_moe.routed_expert_down_proj*",
    "*mlp.routed_expert_down_proj*",
    "*block_sparse_moe.routed_expert_norm*",
    "*mlp.routed_expert_norm*",
    "*self_attn.q_conv1d*",
    "*self_attn.k_conv1d*",
    "*self_attn.v_conv1d*",
    "*self_attn.f_b_proj*",
    "*self_attn.o_norm*",
    "*self_attn.q_a_layernorm*",
    "*self_attn.kv_a_layernorm*",
    "*_res_proj*",
    "*_res_norm*",
    "*layernorm*",
    "language_model.model.layers.0.mlp.*",
    "layers.0.mlp.*",
    "model.layers.0.mlp.*",
    "language_model.lm_head",
    "lm_head",
    "vision_tower*",
    "mm_projector*",
]
KDA_LAYERS, MLA_LAYERS = (1, 2), (3,)
KDA_MXFP8 = (
    "q_proj",
    "k_proj",
    "v_proj",
    "g_proj",
    "b_proj",
    "f_a_proj",
    "o_proj",
    "fused_qkvg_proj",
    "in_proj_qkvgfab",
)
SHARED_MXFP8 = ("gate_proj", "up_proj", "down_proj", "gate_up_proj")


def _export_config() -> dict:
    layers = {}
    for base in ("language_model.model.layers", "model.layers", "layers"):
        for i in (*KDA_LAYERS, *MLA_LAYERS):
            for moe in ("block_sparse_moe", "mlp"):
                layers[f"{base}.{i}.{moe}.experts"] = {
                    "quant_algo": "NVFP4",
                    "group_size": 16,
                }
                for proj in SHARED_MXFP8:
                    layers[f"{base}.{i}.{moe}.shared_experts.{proj}"] = {
                        "quant_algo": "MXFP8"
                    }
        for i in (0, *KDA_LAYERS):
            for proj in KDA_MXFP8:
                layers[f"{base}.{i}.self_attn.{proj}"] = {"quant_algo": "MXFP8"}
    return {
        "quant_method": "modelopt_mixed",
        "quant_algo": "MIXED_PRECISION",
        "group_size": 16,
        "ignore": IGNORE,
        "quantized_layers": layers,
    }


@pytest.fixture(scope="module")
def cfg():
    config = ModelOptMixedPrecisionConfig.from_config(_export_config())
    config.packed_modules_mapping = dict(
        kimi_model.KimiLinearModel.packed_modules_mapping
    )
    return config


def _precision(cfg, prefix):
    if cfg.is_layer_excluded(prefix):
        return "BF16"
    return cfg._resolve_quant_algo(prefix) or "BF16"


L = "language_model.model.layers"


@pytest.mark.parametrize(
    ("prefix", "want"),
    [
        # KDA layer: fused in-projection and o_proj MXFP8, f_b_proj BF16
        (f"{L}.1.self_attn.in_proj_qkvgfab", "MXFP8"),
        (f"{L}.1.self_attn.o_proj", "MXFP8"),
        (f"{L}.1.self_attn.f_b_proj", "BF16"),
        # MLA / DSA layer: all BF16
        (f"{L}.3.self_attn.fused_qkv_a_g_proj", "BF16"),
        (f"{L}.3.self_attn.q_b_proj", "BF16"),
        (f"{L}.3.self_attn.kv_b_proj", "BF16"),
        (f"{L}.3.self_attn.o_proj", "BF16"),
        (f"{L}.3.self_attn.indexer.wq_b", "BF16"),
        # MoE block: experts NVFP4, shared experts MXFP8, router and latent BF16
        (f"{L}.2.block_sparse_moe.experts", "NVFP4"),
        (f"{L}.2.block_sparse_moe.shared_experts.gate_up_proj", "MXFP8"),
        (f"{L}.2.block_sparse_moe.shared_experts.down_proj", "MXFP8"),
        (f"{L}.2.block_sparse_moe.gate", "BF16"),
        (f"{L}.2.block_sparse_moe.routed_expert_up_proj", "BF16"),
        (f"{L}.2.block_sparse_moe.routed_expert_down_proj", "BF16"),
        # first layer's dense MLP, lm_head
        (f"{L}.0.mlp.gate_up_proj", "BF16"),
        (f"{L}.0.mlp.down_proj", "BF16"),
        ("language_model.lm_head", "BF16"),
    ],
)
def test_module_precision(cfg, prefix, want):
    assert _precision(cfg, prefix) == want


def test_mega_moe_format_per_layer(cfg):
    fmt = kimi_model.kimi_k3_mega_moe_weight_format
    for i in (*KDA_LAYERS, *MLA_LAYERS):
        assert fmt(cfg, f"{L}.{i}.block_sparse_moe") is dg.MEGA_MOE_NVFP4
