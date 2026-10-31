# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online quantization presets for Kimi K3 / K3.1 BF16 projections.

The MXFP4 expert checkpoint keeps every non-expert projection in BF16
(``ignore: self_attn, shared_experts, mlp``; the latent MoE up/down are plain
BF16 tensors). These presets select them for online MXFP8 through
``--quantization-config '{"targets": {...}}'``; an activation-only ``moe``
override (``{"moe": {"activation": "mxfp8"}}``) composes with them.

Not targeted: ``kv_b_proj`` (absorbed into W_UK / W_UV at load), the router
``gate``, ``lm_head`` / embeddings, the AttnRes projections, norms and the
K = 128 ``f_b_proj``.
"""

# DSA attention (q_a | kv_a, q_b, output gate, o_proj) and its indexer (wq_b and
# the merged wk | weights_proj), KDA (in_proj_qkvgfab, o_proj).
K31_ATTENTION_MXFP8_TARGETS: dict[str, str] = {
    r"re:.*\.self_attn\.(in_proj_qkvgfab|fused_qkv_a_proj|q_b_proj|g_proj|o_proj)$": (
        "mxfp8"
    ),
    r"re:.*\.self_attn\.dsa_indexer\.(wq_index|wkv_index)$": "mxfp8",
}

# Shared experts, the latent MoE down/up projections and the dense layer-0 MLP.
K31_MLP_MXFP8_TARGETS: dict[str, str] = {
    r"re:.*\.block_sparse_moe\.shared_experts\.(gate_up_proj|down_proj)$": "mxfp8",
    r"re:.*\.block_sparse_moe\.routed_expert_(up|down)_proj$": "mxfp8",
    r"re:.*\.layers\.\d+\.mlp\.(gate_up_proj|down_proj)$": "mxfp8",
}

# Every BF16 non-expert projection of the target model (DEP / TP1 deployments,
# where they are replicated per rank).
K31_DEP_MXFP8_TARGETS: dict[str, str] = {
    **K31_ATTENTION_MXFP8_TARGETS,
    **K31_MLP_MXFP8_TARGETS,
}
