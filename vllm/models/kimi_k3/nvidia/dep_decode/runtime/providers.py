# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-layer providers of the DEP decode: attention (KDA or MLA) from the normalized
layer input to the o-proj output, and the feed-forward block. They read the stock
modules' parameters and caches and never modify the modules."""

import torch

from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update

from ..kda_verify import KdaOutProj, MlaOutProj, kda_cute_verify
from ..moe import DepMoE


def make_attention(layer):
    attention = layer.self_attn
    if hasattr(attention, "in_proj_qkvgfab"):
        return KdaAttention(attention)
    return MlaAttention(attention)


def make_feed_forward(layer):
    module = layer.mlp
    if getattr(module, "use_mega_moe", False):
        return DepMoE(module)
    return module  # the dense first layer runs its stock MLP


class KdaAttention:
    """A KDA layer's decode or verify step: input projections, short convolution,
    the RecoverSSM verify recurrence (CuTe), then the fused gated output norm,
    MXFP8 quantization and o-proj."""

    def __init__(self, module):
        assert module.use_recoverssm, "the DEP decode verifies KDA with RecoverSSM"
        self.module = module
        weight = module.conv1d.weight
        self.conv_weight = weight.view(weight.size(0), weight.size(2))
        self.out_proj = KdaOutProj(module)

    def project(self, x):
        """(mixed_qkv [T, 3HK], g_proj [T, HK], f_b gate [T, HK], beta [T, H])."""
        m = self.module
        overlap = (
            0 < x.shape[0] <= m._projection_overlap_max_tokens
            and m._projection_events is not None
            and m._projection_aux_stream is not None
        )
        if overlap:
            from vllm.models.kimi_k3.nvidia.low_latency_gemm import (
                run_kda_projection_overlap,
            )

            qkvg, gate, beta = run_kda_projection_overlap(
                x,
                m.in_proj_qkvgfab.weight,
                m.f_b_proj.weight,
                m._projection_aux_stream,
                m._projection_events,
            )
            mixed, g_proj = qkvg.split(
                [3 * m.local_projection_size, m.local_projection_size], dim=-1
            )
            return mixed, g_proj, gate, beta
        sizes = [
            3 * m.local_projection_size,
            m.local_projection_size,
            m.head_dim,
            m.local_num_heads,
        ]
        if m.in_proj_padding:
            sizes.append(m.in_proj_padding)
        mixed, g_proj, f_a, beta, *_ = m.in_proj_qkvgfab(x)[0].split(sizes, dim=-1)
        return mixed, g_proj, m.f_b_proj(f_a)[0], beta

    def run(self, x, md, q):
        m = self.module
        mixed, g_proj, gate, beta = self.project(x)
        tokens = md.num_actual_tokens
        conv_state, state, *records = m.kv_cache
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        requests = md.num_spec_decodes
        slots = md.spec_state_indices_tensor[:requests, 0]
        starts = md.spec_query_start_loc[: requests + 1]
        convolved = torch.empty_like(mixed[:tokens])
        causal_conv1d_update(
            mixed[:tokens],
            conv_state,
            self.conv_weight,
            m.conv1d.bias,
            activation="silu",
            conv_state_indices=slots,
            num_accepted_tokens=md.num_accepted_tokens,
            query_start_loc=starts,
            max_query_len=q,
            validate_data=False,
            out=convolved,
        )
        core = x.new_empty((x.shape[0], m.local_num_heads * m.head_dim))
        kda_cute_verify(
            convolved,
            gate[:tokens],
            beta[:tokens],
            m.A_log,
            m.dt_bias,
            m.gate_lower_bound,
            state,
            records[0],
            records[1],
            starts,
            slots,
            q,
            core[:tokens],
        )
        return self.out_proj(core, g_proj)


class MlaAttention:
    """An MLA layer's decode or verify step: the fused q-LoRA / KV-A / gate
    projection, the latent attention, then the fused sigmoid gate, MXFP8
    quantization and o-proj."""

    def __init__(self, module):
        assert module.q_lora_rank is not None and module.use_output_gate
        self.module = module
        self.qkv_rows = (
            module.q_lora_rank + module.kv_lora_rank + module.qk_rope_head_dim
        )
        self.out_proj = MlaOutProj(module)

    def run(self, x, positions):
        m = self.module
        qkv, gate = m.fused_qkv_a_g_proj(x)[0].split(
            [self.qkv_rows, m.num_local_heads * m.v_head_dim], dim=-1
        )
        out = m._apply_q_lora_attention(positions, x, qkv)
        return self.out_proj(out, gate)
