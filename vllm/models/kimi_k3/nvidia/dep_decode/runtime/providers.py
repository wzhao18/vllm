# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-layer providers of the DEP decode: attention (KDA or MLA) from the normalized
layer input to the o-proj output, and the feed-forward block. They read the stock
modules' parameters and caches and never modify the modules."""

import torch

from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.models.common.ops import fused_q_kv_rmsnorm

from ..gemm import streamk_mxfp8_gemm
from ..kda_verify import KdaOutProj, MlaOutProj, kda_cute_verify
from ..linear import fp8_rows
from ..moe import DepMoE


def make_attention(layer):
    attention = layer.self_attn
    if hasattr(attention, "in_proj_qkvgfab"):
        if attention.use_recoverssm:
            return KdaAttention(attention)
        return StockKdaAttention(attention)
    return MlaAttention(attention)


class StockKdaAttention:
    """A KDA layer without RecoverSSM (e.g. P/D disaggregation, whose KV
    connectors do not support it): the stock module, verifying with per-draft
    states."""

    def __init__(self, module):
        self.module = module

    def run(self, x, positions):
        return self.module(hidden_states=x, positions=positions)


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
        # The verify kernel fuses the short convolution (SiLU, no bias), for either
        # conv-state layout (token major, or dim first: VLLM_SSM_CONV_STATE_LAYOUT=DS).
        self.fused_conv = module.conv1d.bias is None
        self.conv_dim_first = is_conv_state_dim_first()
        # The fused q/k/v/gate/f_a/beta projection on the MXFP8 layer input.
        self.in_proj = fp8_rows(module.in_proj_qkvgfab)

    def project(self, x, xq, xs):
        """(mixed_qkv [T, 3HK], g_proj [T, HK], f_b gate [T, HK], beta [T, H])."""
        m = self.module
        sizes = [
            3 * m.local_projection_size,
            m.local_projection_size,
            m.head_dim,
            m.local_num_heads,
        ]
        if m.in_proj_padding:
            sizes.append(m.in_proj_padding)
        if self.in_proj is not None:
            projected = streamk_mxfp8_gemm(xq, xs, *self.in_proj)
            mixed, g_proj, f_a, beta, *_ = projected.split(sizes, dim=-1)
            return mixed, g_proj, m.f_b_proj(f_a)[0], beta
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
        mixed, g_proj, f_a, beta, *_ = m.in_proj_qkvgfab(x)[0].split(sizes, dim=-1)
        return mixed, g_proj, m.f_b_proj(f_a)[0], beta

    def run(self, x, xq, xs, md, q):
        m = self.module
        mixed, g_proj, gate, beta = self.project(x, xq, xs)
        tokens = md.num_actual_tokens
        conv_state, state, *records = m.kv_cache
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        requests = md.num_spec_decodes
        slots = md.spec_state_indices_tensor[:requests, 0]
        starts = md.spec_query_start_loc[: requests + 1]
        core = x.new_empty((x.shape[0], m.local_num_heads * m.head_dim))
        if self.fused_conv:
            # The verify kernel applies the short convolution to the stored state.
            kda_cute_verify(
                mixed[:tokens],
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
                conv=(m.kv_cache[0], self.conv_weight, md.num_accepted_tokens),
                conv_dim_first=self.conv_dim_first,
            )
            return self.out_proj(core, g_proj)
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
    projection, the q / KV norms and q up-projection, the latent cache insert, the
    backend's MLA core (``MlaCore``), the V up-projection, then the
    fused sigmoid gate, MXFP8 quantization and o-proj."""

    def __init__(self, module):
        assert module.q_lora_rank is not None and module.use_output_gate
        assert module.dcp_world_size == 1
        self.module = module
        self.qkv_rows = (
            module.q_lora_rank + module.kv_lora_rank + module.qk_rope_head_dim
        )
        self.out_proj = MlaOutProj(module)
        self.qkv_a_g = fp8_rows(module.fused_qkv_a_g_proj)

    def run(self, x, xq, xs, md, positions, core):
        m = self.module
        if self.qkv_a_g is not None:
            projected = streamk_mxfp8_gemm(xq, xs, *self.qkv_a_g)
        else:
            projected = m.fused_qkv_a_g_proj(x)[0]
        qkv, gate = projected.split(
            [self.qkv_rows, m.num_local_heads * m.v_head_dim], dim=-1
        )
        q_c, kv_c, k_pe = qkv.split(
            [m.q_lora_rank, m.kv_lora_rank, m.qk_rope_head_dim], dim=-1
        )
        q_c, kv_c = fused_q_kv_rmsnorm(
            q_c,
            kv_c,
            m.q_a_layernorm.weight.data,
            m.kv_a_layernorm.weight.data,
            m.rms_norm_eps,
        )
        q = m.q_b_proj(q_c)[0].view(-1, m.num_local_heads, m.qk_head_dim)
        q_nope, q_pe = q.split([m.qk_nope_head_dim, m.qk_rope_head_dim], dim=-1)
        # Absorb q_nope into the latent space: (N, B, P) x (N, P, L) -> (B, N, L).
        ql_nope = torch.bmm(q_nope.transpose(0, 1), m.W_UK_T).transpose(0, 1)
        rope = m.rotary_emb
        query = m._decode_concat_cache(
            ql_nope,
            q_pe,
            kv_c,
            k_pe.unsqueeze(1),
            positions if rope is not None else None,
            rope.cos_sin_cache if rope is not None else None,
            get_forward_context().slot_mapping[m.layer_name][: x.shape[0]],
        )
        out = x.new_empty((x.shape[0], m.num_local_heads * m.v_head_dim))
        m._v_up_proj(core(m, query, md), out=out)
        return self.out_proj(out, gate)
