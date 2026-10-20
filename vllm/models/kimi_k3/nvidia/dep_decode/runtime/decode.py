# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DEP decode executor over the stock Kimi K3 backbone: one routing decision per
step, then every layer's providers in one loop.

Per layer: the AttnRes entering the layer (cluster AttnRes kernel), the attention
provider, the AttnRes after attention, and the feed-forward provider; the final
AttnRes also applies the backbone's final norm.
"""

from typing import Any, NamedTuple

import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.utils.math_utils import round_up

from ...specialized_decode.cutedsl.attn_res.prepared import PreparedAttnRes
from ...specialized_decode.runtime.decode import decode_metadata
from ..gemm import MAX_TILE_N
from ..mla import WaveMla
from .providers import (
    KdaAttention,
    MlaAttention,
    make_attention,
    make_feed_forward,
)

logger = init_logger(__name__)


class Step(NamedTuple):
    q: int  # tokens per request
    metadata: list  # per layer
    positions: torch.Tensor


def post_attention_res(layer):
    """The AttnRes between attention and MLP. At a block-write layer the attention
    output starts the new block: it replaces the prefix instead of adding to it."""
    weights = (
        layer.mlp_res_norm.weight,
        layer.mlp_res_proj.weight,
        layer.post_attention_layernorm.weight,
    )
    write = layer.is_block_write_layer
    key = (layer.prev_valid_blocks + int(write), not write, -1, False)
    eps = (
        layer.mlp_res_norm.variance_epsilon,
        layer.post_attention_layernorm.variance_epsilon,
    )
    return PreparedAttnRes(weights, key, eps)


class DepDecode:
    """Built once the weights are loaded; owns every provider and the AttnRes
    consumers. Reads the backbone, never modifies it."""

    def __init__(self, backbone, config):
        self.backbone, self.q = backbone, config.q
        assert backbone.use_attn_res, "the DEP decode runs the AttnRes backbone"
        # FULL verify graphs round each capture size up to whole requests.
        rounded = {round_up(s, self.q) for s in config.capture_sizes}
        self.batches = tuple(sorted(m for m in rounded if m <= MAX_TILE_N))
        layers = backbone.layers[backbone.start_layer : backbone.end_layer]
        self.layers = layers = tuple(layers)
        self.attention = {d.layer_idx: make_attention(d) for d in layers}
        self.feed_forward = {d.layer_idx: make_feed_forward(d) for d in layers}
        self.before = {
            d.layer_idx: PreparedAttnRes.for_target(d, has_delta=d is not layers[0])
            for d in layers
        }
        self.after = {d.layer_idx: post_attention_res(d) for d in layers}
        # The final AttnRes also applies the backbone's final norm.
        self.final = PreparedAttnRes.for_target(backbone, final=True)
        # One wave-scheduled MLA core (and split plan per step) for every MLA layer.
        mla = [a for a in self.attention.values() if isinstance(a, MlaAttention)]
        self.mla_index = next(
            (
                i
                for i, d in enumerate(layers)
                if isinstance(self.attention[d.layer_idx], MlaAttention)
            ),
            None,
        )
        self.mla_core = WaveMla(mla[0].module, max(self.batches)) if mla else None
        # The AttnRes entering each aux layer (or the final one) also stores the
        # updated prefix: the stock aux stream (prefix sum plus the last MLP output).
        self.aux_layers = tuple(backbone.aux_hidden_state_layers)
        end = layers[-1].layer_idx + 1
        for index in self.aux_layers:
            (self.final if index == end else self.before[index]).bind_aux()
        self.warm_up()

    def warm_up(self):
        """Compile every backend kernel a planned step launches. Planned steps run
        only inside CUDA graph captures, so nothing may compile there: the CuTe
        kernels compile here, and each Triton producer runs once per planned batch
        (Triton specializes on argument values). The stock kernels the providers
        call (projections, convolution, MLA core, router, MegaMoE) compile in the
        stock warmup steps."""
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
            mxfp8_e4m3_quantize,
        )

        from ...specialized_decode.cutedsl.common.host import architecture
        from ..cutedsl.streamk_gemm import SUPPORTED_TILE_N
        from ..elementwise import (
            gated_rms_norm_mxfp8,
            rms_norm_mxfp8,
            sigmoid_gate_mxfp8,
            situ_and_mul_mxfp8,
        )
        from ..gemm import StreamKScratch, _compile
        from ..kda_verify import _compile as compile_verify
        from ..moe import DepMoE

        embed = self.backbone.embed_tokens.weight
        device, dtype = embed.device, embed.dtype
        StreamKScratch.get(device)
        for tile_n in SUPPORTED_TILE_N:
            for residual in (False, True):
                _compile(tile_n, residual, 4)
        kda = [a for a in self.attention.values() if isinstance(a, KdaAttention)]
        mla = [a for a in self.attention.values() if not isinstance(a, KdaAttention)]
        moe = [f for f in self.feed_forward.values() if isinstance(f, DepMoE)]
        if kda:
            heads = kda[0].module.local_num_heads
            compile_verify(heads, self.q, architecture(device))

        def zeros(*shape):
            return torch.zeros(*shape, device=device, dtype=dtype)

        for b in self.batches:
            mxfp8_e4m3_quantize(zeros(b, embed.shape[1]), is_sf_swizzled_layout=True)
            if kda:
                m = kda[0].module
                width = m.local_num_heads * m.head_dim
                # The gate is a column slice of the in-proj output, as in a real step.
                row = 4 * width + m.head_dim + m.local_num_heads + m.in_proj_padding
                gate = zeros(b, row)[:, 3 * width : 4 * width]
                gated_rms_norm_mxfp8(
                    zeros(b, width), gate, m.o_norm.weight, m.o_norm.eps
                )
            if mla:
                m = mla[0].module
                width = m.num_local_heads * m.v_head_dim
                gate = zeros(b, mla[0].qkv_rows + width)[:, -width:]
                sigmoid_gate_mxfp8(zeros(b, width), gate)
            if moe:
                block = moe[0]
                front = zeros(b, block.front_weight.shape[0])
                situ_and_mul_mxfp8(
                    front[:, block.latent :], block.beta, block.linear_beta
                )
                rms_norm_mxfp8(
                    zeros(b, block.latent), block.norm_weight, block.norm_eps
                )
        torch.cuda.synchronize(device)

    def prepare_kv(self):
        """Compile the MLA core's split-count decode against the bound cache (the
        profiling cache, then the real one)."""
        if self.mla_core is None:
            return
        module = self.layers[self.mla_index].self_attn
        if module.kv_cache.numel() == 0:
            return
        for b in self.batches:
            self.mla_core.warm_up(module, b // self.q, self.q)
        torch.cuda.synchronize(module.kv_cache.device)

    def embed_input_ids(self, ids):
        return self.backbone.embed_input_ids(ids)

    def plan(self, positions, intermediate_tensors):
        """This step's decode plan, or None to run the stock forward instead.

        Steps that run the stock forward:
        - pipeline-parallel stage inputs (``intermediate_tensors``);
        - token counts without providers (not a rounded capture size);
        - steps outside a CUDA graph capture (profiling, warmup and the warmup run
          before each capture), so every stock kernel runs before its capture;
        - PIECEWISE steps, keyed by token count alone and so captured as stock;
        - steps with prompt rows, and speculative steps whose requests do not all
          verify q tokens.

        Ranks may disagree: both paths run the same MegaMoE collective per layer.
        """
        b = positions.shape[0]
        if intermediate_tensors is not None or b not in self.batches:
            return None
        if not torch.cuda.is_current_stream_capturing():
            return None
        if not is_forward_context_available():
            return None
        context = get_forward_context()
        if context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
            return None
        by_layer = context.attn_metadata
        rows: list[Any] = list(by_layer.values()) if isinstance(by_layer, dict) else []
        q = self.q if any(getattr(md, "num_spec_decodes", 0) for md in rows) else 1
        if q == 1 or b % q:
            return None  # plain decode steps (no speculation) are not prepared yet
        metadata = [decode_metadata(d.self_attn, context, q, b) for d in self.layers]
        if any(md is None for md in metadata):
            return None
        logger.info_once("Kimi K3 DEP decode runs %d-token verify steps", b)
        return Step(q, metadata, positions)

    def run(self, step, input_ids, inputs_embeds):
        """Normalized hidden states (and the aux streams) of a planned step."""
        x = inputs_embeds
        if x is None:
            x = self.embed_input_ids(input_ids)
        b, hidden = x.shape
        out = torch.empty_like(x)
        aux_out = [torch.empty_like(x) for _ in self.aux_layers]
        aux = dict(zip(self.aux_layers, aux_out))
        prefix = torch.empty_like(x)
        blocks = x.new_empty((b, self.backbone.num_attn_res_blocks, hidden))
        normed = torch.empty_like(x)
        delta = x  # the first AttnRes takes the embeddings as its prefix
        if self.mla_core is not None:
            # Every MLA layer reads the same lengths: one split plan per step.
            md = step.metadata[self.mla_index]
            capacity = self.mla_core.plan(md.decode.seq_lens, step.q)
        for layer, md in zip(self.layers, step.metadata):
            index = layer.layer_idx
            self.before[index].run(delta, prefix, blocks, normed, aux.get(index))
            attention = self.attention[index]
            if isinstance(attention, KdaAttention):
                attn_out = attention.run(normed, md, step.q)
            else:
                attn_out = attention.run(
                    normed, md, step.positions, self.mla_core, capacity
                )
            self.after[index].run(attn_out, prefix, blocks, normed)
            delta = self.feed_forward[index](normed)
        self.final.run(
            delta, prefix, blocks, out, aux.get(self.layers[-1].layer_idx + 1)
        )
        return (out, aux_out) if aux_out else out

    def normalize(self, output):
        """A stock step's output with the backbone's final norm applied, as the DEP
        decode's last AttnRes applies it; the aux streams stay pre-norm. Chunked at
        255 rows to keep the reduction order of the stock RMSNorm (see the
        latency specialized decode)."""
        hidden, aux = output if isinstance(output, tuple) else (output, None)
        norm = self.backbone.norm
        if hidden.ndim == 2 and hidden.shape[0] >= 256:
            chunks = range(0, hidden.shape[0], 255)
            hidden = torch.cat([norm(hidden[i : i + 255], None) for i in chunks])
        else:
            hidden = norm(hidden, None)
        return hidden if aux is None else (hidden, aux)
