# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1's release DSpark draft (Inferact/Kimi-K3.1-Pro-DSpark) as the Qwen3
DSpark GQA drafter (``qwen3_dspark.Qwen3DSparkForCausalLM``: 5 sliding-window GQA
layers of 96 query / 16 KV heads of 64, a 35840 -> 7168 context ``fc``, a rank-512
Markov head, the target's embedding and LM head), with the stock math and the
hooks the DSpark speculator probes for the specialized decode's drafter pieces
(``runtime.draft.GqaDraftFastPath``, built after the weights load).

Differences from the stock class (the same math; summation order aside):
  * o_proj / down_proj keep their partial sums (``reduce_results=False``); the
    all-reduce is fused into the RMSNorm after it (``fused_allreduce_rms_norm``), as
    the preview drafter (``dspark_mla``) does.
  * The context step normalizes in ``combine_hidden_states``: it returns
    ``hidden_norm(fc(aux))`` (stock: ``fc(aux)``, normalized by
    ``precompute_and_store_context_kv``; both are BF16 at the same points), which
    lets the specialized decode fuse the norm into its row gather (CtxGather).
  * The eager context step (``combine_aux_rows``) row-shards the replicated ``fc``
    over the TP group from 256 rows per rank (P/D prefill instances' 16K chunks).
  * The forward is not torch.compiled (``ignore_torch_compile``): its specialized
    pieces run inside the DSpark speculator's FULL graphs, as the preview's.
  * The context K/V buffers (the fused K/V weight of every layer) are built before
    the specialized pieces wrap the qkv projections, also when the load skipped
    ``load_weights`` (dummy weights), where the stock class builds them lazily.

``configure_dense_draft`` (dense_draft.py) routes a speculators ``dspark``
checkpoint (``Qwen3DSparkModel``) on a Kimi K3.1 target to this class.
"""

from collections.abc import Iterable

import torch
import torch.nn as nn

import vllm._custom_ops as ops
from vllm.compilation.decorators import ignore_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3DecoderLayer
from vllm.model_executor.models.qwen3_dspark import (
    Qwen3DSparkForCausalLM,
    Qwen3DSparkModel,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.utils import weights_already_processed
from vllm.models.common.ops.fused_allreduce_rms_norm import fused_allreduce_rms_norm
from vllm.models.common.ops.sequence_parallel import sp_all_gather
from vllm.models.kimi_k3.nvidia.dspark_mla import (
    _CONTEXT_ROW_SHARD_MIN_ROWS,
    _aux_rows,
)
from vllm.models.kimi_k3.nvidia.ops import prefill_glue
from vllm.models.kimi_k3.nvidia.specialized_decode.shapes import count_drafter_stock

logger = init_logger(__name__)


def _head_norm(norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Per-head RMSNorm ``norm(x)`` of [tokens, heads, head_dim]; long context
    steps take the bitwise-equal prefill replay of the stock kernel (whose
    8-thread blocks run ~20x over the byte floor)."""
    weight = norm.weight.data
    if x.dim() == 3 and prefill_glue.use_head_rms_norm(x, weight):
        return prefill_glue.rms_norm(x, weight, norm.variance_epsilon)
    return norm(x)


def stock_attention_core(attn: nn.Module, qkv: torch.Tensor, positions: torch.Tensor):
    """``DFlashQwen3Attention.forward`` between its qkv and o_proj GEMMs: the
    per-head q / k RMSNorm, RoPE, the value scale and the attention (with the K/V
    insert of the query block); [rows, q_size]."""
    q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
    q_shape, k_shape = q.shape, k.shape
    q = _head_norm(
        attn.q_norm,
        q.view(*q_shape[:-1], q_shape[-1] // attn.head_dim, attn.head_dim),
    ).view(q_shape)
    k = _head_norm(
        attn.k_norm,
        k.view(*k_shape[:-1], k_shape[-1] // attn.head_dim, attn.head_dim),
    ).view(k_shape)
    q, k = attn.rotary_emb(positions, q, k)
    if attn.v_scale is not None:
        v = v * attn.v_scale
    out = _eager_window_attention(attn.attn, q, k, v)
    if out is not None:
        return out
    return attn.attn(q, k, v)


def _no_window_attention(why: str, impl, layer) -> None:
    count_drafter_stock(f"draft attention: {why.split(' ')[0]}", 0)
    logger.info_once(
        "Kimi K3.1 DSpark draft attention: stock path (%s: %s %s)",
        why, type(impl).__name__, getattr(impl, "vllm_flash_attn_version", None),
    )  # fmt: skip
    return None


def _eager_window_attention(layer: nn.Module, q, k, v) -> torch.Tensor | None:
    """Draft steps without the specialized GQA attention: the query block's K/V
    insert as the backend does it, then ``draft_attention.attention`` (FlashAttention
    2 math, keys split over the window) in place of FlashAttention 2's one-split
    schedule. None where it does not cover the call (the caller runs the layer)."""
    from vllm.forward_context import get_forward_context, is_forward_context_available
    from vllm.models.kimi_k3.nvidia.specialized_prefill import draft_attention
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends.flash_attn import (
        FlashAttentionImpl,
        FlashAttentionMetadata,
    )

    impl = getattr(layer, "impl", None)
    if (
        not is_forward_context_available()
        or type(impl) is not FlashAttentionImpl
        or impl.vllm_flash_attn_version != 2
        or impl.attn_type != AttentionType.DECODER
        or impl.kv_cache_dtype not in ("auto", "bfloat16")
        or impl.alibi_slopes is not None
        or impl.logits_soft_cap not in (None, 0, 0.0)
        or getattr(impl, "sinks", None) is not None
        or impl.kv_sharing_target_layer_name is not None
        or q.dtype != torch.bfloat16
    ):
        return _no_window_attention("layer/impl", impl, layer)
    meta = get_forward_context().attn_metadata
    if isinstance(meta, dict):
        meta = meta.get(layer.layer_name)
    if (
        type(meta) is not FlashAttentionMetadata
        or not isinstance(meta.causal, bool)
        or meta.max_dcp_context_kv_len
        or meta.rswa_prefix_lens is not None
        or meta.mm_prefix_query_range_tensor is not None
        or meta.num_actual_tokens > q.shape[0]
        or not isinstance(layer.kv_cache, torch.Tensor)
    ):
        why = (
            f"metadata {type(meta).__name__} causal "
            f"{getattr(meta, 'causal', None)!r} rows "
            f"{getattr(meta, 'num_actual_tokens', None)}/{q.shape[0]}"
        )
        return _no_window_attention(why, impl, layer)
    hd, nkv = impl.head_size, impl.num_kv_heads
    key_cache, value_cache = layer.kv_cache.transpose(1, 2).split(hd, dim=-1)
    if not draft_attention.covers(
        meta.max_query_len, impl.num_heads, nkv, hd, key_cache, value_cache
    ):
        return _no_window_attention("shape", impl, layer)
    # FlashAttention's causal mask is its local window with no right side.
    window = tuple(impl.sliding_window)
    if meta.causal:
        window = (window[0], 0)
    tokens = q.shape[0]
    impl.do_kv_cache_update(
        layer, k.view(tokens, nkv, hd), v.view(tokens, nkv, hd), layer.kv_cache,
        meta.slot_mapping,
    )  # fmt: skip
    alloc = torch.empty if meta.num_actual_tokens == tokens else torch.zeros
    out = alloc((tokens, impl.num_heads, hd), dtype=q.dtype, device=q.device)
    draft_attention.attention(
        q.view(tokens, impl.num_heads, hd), key_cache, value_cache,
        meta.block_table, meta.seq_lens, meta.query_start_loc, meta.max_query_len,
        impl.scale, window, out,
    )  # fmt: skip
    return out.view(tokens, impl.num_heads * hd)


def _uncovered(states: torch.Tensor, why: str) -> bool:
    """Log (once) why a long context step keeps the stock K/V path; False."""
    count_drafter_stock(f"context K/V: {why}", states.shape[0])
    logger.info_once(
        "Kimi K3.1 DSpark context K/V: stock path at %d rows (%s)",
        states.shape[0], why,
    )  # fmt: skip
    return False


class K3GqaDSparkDecoderLayer(DFlashQwen3DecoderLayer):
    """A DFlash Qwen3 layer whose o_proj and down_proj partial sums are all-reduced
    by the RMSNorm that follows them (``fused_allreduce_rms_norm``)."""

    def __init__(self, vllm_config: VllmConfig, *, config, layer_idx: int, **kwargs):
        super().__init__(vllm_config, config=config, layer_idx=layer_idx, **kwargs)
        self.layer_idx = layer_idx
        self.self_attn.o_proj.reduce_results = False
        self.mlp.down_proj.reduce_results = False
        # The specialized decode's fused GQA attention (``runtime.draft
        # GqaAttention``): ``forward(index, qkv, positions)`` returns the
        # attention output, or None when this call is not covered.
        self.fast_attention = None

    def attention_core(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """The attention output before o_proj ([rows, q_size])."""
        attn = self.self_attn
        qkv, _ = attn.qkv_proj(hidden_states)
        fast = self.fast_attention
        if fast is not None:
            out = fast.forward(self.layer_idx, qkv, positions)
            if out is not None:
                return out
        return stock_attention_core(attn, qkv, positions)

    def attention(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """This rank's partial o_proj output (not all-reduced)."""
        out, _ = self.self_attn.o_proj(self.attention_core(positions, hidden_states))
        return out

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            # First layer: hidden_states is the (all-reduced) embedding.
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = fused_allreduce_rms_norm(
                hidden_states, residual, self.input_layernorm
            )
        hidden_states = self.attention(positions, hidden_states)
        hidden_states, residual = fused_allreduce_rms_norm(
            hidden_states, residual, self.post_attention_layernorm
        )
        # Partial; reduced by the next layer's input_layernorm or the final norm.
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@ignore_torch_compile
class K3GqaDSparkModel(Qwen3DSparkModel):
    """The Qwen3 DSpark backbone with fused all-reduce norms and the context step's
    norm in ``combine``."""

    decoder_layer_cls = K3GqaDSparkDecoderLayer

    def __init__(
        self, *, vllm_config: VllmConfig, start_layer_id: int = 0, prefix: str = ""
    ) -> None:
        super().__init__(
            vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix
        )
        if not self.use_aux_hidden_state:
            raise ValueError(
                "Kimi K3.1 GQA DSpark drafts read the target's aux hidden states "
                "(use_aux_hidden_state)"
            )
        # The specialized decode's drafter pieces (runtime.draft.GqaDraftFastPath):
        # the all-reduce + norm sites of the forward, the sharded context step
        # (fc shard + row gather + hidden_norm) and the context K/V projection.
        self._fast_ar = None
        self._fast_context = None
        self._context_projection = None
        # The context K/V insert of the fused GQA attention (or None).
        self._fast_context_kv = None

    def combine(
        self, hidden_states: torch.Tensor, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        """hidden_norm(fc(hidden_states)) [rows, hidden], written to ``out`` (a
        contiguous [rows, hidden] buffer) when given."""
        fast = self._fast_context
        if fast is not None:
            combined = fast.combine(hidden_states)
            if combined is not None:
                if out is None:
                    return combined
                return out.copy_(combined)
        projected = self.fc(hidden_states.to(self.fc.params_dtype))
        return self.normalize(projected, out=out)

    def normalize(
        self, rows: torch.Tensor, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``hidden_norm`` with the stock context step's kernel (``ops.rms_norm``),
        or its bitwise-equal prefill replay on long context steps."""
        weight = self.hidden_norm.weight.data
        if out is None:
            out = torch.empty_like(rows)
        if prefill_glue.use_rms_norm(rows, weight):
            return prefill_glue.rms_norm(
                rows, weight, self.hidden_norm.variance_epsilon, out=out
            )
        ops.rms_norm(
            out, rows, self.hidden_norm.weight.data, self.hidden_norm.variance_epsilon
        )
        return out

    def _normalize_context_states(self, context_states: torch.Tensor) -> torch.Tensor:
        # ``combine`` normalized the rows already.
        return context_states

    def _project_context_kv_token_major(
        self,
        context_states: torch.Tensor,
        num_ctx: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        """Context K/V of every layer, ``[num_ctx, L, 2, nkv, hd]``: the split-K
        projection of the fused K/V weight for its prepared rows inside CUDA graph
        captures, else the stock projection."""
        prepared = self._context_projection
        if (
            prepared is not None
            and self._fused_kv_weight is not None
            and num_ctx in prepared.kernels
            and context_states.is_contiguous()
            and torch.cuda.is_current_stream_capturing()
        ):
            out = context_states.new_empty((num_ctx, self._fused_kv_weight.shape[0]))
            prepared.run(context_states, out)
            return out.view(num_ctx, num_layers, 2, num_kv_heads, head_dim)
        return super()._project_context_kv_token_major(
            context_states, num_ctx, num_layers, num_kv_heads, head_dim
        )

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        """The stock context K/V step on the normalized rows (``combine``), or the
        fused GQA attention's K/V insert (per-layer k_norm, RoPE, insert) when it
        covers this call."""
        if self._fused_context_kv(
            context_states, context_positions, context_slot_mapping
        ):
            return
        if self._store_context_kv_rows(
            context_states, context_positions, context_slot_mapping
        ):
            return
        super().precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mapping
        )

    def _store_context_kv_rows(self, states, positions, slot_mapping) -> bool:
        """Context steps without the specialized decode fast path: the stock K/V
        step with the RoPE and the per-layer cache insert in one pass per layer
        (``prefill_glue.rope_store_kv``, bitwise the stock ``rotary_embedding`` +
        ``reshape_and_cache_flash``) and the K norm read from the token-major
        projection (no layer-major copy). False (nothing done) where it does not
        cover the call."""
        from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
        from vllm.v1.attention.backend import AttentionType

        if (
            slot_mapping is None
            or not hasattr(self, "_num_attn_layers")
            or getattr(self.layers[0].self_attn, "v_scale", None) is not None
            or not self._rope_is_neox
            or self._rope_head_size != self._head_dim
            or positions.dtype != torch.int64
        ):
            return _uncovered(
                states,
                f"rope "
                f"{self._rope_is_neox} {self._rope_head_size} positions "
                f"{positions.dtype} slots {slot_mapping is None}",
            )
        num_ctx, layers = states.shape[0], self._num_attn_layers
        nkv, hd = self._num_kv_heads, self._head_dim
        per_layer = isinstance(slot_mapping, (list, tuple))
        slots = [slot_mapping[i] if per_layer else slot_mapping for i in range(layers)]
        cos_sin = self._rope_cos_sin_cache
        if cos_sin.dtype != states.dtype:
            cos_sin = cos_sin.to(dtype=states.dtype)
        caches = []
        for i in range(layers):
            if slots[i] is None:
                caches.append(None)
                continue
            attn = self._attn_layers[i]
            impl = attn.impl
            if (
                type(impl).do_kv_cache_update
                is not FlashAttentionImpl.do_kv_cache_update
                or impl.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER)
                or impl.kv_cache_dtype not in ("auto", "bfloat16")
                or slots[i].numel() > num_ctx
                or not isinstance(attn.kv_cache, torch.Tensor)
            ):
                return _uncovered(
                    states,
                    f"KV cache layer {type(impl).__name__} {impl.attn_type} "
                    f"{impl.kv_cache_dtype} {type(attn.kv_cache).__name__} "
                    f"{slots[i].numel()}/{num_ctx}",
                )
            caches.append(attn.kv_cache.transpose(1, 2).split(hd, dim=-1))
        if all(c is None for c in caches):
            return _uncovered(states, "no slots")
        # Coverage on the K norm buffer (the projection's K / V row views differ
        # only in their row stride, 2 * L * nkv * hd elements: a multiple of 8).
        knorm = states.new_empty((num_ctx, nkv, hd))
        if not prefill_glue.use_head_rms_norm(knorm, self._k_norm_weights[0]):
            return _uncovered(states, "K norm shape")
        for i, cache in enumerate(caches):
            if cache is not None and not prefill_glue.use_rope_store_kv(
                knorm[: slots[i].numel()], knorm[: slots[i].numel()], positions,
                cos_sin, slots[i], *cache,
            ):  # fmt: skip
                return _uncovered(states, "cache insert layout")
        kv = self._project_context_kv_token_major(states, num_ctx, layers, nkv, hd)
        if tuple(kv.shape) != (num_ctx, layers, 2, nkv, hd) or not kv.is_contiguous():
            raise RuntimeError(
                "Kimi K3 DSpark context K/V: projection layout "
                f"{tuple(kv.shape)} / {kv.stride()} after the coverage check"
            )
        weights = [self._k_norm_weights[i] for i in range(layers)]
        if prefill_glue.use_ctx_kv_insert(
            kv, weights, self._rms_norm_eps, positions, cos_sin, slots, caches
        ):
            # Every layer's K norm + RoPE + insert in one launch (bitwise the
            # per-layer pair below).
            prefill_glue.ctx_kv_insert(
                kv, weights, self._rms_norm_eps, positions, cos_sin, slots, caches
            )
            logger.info_once("Kimi K3.1 DSpark context K/V: one-launch insert")
            return True
        for i in range(layers):
            if caches[i] is None:
                continue
            n = slots[i].numel()
            prefill_glue.rms_norm(
                kv[:, i, 0], self._k_norm_weights[i], self._rms_norm_eps, out=knorm
            )
            key_cache, value_cache = caches[i]
            args = (knorm[:n], kv[:n, i, 1], positions, cos_sin, slots[i])
            if not prefill_glue.use_rope_store_kv(*args, key_cache, value_cache):
                raise RuntimeError(
                    f"Kimi K3 DSpark context K/V: layer {i} insert not covered"
                )
            prefill_glue.rope_store_kv(*args, key_cache, value_cache)
        logger.info_once("Kimi K3.1 DSpark context K/V: fused RoPE + insert")
        return True

    def _normalize_context_k(self, all_k: torch.Tensor) -> torch.Tensor:
        """The stock grouped per-layer K RMSNorm ([L, ctx, kv_heads, head_dim],
        weight [L, head_dim]); long context steps replay it per layer
        bitwise (the stock 8-thread blocks run ~20x over the byte floor)."""
        weights = self._k_norm_weights
        if all_k.dim() != 4 or not prefill_glue.use_head_rms_norm(
            all_k[0], weights[0]
        ):
            return super()._normalize_context_k(all_k)
        out = torch.empty_like(all_k)
        for layer in range(all_k.shape[0]):
            prefill_glue.rms_norm(
                all_k[layer], weights[layer], self._rms_norm_eps, out=out[layer]
            )
        return out

    def _fused_context_kv(self, states, positions, slot_mapping) -> bool:
        fast = self._fast_context_kv
        if (
            fast is None
            or slot_mapping is None
            or not hasattr(self, "_num_attn_layers")
            or not torch.cuda.is_current_stream_capturing()
        ):
            return False
        if isinstance(slot_mapping, (list, tuple)):
            if any(s is None for s in slot_mapping):
                return False
            if len({s.data_ptr() for s in slot_mapping}) != 1:
                fast.uncovered_context(
                    states.shape[0], "draft layers in several KV-cache groups"
                )
                return False
            slot_mapping = slot_mapping[0]
        num_ctx = states.shape[0]
        kv = self._project_context_kv_token_major(
            states, num_ctx, self._num_attn_layers, self._num_kv_heads, self._head_dim
        )
        return fast.store_context(kv.view(num_ctx, -1), positions, slot_mapping)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fast = self._fast_ar
        if fast is not None and input_embeds is None:
            # The specialized decode's Lamport all-reduce + norm sites.
            out = fast.run(self, input_ids, positions)
            if out is not None:
                return out
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)
        hidden_states, residual = input_embeds, None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = fused_allreduce_rms_norm(hidden_states, residual, self.norm)
        return hidden_states


class K3GqaDSparkForCausalLM(Qwen3DSparkForCausalLM):
    """The release DSpark draft on Kimi K3.1 (module docstring)."""

    model_cls = K3GqaDSparkModel

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        self.draft_model_config = speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        self.target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != self.target_vocab_size:
            raise ValueError(
                "Kimi K3.1 GQA DSpark drafts share the target's vocabulary "
                f"({self.target_vocab_size}), got {self.config.draft_vocab_size}"
            )
        target_layer_num = vllm_config.model_config.get_total_num_hidden_layers()
        self.model = self.model_cls(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )
        # Assigned by load_dspark_model from the target (no transient full-vocab
        # shard allocation).
        self.lm_head: nn.Module | None = None
        self.draft_id_to_target_id = None
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size,
            scale=getattr(self.config, "logit_scale", 1.0),
        )
        self._vllm_config = vllm_config
        # The specialized decode's drafter pieces (built after the weights load).
        self._fast_path = None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        def checked(items):
            for name, weight in items:
                if "lm_head" in name or "embed_tokens" in name or "d2t" in name:
                    raise ValueError(
                        "Kimi K3.1 GQA DSpark drafts share the target's embedding and "
                        f"LM head; the checkpoint ships {name}"
                    )
                yield name, weight

        return super().load_weights(checked(weights))

    def combine_hidden_states(
        self, hidden_states: torch.Tensor, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        """hidden_norm(fc(cat(aux))): the context rows, normalized (module
        docstring); written to ``out`` ([rows, hidden], contiguous) when given."""
        needs_squeeze = hidden_states.dim() == 1
        if needs_squeeze:
            hidden_states = hidden_states.unsqueeze(0)
        expected = self.model.fc.input_size
        if hidden_states.shape[-1] != expected:
            raise ValueError(
                f"DSpark drafter expects {expected} concatenated aux hidden "
                f"features but received {hidden_states.shape[-1]}."
            )
        if out is not None and (
            needs_squeeze
            or not out.is_contiguous()
            or out.shape != (hidden_states.shape[0], self.model.fc.output_size)
        ):
            raise ValueError(
                f"DSpark combine: out {tuple(out.shape)} / {out.stride()} for "
                f"{tuple(hidden_states.shape)} rows"
            )
        result = self.model.combine(hidden_states, out=out)
        return result.squeeze(0) if needs_squeeze else result

    def combine_aux_rows(
        self,
        aux_hidden_states: list[torch.Tensor],
        start: int,
        end: int,
        out: torch.Tensor,
    ) -> None:
        """Eager context step: ``out`` [end - start, hidden] =
        ``combine_hidden_states(cat(aux rows [start, end)))``. Every TP rank holds the
        same target rows and ``fc`` is replicated, so from
        ``_CONTEXT_ROW_SHARD_MIN_ROWS`` rows per rank each rank combines a contiguous
        1/tp of the rows and the TP group all-gathers the result. All TP ranks call
        this with the same rows. Sequence-shard aux states (``SpLocalAux``: the
        target under sequence parallelism hands each rank its own 1/tp shard,
        ops/sp_local_aux.py) take the base class's shard-local projection and
        all-gather instead."""
        from vllm.models.kimi_k3.nvidia.ops import sp_local_aux

        if isinstance(aux_hidden_states, sp_local_aux.SpLocalAux):
            super().combine_aux_rows(aux_hidden_states, start, end, out)
            return
        num_rows = end - start
        if num_rows <= 0:
            return
        tp = get_tensor_model_parallel_world_size()
        if tp == 1 or num_rows < tp * _CONTEXT_ROW_SHARD_MIN_ROWS:
            rows = _aux_rows(aux_hidden_states, start, end)
            if out.is_contiguous():
                # The norm writes the speculator's rows directly (no copy).
                self.combine_hidden_states(rows, out=out)
            else:
                out.copy_(self.combine_hidden_states(rows))
            return
        chunk = -(-num_rows // tp)
        lo = start + get_tensor_model_parallel_rank() * chunk
        hi = min(end, lo + chunk)
        local = self.combine_hidden_states(_aux_rows(aux_hidden_states, lo, hi))
        if hi - lo < chunk:  # the last rank's short shard: pad to the gather size
            padded = local.new_zeros((chunk, local.shape[-1]))
            padded[: hi - lo].copy_(local)
            local = padded
        out.copy_(sp_all_gather(local)[:num_rows])

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.lm_head is not None
        return self.logits_processor(self.lm_head, hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.compute_draft_logits(hidden_states)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids

    def process_weights_after_loading(self) -> None:
        from vllm.models.kimi_k3.nvidia import cute_cache
        from vllm.models.kimi_k3.nvidia.mxfp8 import enable_kimi_k3_mxfp8_dispatch
        from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.draft import (
            GqaDraftFastPath,
        )

        if not hasattr(self.model, "_num_attn_layers"):
            # The context step's fused K/V buffers: the stock class builds them in
            # load_weights, which a dummy-weight load skips; its lazy build (first
            # context step) would come after the fast path wraps the qkv projections
            # and miss the fused K/V weight (every layer's whole qkv GEMM over the
            # context rows). Built here as the lazy build does, before the fast path.
            with weights_already_processed():
                self.model._build_fused_kv_buffers()
        if switched := enable_kimi_k3_mxfp8_dispatch(self):
            logger.info_once(
                "Kimi K3.1 GQA DSpark drafter: %d online MXFP8 linears use the CuTe "
                "DSL (<= 16 rows) / cuBLASLt MXFP8 dispatch.",
                switched,
            )
        # Ranks split the drafter's cold compiles; all loaded when the build returns.
        with cute_cache.shared_phase("specialized GQA drafter built"):
            self._fast_path = GqaDraftFastPath.build(self._vllm_config, self)

    def context_overlap(self, num_reqs: int) -> bool:
        """Whether a captured draft of ``num_reqs`` requests may run its context step
        on a side stream, joined before the first draft layer's attention core
        (``defer_context_join``)."""
        fast = self._fast_path
        return fast is not None and fast.context_overlap(self, num_reqs)

    def defer_context_join(self, ready: torch.cuda.Event) -> None:
        self._fast_path.defer_context_join(self, ready)

    def greedy_markov_draft(
        self, hidden: torch.Tensor, anchors: torch.Tensor, draft: torch.Tensor
    ) -> bool:
        """Greedy sequential Markov drafting of every position in one fused call
        (``K3DSparkForCausalLM.greedy_markov_draft``)."""
        fast = self._fast_path
        return fast is not None and fast.greedy_markov(self, hidden, anchors, draft)

    def sampled_markov_draft(
        self,
        hidden: torch.Tensor,
        anchors: torch.Tensor,
        draft: torch.Tensor,
        slots: torch.Tensor,
        positions: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        draft_logits: torch.Tensor,
    ) -> bool:
        """Probabilistic sequential Markov drafting in one fused call, drawing as the
        stock loop's ``gumbel_sample`` (``K3DSparkForCausalLM.sampled_markov_draft``);
        this rank writes its vocabulary columns of the draft-logits cache."""
        fast = self._fast_path
        sampling = (temperature, seeds, slots, positions, draft_logits)
        return fast is not None and fast.greedy_markov(
            self, hidden, anchors, draft, sampling
        )

    def gather_draft_logits(self, draft_logits: torch.Tensor, slots: torch.Tensor):
        """Complete ``draft_logits`` rows of ``slots`` written by
        ``sampled_markov_draft`` (each rank's columns) on every rank."""
        fast = self._fast_path
        if fast is not None:
            fast.gather_draft_logits(draft_logits, slots)

    @property
    def requires_fused_markov(self) -> bool:
        """Whether every Markov draft must take the fused path: the strict
        (target-topology) drafter refuses a speculator-side stock-loop bypass."""
        fast = self._fast_path
        return fast is not None and fast.strict

    @property
    def shards_draft_logits(self) -> bool:
        """Whether probabilistic drafting may leave only this rank's columns of the
        draft-logits cache rows (``sampled_markov_draft``)."""
        fast = self._fast_path
        return fast is not None and fast.sampled and bool(fast.batches)
