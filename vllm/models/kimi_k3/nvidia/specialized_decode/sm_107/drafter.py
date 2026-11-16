# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSpark drafter of the SM107 specialized decode.

The source branch wired the drafter's fast path (``runtime.draft.DraftFastPath``:
fused Markov sampling, LM-head GEMV, sharded context step with the fused
context-KV epilogue, Lamport all-reduce sites, the drafter's DCP attention) through
edits to the stock ``dspark_mla.py``, ``mla.py`` and the DSpark speculator. Here
those files stay unchanged: ``register`` makes vLLM build
``Sm107K3DSparkForCausalLM`` (a subclass of the stock drafter, same weights and
checkpoint names) for the DSpark draft architecture, its backbone and MLA layers are
class-swapped to the subclasses below, and the speculator's own extension point
(``sample_sequential``) runs the fused Markov drafting.
"""

import weakref
from typing import cast

import torch

import vllm._custom_ops as ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.kimi_k3.nvidia.dspark_mla import (
    _GROUPED_KV_CACHE_DTYPES,
    K3DSparkForCausalLM,
    K3DSparkModel,
)
from vllm.models.kimi_k3.nvidia.mla import MultiHeadLatentAttention
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

# The DSpark draft checkpoint's architecture (``ModelRegistry``).
DRAFT_ARCH = "K3DSparkModel"

# The drafter of this process (one per worker), for the target's rejection sampler:
# a verification that falls back to the stock sampler completes the draft-logits
# rows the fused probabilistic drafting left sharded (``gather_draft_logits``).
_CURRENT: list = [None]


def current_drafter() -> "Sm107K3DSparkForCausalLM | None":
    ref = _CURRENT[0]
    return None if ref is None else ref()


def register() -> None:
    """Serve the DSpark draft with ``Sm107K3DSparkForCausalLM`` in this process
    (``ModelRegistry.register_model``). The target model is built (and its decode
    config calls this) before the speculator loads the drafter; a draft class
    resolved earlier in this process is dropped so the load resolves again."""
    from vllm.model_executor.model_loader import utils as loader_utils
    from vllm.model_executor.models.registry import ModelRegistry

    ModelRegistry.register_model(DRAFT_ARCH, Sm107K3DSparkForCausalLM)
    cached = loader_utils._MODEL_ARCH_BY_HASH
    for key, (cls, _) in list(cached.items()):
        if (
            isinstance(cls, type)
            and issubclass(cls, K3DSparkForCausalLM)
            and cls is not Sm107K3DSparkForCausalLM
        ):
            del cached[key]


class Sm107DraftMLA(MultiHeadLatentAttention):
    """A drafter MLA layer with the source branch's ``draft_dcp`` hooks (set after
    the weights load by ``cutedsl.draft.attention``; None keeps the stock path)."""

    draft_dcp = None

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        super().bind_kv_cache(kv_cache)
        if self.draft_dcp is not None:  # the drafter's fast DCP attention
            self.draft_dcp.bind(self._attn_read_kv_cache())

    def _apply_q_lora_attention(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        qkv_lora: torch.Tensor,
    ) -> torch.Tensor:
        if self.draft_dcp is not None:
            # The fused query kernel from the in-projection rows (None: not covered).
            attn_out = self.draft_dcp.attend_projected(self, positions, qkv_lora)
            if attn_out is not None:
                return attn_out
        return super()._apply_q_lora_attention(positions, hidden_states, qkv_lora)

    # The stock ``MultiHeadLatentAttention._attention`` (this tree's, verbatim) with
    # the source branch's ``draft_dcp.attend`` after the decode query concat.
    @eager_break_during_capture
    def _attention(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        attn_out: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_by_layer = forward_context.attn_metadata
        if attn_metadata_by_layer is None:
            attn_out.zero_()
            return
        assert isinstance(attn_metadata_by_layer, dict)
        attn_metadata = cast(
            "MLACommonMetadata", attn_metadata_by_layer[self.layer_name]
        )

        num_actual_toks = attn_metadata.num_actual_tokens
        slot_mapping_by_layer = forward_context.slot_mapping
        assert isinstance(slot_mapping_by_layer, dict)
        slot_mapping = slot_mapping_by_layer[self.layer_name]

        q = q[:num_actual_toks]
        kv_c_normed = kv_c_normed[:num_actual_toks]
        k_pe = k_pe[:num_actual_toks]
        positions = positions[:num_actual_toks]
        attn_out = attn_out[:num_actual_toks]

        cos_sin_cache = None
        rope_positions = None
        if self.rotary_emb is not None:
            # Pass the fp32 cos/sin table straight to the fused epilogue (it reads
            # fp32 and does the RoPE math in fp32) -- no per-forward dtype cast.
            cos_sin_cache = self.rotary_emb.cos_sin_cache
            rope_positions = positions

        # Decode tokens are laid out first, prefill tokens after. The fused
        # prefill covers every supported config (bf16 / plain-fp8 /
        # fp8_ds_mla), so there is no dense-MHA (forward_mha) fallback.
        num_mqa_tokens = attn_metadata.num_decode_tokens
        num_mha_tokens = q.size(0) - num_mqa_tokens

        # Both the prefill and decode fused epilogues write their own cache
        # slice, so there is no separate do_kv_cache_update.

        # ---- Prefill: fused key-concat + cache-insert + attention ----
        if num_mha_tokens > 0:
            self._forward_prefill_fused(
                q[num_mqa_tokens:],
                kv_c_normed[num_mqa_tokens:],
                k_pe[num_mqa_tokens:],
                rope_positions[num_mqa_tokens:] if rope_positions is not None else None,
                cos_sin_cache,
                slot_mapping[num_mqa_tokens:num_actual_toks],
                attn_metadata,
                attn_out[num_mqa_tokens:],
            )

        # ---- Decode: latent multi-query attention ----
        if num_mqa_tokens > 0:
            mqa_q_nope, mqa_q_pe = q[:num_mqa_tokens].split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )
            # BMM1: absorb q_nope into latent space. (N,B,P) x (N,P,L) -> (B,N,L)
            ql_nope = torch.bmm(mqa_q_nope.transpose(0, 1), self.W_UK_T).transpose(0, 1)
            # Fused: concat mqa_q = [ql_nope | q_pe] and insert the decode-token
            # latent into the paged cache (one launch, right before forward_mqa).
            mqa_q = self._decode_concat_cache(
                ql_nope,
                mqa_q_pe,
                kv_c_normed[:num_mqa_tokens],
                k_pe[:num_mqa_tokens],
                rope_positions[:num_mqa_tokens] if rope_positions is not None else None,
                cos_sin_cache,
                slot_mapping[:num_mqa_tokens],
            )
            # The drafter's DCP attention on the specialized decode's Lamport
            # mailboxes (query gather, core, LSE combine + W_UV); False: not covered.
            if self.draft_dcp is not None and self.draft_dcp.attend(
                self, mqa_q, attn_metadata, attn_out[:num_mqa_tokens]
            ):
                return
            if self.dcp_world_size > 1:
                assert self.dcp_manager is not None
                assert self.dcp_manager.query_gather is not None
                mqa_q = self.dcp_manager.query_gather(mqa_q)
            latent_out, lse = self.impl.forward_mqa(  # type: ignore[attr-defined]
                mqa_q, self._attn_read_kv_cache(), attn_metadata, self
            )
            if self.dcp_world_size > 1:
                assert lse is not None
                assert self.dcp_manager is not None
                assert attn_metadata.decode is not None
                latent_out = self.dcp_manager.combine(
                    latent_out,
                    lse,
                    seq_lens=attn_metadata.decode.seq_lens,
                    query_start_loc=attn_metadata.query_start_loc[
                        : attn_metadata.num_decodes + 1
                    ],
                )
            self._v_up_proj(latent_out, out=attn_out[:num_mqa_tokens])



class Sm107K3DSparkModel(K3DSparkModel):
    """``K3DSparkModel`` with the source branch's fast-path hooks: the sharded
    context step (``_fast_context``), its fused context-KV epilogue, and the Lamport
    all-reduce + norm sites (``_fast_ar``). Each is set by ``DraftFastPath.build``
    after the weights load; unset or not covering a call, the stock code runs."""

    _fast_context = None
    _fast_ar = None

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        fast = self._fast_context
        if fast is not None:
            combined = fast.combine(hidden_states)
            if combined is not None:
                return combined
        return super().combine_hidden_states(hidden_states)

    def _precompute_fused_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None,
    ) -> None:
        if self._fused_context_ready(context_positions, context_slot_mapping):
            # One KV-only GEMM, then the specialized decode's fused epilogue
            # (per-layer RMSNorm, RoPE, FP8 insert of the owned slots).
            all_kv = self.context_kv_proj(context_states)
            if self._fused_context_kv(all_kv, context_positions, context_slot_mapping):
                return
            # Declined after the GEMM: the stock epilogue on the same rows (the
            # source branch's order: one GEMM either way).
            logger.warning_once(
                "Kimi K3 SM107 drafter: the fused context-KV epilogue declined a "
                "captured context step (%d rows); it runs the stock epilogue",
                all_kv.shape[0],
            )
            self._stock_context_kv_epilogue(
                all_kv, context_positions, context_slot_mapping
            )
            return
        super()._precompute_fused_context_kv(
            context_states, context_positions, context_slot_mapping
        )

    def _stock_context_kv_epilogue(
        self,
        all_kv: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None,
    ) -> None:
        """``K3DSparkModel._precompute_fused_context_kv`` after its KV-only GEMM
        (``all_kv`` = ``context_kv_proj(context_states)``): this tree's
        ``dspark_mla.py`` body after the GEMM, verbatim."""
        num_ctx = all_kv.shape[0]
        num_layers = self._num_context_layers

        all_kv = all_kv.view(num_ctx, num_layers, self._context_kv_width)
        all_kv_c = all_kv[..., : self._context_kv_lora_rank]
        all_k_pe = all_kv[..., self._context_kv_lora_rank :]

        # Layer-major layout lets the 2-D RMSNorm weights select a distinct row
        # for each draft layer in one grouped kernel.
        all_kv_c = all_kv_c.permute(1, 0, 2).contiguous()
        all_kv_c_normed = torch.empty_like(all_kv_c)
        ops.rms_norm(
            all_kv_c_normed,
            all_kv_c,
            self._context_kv_norm_weights,
            self._context_rms_norm_eps,
        )

        all_k_pe = all_k_pe.permute(1, 0, 2).contiguous()
        all_k_pe_flat = all_k_pe.view(num_layers * num_ctx, 1, self._context_rope_dim)
        (repeated_positions,) = current_workspace_manager().get_simultaneous(
            ((num_layers * self._max_num_context_tokens,), torch.int64),
        )
        repeated_positions = repeated_positions[: num_layers * num_ctx]
        repeated_positions.view(num_layers, num_ctx).copy_(context_positions)
        # Keep the single-tensor context RoPE on vLLM's optimized CUDA op;
        # DeepSeek YaRN's FlashInfer wrapper assumes a non-null key tensor.
        rotary_emb = self.layers[0].self_attn.rotary_emb
        assert rotary_emb is not None
        ops.rotary_embedding(
            repeated_positions,
            all_k_pe_flat,
            None,
            rotary_emb.head_size,
            rotary_emb.cos_sin_cache,
            rotary_emb.is_neox_style,
        )
        all_k_pe = all_k_pe_flat.view(num_layers, num_ctx, 1, self._context_rope_dim)

        if context_slot_mapping is None:
            return

        cache_layers = [layer.self_attn for layer in self.layers]
        cache_dtype = cache_layers[0].kv_cache_dtype
        if (
            cache_dtype in _GROUPED_KV_CACHE_DTYPES
            and all(cl.kv_cache_dtype == cache_dtype for cl in cache_layers)
            and self._has_uniform_block_layout(cache_layers)
            and (
                isinstance(context_slot_mapping, torch.Tensor)
                or all(s is not None for s in context_slot_mapping)
            )
        ):
            # Grouped context KV insert assumes that all layers share the same
            # cache dtype and block layout.

            if isinstance(context_slot_mapping, (list, tuple)):
                per_layer_slot_mappings = [
                    s for s in context_slot_mapping if s is not None
                ]
                if len({s.data_ptr() for s in per_layer_slot_mappings}) == 1:
                    # All rows alias to the same slot mapping.
                    slot_mapping = (
                        per_layer_slot_mappings[0].unsqueeze(0).expand(num_layers, -1)
                    )
                else:
                    slot_mapping = torch.stack(per_layer_slot_mappings, dim=0)
            else:
                # Broadcast the single shared context_slot_mapping tensor.
                slot_mapping = context_slot_mapping.unsqueeze(0).expand(num_layers, -1)

            ref_cache = cache_layers[0].kv_cache
            ops.concat_and_cache_mla_grouped(
                all_kv_c_normed,
                all_k_pe.squeeze(2),
                self._get_context_kv_cache_ptrs(cache_layers),
                slot_mapping,
                ref_cache.size(1),
                ref_cache.stride(0),
                ref_cache.stride(1),
                self._context_kv_scales,
                cache_dtype,
            )
            return

        for layer_idx, layer in enumerate(self.layers):
            slot_mapping = (
                context_slot_mapping[layer_idx]
                if isinstance(context_slot_mapping, (list, tuple))
                else context_slot_mapping
            )
            if slot_mapping is None:
                continue
            attn = layer.self_attn
            attn.impl.do_kv_cache_update(
                all_kv_c_normed[layer_idx],
                all_k_pe[layer_idx],
                attn.kv_cache,
                slot_mapping,
                attn.kv_cache_dtype,
                attn._k_scale,
            )

    def _fused_context_ready(self, positions, slot_mapping) -> bool:
        """The source branch's ``_fused_context_kv`` checks that do not need the
        projected rows (only inside CUDA graph captures, per-tensor FP8 caches of one
        block layout, int64 positions and slots)."""
        if (
            self._fast_context is None
            or slot_mapping is None
            or not torch.cuda.is_current_stream_capturing()
            or getattr(self, "_context_kv_scales", None) is None
        ):
            return False
        slots = self._single_slot_mapping(slot_mapping)
        if slots is None:
            return False
        cache_layers = [layer.self_attn for layer in self.layers]
        cache_dtype = cache_layers[0].kv_cache_dtype
        return (
            cache_dtype in ("fp8", "fp8_e4m3")
            and cache_dtype in _GROUPED_KV_CACHE_DTYPES
            and all(cl.kv_cache_dtype == cache_dtype for cl in cache_layers)
            and self._has_uniform_block_layout(cache_layers)
            and positions.dtype == torch.int64
            and slots.dtype == torch.int64
        )

    @staticmethod
    def _single_slot_mapping(slot_mapping):
        if isinstance(slot_mapping, (list, tuple)):
            if any(s is None for s in slot_mapping):
                return None
            if len({s.data_ptr() for s in slot_mapping}) != 1:
                return None
            return slot_mapping[0]
        return slot_mapping

    def _fused_context_kv(self, all_kv, positions, slot_mapping) -> bool:
        """The source branch's fused context-KV epilogue, when it covers this call."""
        if not all_kv.is_contiguous():
            return False
        slots = self._single_slot_mapping(slot_mapping)
        cache_layers = [layer.self_attn for layer in self.layers]
        ref = cache_layers[0].kv_cache
        return self._fast_context.store_kv(
            self,
            all_kv,
            positions,
            slots,
            self._get_context_kv_cache_ptrs(cache_layers),
            self._context_kv_scales,
            ref.size(1),
            ref.stride(0) * ref.element_size(),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fast = self._fast_ar
        if fast is not None and inputs_embeds is None:
            # The specialized decode's Lamport all-reduce + norm sites.
            out = fast.run(self, input_ids, positions)
            if out is not None:
                return out
        return super().forward(input_ids, positions, inputs_embeds)


class Sm107K3DSparkForCausalLM(K3DSparkForCausalLM):
    """The stock DSpark drafter with the source branch's specialized pieces, built in
    place of ``K3DSparkForCausalLM`` (``register``). Same weights, same checkpoint
    names; every hook falls through to the stock code where the fast path does not
    cover a call (the fast path itself refuses uncovered captured batches on the
    target topology, NQ-NOFALLBACK)."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._vllm_config = vllm_config
        self._fast_path = None
        self.model.__class__ = Sm107K3DSparkModel
        for layer in self.model.layers:
            attn = layer.self_attn
            if type(attn) is not MultiHeadLatentAttention:
                raise TypeError(
                    "Kimi K3 SM107 DSpark drafter: draft layers must be the stock "
                    f"MultiHeadLatentAttention, got {type(attn).__name__}"
                )
            attn.__class__ = Sm107DraftMLA

    def process_weights_after_loading(self) -> None:
        super().process_weights_after_loading()
        from .runtime.draft import DraftFastPath

        self._fast_path = DraftFastPath.build(self._vllm_config, self)
        _CURRENT[0] = weakref.ref(self)

    # -- the speculator's extension point -------------------------------------------
    def sample_sequential(self, speculator, num_reqs: int, head_hidden) -> bool:
        """The source branch's ``DSparkSpeculator._fused_greedy_markov``: every Markov
        position in one fused call (``greedy_markov_draft`` for greedy drafting,
        ``sampled_markov_draft`` for probabilistic drafting). Only plain
        anchor-as-first drafting qualifies; anything else, or a False from the fused
        path, keeps the speculator's loop (nothing launched)."""
        s = speculator
        if (
            getattr(s, "_draft_topk", None) is not None
            or getattr(s, "use_confidence_head", False)
            or getattr(s, "acceptance_estimator", None) is not None
            or getattr(s, "draft_watermarker", None) is not None
            or not s.sample_from_anchor
            or s.num_query_per_req != s.num_speculative_steps
        ):
            return False
        num_sample = num_reqs * s.num_speculative_steps
        # Anchor-as-first: sample row k is query row k, and each request's anchor is
        # its first query token.
        step = s.num_query_per_req
        anchors = s.input_buffers.input_ids[: num_reqs * step : step]
        draft = s.draft_tokens[:num_reqs]
        if s.draft_logits is None:
            return self.greedy_markov_draft(head_hidden[:num_sample], anchors, draft)
        if (
            s.use_fp64_gumbel
            or s._d2t_scatter_index is not None
            # The fused drafting writes rows by request state index.
            or s.draft_logits.row_mapping is not None
        ):
            return False
        return self.sampled_markov_draft(
            head_hidden[:num_sample],
            anchors,
            draft,
            s.sample_idx_mapping[:num_sample],
            s.sample_pos[:num_sample],
            s.temperature,
            s.seeds,
            s.draft_logits.values,
        )

    def greedy_markov_draft(
        self, hidden: torch.Tensor, anchors: torch.Tensor, draft: torch.Tensor
    ) -> bool:
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
    def shards_draft_logits(self) -> bool:
        fast = self._fast_path
        return fast is not None and fast.sampled and bool(fast.batches)

    # The source branch's context-step overlap hooks; this tree's speculator does
    # not call them (the overlap is not wired: see the port's notes).
    def context_overlap(self, num_reqs: int) -> bool:
        fast = self._fast_path
        return fast is not None and fast.context_overlap(self, num_reqs)

    def defer_context_join(self, ready: torch.cuda.Event) -> None:
        self._fast_path.defer_context_join(self, ready)
