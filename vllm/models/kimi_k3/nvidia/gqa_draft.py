# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GQA drafts (e.g. a Qwen3 sliding-window DSpark) on Kimi-K3: their KV cache grouping.

Dense MLA drafts keep their own MLA cache (``dense_draft.py``)."""

from dataclasses import replace

from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import SupportsKVCacheGrouping
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)

# A multiple of K3.1's 64-token prefix-hash unit. Draft blocks also divide the
# target's block, so the scheduler block (their LCM) does not change.
_DRAFT_BLOCK_ALIGN = 64


def _fit_draft_page(spec: AttentionSpec, page_size: int) -> AttentionSpec:
    """Shrink a draft's block until its page fits under ``page_size``."""
    for block_size in range(spec.block_size, 0, -_DRAFT_BLOCK_ALIGN):
        fitted = replace(spec, block_size=block_size)
        if spec.block_size % block_size == 0 and fitted.page_size_bytes <= page_size:
            return fitted
    raise ValueError(
        f"No draft block size fits the target's {page_size}-byte KV page: {spec}"
    )


def _flash_attn_key_tile(head_size: int) -> int:
    """Keys per tile of FlashAttention-2's paged (split-KV) forward
    (run_mha_fwd_splitkv_dispatch's kBlockN), its tiles aligned to it."""
    return 256 if head_size <= 64 else 128 if head_size <= 128 else 64


def _hit_unit_block(spec: AttentionSpec, hit_unit: int | None) -> AttentionSpec:
    """The draft's block at the prefix-cache hit granularity, when it divides the
    draft's block and a backend kernel takes it whole (packed blocks are never
    split into kernel blocks).

    A sliding window then keeps one more block below it when the attention
    kernel's key tile is wider than the block. FlashAttention (the draft's
    backend on P) loads whole tiles: a window starting in a tile's upper half
    loads the block below its first block too, masked, and a masked key still
    multiplies its V by 0. That block must hold finite bytes, so the KV cache
    manager keeps it (it frees blocks, and a hit restores them, by the
    ``sliding_window`` of the group's spec; the layers' attention window stays
    their own). Released, it may hold another group's bytes (packed slots share
    the target's rows: FP8 latents, FP32 KDA states); nulled, the null slot is in
    the null row every group's padding writes (the KDA states), and under DCP the
    worker does not null it at all.
    """
    alignment = spec.block_stride_alignment
    if (
        not hit_unit
        or hit_unit >= spec.block_size
        or spec.block_size % hit_unit
        or (isinstance(alignment, MultipleOf) and hit_unit > alignment.base)
    ):
        return spec
    blocked = replace(spec, block_size=hit_unit)
    if isinstance(spec, SlidingWindowSpec):
        tile = _flash_attn_key_tile(spec.head_size)
        if tile > hit_unit:
            # The tile holding the window start begins at most tile - hit_unit
            # tokens below the window's first block.
            blocked = replace(
                blocked, sliding_window=spec.sliding_window + tile - hit_unit
            )
    return blocked
def _dsa_dp_kv(vllm_config: VllmConfig) -> bool:
    from vllm.models.kimi_k3.nvidia.ops import dsa_dp

    pc = vllm_config.parallel_config
    return (
        dsa_dp.KV_ENABLED
        and pc.tensor_parallel_size > 1
        and pc.decode_context_parallel_size == 1
    )


class GQADraftKVCacheGrouping(SupportsKVCacheGrouping):
    """Kimi-K3's KV cache grouping when a GQA draft is attached; otherwise the
    default grouping.

    Kimi-K3's own layers are KDA (Mamba) and MLA, so any other attention layer
    belongs to a draft. Grouping the draft on its own keeps the target's groups:
    the default grouping sizes every group by the draft's few layers (20 groups
    instead of 4 + 1 for a 5-layer draft).

    Layer-outer layouts give all groups one page size and MLA pages cannot be
    padded, so the draft's page must fit under the target's: with all KV heads
    on one rank (attention TP1 under DP) it does not, and the draft gets the
    largest block that fits.

    Block-outermost layouts (K3.1's packed DSA rows) size every block by the
    widest group, the DSA row, which a draft block would take whole: 3-4 rows a
    request for a 2048-token window. The draft's blocks are cut to the
    prefix-cache hit unit instead and packed as slots of the rows
    (``blocks_per_pool_block``): a request's window takes a fraction of a row,
    and a window tail ends on every hit boundary, so the draft keeps the
    target's fine-grained prefix-cache hits.
    """

    @classmethod
    def get_kv_cache_groups(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: dict[str, KVCacheSpec],
    ) -> list[KVCacheGroupSpec]:
        from vllm.v1.core.kv_cache_planning import (
            _get_kv_cache_bytes_per_block,
            get_group_slot_alignment,
            get_group_slot_bytes,
            get_kv_cache_groups,
            get_slots_per_pool_block,
            unify_kv_cache_spec_page_size,
        )

        if _dsa_dp_kv(vllm_config):
            # DSA-DP per-card DSA KV: the draft's MLA layers run on every card,
            # so they must not share a block table with the owner-local DSA
            # layers (one uniform-type group by default).
            mla_draft = {
                n: s
                for n, s in kv_cache_spec.items()
                if isinstance(s, MLAAttentionSpec) and s.is_draft
            }
            if mla_draft and len(mla_draft) < len(kv_cache_spec):
                target = {n: s for n, s in kv_cache_spec.items() if n not in mla_draft}
                return get_kv_cache_groups(vllm_config, target) + get_kv_cache_groups(
                    vllm_config, mla_draft
                )
        draft = {
            name: spec
            for name, spec in kv_cache_spec.items()
            if isinstance(spec, AttentionSpec)
            and not isinstance(spec, MLAAttentionSpec)
        }
        target = {n: s for n, s in kv_cache_spec.items() if n not in draft}
        if not draft or not target:
            return get_kv_cache_groups(vllm_config, kv_cache_spec)
        layout = vllm_config.cache_config.get_resolved_kv_cache_layout()
        if not layout.is_block_outermost:
            page_size = max(spec.page_size_bytes for spec in target.values())
            draft = {n: _fit_draft_page(s, page_size) for n, s in draft.items()}
            specs = unify_kv_cache_spec_page_size(target | draft)
            return get_kv_cache_groups(
                vllm_config, {n: specs[n] for n in target}
            ) + get_kv_cache_groups(vllm_config, {n: specs[n] for n in draft})

        hit_unit = vllm_config.cache_config.prefix_match_unit
        draft = {n: _hit_unit_block(s, hit_unit) for n, s in draft.items()}
        # Like the target beside an MLA draft, whose pages made its DSA bucket
        # unbalanced: the KDA states take the fewest groups the row fits.
        groups = get_kv_cache_groups(
            vllm_config, target | draft, fewest_state_groups=True
        )
        draft_groups = [g for g in groups if draft.keys() >= set(g.layer_names)]
        pool_block_bytes = _get_kv_cache_bytes_per_block(
            [g for g in groups if g not in draft_groups], layout
        )
        for group in draft_groups:
            group.blocks_per_pool_block = get_slots_per_pool_block(
                get_group_slot_bytes(group),
                pool_block_bytes,
                get_group_slot_alignment(group),
            )
        return groups
