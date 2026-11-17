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


def _hit_unit_block(spec: AttentionSpec, hit_unit: int | None) -> AttentionSpec:
    """The draft's block at the prefix-cache hit granularity, when it divides the
    draft's block and a backend kernel takes it whole (packed blocks are never
    split into kernel blocks)."""
    alignment = spec.block_stride_alignment
    if (
        not hit_unit
        or hit_unit >= spec.block_size
        or spec.block_size % hit_unit
        or (isinstance(alignment, MultipleOf) and hit_unit > alignment.base)
    ):
        return spec
    return replace(spec, block_size=hit_unit)


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
