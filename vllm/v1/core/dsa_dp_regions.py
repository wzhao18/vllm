# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-card DSA KV regions for DSA-DP (``models/kimi_k3/nvidia/ops/dsa_dp.py``).

Under DSA-DP (``models/kimi_k3/nvidia/ops/dsa_dp.py``) each request has one
owner card in its TP group, and only the owner reads or writes the request's
DSA KV. The block-id space is split in two:

- the shared region ``[0, S)`` (the coordinator's ``BlockPool``): KDA state,
  the drafter and any other group, backed on every card;
- one region per owner card ``c``: ``[S + c * D, S + (c + 1) * D)``, holding the
  DSA groups' blocks of the requests ``c`` owns. Card ``c`` backs it at its
  local pages ``[S, S + D)``.

Each card allocates ``S + D`` pages; the scheduler addresses ``S + tp * D``.
"""

import weakref
from collections.abc import Iterable, Sequence
from typing import Any

from vllm.v1.core.block_pool import BlockHashToBlockMap, BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.hisparse.block_pool import SharedEventQueueBlockPool


def _pool_of(block: KVCacheBlock) -> BlockPool | None:
    """The pool that allocated ``block`` (``block.pool`` is a weak reference to it
    since KVCacheBlock stays out of the cyclic GC)."""
    pool = block.pool
    return pool() if isinstance(pool, weakref.ref) else pool


class _OffsetBlocks:
    """``BlockPool.blocks`` of a region: indexed by global block id."""

    def __init__(self, blocks: list[KVCacheBlock], first_block_id: int) -> None:
        self._blocks = blocks
        self._first = first_block_id

    def __getitem__(self, block_id: int) -> KVCacheBlock:
        return self._blocks[block_id - self._first]

    def __iter__(self):
        return iter(self._blocks)

    def __len__(self) -> int:
        return len(self._blocks)


class DSARegionBlockPool(SharedEventQueueBlockPool):
    """``num_blocks`` blocks with ids ``[first_block_id, first_block_id +
    num_blocks)``. The null block is the shared pool's (id 0)."""

    def __init__(
        self,
        first_block_id: int,
        num_blocks: int,
        shared_pool: BlockPool,
    ) -> None:
        super().__init__(
            num_gpu_blocks=num_blocks + 1,
            enable_caching=shared_pool.enable_caching,
            hash_block_size=shared_pool.hash_block_size,
            enable_kv_cache_events=shared_pool.enable_kv_cache_events,
            num_hash_lookahead_tokens=shared_pool.num_hash_lookahead_tokens,
            medium=shared_pool.medium,
            event_owner=shared_pool,
        )
        # Block 0 was popped as this pool's null block; drop it and renumber
        # the others into the region.
        blocks = self.blocks[1:]
        for i, block in enumerate(blocks):
            block.block_id = first_block_id + i
        self.blocks = _OffsetBlocks(blocks, first_block_id)  # type: ignore[assignment]
        # The hash map resolves cached block ids through the block list.
        self.cached_block_hash_to_block = BlockHashToBlockMap(self.blocks)
        self.null_block = shared_pool.null_block
        self.first_block_id = first_block_id
        self.demotion_gate = shared_pool.demotion_gate

    def evict_blocks(self, block_ids: set[int]) -> None:
        for block_id in block_ids:
            self._maybe_evict_cached_block(self.blocks[block_id])


class _BlocksById:
    def __init__(self, regions: "DSARegionPools") -> None:
        self._regions = regions

    def __getitem__(self, block_id: int) -> KVCacheBlock:
        return self._regions.pool_of(block_id).blocks[block_id]


class DSABlockIdView:
    """The shared pool and every DSA region addressed by global block id, for
    callers that hold block ids of all groups (the KV connectors). Other
    attributes are the shared pool's."""

    def __init__(self, regions: "DSARegionPools") -> None:
        self._regions = regions
        self.blocks = _BlocksById(regions)

    def _by_pool(
        self, blocks: Iterable[KVCacheBlock]
    ) -> dict[BlockPool, list[KVCacheBlock]]:
        out: dict[BlockPool, list[KVCacheBlock]] = {}
        for block in blocks:
            out.setdefault(self._regions.pool_of(block.block_id), []).append(block)
        return out

    def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
        for pool, group in self._by_pool(blocks).items():
            pool.touch(group)

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        for pool, group in self._by_pool(ordered_blocks).items():
            pool.free_blocks(group)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__dict__["_regions"].shared_pool, name)


class DSARegionPools:
    """The DSA groups' ``block_pool``: one ``DSARegionBlockPool`` per owner card.

    Calls that take blocks go to the blocks' own region; the others (new
    blocks, free counts, prefix-cache lookups) go to the bound owner's region.
    The KV cache manager binds the owner of the request it serves
    (``bind`` / ``bind_request_id``).
    """

    def __init__(
        self, shared_pool: BlockPool, shared_blocks: int, region_blocks: int, tp: int
    ) -> None:
        assert shared_pool.num_gpu_blocks == shared_blocks
        self.shared_pool = shared_pool
        self.shared_blocks = shared_blocks
        self.region_blocks = region_blocks
        self.regions = [
            DSARegionBlockPool(
                shared_blocks + c * region_blocks, region_blocks, shared_pool
            )
            for c in range(tp)
        ]
        self.owner = 0
        self.owner_of: dict[str, int] = {}
        self.null_block = shared_pool.null_block
        self.by_block_id = DSABlockIdView(self)

    # Owner binding.
    def bind(self, request_id: str, owner: int) -> None:
        self.owner_of[request_id] = owner
        self.owner = owner

    def bind_request_id(self, request_id: str) -> None:
        owner = self.owner_of.get(request_id)
        if owner is not None:
            self.owner = owner

    def forget(self, request_id: str) -> None:
        self.owner_of.pop(request_id, None)

    @property
    def pool(self) -> DSARegionBlockPool:
        return self.regions[self.owner]

    def region_of(self, block_id: int) -> int:
        """Owner card of a region block id, or -1 for a shared id."""
        if block_id < self.shared_blocks:
            return -1
        return (block_id - self.shared_blocks) // self.region_blocks

    def pool_of(self, block_id: int) -> BlockPool:
        c = self.region_of(block_id)
        return self.shared_pool if c < 0 else self.regions[c]

    def __getattr__(self, name: str) -> Any:
        # Everything not routed by block goes to the bound owner's region.
        return getattr(self.regions[self.__dict__["owner"]], name)

    # Calls routed by the blocks they take.
    def _by_region(
        self, blocks: Iterable[KVCacheBlock]
    ) -> dict[BlockPool, list[KVCacheBlock]]:
        out: dict[BlockPool, list[KVCacheBlock]] = {}
        for block in blocks:
            if block.is_null:
                continue
            out.setdefault(_pool_of(block) or self.pool, []).append(block)
        return out

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        for pool, blocks in self._by_region(ordered_blocks).items():
            pool.free_blocks(blocks)

    def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
        for pool, group in self._by_region(blocks).items():
            pool.touch(group)

    def demote(self, blocks: Iterable[KVCacheBlock]) -> None:
        for pool, group in self._by_region(blocks).items():
            pool.demote(group)

    def mark_low_priority(self, blocks: Iterable[KVCacheBlock]) -> None:
        for pool, group in self._by_region(blocks).items():
            pool.mark_low_priority(group)

    def evict_cached(self, blocks: Iterable[KVCacheBlock], *args, **kwargs) -> int:
        return sum(
            pool.evict_cached(group, *args, **kwargs)
            for pool, group in self._by_region(blocks).items()
        )

    def unpin_blocks(self, blocks: Iterable[KVCacheBlock], on_reuse) -> None:
        for pool, group in self._by_region(blocks).items():
            pool.unpin_blocks(group, on_reuse)

    def is_block_writable(self, block: KVCacheBlock) -> bool:
        return (_pool_of(block) or self.pool).is_block_writable(block)

    def move_block_hashes(self, src_block: KVCacheBlock, dst_block: KVCacheBlock):
        pool = _pool_of(src_block)
        assert pool is _pool_of(dst_block)
        return pool.move_block_hashes(src_block, dst_block)

    def cache_full_blocks(self, request, blocks, *args, **kwargs) -> None:
        pool = next((_pool_of(b) for b in blocks if not b.is_null), None) or self.pool
        pool.cache_full_blocks(request, blocks, *args, **kwargs)

    def evict_blocks(self, block_ids: set[int]) -> None:
        by_region: dict[int, set[int]] = {}
        for block_id in block_ids:
            by_region.setdefault(self.region_of(block_id), set()).add(block_id)
        for c, ids in by_region.items():
            (self.shared_pool if c < 0 else self.regions[c]).evict_blocks(ids)

    def reset_prefix_cache(self) -> bool:
        return all([pool.reset_prefix_cache() for pool in self.regions])

    def take_events(self) -> list:
        return []

    def get_num_free_blocks_of(self, owner: int) -> int:
        return self.regions[owner].get_num_free_blocks()

    def usage_by_owner(self) -> list[float]:
        return [pool.get_usage() for pool in self.regions]


# Shared-region pages reserved per sequence for a Mamba (KDA state) group when
# its spec cannot say (``MambaSpec.max_memory_usage_bytes``).
_SHARED_BLOCKS_PER_SEQ = 6


def _mamba_pages_per_seq(spec: Any, vllm_config: Any) -> int:
    """State pages one sequence holds in a Mamba group (align mode: the state,
    its speculative and prefill-checkpoint copies)."""
    try:
        return max(
            1, -(-spec.max_memory_usage_bytes(vllm_config) // spec.page_size_bytes)
        )
    except AttributeError:
        return _SHARED_BLOCKS_PER_SEQ


def region_sizes(
    num_blocks_per_card: int,
    tp: int,
    num_dsa_groups: int,
    num_shared_token_groups: float,
    shared_reserve_blocks: int,
    region_blocks_override: int = 0,
) -> tuple[int, int]:
    """Split each card's ``num_blocks_per_card`` pages into the shared region
    ``S`` and the card's DSA region ``D`` (``S + D == num_blocks_per_card``).

    Every block of context costs one block in each token group: the DSA groups
    on the owner card only, the shared token groups (e.g. the drafter's KV) on
    every card. Per-request state (KDA) takes ``shared_reserve_blocks`` of the
    shared region. The rest is split so that the TP group's contexts fill both
    regions at once: ``D / (S - reserve) = (num_dsa / tp) / num_shared_tok``.
    """
    m = num_blocks_per_card
    if region_blocks_override > 0:
        d = region_blocks_override
    else:
        rest = max(m - 1 - shared_reserve_blocks, 0)
        dsa_per_card = num_dsa_groups / tp
        d = int(rest * dsa_per_card / (dsa_per_card + num_shared_token_groups))
    d = max(1, min(d, m - 2))
    return m - d, d


def apply_dsa_dp_regions(
    vllm_config: Any,
    kv_cache_specs: list[dict[str, Any]],
    kv_cache_configs: list[Any],
    scheduler_kv_cache_config: Any,
) -> None:
    """Engine core, DSA-DP layouts (``dsa_dp.supported``): split the per-card pages
    into the shared region and the card's DSA region, and record the split in every
    config. The scheduler's pool becomes the shared region (``num_blocks = S``);
    the workers keep ``num_blocks = S + D`` pages."""
    from vllm.logger import init_logger
    from vllm.models.kimi_k3.nvidia.ops import dsa_dp
    from vllm.v1.kv_cache_interface import (
        AttentionSpec,
        MambaSpec,
        MLAAttentionSpec,
        SlidingWindowSpec,
    )

    pc = vllm_config.parallel_config
    tp = pc.tensor_parallel_size
    if not dsa_dp.configure(vllm_config) or not dsa_dp.KV_ENABLED:
        return
    # Per-layer specs: a group's merged spec drops ``is_draft``.
    layer_specs = {n: sp for specs in kv_cache_specs for n, sp in specs.items()}
    groups = kv_cache_configs[0].kv_cache_groups
    dsa_ids, shared_tok, mamba, swa_blocks = [], 0.0, 0, 0
    for i, group in enumerate(groups):
        specs = [layer_specs[n] for n in group.layer_names]
        # Slot-packed groups (the release drafter's KV, gqa_draft.py) take
        # ``blocks_per_pool_block`` of their blocks per page of the shared region.
        per_page = max(getattr(group, "blocks_per_pool_block", 1) or 1, 1)
        if all(isinstance(s, MLAAttentionSpec) and not s.is_draft for s in specs):
            dsa_ids.append(i)
        elif any(isinstance(s, MLAAttentionSpec) and not s.is_draft for s in specs):
            raise ValueError(
                f"DSA-DP KV: cache group {i} mixes DSA and other layers "
                f"{group.layer_names[:4]}"
            )
        elif all(isinstance(s, MambaSpec) for s in specs):
            mamba += _mamba_pages_per_seq(specs[0], vllm_config)
        elif all(isinstance(s, SlidingWindowSpec) for s in specs):
            # A sliding-window group (e.g. the SWA DSpark) holds a fixed number
            # of blocks per request, not one per block of context.
            bs = group.kv_cache_spec.block_size
            slots = -(-specs[0].sliding_window // bs) + 1
            # Packed slots: the pages a window spans, plus one partly shared page.
            swa_blocks += slots if per_page == 1 else -(-slots // per_page) + 1
        elif all(isinstance(s, AttentionSpec) for s in specs):
            shared_tok += 1.0 / per_page
        else:
            raise ValueError(f"DSA-DP KV: unsupported cache group {i}")
    if not dsa_ids:
        raise ValueError("DSA-DP KV: no DSA cache group")
    m = scheduler_kv_cache_config.num_blocks
    per_seq = mamba + swa_blocks
    max_num_seqs = vllm_config.scheduler_config.max_num_seqs
    reserve = max_num_seqs * per_seq
    # Large hybrid pages (unified with the FP32 KDA state: thousands of tokens a
    # page) can make max_num_seqs sequences' state pages exceed the card. Then
    # the shared region keeps at most half the pages and the DSA region the rest,
    # instead of a 1-page DSA region.
    cap = (m - 1) // 2
    if reserve > cap:
        init_logger(__name__).warning(
            "DSA-DP KV regions: the per-sequence state of max_num_seqs=%d "
            "sequences (%d pages each) exceeds half of the %d pages per card; the "
            "shared region holds about %d sequences' state.",
            max_num_seqs,
            per_seq,
            m,
            cap // max(per_seq, 1),
        )
        reserve = cap
    s, d = region_sizes(
        m,
        tp,
        num_dsa_groups=len(dsa_ids),
        num_shared_token_groups=shared_tok,
        shared_reserve_blocks=reserve,
    )
    for cfg in (*kv_cache_configs, scheduler_kv_cache_config):
        assert cfg.num_blocks == m
        cfg.dsa_dp_group_ids = tuple(dsa_ids)
        cfg.dsa_dp_shared_blocks = s
        cfg.dsa_dp_region_blocks = d
        cfg.dsa_dp_num_regions = tp
    scheduler_kv_cache_config.num_blocks = s
    bs = groups[dsa_ids[0]].kv_cache_spec.block_size
    init_logger(__name__).info(
        "DSA-DP KV regions: %d pages/card = shared %d + DSA region %d (tp %d, "
        "DSA groups %s, shared token groups %.2f, Mamba state pages/seq %d, "
        "reserve %d = %d requests x %d pages). "
        "Per card: DSA KV %d tokens (owned requests), shared-group KV %d tokens "
        "(all requests); TP group DSA KV %d tokens. Without regions: %d tokens "
        "of every group per card.",
        m,
        s,
        d,
        tp,
        dsa_ids,
        shared_tok,
        mamba,
        reserve,
        reserve // max(per_seq, 1),
        per_seq,
        d * bs,
        max(s - 1 - reserve, 0) * bs,
        tp * d * bs,
        (m - 1) * bs,
    )
