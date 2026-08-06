# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""External-store cache-hit coordinator for MooncakeStoreConnector."""

from collections.abc import Sequence
from typing import cast

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    chunk_hashes_for_block_size,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import (
    group_kv_cache_specs,
    reconcile_kv_cache_hits,
)
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    KVCacheBlock,
    get_prefix_cache_hit_limit,
    get_prefix_cache_replay_config,
    get_prefix_replay_checkpoint,
)
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    KVCacheSpec,
)


class ExternalCachedBlockPool:
    """Duck-typed BlockPool backed by a ``(group_id, hash)`` exists set."""

    def __init__(
        self,
        hash_block_size: int,
        exists: set[tuple[int, bytes]] | None = None,
    ) -> None:
        # ``exists=None`` is used on the recv side where hit_length is already
        # determined and we just want each spec's manager to apply its own mask.
        self._exists = exists
        self.hash_block_size = hash_block_size
        self.null_block = KVCacheBlock(block_id=0)
        # Dummy ID 1 for present block for duck-typing.
        self._present_block = KVCacheBlock(block_id=1)

    def get_cached_block(
        self,
        block_hash: BlockHash,
        group_ids: list[int],
    ) -> list[KVCacheBlock] | None:
        # Mirrors BlockPool.get_cached_block: hit only when every group_id
        # (groups sharing a spec) has the hash cached.
        if self._exists is None:
            return [self._present_block] * len(group_ids)
        h = bytes(block_hash)
        if all((g, h) in self._exists for g in group_ids):
            return [self._present_block] * len(group_ids)
        return None


class MooncakeStoreCoordinator:
    """Apply core prefix-cache policies to MooncakeStore entries."""

    def __init__(
        self,
        kv_cache_groups: list[KVCacheGroupSpec],
        scheduler_block_size: int,
        hash_block_size: int,
        use_eagle: bool = False,
        retention_interval: int | None = None,
        enable_caching: bool = True,
    ) -> None:
        assert all(
            g.kv_cache_spec.block_size % hash_block_size == 0 for g in kv_cache_groups
        ), "block_size must be divisible by hash_block_size"
        assert scheduler_block_size % hash_block_size == 0, (
            f"scheduler_block_size ({scheduler_block_size}) must be a multiple of "
            f"hash_block_size ({hash_block_size})"
        )
        assert all(
            scheduler_block_size % g.kv_cache_spec.block_size == 0
            for g in kv_cache_groups
        ), "scheduler_block_size must be a multiple of each group's block_size"
        self.kv_cache_groups = kv_cache_groups
        self.hash_block_size = hash_block_size
        self.lcm_block_size = scheduler_block_size
        replay_config = get_prefix_cache_replay_config(
            kv_cache_groups,
            scheduler_block_size,
            hash_block_size,
            use_eagle=use_eagle,
            enable_caching=enable_caching,
        )
        self.enable_partial_hash_hits = replay_config.enable_partial_hash_hits
        self.replay_alignment_tokens = replay_config.replay_alignment_tokens
        self.eagle_rewind_tokens = replay_config.eagle_rewind_tokens
        self.retention_interval = retention_interval
        self.attention_groups = group_kv_cache_specs(
            [group.kv_cache_spec for group in kv_cache_groups]
        )

    def get_lookup_limit(self, num_tokens: int) -> int:
        return get_prefix_replay_checkpoint(
            get_prefix_cache_hit_limit(num_tokens, self.eagle_rewind_tokens),
            self.replay_alignment_tokens,
            0,
        )

    def find_longest_cache_hit(
        self,
        block_hashes: Sequence[BlockHash],
        max_length: int,
        cached_block_pool: ExternalCachedBlockPool,
    ) -> int:
        """Return the replay-capped hit shared by every cache group."""
        _, hit_length, _ = reconcile_kv_cache_hits(
            self.attention_groups,
            cast(BlockHashList, block_hashes),
            max_length,
            cast(BlockPool, cached_block_pool),
            num_groups=len(self.kv_cache_groups),
            alignment_tokens=self.replay_alignment_tokens,
            eagle_rewind_tokens=self.eagle_rewind_tokens,
        )
        return hit_length

    def load_mask(
        self,
        block_hashes: Sequence[BlockHash],
        hit_length: int,
    ) -> tuple[list[bool], ...]:
        """Reconstruct masks for a hit returned by external lookup.

        ``hit_length`` must already be replay-capped and reconciled across
        groups by ``find_longest_cache_hit``. The assertion below protects
        that scheduler-to-receiver contract.
        """
        cached_block_pool = ExternalCachedBlockPool(self.hash_block_size)
        blocks_per_group, reconstructed_hit_length, _ = reconcile_kv_cache_hits(
            self.attention_groups,
            cast(BlockHashList, block_hashes),
            hit_length,
            cast(BlockPool, cached_block_pool),
            num_groups=len(self.kv_cache_groups),
            alignment_tokens=self.replay_alignment_tokens,
        )
        assert reconstructed_hit_length == hit_length, (
            f"Load hit length changed from {hit_length} to "
            f"{reconstructed_hit_length} while reconstructing its group masks"
        )
        return tuple(
            [blk is not cached_block_pool.null_block for blk in blocks]
            for blocks in blocks_per_group
        )

    def store_mask(
        self,
        aligned_token_len: int,
        start_token: int = 0,
        num_prompt_tokens: int | None = None,
    ) -> tuple[list[bool] | None, ...]:
        """Per-group store masks for the suffix starting at ``start_token``.

        ``mask[g][i]`` is True iff the i-th chunk of group ``g`` *after*
        ``start_token`` should be written to the store so a future cache hit
        can consume it. ``None`` is the all-True sentinel for the suffix.

        Reuses the engine's ``SingleTypeKVCacheManager.reachable_block_mask``
        so the store retains exactly the blocks the local prefix cache would.
        """
        return self._reachable_masks(
            aligned_token_len,
            start_token,
            retention_interval=self.retention_interval,
            num_prompt_tokens=num_prompt_tokens,
        )

    def lookup_mask(
        self,
        aligned_token_len: int,
    ) -> tuple[list[bool] | None, ...]:
        """Per-group lookup masks.

        ``mask[g][i]`` is True iff chunk ``i`` of group ``g`` should be
        looked up as an aligned hit boundary. ``None`` is the all-True
        sentinel.
        """
        return self._reachable_masks(
            aligned_token_len,
            0,
            retention_interval=None,
            num_prompt_tokens=None,
        )

    def _reachable_masks(
        self,
        aligned_token_len: int,
        start_token: int,
        *,
        retention_interval: int | None,
        num_prompt_tokens: int | None,
    ) -> tuple[list[bool] | None, ...]:
        assert aligned_token_len % self.replay_alignment_tokens == 0, (
            f"aligned_token_len ({aligned_token_len}) must be a multiple of "
            f"{self.replay_alignment_tokens}"
        )
        masks: list[list[bool] | None] = [None] * len(self.kv_cache_groups)
        for spec, group_ids, manager_cls in self.attention_groups:
            end_chunk = aligned_token_len // spec.block_size
            start_chunk = min(end_chunk, max(0, cdiv(start_token, spec.block_size)))
            mask = manager_cls.reachable_block_mask(
                start_block=start_chunk,
                end_block=end_chunk,
                alignment_tokens=self.lcm_block_size,
                kv_cache_spec=spec,
                retention_interval=retention_interval,
                num_prompt_tokens=num_prompt_tokens,
                replay_alignment_tokens=self.replay_alignment_tokens,
                eagle_rewind_tokens=self.eagle_rewind_tokens,
            )
            if mask is not None:
                assert len(mask) == end_chunk - start_chunk
            for group_id in group_ids:
                masks[group_id] = mask
        return tuple(masks)

    def block_hashes_for_spec(
        self, block_hashes: Sequence[BlockHash], spec: KVCacheSpec
    ) -> Sequence[BlockHash]:
        return chunk_hashes_for_block_size(
            block_hashes, self.hash_block_size, spec.block_size
        )
