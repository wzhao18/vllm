# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-priority Mamba states of short prompts.

``CacheConfig.mamba_low_priority_snapshot_prompt_len`` (TensorRT-LLM
``TLLM_V2_LOW_PRIORITY_SNAPSHOT_PROMPT_LEN``): the Mamba states a request with a
shorter prompt caches go behind the uncached free blocks and ahead of every
other cached block when freed, whatever the cache pressure. A state is reusable
only at its exact boundary and takes a full slot, while a short prompt is cheap
to recompute. Attention blocks keep LRU order. Only the eviction order changes.
"""

import pytest

from tests.v1.core.prefix_cache.test_mamba_eagle_resume_checkpoint import (
    PREFIX,
    _manager,
    _prefill,
    _stub,
)
from tests.v1.core.prefix_cache.test_stale_block_demotion import (
    _cache,
    _cached_free_order,
    _first_cached_index,
)
from tests.v1.core.test_prefix_caching import (
    _make_hybrid_kv_cache_config,
    make_kv_cache_manager,
    make_request,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import get_block_hash, get_group_id, init_none_hash
from vllm.v1.core.single_type_kv_cache_manager import MambaManager

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _auto_init_hash_fn():
    init_none_hash(sha256)


def _pool_with_cached(num: int) -> tuple[BlockPool, list]:
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=16)
    blocks = pool.get_new_blocks(num)
    for i, block in enumerate(blocks):
        _cache(pool, block, i)
    return pool, blocks


def test_low_priority_blocks_are_evicted_before_other_cached_blocks():
    pool, blocks = _pool_with_cached(4)
    pool.mark_low_priority([blocks[1], blocks[3]])
    pool.free_blocks(blocks)
    assert _cached_free_order(pool) == [
        blocks[1].block_id,
        blocks[3].block_id,
        blocks[0].block_id,
        blocks[2].block_id,
    ]
    # Still cached and hit-able until reused.
    assert pool.get_cached_block(get_block_hash(blocks[1].block_hash), [0])
    num_uncached = _first_cached_index(pool)
    taken = pool.get_new_blocks(num_uncached + 1)
    assert taken[-1] is blocks[1]


def test_low_priority_applies_without_cache_pressure():
    """Unlike stale-block demotion, no horizon gate."""
    pool, blocks = _pool_with_cached(3)
    now = [0.0]
    pool._clock = lambda: now[0]
    pool.free_blocks(blocks[:2])
    now[0] = pool._DEMOTE_HORIZON_S + 100  # the LRU head is old: no pressure
    pool.mark_low_priority([blocks[2]])
    pool.free_blocks([blocks[2]])
    assert _cached_free_order(pool)[0] == blocks[2].block_id


def test_low_priority_goes_ahead_of_demoted_blocks():
    pool, blocks = _pool_with_cached(3)
    pool.demote([blocks[0]])
    pool.mark_low_priority([blocks[1]])
    pool.free_blocks(blocks)
    assert _cached_free_order(pool) == [
        blocks[1].block_id,
        blocks[0].block_id,
        blocks[2].block_id,
    ]


def test_mark_is_consumed_by_the_free():
    pool, blocks = _pool_with_cached(2)
    pool.mark_low_priority([blocks[0]])
    pool.free_blocks(blocks)
    pool.touch([blocks[0]])  # a later request hits it and frees it again
    pool.free_blocks([blocks[0]])
    assert _cached_free_order(pool) == [blocks[1].block_id, blocks[0].block_id]


def test_uncached_and_free_blocks_are_not_marked():
    pool, blocks = _pool_with_cached(1)
    uncached = pool.get_new_blocks(1)[0]
    pool.mark_low_priority([uncached])
    pool.free_blocks(blocks)
    pool.mark_low_priority(blocks)  # already free: nothing to consume the mark
    assert not pool._low_priority_on_free


def test_kv_cache_manager_arms_only_mamba_managers():
    config = _make_hybrid_kv_cache_config(512, 64, ["full", "mamba"])
    manager = make_kv_cache_manager(
        kv_cache_config=config,
        max_model_len=1 << 16,
        enable_caching=True,
        hash_block_size=32,
        mamba_low_priority_snapshot_prompt_len=2048,
    )
    lens = {
        type(m).__name__: m.low_priority_prompt_len
        for m in manager.coordinator.single_type_managers
    }
    assert lens == {"FullAttentionManager": 0, "MambaManager": 2048}


@pytest.mark.parametrize("prompt_len", [1200, 3000])
def test_manager_marks_only_short_prompt_mamba_states(prompt_len):
    """Prefill to completion through the real split and manager, then free."""
    block_size, hash_block_size = 512, 32
    manager = _manager(block_size, hash_block_size)
    for m in manager.coordinator.single_type_managers:
        if isinstance(m, MambaManager):
            m.low_priority_prompt_len = 2048
            mamba_group = m.kv_cache_group_id
    stub = _stub(manager, block_size, hash_block_size)
    # An older, long request: its cached blocks are ahead in LRU order.
    old = make_request("old", [-1] * 4000, hash_block_size, sha256)
    _prefill(manager, stub, old)
    manager.free(old)
    old_ids = {
        b.block_id
        for b in manager.block_pool.free_block_queue.get_all_free_blocks()
        if b.block_hash is not None
    }

    request = make_request("r", PREFIX[:prompt_len], hash_block_size, sha256)
    _prefill(manager, stub, request)
    manager.free(request)

    cached = [
        b
        for b in manager.block_pool.free_block_queue.get_all_free_blocks()
        if b.block_hash is not None
    ]
    new_mamba = [
        i
        for i, b in enumerate(cached)
        if b.block_id not in old_ids and get_group_id(b.block_hash) == mamba_group
    ]
    new_attention = [
        i
        for i, b in enumerate(cached)
        if b.block_id not in old_ids and get_group_id(b.block_hash) != mamba_group
    ]
    first_old = min(i for i, b in enumerate(cached) if b.block_id in old_ids)
    assert new_mamba and new_attention
    # A short prompt's Mamba states go ahead of every cached block, even the
    # older request's; its attention blocks keep LRU order behind them.
    assert (max(new_mamba) < first_old) == (prompt_len < 2048)
    assert min(new_attention) > first_old
    assert not manager.block_pool._low_priority_on_free
