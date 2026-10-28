# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``CacheConfig.prefix_cache_policy = "v2"``: one eviction order.

- Demotion has no cache-pressure horizon: the copy-on-write source of a
  partial-tail hit and generated blocks past a divergence always go
  evict-first (behind the uncached free blocks, ahead of every LRU block).
- A turn's superseded prompt checkpoint stays in plain LRU.
- Short-prompt Mamba states are low priority (16384 unless set).
- ``mamba_conversation_max_turns`` retires old turns by demotion ("demote"
  mode): still hit-able until reused; a hit cancels it.

Integrity: only the eviction order changes; marks never follow a block into
its next life; a hit cancels a demotion; blocks a connector has yet to read
are left alone; the free queue stays consistent with mixed marks.
"""

import pytest

from tests.v1.core.prefix_cache.test_stale_block_demotion import (
    GLOBAL_BLOCK,
    HASH,
    _cache,
    _cached_free_order,
    _free_order,
    _run,
)
from tests.v1.core.prefix_cache.test_superseded_state_drop import (
    PROMPT,
    _consistent,
    _mamba_hit,
    _manager,
    _turn,
)
from tests.v1.core.test_prefix_caching import make_request
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import V2_LOW_PRIORITY_SNAPSHOT_PROMPT_LEN
from vllm.v1.core.kv_cache_utils import init_none_hash

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _auto_init_hash_fn():
    init_none_hash(sha256)


def _old_pool(num: int = 4, gate: bool = True):
    """Cached blocks freed long ago: the LRU head is past the horizon."""
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=16)
    pool.demotion_gate = gate
    now = [0.0]
    pool._clock = lambda: now[0]
    blocks = pool.get_new_blocks(num)
    for i, block in enumerate(blocks):
        _cache(pool, block, i)
    return pool, blocks, now


# ---------------------------------------------------------------------------
# BlockPool: the gate and mixed marks


@pytest.mark.parametrize("gate", [True, False])
def test_v2_demotes_without_cache_pressure(gate):
    pool, blocks, now = _old_pool(3, gate)
    pool.free_blocks(blocks)
    now[0] = pool._DEMOTE_HORIZON_S + 100  # no pressure
    pool.demote([blocks[2]])
    first = _cached_free_order(pool)[0]
    assert (first == blocks[2].block_id) == (not gate)
    assert _consistent(pool)


def test_v2_referenced_demotion_applies_at_free_without_pressure():
    pool, blocks, now = _old_pool(3, gate=False)
    pool.free_blocks(blocks[:2])
    now[0] = 10_000.0
    pool.demote([blocks[2]])  # still referenced
    pool.free_blocks([blocks[2]])
    assert _cached_free_order(pool)[0] == blocks[2].block_id


def test_v2_free_queue_order_with_mixed_marks():
    """Uncached, then low-priority, then demoted, then LRU; evicted blocks
    join the uncached ones. The queue stays intact."""
    pool, blocks, now = _old_pool(5, gate=False)
    plain, low, demoted, evicted, both = blocks
    pool.mark_low_priority([low, both])
    pool.demote([demoted, both])
    pool.evict_cached([evicted])
    now[0] = 10_000.0
    pool.free_blocks(blocks)
    order = _cached_free_order(pool)
    assert order == [low.block_id, both.block_id, demoted.block_id, plain.block_id]
    assert evicted.block_hash is None
    assert _free_order(pool).index(evicted.block_id) < _free_order(pool).index(
        low.block_id
    )
    assert not (pool._demote_on_free | pool._low_priority_on_free | pool._evict_on_free)
    assert _consistent(pool)


@pytest.mark.parametrize("with_lru", [False, True])
def test_v2_redemoting_the_last_demoted_block_keeps_the_queue_intact(with_lru):
    """The K4 crash: the block demoted last sits right before the LRU head
    (or at the tail with no LRU blocks); demoting it again must not use it as
    its own anchor."""
    pool, blocks, now = _old_pool(4, gate=False)
    pool.free_blocks(blocks if with_lru else [])
    if not with_lru:
        pool.free_blocks(blocks)
        pool.demote(blocks)  # every cached block demoted: no LRU block left
    num_free = pool.free_block_queue.num_free_blocks
    for _ in range(3):
        pool.demote([blocks[1]])
        pool.demote([blocks[1], blocks[1]])
        pool.demote([blocks[3], blocks[1]])
        assert pool.free_block_queue.num_free_blocks == num_free
        assert _consistent(pool)


def test_v2_a_hit_cancels_a_pending_demotion():
    pool, blocks, now = _old_pool(2, gate=False)
    a, b = blocks
    pool.demote([a])
    pool.touch([a])  # another request hits it: not stale
    pool.free_blocks([a])
    pool.free_blocks([a, b])
    assert _cached_free_order(pool) == [a.block_id, b.block_id]


def test_v2_marks_do_not_follow_a_block_into_its_next_life():
    pool, blocks, now = _old_pool(1, gate=False)
    pool.demote(blocks)  # referenced: marked
    pool._maybe_evict_cached_block(blocks[0])  # evicted some other way
    pool.free_blocks(blocks)
    pool.get_new_blocks(pool.get_num_free_blocks())
    assert not pool._demote_on_free


def test_demote_by_key_follows_cow_moves_and_skips_pending_reads():
    pool, blocks, now = _old_pool(3, gate=False)
    spare = pool.get_new_blocks(1)[0]
    key = blocks[0].block_hash
    pool.move_block_hashes(blocks[0], spare)
    pool._reuse_watchers[blocks[2].block_id] = lambda block: None
    pool.free_blocks([*blocks, spare])
    keys = [key, blocks[1].block_hash, blocks[2].block_hash]
    assert pool.demote_cached_hashes(keys, exclude_block_ids={blocks[1].block_id}) == 1
    assert _cached_free_order(pool)[0] == spare.block_id
    assert spare.block_hash == key  # demoted, still cached
    assert _consistent(pool)


# ---------------------------------------------------------------------------
# Hybrid manager: the agentic two-turn flow under v2


@pytest.mark.parametrize("policy", ["legacy", "v2"])
def test_v2_stale_cow_source_and_generated_chain_go_first(policy):
    """Without cache pressure, legacy keeps LRU order; v2 sends the stale CoW
    tail source and the diverged generated blocks evict-first. The parent's
    superseded prompt checkpoint stays in LRU order under v2."""
    manager = _manager(prefix_cache_policy=policy)
    pool = manager.block_pool
    now = [0.0]
    pool._clock = lambda: now[0]
    other = make_request("other", list(range(50_000, 50_700)), HASH, sha256)
    _run(manager, other, list(range(60_000, 60_010)))
    full_mgr = manager.coordinator.single_type_managers[0]
    other_full = [b.block_id for b in full_mgr.req_to_blocks["other"] if b.block_hash]
    manager.free(other)

    parent = make_request("parent", PROMPT, HASH, sha256)
    _run(manager, parent, list(range(70_000, 70_300)))
    parent_full = list(full_mgr.req_to_blocks["parent"])
    prompt_end = (len(PROMPT) - 1) // GLOBAL_BLOCK
    stale = [parent_full[prompt_end], *parent_full[prompt_end + 1 : 10]]
    manager.free(parent)

    now[0] = 10_000.0  # the cache holds far more than the horizon
    child = make_request("child", PROMPT + list(range(80_000, 80_200)), HASH, sha256)
    blocks, _, _ = manager.get_computed_blocks(child)
    resumed_state = next(b for b in reversed(blocks.blocks[1]) if not b.is_null)
    _run(manager, child, [])

    order = _free_order(pool)
    position = {block_id: i for i, block_id in enumerate(order)}
    stale_ids = [b.block_id for b in stale if b.block_id in position]
    assert stale_ids
    first_other = min(position[b] for b in other_full if b in position)
    ahead = all(position[b] < first_other for b in stale_ids)
    assert ahead == (policy == "v2")
    # The superseded prompt checkpoint is never ahead of older live blocks
    # under v2 (plain LRU), and is still cached.
    if policy == "v2" and resumed_state.block_id in position:
        assert position[resumed_state.block_id] > first_other
        assert resumed_state.block_hash is not None
    assert _consistent(pool)


def test_v2_defaults():
    v2 = _manager(prefix_cache_policy="v2")
    legacy = _manager()
    mamba_v2 = v2.mamba_state_managers[0]
    assert mamba_v2.low_priority_prompt_len == V2_LOW_PRIORITY_SNAPSHOT_PROMPT_LEN
    assert mamba_v2.turn_drop_mode == "demote"
    assert not mamba_v2.demote_superseded_checkpoints
    assert not v2.block_pool.demotion_gate
    mamba_legacy = legacy.mamba_state_managers[0]
    assert mamba_legacy.low_priority_prompt_len == 0
    assert mamba_legacy.turn_drop_mode == "drop"
    assert legacy.block_pool.demotion_gate
    explicit = _manager(
        prefix_cache_policy="v2",
        mamba_low_priority_snapshot_prompt_len=4096,
        mamba_conversation_max_turns_mode="drop",
    )
    assert explicit.mamba_state_managers[0].low_priority_prompt_len == 4096
    assert explicit.mamba_state_managers[0].turn_drop_mode == "drop"


@pytest.mark.parametrize("mode", ["drop", "demote"])
def test_turn_demote_keeps_old_states_hittable(mode):
    manager = _manager(
        prefix_cache_policy="v2",
        mamba_conversation_max_turns=1,
        mamba_conversation_max_turns_mode=mode,
    )
    _turn(manager, "t1", PROMPT)
    rewind = PROMPT + list(range(90_000, 90_050))
    before = _mamba_hit(manager, rewind)
    _, retired = _turn(manager, "t2", PROMPT + list(range(80_000, 80_200)))
    assert retired > 0
    after = _mamba_hit(manager, rewind)
    if mode == "drop":
        assert after < before
    else:
        assert after == before  # demoted, not dropped
        # A rewind that uses the state cancels its demotion.
        rewind_req = make_request("rw", rewind, HASH, sha256)
        blocks = manager.get_computed_blocks(rewind_req)[0]
        state = next(b for b in reversed(blocks.blocks[1]) if not b.is_null)
        manager.block_pool.touch([state])
        assert state.block_id not in manager.block_pool._demote_on_free
        manager.block_pool.free_blocks([state])
    assert _consistent(manager.block_pool)


def test_turn_demote_leaves_pending_hand_offs_alone():
    manager = _manager(
        prefix_cache_policy="v2",
        mamba_conversation_max_turns=1,
    )
    _turn(manager, "t1", PROMPT)
    mamba = manager.mamba_state_managers[0]
    for block in manager.block_pool.blocks:
        if block.block_hash is not None:
            mamba._pending_boundary_state_offloads.append(("x", 1, block, 0))
    _, retired = _turn(manager, "t2", PROMPT + list(range(80_000, 80_200)))
    assert retired == 0
