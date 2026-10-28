# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dropping superseded prefix-cache states.

- ``CacheConfig.mamba_conversation_max_turns`` (TensorRT-LLM
  ``per_conversation`` / ``max_num_turns``): a conversation keeps the Mamba
  states of its last N turns; older turns' states leave the prefix cache.

Integrity: a dropped block is resolved by its key at drop time, a referenced
one leaves the cache only at its last free (a new hit does not cancel that),
and blocks a connector has yet to read are never dropped.
"""

import pytest
import torch

from tests.v1.core.prefix_cache.test_stale_block_demotion import (
    BLOCK,
    DCP,
    GLOBAL_BLOCK,
    HASH,
    _cache,
    _cached_free_order,
    _free_order,
    _run,
)
from tests.v1.core.test_prefix_caching import make_kv_cache_manager, make_request
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.conversation_state import ConversationStateTracker
from vllm.v1.core.kv_cache_utils import get_block_hash, init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _auto_init_hash_fn():
    init_none_hash(sha256)


# ---------------------------------------------------------------------------
# BlockPool.evict_cached


def _pool(num: int = 3) -> tuple[BlockPool, list]:
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=16)
    blocks = pool.get_new_blocks(num)
    for i, block in enumerate(blocks):
        _cache(pool, block, i)
    return pool, blocks


def _consistent(pool: BlockPool) -> bool:
    queue = pool.free_block_queue
    return len(queue.get_all_free_blocks()) == queue.num_free_blocks


def test_evicting_a_free_cached_block_uncaches_it_and_moves_it_first():
    pool, blocks = _pool()
    pool.free_blocks(blocks)
    key = blocks[1].block_hash
    assert pool.evict_cached([blocks[1]]) == 1
    assert blocks[1].block_hash is None
    assert pool.cached_block_hash_to_block.get_one_block(key) is None
    assert _free_order(pool)[0] == blocks[1].block_id
    assert _cached_free_order(pool) == [blocks[0].block_id, blocks[2].block_id]
    assert _consistent(pool)


def test_a_referenced_block_leaves_the_cache_at_its_last_free():
    pool, blocks = _pool()
    assert pool.evict_cached([blocks[0]]) == 1
    # Still cached and hit-able while referenced; a new hit does not cancel.
    assert pool.get_cached_block(get_block_hash(blocks[0].block_hash), [0])
    pool.touch([blocks[0]])
    pool.free_blocks([blocks[0]])
    assert blocks[0].block_hash is not None  # one reference left
    pool.free_blocks(blocks)
    assert blocks[0].block_hash is None
    assert _cached_free_order(pool) == [blocks[1].block_id, blocks[2].block_id]
    assert _consistent(pool)


def test_excluded_and_watched_blocks_are_left_alone():
    pool, blocks = _pool()
    pool.free_blocks(blocks)
    pool._reuse_watchers[blocks[2].block_id] = lambda block: None
    assert pool.evict_cached(blocks, exclude_block_ids={blocks[0].block_id}) == 1
    assert blocks[0].block_hash is not None  # pending hand-off
    assert blocks[1].block_hash is None
    assert blocks[2].block_hash is not None  # still being read


def test_eviction_wins_over_demotion_and_low_priority():
    pool, blocks = _pool()
    pool.demote([blocks[0]])
    pool.mark_low_priority([blocks[1]])
    pool.evict_cached(blocks[:2])
    pool.free_blocks(blocks)
    assert blocks[0].block_hash is None and blocks[1].block_hash is None
    assert not (pool._demote_on_free | pool._low_priority_on_free | pool._evict_on_free)
    assert _cached_free_order(pool) == [blocks[2].block_id]
    assert _consistent(pool)


def test_evicting_a_demoted_free_block_keeps_the_queue_intact():
    pool, blocks = _pool(4)
    now = [0.0]
    pool._clock = lambda: now[0]
    pool.free_blocks(blocks)
    now[0] = 10.0  # pressure: demotion applies
    pool.demote([blocks[2]])
    pool.evict_cached([blocks[2]])
    pool.evict_cached([blocks[2]])  # again: already uncached
    assert _consistent(pool)
    assert _cached_free_order(pool) == [
        blocks[0].block_id,
        blocks[1].block_id,
        blocks[3].block_id,
    ]


def test_evict_by_key_resolves_the_current_block():
    """A copy-on-write moves a key to another block; the drop follows it."""
    pool, blocks = _pool()
    spare = pool.get_new_blocks(1)[0]
    key = blocks[0].block_hash
    pool.move_block_hashes(blocks[0], spare)
    pool.free_blocks([*blocks, spare])
    assert blocks[0].block_hash is None
    assert pool.evict_cached_hashes([key]) == 1
    assert spare.block_hash is None
    assert pool.cached_block_hash_to_block.get_one_block(key) is None
    assert pool.evict_cached_hashes([key]) == 0  # gone: nothing to do


def test_a_reused_block_carries_no_eviction_mark():
    pool, blocks = _pool(1)
    pool.evict_cached(blocks)  # referenced: marked
    pool._maybe_evict_cached_block(blocks[0])  # evicted some other way
    pool.free_blocks(blocks)
    taken = pool.get_new_blocks(pool.get_num_free_blocks())
    assert blocks[0] in taken and not pool._evict_on_free


# ---------------------------------------------------------------------------
# ConversationStateTracker


def _request(req_id: str, session: str | None = None, conv: str | None = None):
    params = SamplingParams(max_tokens=4)
    if conv is not None:
        params.extra_args = {"conversation_id": conv}
    return Request(
        request_id=req_id,
        prompt_token_ids=[1, 2, 3],
        sampling_params=params,
        pooling_params=None,
        session_id=session,
    )


def test_conversation_id_sources():
    cid = ConversationStateTracker.conversation_id
    assert cid(_request("a", session="s")) == "s"
    assert cid(_request("a", conv=" c ")) == "c"
    assert cid(_request("a", session="s", conv="c")) == "s"
    assert cid(_request("a")) is None


def test_tracker_drops_the_oldest_turn_minus_retained_states():
    tracker = ConversationStateTracker(max_num_turns=2)
    turns = [{"a", "b"}, {"c"}, {"d", "b"}]
    dropped = []
    for i, states in enumerate(turns):
        request = _request(f"t{i}", session="s")
        tracker.prepare(request)
        dropped.append(tracker.finish(request, states, completed=True))
    # Turn 0 leaves when turn 2 completes; "b" is also turn 2's.
    assert dropped == [set(), set(), {"a"}]
    assert tracker.num_turns_dropped == 1


def test_tracker_ignores_concurrent_and_incomplete_requests():
    tracker = ConversationStateTracker(max_num_turns=1)
    first, second = _request("1", session="s"), _request("2", session="s")
    tracker.prepare(first)
    tracker.prepare(second)  # the conversation already has a current request
    assert tracker.finish(second, {"x"}, completed=True) == set()
    assert tracker.finish(first, {"a"}, completed=False) == set()  # aborted
    third = _request("3", session="s")
    tracker.prepare(third)
    assert tracker.finish(third, {"b"}, completed=True) == set()
    fourth = _request("4", session="s")
    tracker.prepare(fourth)
    assert tracker.finish(fourth, {"c"}, completed=True) == {"b"}
    assert tracker.finish(_request("5"), {"z"}, completed=True) == set()


def test_tracker_forgets_rather_than_drops_past_its_bound():
    tracker = ConversationStateTracker(max_num_turns=1, max_conversations=2)
    for name in ("a", "b", "c"):
        request = _request(name, session=name)
        tracker.prepare(request)
        tracker.finish(request, {name}, completed=True)
    assert len(tracker._conversations) == 2


# ---------------------------------------------------------------------------
# Hybrid manager, agentic turns


def _manager(**kwargs):
    groups = [
        KVCacheGroupSpec(
            ["full"],
            FullAttentionSpec(
                block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32
            ),
        ),
        KVCacheGroupSpec(
            ["mamba"],
            MambaSpec(
                block_size=GLOBAL_BLOCK,
                shapes=(1, 1),
                dtypes=(torch.float32,),
                mamba_cache_mode="align",
                num_speculative_blocks=2,
                num_prefill_checkpoint_blocks=1,
                prefill_checkpoint_alignment=16,
            ),
        ),
    ]
    manager = make_kv_cache_manager(
        kv_cache_config=KVCacheConfig(
            num_blocks=256, kv_cache_tensors=[], kv_cache_groups=groups
        ),
        max_model_len=1 << 15,
        enable_caching=True,
        dcp_world_size=DCP,
        scheduler_block_size=GLOBAL_BLOCK,
        hash_block_size=HASH,
        use_eagle=True,
        **kwargs,
    )
    manager.coordinator.retention_interval = 0
    return manager


def _turn(manager, req_id, prompt, session="conv"):
    request = make_request(req_id, prompt, HASH, sha256, session_id=session)
    manager.prepare_conversation_turn(request)
    _run(manager, request, [])
    dropped = manager.finish_conversation_turn(request)
    manager.free(request)
    return request, dropped


def _mamba_hit(manager, prompt) -> int:
    probe = make_request("probe", prompt, HASH, sha256)
    return manager.get_computed_blocks(probe)[1]


PROMPT = list(range(1000))


@pytest.mark.parametrize("max_turns", [0, 1, 2])
def test_old_turns_states_leave_the_cache(max_turns):
    manager = _manager(mamba_conversation_max_turns=max_turns)
    _turn(manager, "t1", PROMPT)
    rewind = PROMPT + list(range(90_000, 90_050))  # resumes at turn 1's tail
    before = _mamba_hit(manager, rewind)
    assert before > 7 * GLOBAL_BLOCK
    _, dropped = _turn(manager, "t2", PROMPT + list(range(80_000, 80_200)))
    after = _mamba_hit(manager, rewind)
    if max_turns == 1:
        assert dropped > 0 and after < before
    else:
        assert dropped == 0 and after == before
    # The newest turn's own states stay.
    assert _mamba_hit(manager, PROMPT + list(range(80_000, 80_200)) + [7] * 50) > 1000


def test_turns_of_other_conversations_are_independent():
    manager = _manager(mamba_conversation_max_turns=1)
    _turn(manager, "a1", PROMPT, session="a")
    rewind = PROMPT + list(range(90_000, 90_050))
    before = _mamba_hit(manager, rewind)
    _turn(manager, "b1", list(range(5000, 6000)), session="b")
    _turn(manager, "b2", list(range(5000, 6200)), session="b")
    assert _mamba_hit(manager, rewind) == before


def test_states_awaiting_a_connector_hand_off_are_kept():
    manager = _manager(mamba_conversation_max_turns=1)
    _turn(manager, "t1", PROMPT)
    rewind = PROMPT + list(range(90_000, 90_050))
    before = _mamba_hit(manager, rewind)
    mamba = manager.mamba_state_managers[0]
    # Pretend every cached Mamba block still has a pending offload.
    for block in manager.block_pool.blocks:
        if block.block_hash is not None:
            mamba._pending_boundary_state_offloads.append(("x", 1, block, 0))
    _turn(manager, "t2", PROMPT + list(range(80_000, 80_200)))
    assert _mamba_hit(manager, rewind) == before


def test_flag_off_records_nothing():
    manager = _manager()
    assert manager.conversation_tracker is None
    _turn(manager, "t1", PROMPT)
    assert not manager.mamba_state_managers[0]._state_hashes
