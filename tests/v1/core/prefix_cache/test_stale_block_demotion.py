# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stale cached blocks are demoted to the evict-first end of the free queue.

An agentic turn resumes from its parent's prompt checkpoint and leaves three
kinds of cached blocks that will not be hit again:
- the parent's Mamba checkpoint, once the turn registers its own prompt
  checkpoint;
- the parent's attention tail block, which the turn copies on write;
- the parent's generated-token blocks, where the turn's prompt diverged.
Demotion only reorders eviction: the blocks stay cached and hit-able until the
pool reuses them.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.core.test_prefix_caching import make_kv_cache_manager, make_request
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    get_block_hash,
    init_none_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _auto_init_hash_fn():
    init_none_hash(sha256)


def _free_order(pool: BlockPool) -> list[int]:
    return [b.block_id for b in pool.free_block_queue.get_all_free_blocks()]


def _cached_free_order(pool: BlockPool) -> list[int]:
    """Free cached blocks in eviction order."""
    return [
        b.block_id
        for b in pool.free_block_queue.get_all_free_blocks()
        if b.block_hash is not None
    ]


def _cache(pool: BlockPool, block, key: int) -> None:
    block_hash = make_block_hash_with_group_id(key.to_bytes(4, "little"), 0)
    block.set_block_hash(block_hash, 16)
    pool.cached_block_hash_to_block.insert(block_hash, block)
    pool.cached_block_hashes_by_block.setdefault(block.block_id, set()).add(block_hash)


def _first_cached_index(pool: BlockPool) -> int:
    order = pool.free_block_queue.get_all_free_blocks()
    return next(i for i, b in enumerate(order) if b.block_hash is not None)


def test_demote_moves_a_free_cached_block_ahead_of_cached_blocks():
    pool = BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=16)
    blocks = pool.get_new_blocks(4)
    for i, block in enumerate(blocks):
        _cache(pool, block, i)
    pool.free_blocks(blocks)  # cached: appended in LRU order
    assert _cached_free_order(pool) == [b.block_id for b in blocks]
    pool.demote([blocks[2]])
    assert _cached_free_order(pool) == [
        blocks[2].block_id,
        blocks[0].block_id,
        blocks[1].block_id,
        blocks[3].block_id,
    ]
    # Behind the uncached free blocks, whose reuse costs no hit.
    assert all(
        b.block_hash is None
        for b in pool.free_block_queue.get_all_free_blocks()[
            : _first_cached_index(pool)
        ]
    )
    # Still cached and hit-able.
    assert pool.get_cached_block(get_block_hash(blocks[2].block_hash), [0])
    # The next allocations take the uncached blocks, then the demoted one.
    num_uncached = _first_cached_index(pool)
    taken = pool.get_new_blocks(num_uncached + 1)
    assert taken[-1] is blocks[2] and blocks[2].block_hash is None


def test_no_demotion_while_the_cache_holds_a_full_horizon():
    """When the oldest evictable cached block is older than the horizon, the
    cache is not under pressure: LRU keeps the stale block behind it."""
    pool = BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=16)
    now = [0.0]
    pool._clock = lambda: now[0]
    blocks = pool.get_new_blocks(3)
    for i, block in enumerate(blocks):
        _cache(pool, block, i)
    pool.free_blocks(blocks)  # all join the LRU at t = 0
    pool.get_new_blocks(_first_cached_index(pool))  # leave no uncached slack
    now[0] = pool._DEMOTE_HORIZON_S + 1
    pool.demote([blocks[2]])
    assert _cached_free_order(pool) == [b.block_id for b in blocks]
    now[0] = pool._DEMOTE_HORIZON_S - 1  # a younger LRU head: pressure
    pool.demote([blocks[2]])
    assert _cached_free_order(pool)[0] == blocks[2].block_id


def test_demoting_the_last_demoted_block_again_keeps_the_free_queue_intact():
    """Two requests that resumed from one checkpoint each demote it when their
    own checkpoint registers. The block demoted last sits right before the LRU
    head, so the second demotion must not use it as its own anchor."""
    pool = BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=16)
    now = [0.0]
    pool._clock = lambda: now[0]
    blocks = pool.get_new_blocks(4)
    for i, block in enumerate(blocks):
        _cache(pool, block, i)
    pool.free_blocks(blocks)
    now[0] = 10.0  # a young LRU head: pressure
    num_free = pool.free_block_queue.num_free_blocks
    want = [b.block_id for b in (blocks[2], blocks[0], blocks[1], blocks[3])]
    pool.demote([blocks[2]])
    assert _cached_free_order(pool) == want
    pool.demote([blocks[2]])
    assert _cached_free_order(pool) == want
    pool.demote([blocks[2], blocks[2]])  # a repeated block in one call
    assert _cached_free_order(pool) == want
    pool.demote([blocks[0], blocks[2]])  # with another block demoted alongside
    assert _cached_free_order(pool) == [
        blocks[0].block_id,
        blocks[2].block_id,
        blocks[1].block_id,
        blocks[3].block_id,
    ]
    assert pool.free_block_queue.num_free_blocks == num_free
    assert len(pool.free_block_queue.get_all_free_blocks()) == num_free


def test_demote_of_a_referenced_block_applies_when_it_is_freed():
    pool = BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=16)
    a, b, c = pool.get_new_blocks(3)
    for i, block in enumerate((a, b, c)):
        _cache(pool, block, i)
    pool.demote([b])
    assert b.ref_cnt == 1 and b.prev_free_block is None  # still in use
    pool.free_blocks([a, b, c])
    assert _cached_free_order(pool) == [b.block_id, a.block_id, c.block_id]


def test_a_new_hit_cancels_a_pending_demotion():
    pool = BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=16)
    a, b = pool.get_new_blocks(2)
    _cache(pool, a, 0)
    _cache(pool, b, 1)
    pool.demote([a])
    pool.touch([a])  # another request hits it: not stale
    pool.free_blocks([a])
    pool.free_blocks([a, b])
    assert _cached_free_order(pool) == [a.block_id, b.block_id]


# ---------------------------------------------------------------------------
# Agentic two-turn flow on a hybrid (full attention + Mamba "align") manager.

HASH, BLOCK, DCP = 16, 64, 2
GLOBAL_BLOCK = BLOCK * DCP  # 128 tokens per scheduler block


def _manager(num_blocks: int = 256):
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
            num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups
        ),
        max_model_len=1 << 15,
        enable_caching=True,
        dcp_world_size=DCP,
        scheduler_block_size=GLOBAL_BLOCK,
        hash_block_size=HASH,
        use_eagle=True,
    )
    manager.coordinator.retention_interval = 0
    return manager


_SCHED = SimpleNamespace(
    block_size=GLOBAL_BLOCK,
    cache_config=SimpleNamespace(block_size=BLOCK),
    hash_block_size=HASH,
    max_num_scheduled_tokens=512,
    scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
    use_eagle_block_drop=True,
    mamba_partial_cache_hit=True,
    mamba_shared_prefix_checkpoint=False,
    mamba_has_prefill_checkpoint_blocks=True,
    mamba_prefill_checkpoint_alignment=16,
)


def _allocate(manager, request, num_new_tokens, *hit) -> None:
    """One scheduler step: allocate, account the tokens, run the copies."""
    manager.new_step_starts()
    assert manager.allocate_slots(request, num_new_tokens, *hit) is not None
    request.num_computed_tokens += num_new_tokens + (hit[0] if hit else 0)
    _, retained = manager.take_kv_cache_block_copies()
    manager.block_pool.free_blocks(retained)  # the copy step has run


def _run(manager, request, output: list[int]) -> None:
    """Prefill in mamba-aligned chunks from the prefix hit, then decode."""
    blocks, hit, _ = manager.get_computed_blocks(request)
    first: tuple = (hit, blocks)
    while request.num_computed_tokens + (first[0] if first else 0) < (
        request.num_prompt_tokens
    ):
        local = first[0] if first else 0
        todo = request.num_prompt_tokens - request.num_computed_tokens - local
        n = Scheduler._mamba_block_aligned_split(_SCHED, request, todo, local)
        assert n > 0
        _allocate(manager, request, n, *first)
        first = ()
    for tok in output:
        request.append_output_token_ids([tok])
        _allocate(manager, request, 1)


def test_agentic_turn_demotes_the_parents_stale_blocks_only():
    manager = _manager()
    pool = manager.block_pool
    # An unrelated, older session: its cached blocks must outlive the stale ones.
    other = make_request("other", list(range(50_000, 50_700)), HASH, sha256)
    _run(manager, other, list(range(60_000, 60_010)))
    other_full = [
        b.block_id
        for b in manager.coordinator.single_type_managers[0].req_to_blocks["other"]
        if b.block_hash is not None
    ]
    manager.free(other)

    # Turn 1: 1000-token prompt, 300 generated tokens (two generated-only blocks).
    prompt = list(range(1000))
    parent = make_request("parent", prompt, HASH, sha256)
    _run(manager, parent, list(range(70_000, 70_300)))
    full_mgr, mamba_mgr = manager.coordinator.single_type_managers
    parent_full = list(full_mgr.req_to_blocks["parent"])
    prompt_end = (len(prompt) - 1) // GLOBAL_BLOCK  # block 7 holds the prompt end
    tail, generated = parent_full[prompt_end], parent_full[prompt_end + 1 : 10]
    assert all(b.block_hash is not None for b in [tail, *generated])
    manager.free(parent)

    # Turn 2: the parent's prompt plus a different continuation (an agent turn
    # carries canned text, not the generated tokens).
    child = make_request("child", prompt + list(range(80_000, 80_200)), HASH, sha256)
    blocks, hit, _ = manager.get_computed_blocks(child)
    assert prompt_end * GLOBAL_BLOCK < hit < len(prompt)  # resumes inside `tail`
    resumed_state = next(b for b in reversed(blocks.blocks[1]) if not b.is_null)
    _run(manager, child, [])

    order = _free_order(pool)
    position = {block_id: i for i, block_id in enumerate(order)}
    stale = [tail.block_id, *(b.block_id for b in generated), resumed_state.block_id]
    for block_id in stale:
        assert block_id in position, block_id
    # Stale blocks are evicted before the older session's live cached blocks...
    assert max(position[b] for b in stale) < min(
        position[b] for b in other_full if b in position
    )
    # ...and are still cached (hit-able until reused).
    assert all(pool.blocks[b].block_hash is not None for b in stale)
    # The parent's prompt blocks the child shares are in use, not demoted.
    assert all(b.ref_cnt > 0 for b in parent_full[:prompt_end])


def test_replaying_the_generated_tokens_keeps_them_live():
    """Full attention alone can hit through generated tokens a chat turn
    resends; those blocks stay live, and only the ones past the divergence are
    demoted."""
    manager = make_kv_cache_manager(
        kv_cache_config=KVCacheConfig(
            num_blocks=64,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["full"],
                    FullAttentionSpec(
                        block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32
                    ),
                )
            ],
        ),
        max_model_len=4096,
        enable_caching=True,
        hash_block_size=16,
    )
    pool = manager.block_pool
    prompt, generated = list(range(100)), list(range(1000, 1060))
    parent = make_request("parent", prompt, 16, sha256)
    manager.allocate_slots(parent, len(prompt))
    parent.num_computed_tokens = len(prompt)
    for tok in generated:
        parent.append_output_token_ids([tok])
        manager.allocate_slots(parent, 1)
        parent.num_computed_tokens += 1
    parent_blocks = list(
        manager.coordinator.single_type_managers[0].req_to_blocks["parent"]
    )
    manager.free(parent)

    # Resends the first 40 generated tokens, then diverges inside block 8
    # (tokens 128-143): it hits blocks 0-7, generated-token blocks 7 included,
    # so blocks 8 and later are past the divergence.
    child = make_request("child", prompt + generated[:40] + [5000] * 20, 16, sha256)
    blocks, hit, _ = manager.get_computed_blocks(child)
    assert hit == 8 * 16
    manager.allocate_slots(child, len(child.prompt_token_ids) - hit, hit, blocks)
    assert all(b.ref_cnt > 0 for b in parent_blocks[:8])  # hit: live, in use
    stale = [b.block_id for b in parent_blocks[8:] if b.block_hash is not None]
    assert len(stale) == 2  # blocks 8 and 9 (tokens 128-159)
    assert set(_cached_free_order(pool)[: len(stale)]) == set(stale)
