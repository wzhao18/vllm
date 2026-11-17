# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A KV cache group packed as slots of the shared pool's blocks.

``SlotBlockPool`` hands out one group's blocks as equal slots of the parent
pool's blocks (K3.1's block-outermost rows): parent blocks are taken on demand,
given back when no slot is referenced, kept in the parent's LRU while a slot is
cached, and a parent reuse evicts those slots. The coordinator wiring packs a
sliding-window draft group beside a Mamba-hybrid target (Kimi K3.1 with the
release GQA DSpark draft, scaled down): hits keep the hash-unit granularity and
restore exactly the draft window the writer computed.
"""

import itertools
import random

import pytest
import torch

from tests.v1.core.test_prefix_caching import make_request
from vllm.distributed.kv_events import BlockRemoved
from vllm.utils.hashing import sha256
from vllm.v1.core import block_pool as block_pool_module
from vllm.v1.core.block_pool import BlockPool, SlotBlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)

pytestmark = pytest.mark.cpu_test

HASH = 16
SLOTS = 4
GROUP = 1


@pytest.fixture(autouse=True)
def _auto_init_hash_fn():
    init_none_hash(sha256)


@pytest.fixture
def clock(monkeypatch):
    """LRU join times that strictly increase."""
    ticks = itertools.count(1)
    monkeypatch.setattr(block_pool_module.time, "monotonic", lambda: next(ticks))


def _pools(num_blocks: int = 6, events: bool = False):
    parent = BlockPool(
        num_blocks,
        enable_caching=True,
        hash_block_size=HASH,
        enable_kv_cache_events=events,
    )
    return parent, SlotBlockPool(parent, SLOTS)


def _cache(pool: BlockPool, request, blocks) -> None:
    """Cache ``blocks`` as the request's first blocks."""
    pool.cache_full_blocks(request, blocks, 0, len(blocks), HASH, GROUP)


def _request(name: str, num_blocks: int, offset: int = 0):
    tokens = list(range(offset, offset + num_blocks * HASH))
    return make_request(name, tokens, HASH, sha256)


def _rows(blocks) -> set[int]:
    return {block.block_id // SLOTS for block in blocks}


def test_slots_share_a_parent_block_until_all_are_free():
    parent, pool = _pools()
    num_free = parent.get_num_free_blocks()
    first = pool.get_new_blocks(3)
    (row,) = _rows(first)
    assert row != 0 and parent.get_num_free_blocks() == num_free - 1
    second = pool.get_new_blocks(2)
    # The row's last slot, then a slot of a second row.
    assert second[0].block_id // SLOTS == row and len(_rows(second)) == 2
    assert parent.get_num_free_blocks() == num_free - 2
    pool.free_blocks(first)
    assert parent.get_num_free_blocks() == num_free - 2
    pool.free_blocks(second)
    assert parent.get_num_free_blocks() == num_free
    assert pool.get_num_free_blocks() == 0


def test_cached_slots_age_with_their_row_until_the_parent_reuses_it():
    parent, pool = _pools(events=True)
    num_free = parent.get_num_free_blocks()
    request = _request("a", SLOTS)
    slots = pool.get_new_blocks(SLOTS)
    _cache(pool, request, slots)
    pool.free_blocks(reversed(slots))
    # Given back to the parent, counted free, its slots still hit.
    assert parent.get_num_free_blocks() == num_free
    hit = pool.get_cached_block(request.block_hashes[1], [GROUP])
    assert hit == [slots[1]]
    # A hit takes the row back, the other slots still cached.
    pool.touch(hit)
    assert parent.get_num_free_blocks() == num_free - 1
    assert pool.get_cached_block(request.block_hashes[3], [GROUP]) == [slots[3]]
    pool.free_blocks(hit)
    assert parent.get_num_free_blocks() == num_free

    # Reusing the row evicts every cached slot, with an event each.
    parent.kv_event_queue.clear()
    parent.get_new_blocks(num_free)
    assert all(
        pool.get_cached_block(block_hash, [GROUP]) is None
        for block_hash in request.block_hashes
    )
    removed = [e for e in parent.take_events() if isinstance(e, BlockRemoved)]
    assert len(removed) == SLOTS and {e.group_idx for e in removed} == {GROUP}


def test_uncached_free_slots_and_free_parent_blocks_come_before_cached_slots():
    parent, pool = _pools()
    request = _request("a", 1)
    row_a = pool.get_new_blocks(SLOTS)
    _cache(pool, request, row_a[:1])
    pool.free_blocks(row_a[:1])
    # The parent has free blocks: a new slot takes one rather than evicting.
    (new,) = pool.get_new_blocks(1)
    assert _rows([new]) != _rows(row_a)
    assert pool.get_cached_block(request.block_hashes[0], [GROUP]) == [row_a[0]]
    # Uncached free slots of held rows are used first.
    pool.free_blocks(row_a[1:2])
    assert pool.get_new_blocks(1) == [row_a[1]]


def test_one_lru_order_across_the_pools(clock):
    """With no free uncached block anywhere, a new slot evicts whichever
    cached content joined the LRU first: a parent block or a cached slot."""
    parent, pool = _pools(num_blocks=3)  # 2 usable blocks

    def cache_in_parent(name: str, offset: int):
        request = _request(name, 1, offset=offset)
        (block,) = parent.get_new_blocks(1)
        parent.cache_full_blocks(request, [block], 0, 1, HASH, 0)
        parent.free_blocks([block])
        return request

    older = cache_in_parent("older", 1000)
    slot_request = _request("slot", 1, offset=2000)
    slots = pool.get_new_blocks(SLOTS)
    _cache(pool, slot_request, slots[:1])
    pool.free_blocks(slots[:1])
    # The parent's cached block is older than the cached slot: it goes.
    (new,) = pool.get_new_blocks(1)
    assert new.block_id // SLOTS != slots[0].block_id // SLOTS
    assert parent.get_cached_block(older.block_hashes[0], [0]) is None
    assert pool.get_cached_block(slot_request.block_hashes[0], [GROUP]) == [slots[0]]

    # Now the parent's cached block is newer than the slot: the slot goes.
    pool.free_blocks([new])
    newer = cache_in_parent("newer", 3000)
    assert pool.get_new_blocks(1) == [slots[0]]
    assert pool.get_cached_block(slot_request.block_hashes[0], [GROUP]) is None
    assert parent.get_cached_block(newer.block_hashes[0], [0]) is not None


def test_parent_block_count_bounds_what_an_allocation_takes():
    parent, pool = _pools(num_blocks=8)
    request = _request("a", SLOTS)
    held = pool.get_new_blocks(SLOTS + 1)  # one full row and a slot of another
    _cache(pool, request, held[:SLOTS])
    pool.free_blocks(held[:SLOTS])  # its row goes back to the parent, cached
    hits = [pool.get_cached_block(h, [GROUP])[0] for h in request.block_hashes]
    for num_new in range(0, 3 * SLOTS):
        bound = pool.get_num_new_parent_blocks(num_new, hits)
        num_free = parent.get_num_free_blocks()
        pool.touch(hits)
        new = pool.get_new_blocks(num_new)
        assert num_free - parent.get_num_free_blocks() <= bound
        pool.free_blocks(new + hits)
    pool.free_blocks(held[SLOTS:])
    assert parent.get_num_free_blocks() == 7


def test_reset_drops_every_slot_hash():
    parent, pool = _pools()
    request = _request("a", SLOTS)
    slots = pool.get_new_blocks(SLOTS)
    _cache(pool, request, slots)
    assert not pool.reset_prefix_cache()  # referenced
    pool.free_blocks(slots)
    assert parent.reset_prefix_cache() and pool.reset_prefix_cache()
    assert all(
        pool.get_cached_block(block_hash, [GROUP]) is None
        for block_hash in request.block_hashes
    )


# Kimi K3.1 with the release draft, scaled down 64x: DSA rows of 64 tokens per
# rank at DCP 4 (256 per row, the scheduler block), KDA "align" states every 64
# tokens, prefix-hash unit 16 (128), a 128-token draft window (2048).
BLOCK, DCP, WINDOW = 64, 4, 128
DSA, KDA, DRAFT = 0, 1, 2


def _manager(packed: bool, num_rows: int = 64) -> KVCacheManager:
    draft_block = HASH if packed else BLOCK
    config = KVCacheConfig(
        num_blocks=num_rows,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["dsa"],
                MLAAttentionSpec(
                    block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32
                ),
            ),
            KVCacheGroupSpec(
                ["kda"],
                MambaSpec(
                    block_size=BLOCK,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    num_prefill_checkpoint_blocks=1,
                    prefill_checkpoint_alignment=HASH,
                ),
            ),
            KVCacheGroupSpec(
                ["draft"],
                SlidingWindowSpec(
                    block_size=draft_block,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=WINDOW,
                    dcp_sharded=False,
                ),
                blocks_per_pool_block=SLOTS if packed else 1,
            ),
        ],
    )
    return KVCacheManager(
        config,
        max_model_len=4096,
        enable_caching=True,
        hash_block_size=HASH,
        scheduler_block_size=BLOCK * DCP,
        dcp_world_size=DCP,
    )


def _prefill(manager: KVCacheManager, request, chunks: list[int]) -> None:
    blocks, num_hit, _ = manager.get_computed_blocks(request)
    for i, num_new in enumerate(chunks):
        assert manager.allocate_slots(
            request, num_new, num_hit if i == 0 else 0, blocks if i == 0 else None
        )
        request.num_computed_tokens += (num_hit if i == 0 else 0) + num_new


def _draft_hit(manager: KVCacheManager, request) -> tuple[int, list[int]]:
    blocks, num_hit, _ = manager.get_computed_blocks(request)
    return num_hit, [b.block_id for b in blocks.blocks[DRAFT] if not b.is_null]


@pytest.mark.parametrize("packed", [True, False])
def test_prompt_end_hit_restores_the_writers_draft_window(packed):
    """Unpacked, the draft group's 64-token blocks turn fine-grained hits off:
    the next turn resumes at the scheduler block (512). Packed, it resumes at
    the prompt's last hash boundary (592) with exactly the writer's draft
    window blocks."""
    manager = _manager(packed)
    tokens = list(range(1000, 1600))
    writer = make_request("writer", tokens, HASH, sha256)
    _prefill(manager, writer, [600])
    window = manager.get_block_ids("writer")[DRAFT]
    manager.free(writer)

    turn = make_request("turn", tokens + list(range(5000, 5100)), HASH, sha256)
    num_hit, draft_blocks = _draft_hit(manager, turn)
    if not packed:
        assert num_hit == 512
        return
    assert manager.coordinator.enable_partial_hash_hits
    assert num_hit == 592
    tail = range(592 // HASH - WINDOW // HASH, 592 // HASH)
    assert draft_blocks == [window[i] for i in tail]


def test_draft_tails_follow_the_state_checkpoints():
    """Only window tails ending where KDA keeps a state are cached: here the
    first chunk's end (512) and the prompt checkpoint (592). A sibling that
    diverges after 560 tokens resumes at 512."""
    manager = _manager(packed=True)
    tokens = list(range(1000, 1600))
    writer = make_request("writer", tokens, HASH, sha256)
    _prefill(manager, writer, [512, 88])
    manager.free(writer)
    draft_pool = manager.coordinator.single_type_managers[DRAFT].block_pool
    cached = len(draft_pool.cached_block_hash_to_block)
    # Tails [384, 512) and [464, 592): 13 blocks, not the prompt's 37.
    assert cached == 13

    sibling = make_request("sibling", tokens[:560] + [7] * 80, HASH, sha256)
    assert _draft_hit(manager, sibling)[0] == 512


def _chunks(start: int, end: int) -> list[int]:
    """Chunk lengths as the Mamba-aligned scheduler splits a prefill: a
    mid-block start runs to its block's end first, then scheduler blocks."""
    bounds, pos = [], start
    while pos < end:
        step = BLOCK if pos % BLOCK else BLOCK * DCP
        pos = min(end, (pos // step + 1) * step)
        bounds.append(pos)
    return [b - a for a, b in zip([start, *bounds], bounds)]


def test_random_traffic_hits_are_exact_and_rows_balance():
    """Shared-prefix traffic with chunked prefills, decodes, frees and pool
    pressure: every hit's draft blocks hold the KV of exactly the hitting
    request's prefix, and the pools' block accounting stays consistent."""
    rng = random.Random(0)
    manager = _manager(packed=True, num_rows=48)
    pool = manager.coordinator.single_type_managers[DRAFT].block_pool
    parent = manager.block_pool
    # Draft slot -> prefix hash of the 16 tokens the worker last wrote there.
    written: dict[int, bytes] = {}

    def write(request) -> None:
        table = manager.get_block_ids(request.request_id)[DRAFT]
        for j in range(min(request.num_computed_tokens // HASH, len(table))):
            if table[j]:
                written[table[j]] = request.block_hashes[j]

    prefixes = [[rng.randrange(1000) for _ in range(700)] for _ in range(3)]
    running: list = []
    num_hits = 0
    for step in range(150):
        if running and rng.random() < 0.35:
            manager.free(running.pop(rng.randrange(len(running))))
        length = rng.randrange(100, 700)
        tokens = rng.choice(prefixes)[: length - 40]
        tokens += [rng.randrange(1000) for _ in range(40)]
        request = make_request(f"r{step}", tokens, HASH, sha256)
        blocks, num_hit, _ = manager.get_computed_blocks(request)
        assert num_hit % HASH == 0
        num_hits += num_hit > 0
        for i, block in enumerate(blocks.blocks[DRAFT]):
            if not block.is_null:
                assert written[block.block_id] == request.block_hashes[i]
        admitted = True
        for i, num_new in enumerate(_chunks(num_hit, len(tokens))):
            hit_tokens = num_hit if i == 0 else 0
            if not manager.allocate_slots(
                request, num_new, hit_tokens, blocks if i == 0 else None
            ):
                admitted = False
                break
            request.num_computed_tokens += hit_tokens + num_new
            write(request)
        if not admitted:
            manager.free(request)
            continue
        for _ in range(rng.randrange(3)):  # a few decode tokens
            request.append_output_token_ids(rng.randrange(1000))
            if not manager.allocate_slots(request, 1):
                break
            request.num_computed_tokens += 1
            write(request)
        running.append(request)

        # Held rows count their referenced slots; nothing leaks.
        for row, used in pool._num_used.items():
            assert used == sum(slot.ref_cnt > 0 for slot in pool._slots(row)) > 0
        assert not pool._held.keys() & pool._released.keys()
        used_rows = parent.num_gpu_blocks - 1 - parent.get_num_free_blocks()
        assert used_rows >= len(pool._held)
    assert num_hits > 30
    for request in running:
        manager.free(request)
    assert not pool._held
    assert parent.get_num_free_blocks() == parent.num_gpu_blocks - 1
