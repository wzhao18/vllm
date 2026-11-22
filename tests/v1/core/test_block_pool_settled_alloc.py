# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Settled allocation: blocks for an async KV load avoid blocks the step still in
flight may use (``BlockFreeClock.freed_recently``), so the connector can issue
the load at allocation time instead of waiting for that step."""

import pytest

from vllm.v1.core.block_pool import BLOCK_FREE_CLOCK, BlockPool, SlotBlockPool

pytestmark = pytest.mark.cpu_test

clock = BLOCK_FREE_CLOCK


def _recent_and_settled(pool: BlockPool, n_old: int, n_recent: int):
    """Free ``n_old`` blocks before a step is built (settled) and ``n_recent``
    after (recent); uncached frees go LIFO to the front of the free queue."""
    old = pool.get_new_blocks(n_old)
    recent = pool.get_new_blocks(n_recent)
    clock.tick()
    pool.free_blocks(old)
    clock.step_built()  # a step is in flight from here on
    pool.free_blocks(recent)
    clock.tick()  # next schedule(): that step may still run
    return old, recent


def test_plain_allocation_reuses_recent_blocks_first():
    pool = BlockPool(num_gpu_blocks=6, enable_caching=False, hash_block_size=16)
    _, recent = _recent_and_settled(pool, n_old=2, n_recent=2)
    got = pool.get_new_blocks(2)
    assert {b.block_id for b in got} == {b.block_id for b in recent}
    assert all(clock.freed_recently(b) for b in got)


def test_settled_allocation_passes_over_recent_blocks():
    pool = BlockPool(num_gpu_blocks=6, enable_caching=False, hash_block_size=16)
    old, recent = _recent_and_settled(pool, n_old=2, n_recent=2)
    num_free = pool.get_num_free_blocks()
    with clock.settled():
        got = pool.get_new_blocks(2)
    assert not any(clock.freed_recently(b) for b in got)
    assert {b.block_id for b in got}.isdisjoint(b.block_id for b in recent)
    assert pool.get_num_free_blocks() == num_free - 2
    # The recent blocks stay free, at the front, for the next plain allocation.
    nxt = pool.get_new_blocks(2)
    assert {b.block_id for b in nxt} == {b.block_id for b in recent}


def test_settled_allocation_takes_recent_blocks_only_when_short():
    pool = BlockPool(num_gpu_blocks=4, enable_caching=False, hash_block_size=16)
    _recent_and_settled(pool, n_old=0, n_recent=3)
    # Only the null block's neighbours: 3 free, all recent (block 0 is null).
    with clock.settled():
        got = pool.get_new_blocks(3)
    assert len(got) == 3
    assert pool.get_num_free_blocks() == 0
    assert all(b.ref_cnt == 1 for b in got)


def _parent_with_recent_and_settled():
    parent = BlockPool(num_gpu_blocks=4, enable_caching=False, hash_block_size=16)
    a, b = parent.get_new_blocks(2)
    clock.tick()
    parent.free_blocks([a])  # settled
    clock.step_built()
    parent.free_blocks([b])  # recent, reused first (LIFO)
    clock.tick()
    return parent, a, b


def test_plain_slot_allocation_takes_the_recent_parent_block_as_row():
    parent, a, b = _parent_with_recent_and_settled()
    slots = SlotBlockPool(parent, slots_per_block=2)
    (slot,) = slots.get_new_blocks(1)
    assert slots.row_block(slot.block_id) is b
    assert clock.freed_recently(slot)


def test_settled_slot_allocation_takes_a_settled_row():
    parent, a, b = _parent_with_recent_and_settled()
    slots = SlotBlockPool(parent, slots_per_block=2)
    with clock.settled():
        (slot,) = slots.get_new_blocks(1)
    assert slots.row_block(slot.block_id) is not b
    assert not clock.freed_recently(slot)


def test_settled_slot_allocation_opens_a_row_when_free_slots_are_recent():
    parent, a, b = _parent_with_recent_and_settled()
    slots = SlotBlockPool(parent, slots_per_block=2)
    (s0,) = slots.get_new_blocks(1)  # plain: row = recent parent block b
    assert slots.row_block(s0.block_id) is b
    # The other slot of row b is free but recent (its row is).
    with clock.settled():
        (slot,) = slots.get_new_blocks(1)
    assert slots.row_block(slot.block_id) is not b
    assert not clock.freed_recently(slot)


def test_settled_context_nests_and_resets():
    assert not clock.wants_settled
    with clock.settled():
        assert clock.wants_settled
        with clock.settled():
            assert clock.wants_settled
        assert clock.wants_settled
    assert not clock.wants_settled
