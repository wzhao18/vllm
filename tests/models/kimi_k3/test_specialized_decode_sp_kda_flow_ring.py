# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KdaFlow state-page ring protocol (CPU model, no GPU).

KdaFlow's two warp groups share a ring of ``flow.SLOTS`` page slots: group g consumes
items g, g + 2, ...; item j's page lands in slot j % SLOTS, refilled by the consumer
of item j - SLOTS (the other group) once it has read it, and each consumer waits on
an mbarrier with a parity wait. ``mbarrier.try_wait.parity`` only tells the current
phase from the one before it: a wait whose phase is still one behind (the previous
page of that slot not landed yet) sees the parity of the phase two back, which is
complete, and passes.

This model replays every interleaving of a bounded schedule (each group may run
ahead over skipped items while a TMA is in flight) against the real mbarrier
semantics, and checks the two invariants the kernel relies on: a consumer passes
its wait only after its page has landed, and no barrier phase ever receives more
than its expected arrivals ('Warp Barrier Arrival Mismatch', BUG RHEADEC)."""

import itertools

import pytest

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda import flow

SLOTS = flow.SLOTS
GROUPS = 2
ARRIVALS = 4  # one per warp quarter


class Barrier:
    """mbarrier with an arrival count and a transaction count (one page = 1 unit)."""

    def __init__(self):
        self.phase, self.pending, self.tx = 0, ARRIVALS, 0

    def arrive_expect(self):
        if self.pending == 0:
            raise AssertionError("arrival mismatch: arrival into a phase with none left")
        self.pending -= 1
        self.tx += 1
        self._advance()

    def complete_tx(self):
        self.tx -= 1
        self._advance()

    def _advance(self):
        if self.pending == 0 and self.tx == 0:
            self.phase += 1
            self.pending = ARRIVALS

    def try_wait_parity(self, parity):
        # PTX: true once the phase of this parity (current or the one before) is
        # complete, i.e. the current phase has the other parity.
        return self.phase % 2 != parity


def barrier_of(j, fixed):
    if fixed:
        return j % SLOTS + SLOTS * ((j // SLOTS) % 2)
    return j % SLOTS


def parity_of(j, fixed):
    return (j // (2 * SLOTS)) % 2 if fixed else (j // SLOTS) % 2


def run(count, order, fixed):
    """Replay one schedule: ``order`` picks, at each step, which actor moves (group 0,
    group 1, or a TMA completion of the oldest / newest page in flight). Returns the
    violations seen."""
    nb = 2 * SLOTS if fixed else SLOTS
    bars = [Barrier() for _ in range(nb)]
    landed = set()
    inflight = []  # (item) pages issued, not landed
    # Prologue: pages of items 0..SLOTS-1 issued (4 quarter arrivals each).
    for j in range(min(SLOTS, count)):
        for _ in range(ARRIVALS):
            bars[barrier_of(j, fixed)].arrive_expect()
        inflight.append(j)
    nxt = [g for g in range(GROUPS)]  # next item per group
    violations = []
    for actor in order:
        if actor in ("t0", "t1"):  # TMA completion, not in issue order
            if inflight:
                j = inflight.pop(0 if actor == "t0" else -1)
                for _ in range(ARRIVALS):
                    bars[barrier_of(j, fixed)].complete_tx()
                landed.add(j)
            continue
        g = actor
        j = nxt[g]
        if j >= count:
            continue
        b = bars[barrier_of(j, fixed)]
        if not b.try_wait_parity(parity_of(j, fixed)):
            continue  # blocked
        if j not in landed:
            violations.append(f"group {g} passed item {j}'s wait before its page landed")
        nxt[g] = j + GROUPS
        if j + SLOTS < count:  # refill the slot with item j + SLOTS's page
            try:
                for _ in range(ARRIVALS):
                    bars[barrier_of(j + SLOTS, fixed)].arrive_expect()
            except AssertionError as e:
                violations.append(f"group {g} refilling item {j + SLOTS}: {e}")
                return violations
            inflight.append(j + SLOTS)
    return violations


def schedules(length):
    for order in itertools.product((0, 1, "t0", "t1"), repeat=length):
        yield order


@pytest.mark.parametrize("count", [5, 7, 9])
def test_single_barrier_ring_admits_an_early_wait(count):
    """The pre-fix protocol (one barrier per slot): some schedule lets a group pass
    its wait before the page landed and then arrive into the wrong phase."""
    found = any(run(count, order, fixed=False) for order in schedules(10))
    assert found


@pytest.mark.parametrize("count", [5, 9])
def test_generation_barriers_are_exact(count):
    """The fixed protocol (two barriers per slot by generation parity): no schedule
    passes a wait early or over-arrives a phase."""
    for order in schedules(10):
        assert run(count, order, fixed=True) == [], order


def test_kernel_uses_the_generation_barriers():
    assert flow.page_parity(0) == 0
    assert flow.page_parity(SLOTS) == 0
    assert flow.page_parity(2 * SLOTS) == 1
    for j in range(4 * SLOTS):
        assert flow.page_barrier(0, j) == j % SLOTS + SLOTS * ((j // SLOTS) % 2)
