# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCP pair exchange step budget (deferred admission): a step emits at most the
workers' ``PAIR_RING_SLOTS`` chunks of exchange, oldest ready plans first (and
always one), so a burst of pulls spreads over the next steps.

The budget is exact: every plan whose pull succeeded and whose request is
still alive is emitted once, in the order the pulls became ready, by a step
no earlier than the one its pull's end allows; a request aborted before its
plan is emitted never has it emitted; and no request is promoted before the
step after the one emitting its plan.
"""

import math
import random
from collections import deque

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.nixl import dcp_pair_exchange as dpe
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.dcp_pair_exchange import (
    PairUnit,
)
from vllm.v1.core.sched.output import SchedulerOutput

from .test_nixl_pair_exchange_scheduler import (
    D_DCP,
    P_BLOCK,
    _Engine,
    _make_scheduler,
    _pull_request,
    _pulled_units,
)

pytestmark = pytest.mark.cpu_test


def test_exchange_chunks_counts_every_stage():
    units = [PairUnit(0, b, True) for b in range(9)]
    # Ratio 2: one stage, 9 units in chunks of 4.
    assert dpe.exchange_chunks(units, 16, 4) == 3
    # Ratio 4 (P DCP 4 at D DCP 16): two stages.
    staged = [PairUnit(0, b, True, 4, 4) for b in range(5)]
    assert dpe.exchange_chunks(staged, 16, 4) == 2 * 2
    assert dpe.exchange_chunks(units + staged, 16, 4) == 3 + 4
    assert dpe.exchange_chunks([], 16, 4) == 0


def test_budget_takes_the_oldest_plans_that_fit_and_at_least_one(monkeypatch):
    monkeypatch.setattr(dpe, "PAIR_RING_SLOTS", 2)
    engine = _Engine(_make_scheduler(), defer=True)
    nixl = engine.nixl
    nixl._pair_chunk_units = 2

    def plan(n):
        return [PairUnit(0, b, True) for b in range(n)]

    # Budget 2 chunks x 2 units = 4 units: a (3) fits, b (2) would make 5.
    nixl._pair_ready = {"a": plan(3), "b": plan(2), "c": plan(1)}
    assert list(nixl._take_pair_budget()) == ["a"]
    assert list(nixl._pair_ready) == ["b", "c"]
    # b + c = 3 units, 2 chunks: both.
    assert list(nixl._take_pair_budget()) == ["b", "c"]
    assert not nixl._pair_ready
    # A plan above the budget alone still goes (the ring wraps for it).
    nixl._pair_ready = {"big": plan(9), "d": plan(1)}
    assert list(nixl._take_pair_budget()) == ["big"]
    assert list(nixl._pair_ready) == ["d"]


@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("seed", range(6))
def test_random_bursts_are_spread_exactly(
    seed: int, async_scheduling: bool, monkeypatch
):
    """Random arrivals, pull ends (several at once), failures, aborts and
    paused steps under a budget of 2 one-unit chunks per step."""
    monkeypatch.setattr(dpe, "PAIR_RING_SLOTS", 2)
    rng = random.Random(seed)
    engine = _Engine(
        _make_scheduler(
            async_scheduling=async_scheduling,
            max_num_seqs=4,
            kv_load_failure_policy="recompute",
            num_blocks=512,
        ),
        defer=True,
    )
    engine.nixl._pair_chunk_units = 1
    budget = dpe.PAIR_RING_SLOTS
    arrivals = [
        _pull_request(i, rng.randint(1, 6 * P_BLOCK), max_tokens=rng.randint(1, 4))
        for i in range(1, 15)
    ]
    plans: dict[str, list[PairUnit]] = {}
    ends: dict[str, int] = {}
    failed: set[str] = set()
    ready_at: dict[str, int] = {}  # First schedule() that may emit the plan.
    aborted_at: dict[str, int] = {}
    in_flight: deque[tuple[int, SchedulerOutput]] = deque()
    depth = 2 if async_scheduling else 1
    step = 0
    while arrivals or engine.scheduler.requests or in_flight:
        assert step < 600, "the requests did not drain"
        for _ in range(rng.choice((0, 0, 1, 3))):
            if arrivals:
                engine.add(arrivals.pop())
        live = [
            r.request_id
            for r in engine.scheduler.requests.values()
            if not r.is_finished()
        ]
        if live and rng.random() < 0.06:
            victim = rng.choice(live)
            aborted_at.setdefault(victim, len(engine.emitted))
            engine.abort(victim)
        output = engine.schedule()
        for req_id, recv in output.kv_connector_metadata.reqs_to_recv.items():
            if recv.awaiting_kvs:
                plans[req_id] = _pulled_units(output, req_id)
                # Bursts: pulls end together.
                ends[req_id] = step + rng.choice((1, 1, 2, 3))
                if rng.random() < 0.1:
                    failed.add(req_id)
        in_flight.append((step, output))
        drain = not arrivals and not engine.scheduler.requests
        while in_flight and (len(in_flight) >= depth or drain):
            index, done = in_flight.popleft()
            ended = {r for r, end in ends.items() if end == index}
            engine.update(done, finished=ended - failed, failed=ended & failed)
            for r in sorted(ended - failed):
                ready_at[r] = len(engine.emitted)
        step += 1

    emitted_order = [r for plans_i in engine.emitted for r in plans_i]
    # Each plan at most once; only plans of successful pulls.
    assert len(emitted_order) == len(set(emitted_order))
    assert set(emitted_order) <= set(ready_at)
    for req_id in ready_at:
        dead_first = aborted_at.get(req_id, math.inf) <= ready_at[req_id]
        emissions = engine.emissions(req_id)
        if dead_first:
            assert emissions == [], req_id
            continue
        if not emissions:
            # Only an abort while it waited for the budget drops it.
            assert req_id in aborted_at, req_id
            continue
        [at] = emissions
        assert at >= ready_at[req_id], req_id
        assert engine.emitted[at][req_id] == plans[req_id]
    # Oldest ready first: a plan is never emitted ahead of one that became
    # ready in an earlier step (pulls ending together have no order).
    readiness = [ready_at[r] for r in emitted_order]
    assert readiness == sorted(readiness)
    # The budget: a step carrying more than one plan stays within it.
    for plans_i in engine.emitted:
        if len(plans_i) > 1:
            units = [u for p in plans_i.values() for u in p]
            assert dpe.exchange_chunks(units, D_DCP, 1) <= budget
    # Promotion only after the step emitting the plan (deferred admission).
    for req_id, promoted_at in engine.promoted.items():
        if req_id in failed or not engine.emissions(req_id):
            continue
        assert engine.emissions(req_id)[0] < promoted_at, req_id
    # Some step deferred a ready plan (the budget was exercised).
    assert any(
        engine.emissions(r) and engine.emissions(r)[0] > ready_at[r]
        for r in ready_at
    ), "no plan waited for the budget: the test did not exercise it"
    engine.assert_idle()


def test_exchange_after_the_forward_is_bound_raises():
    """The exchange's deadlock freedom needs its packs ahead of the step's
    forward on the compute stream: queueing it after the forward was bound
    (check_pair_exchanges_ran) raises."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlConnectorMetadata,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
        NixlPullConnectorWorker,
    )

    worker = object.__new__(NixlPullConnectorWorker)
    worker._pair_consumed = None
    worker._pair_forward_bound = None
    meta = NixlConnectorMetadata()
    meta.reqs_to_pair_exchange = {"a": [PairUnit(0, 1, True)]}
    worker._pair_consumed = meta  # finalize_loaded_kv ran
    worker.check_pair_exchanges_ran(meta)  # the forward is bound
    with pytest.raises(RuntimeError, match="after this step's forward"):
        worker.run_pair_exchanges(meta)
