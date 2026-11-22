# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load-ahead: with the step's token budget used up, the scheduler still starts
the async KV loads of the next fresh waiting requests (an async load takes no
compute budget), bounded by LOAD_AHEAD_MAX_REQS and a free-block reserve."""

import pytest

from vllm.v1.core.sched import scheduler as sched_mod
from vllm.v1.request import RequestStatus

from ..kv_connector.unit.utils import (
    create_mock_connector,
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)

pytestmark = pytest.mark.cpu_test

BS = 16


def _scheduler(num_blocks=200, mnbt=64):
    """Mock connector: a request whose id is in ``scheduler.test_hits`` has that
    many external tokens to load asynchronously; others miss."""
    vllm_config = create_vllm_config(max_num_batched_tokens=mnbt, block_size=BS)
    scheduler = create_scheduler(vllm_config, num_blocks=num_blocks)
    scheduler.connector = create_mock_connector()
    hits: dict[str, int] = {}
    scheduler.test_hits = hits

    def matched(request, _num_computed):
        n = hits.get(request.request_id, 0)
        return (n, True) if n else (0, False)

    scheduler.connector.get_num_new_matched_tokens.side_effect = matched
    scheduler.connector.get_loaded_kv_cache_group_ids.return_value = (0,)
    scheduler.connector.request_finished.return_value = (False, None)
    scheduler.connector.take_events.return_value = ()
    scheduler.connector.take_deferred_recving.return_value = set()
    return scheduler


def _add(scheduler, n_tokens, hit_tokens=0):
    r = create_request(num_tokens=n_tokens, max_tokens=1, block_size=BS)
    if hit_tokens:
        scheduler.test_hits[r.request_id] = hit_tokens
    scheduler.add_request(r)
    return r


def _fill_budget(scheduler):
    """A long fresh prompt with no hit uses the whole step budget (64)."""
    return _add(scheduler, 40 * BS)


def _outputs(scheduler, out, **kw):
    reqs = [scheduler.requests[i] for i in out.num_scheduled_tokens]
    return create_model_runner_output(reqs, **kw)


def test_loads_start_while_the_budget_is_used_up():
    scheduler = _scheduler()
    long = _fill_budget(scheduler)
    a = _add(scheduler, 6 * BS, hit_tokens=5 * BS)
    b = _add(scheduler, 6 * BS, hit_tokens=5 * BS)
    out = scheduler.schedule()
    assert out.num_scheduled_tokens == {long.request_id: 64}
    assert a.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert b.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert scheduler._load_ahead == {a, b}
    scheduler.update_from_output(
        out,
        _outputs(scheduler, out, finished_recving={a.request_id, b.request_id}),
    )


def test_without_load_ahead_budget_blocks_the_lookup(monkeypatch):
    monkeypatch.setattr(sched_mod, "LOAD_AHEAD_MAX_REQS", 0)
    scheduler = _scheduler()
    _fill_budget(scheduler)
    a = _add(scheduler, 6 * BS, hit_tokens=5 * BS)
    scheduler.schedule()
    assert a.status == RequestStatus.WAITING
    assert scheduler.connector.get_num_new_matched_tokens.call_count == 1


def test_request_without_a_hit_waits_for_budget_in_order():
    scheduler = _scheduler()
    _fill_budget(scheduler)
    miss = _add(scheduler, 6 * BS)
    b = _add(scheduler, 6 * BS, hit_tokens=5 * BS)
    scheduler.schedule()
    assert miss.status == RequestStatus.WAITING
    assert miss.num_computed_tokens == 0
    assert scheduler.waiting.peek_request() is miss
    assert b.status == RequestStatus.WAITING_FOR_REMOTE_KVS


def test_at_most_load_ahead_max_reqs_are_loaded_ahead(monkeypatch):
    monkeypatch.setattr(sched_mod, "LOAD_AHEAD_MAX_REQS", 2)
    scheduler = _scheduler()
    _fill_budget(scheduler)
    reqs = [_add(scheduler, 6 * BS, hit_tokens=5 * BS) for _ in range(4)]
    scheduler.schedule()
    assert [r.status for r in reqs] == [
        RequestStatus.WAITING_FOR_REMOTE_KVS,
        RequestStatus.WAITING_FOR_REMOTE_KVS,
        RequestStatus.WAITING,
        RequestStatus.WAITING,
    ]
    assert len(scheduler._load_ahead) == 2


@pytest.mark.parametrize("num_blocks,loads_ahead", [(49, False), (53, True)])
def test_reserve_keeps_a_step_chunk_free_for_compute(num_blocks, loads_ahead):
    # After its first chunk (4 blocks) the long prompt still reserves 36. A load
    # ahead (6 blocks) must also leave the 4-block step-chunk reserve free:
    # 48 usable blocks leave 44 - 6 = 38 < 36 + 4 (refused), 52 leave 42.
    # Without the reserve both would admit it.
    scheduler = _scheduler(num_blocks=num_blocks)
    assert scheduler._load_ahead_reserve_blocks == 4  # cdiv(64, 16)
    _fill_budget(scheduler)
    a = _add(scheduler, 6 * BS, hit_tokens=5 * BS)
    scheduler.schedule()
    assert (a.status == RequestStatus.WAITING_FOR_REMOTE_KVS) is loads_ahead
    assert (a in scheduler._load_ahead) is loads_ahead


def test_running_request_is_never_preempted_for_a_load_ahead():
    scheduler = _scheduler(num_blocks=60)
    long = _fill_budget(scheduler)
    loaders = [_add(scheduler, 6 * BS, hit_tokens=5 * BS) for _ in range(2)]
    pending = {r.request_id for r in loaders}
    for _ in range(25):
        out = scheduler.schedule()
        assert not out.preempted_req_ids
        if long.is_finished():
            break
        done = {
            r.request_id
            for r in loaders
            if r.request_id in pending
            and r.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        }
        pending -= done
        scheduler.update_from_output(
            out, _outputs(scheduler, out, finished_recving=done)
        )
    assert long.num_preemptions == 0


def test_loaded_ahead_request_runs_once_budget_frees_and_leaves_the_set():
    scheduler = _scheduler(mnbt=64)
    long = _add(scheduler, 4 * BS)  # one step uses the whole budget
    a = _add(scheduler, 6 * BS, hit_tokens=5 * BS)
    out = scheduler.schedule()
    assert a in scheduler._load_ahead
    scheduler.update_from_output(
        out, _outputs(scheduler, out, finished_recving={a.request_id})
    )
    out = scheduler.schedule()
    assert a.request_id in out.num_scheduled_tokens
    assert a not in scheduler._load_ahead
    assert long.is_finished() or long.request_id not in out.num_scheduled_tokens


def test_abort_while_loading_ahead_drops_it_from_the_set():
    scheduler = _scheduler()
    _fill_budget(scheduler)
    a = _add(scheduler, 6 * BS, hit_tokens=5 * BS)
    out = scheduler.schedule()
    assert a in scheduler._load_ahead
    scheduler.finish_requests(a.request_id, RequestStatus.FINISHED_ABORTED)
    assert a not in scheduler._load_ahead
    # The transfer still reports in; then its blocks are freed.
    scheduler.update_from_output(
        out, _outputs(scheduler, out, finished_recving={a.request_id})
    )
    assert a.request_id not in scheduler.requests



def test_load_ahead_only_for_requests_the_dp_plan_bound_here():
    """Bind at load-issue time: with a DP plan, load-ahead loads only requests
    the plan bound to this rank; others stay uncommitted (WAITING, movable)."""
    scheduler = _scheduler()
    long = _fill_budget(scheduler)
    a = _add(scheduler, 6 * BS, hit_tokens=5 * BS)
    scheduler.connector.dp_move_tokens.side_effect = lambda r: r.num_prompt_tokens
    scheduler.dp_bound_here = set()
    out = scheduler.schedule()
    assert out.num_scheduled_tokens == {long.request_id: 64}
    assert a.status == RequestStatus.WAITING
    assert scheduler._load_ahead == set()
    assert [rid for rid, _ in scheduler.dp_movable_requests(4)] == [a.request_id]
    # The next plan binds it here: loaded ahead while the budget is still full.
    scheduler.dp_bound_here = {a.request_id}
    out = scheduler.schedule()
    assert out.num_scheduled_tokens == {long.request_id: 64}
    assert a.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert scheduler._load_ahead == {a}
