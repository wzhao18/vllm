# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same-step admission of async KV loads (join_load_in_step).

An async load is normally parked in WAITING_FOR_REMOTE_KVS until the worker
reports it, so its request computes at least one step after the step that
allocated its blocks. A connector may instead join the load to that step when
it can have the KV in the blocks before the step's forward reads them (its
worker waits on that load). These tests pin:

- the scheduler admits a joined request in its allocation step, only when it
  has compute budget there (never a load-ahead), and caches its blocks only
  once the load reports clean, so no other request can hit KV that may still
  fail (d);
- a failed joined load drops the request's output and recomputes it, with
  nothing cached (c);
- the Mooncake worker waits, before the forward, on exactly the joined loads
  of the step's new requests, by request (a); a failure does not raise (the
  step and its collectives run), is reported per request, and the request's
  KV is never stored (c);
- an early-issued joined load is not issued again with the step's metadata
  (b);
- the load-time model and the in-flight step clock the decision uses.
"""

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
    metrics,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
    worker as store_worker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    LoadSpec,
    MooncakeStoreConnectorMetadata,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.metrics import (
    MooncakeStoreConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler import (
    LoadTimeModel,
)
from vllm.v1.core.sched.scheduler import InflightStepClock
from vllm.v1.request import RequestStatus

from .test_mooncake_store_early_load import _blocks, _Channel
from .test_mooncake_store_scheduler import _make_bare_scheduler, _make_new_scheduler_output
from .test_mooncake_store_worker import _make_bare_worker
from .utils import MockKVConfig, create_model_runner_output

pytestmark = pytest.mark.cpu_test

BS = 16


def _scheduler(join: bool = True, matched: int = 3 * BS, **kwargs):
    return create_scheduler(
        enable_prefix_caching=True,
        use_kv_connector=MockKVConfig(
            matched_tokens=matched, is_async=True, join_in_step=join
        ),
        block_size=BS,
        **kwargs,
    )


def _request(req_id: str = "r", num_tokens: int = 4 * BS, max_tokens: int = 1):
    (request,) = create_requests(
        num_requests=1,
        num_tokens=num_tokens,
        max_tokens=max_tokens,
        block_size=BS,
        req_ids=[req_id],
        same_prompt=True,
    )
    return request


def _cached_tokens(scheduler, prompt_twin) -> int:
    """Local prefix-cache hit a request with this prompt would get now."""
    return scheduler.kv_cache_manager.get_computed_blocks(prompt_twin)[1]


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def test_joined_load_computes_in_its_allocation_step():
    scheduler = _scheduler()
    request = _request()
    scheduler.add_request(request)

    output = scheduler.schedule()

    assert request.status == RequestStatus.RUNNING
    assert [r.req_id for r in output.scheduled_new_reqs] == ["r"]
    assert output.scheduled_new_reqs[0].num_computed_tokens == 3 * BS
    assert output.num_scheduled_tokens == {"r": BS}
    (query,) = scheduler.connector.join_queries
    assert query[:2] == ("r", 3 * BS)
    # Not cached while the load may still fail.
    assert _cached_tokens(scheduler, _request("twin")) == 0

    # The clean report (here with the step's output) caches the blocks.
    scheduler.update_from_output(
        output, create_model_runner_output([request], finished_recving={"r"})
    )
    assert request.is_finished()
    assert _cached_tokens(scheduler, _request("twin")) >= 3 * BS
    assert not scheduler._joined_loads


def test_declined_join_parks_the_load_as_before():
    scheduler = _scheduler(join=False)
    request = _request()
    scheduler.add_request(request)

    output = scheduler.schedule()

    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert output.num_scheduled_tokens == {}
    assert not scheduler._joined_loads


def test_no_join_for_a_load_ahead():
    # The first request's chunk takes the whole budget; the second only
    # starts its load ahead, so it is never offered a join.
    scheduler = _scheduler(max_num_batched_tokens=BS, max_model_len=8 * BS)
    first, second = _request("a"), _request("b")
    scheduler.add_request(first)
    scheduler.add_request(second)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {"a": BS}
    assert [q[0] for q in scheduler.connector.join_queries] == ["a"]
    assert second.status == RequestStatus.WAITING_FOR_REMOTE_KVS


def test_pending_blocks_are_not_hit_by_other_requests_in_the_step():
    # (d) Two requests with the same prompt in one step: the second does not
    # take the first's pending blocks as a local hit (they are not cached),
    # so it never reads KV still in flight; it loads its own.
    scheduler = _scheduler()
    first, second = _request("a"), _request("b")
    scheduler.add_request(first)
    scheduler.add_request(second)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {"a": BS, "b": BS}
    assert [q[:2] for q in scheduler.connector.join_queries] == [
        ("a", 3 * BS),
        ("b", 3 * BS),
    ]
    first_blocks = set(scheduler.kv_cache_manager.get_block_ids("a")[0])
    second_blocks = set(scheduler.kv_cache_manager.get_block_ids("b")[0])
    assert not first_blocks & second_blocks
    assert _cached_tokens(scheduler, _request("twin")) == 0


@pytest.mark.parametrize("recompute", [True, False])
def test_failed_joined_load_drops_output(recompute):
    # (c) The load failed after the request was admitted: its output of the
    # step is dropped and nothing it wrote is cached. Under "recompute" it
    # starts over; under "fail" it fails like any failed load.
    scheduler = _scheduler()
    scheduler.recompute_kv_load_failures = recompute
    request = _request(max_tokens=4)
    scheduler.add_request(request)
    output = scheduler.schedule()
    assert request.status == RequestStatus.RUNNING

    engine_outputs = scheduler.update_from_output(
        output,
        create_model_runner_output(
            [request], finished_recving={"r"}, failed_recving={"r"}
        ),
    )

    delivered = [o for outs in engine_outputs.values() for o in outs.outputs]
    assert all(o.new_token_ids == [] for o in delivered)
    assert request.num_output_tokens == 0
    assert _cached_tokens(scheduler, _request("twin")) == 0
    assert not scheduler._joined_loads
    if not recompute:
        assert request.status == RequestStatus.FINISHED_ERROR
        return
    assert not delivered
    assert request.status == RequestStatus.PREEMPTED
    assert request.num_computed_tokens == 0
    # Recomputed from scratch (a fresh lookup, which may load again).
    output = scheduler.schedule()
    assert "r" in output.num_scheduled_tokens


def test_report_after_the_request_ended_is_ignored():
    scheduler = _scheduler()
    request = _request(max_tokens=4)
    scheduler.add_request(request)
    output = scheduler.schedule()
    scheduler.finish_requests("r", RequestStatus.FINISHED_ABORTED)

    scheduler.update_from_output(
        output, create_model_runner_output([], finished_recving={"r"})
    )

    assert not scheduler._joined_loads


# ---------------------------------------------------------------------------
# Mooncake worker
# ---------------------------------------------------------------------------


def _bare_worker():
    worker = _make_bare_worker(kv_role="kv_both")
    worker._kv_connector_stats_lock = threading.Lock()
    worker.kv_connector_stats = MooncakeStoreConnectorStats()
    worker.joined_loads = store_worker.JoinedLoads()
    worker._join_poisoned = set()
    worker._join_wait_timeout_s = 10.0
    return worker


def _meta(join_wait: list[str], requests: list[ReqMeta] = ()):
    meta = MooncakeStoreConnectorMetadata(set(), set())
    meta.join_wait = list(join_wait)
    for request in requests:
        meta.add_request(request)
    return meta


def test_bind_waits_on_exactly_its_own_joined_loads():
    # (a) The step's bind waits for its new requests' joined loads, each on
    # its own completion: another joined load still in flight does not hold
    # it, and its own holds it until that load is reported.
    worker = _bare_worker()
    worker.joined_loads.register("mine")
    worker.joined_loads.register("other")
    timer = threading.Timer(0.2, worker.joined_loads.complete, ("mine",))
    timer.start()
    start = time.monotonic()

    worker.wait_joined_loads(_meta(["mine"]))

    assert time.monotonic() - start >= 0.15
    assert worker.joined_loads.is_joined("other")
    assert not worker._join_poisoned
    counts = worker.kv_connector_stats.data[metrics._EVENT_COUNTS]
    assert counts["join_loads"] == 1 and counts["join_wait_stalled"] == 1


def test_bind_waits_for_a_joined_load_not_yet_registered():
    # The early-load channel and the step's metadata travel separately: the
    # bind may see the request before the worker queued its load.
    worker = _bare_worker()

    def issue_then_finish():
        worker.joined_loads.register("r")
        worker.joined_loads.complete("r")

    threading.Timer(0.1, issue_then_finish).start()
    worker.wait_joined_loads(_meta(["r"]))
    assert not worker._join_poisoned


def test_unreported_joined_load_raises_instead_of_hanging():
    worker = _bare_worker()
    worker._join_wait_timeout_s = 0.05
    with pytest.raises(RuntimeError, match="not reported"):
        worker.wait_joined_loads(_meta(["r"]))


def _recv_thread(worker):
    thread = object.__new__(store_worker.KVCacheStoreRecvingThread)
    thread.done_task_lock = threading.Lock()
    thread.finished_requests = set()
    thread._clean_finished = set()
    thread._failed_requests = set()
    thread._invalid_block_ids_lock = threading.Lock()
    thread._invalid_block_ids = set()
    thread._is_hma_required = False
    thread._load_error_reqs = set()
    thread.joined = worker.joined_loads
    return thread


def test_failed_joined_load_is_reported_per_request_and_never_stored():
    # (c) Without HMA a failed load reports its blocks; a joined one is
    # already admitted, so it fails as a whole request. The bind does not
    # raise (the step's forward and collectives run) and the request's store
    # job is dropped but reported done (its blocks are released).
    worker = _bare_worker()
    thread = _recv_thread(worker)
    worker.joined_loads.register("r")

    thread._report_load_error("r", [3, 4])
    thread.set_finished_request("r")
    worker.wait_joined_loads(_meta(["r"]))

    assert thread.get_and_clear_failed_requests() == {"r"}
    assert thread.get_and_clear_block_ids_with_load_errors() == set()
    assert worker._join_poisoned == {"r"}

    worker.kv_send_thread = MagicMock()
    save = ReqMeta(
        req_id="r",
        token_len_chunk=64,
        block_ids=([1, 2, 3, 4],),
        block_hashes=[bytes([i]) * 4 for i in range(4)],
        can_save=True,
        store_job_id=7,
    )
    worker.wait_for_save(_meta([], [save]))
    worker.kv_send_thread.finish_store_job.assert_called_once_with(save)
    worker.kv_send_thread.add_request.assert_not_called()

    # Preemption restarts it: its recompute stores again.
    worker.can_put = False
    worker.get_finished(set(), SimpleNamespace(preempted_req_ids={"r"}))
    assert not worker._join_poisoned


def test_unjoined_failure_still_reports_blocks():
    worker = _bare_worker()
    thread = _recv_thread(worker)
    thread._report_load_error("r", [3, 4])
    assert thread.get_and_clear_block_ids_with_load_errors() == {3, 4}
    assert thread.get_and_clear_failed_requests() == set()


# ---------------------------------------------------------------------------
# Mooncake scheduler
# ---------------------------------------------------------------------------


def _join_scheduler():
    scheduler = _make_bare_scheduler()
    scheduler._early_loads = _Channel()
    scheduler._group_block_sizes = [16]
    scheduler._joined = set()
    scheduler._early_tokens = {}
    scheduler._load_model = LoadTimeModel()
    scheduler.join_stats = {"joined": 0, "no_headroom": 0, "cold": 0, "recent": 0}
    request = SimpleNamespace(
        block_hashes=[b"h0", b"h1"], num_prompt_tokens=32, num_tokens=32
    )
    scheduler._unfinished_requests["req-0"] = (request, ([0, 1],))
    scheduler.load_specs["req-0"] = LoadSpec(
        vllm_cached_tokens=0, kvpool_cached_tokens=32, can_load=True
    )
    return scheduler


def _warm(model: LoadTimeModel, seconds: float = 0.05):
    for i in range(16):
        model.add(1000 * (i + 1), seconds)


def test_early_joined_load_is_not_issued_again_with_the_step():
    # (b) Issued early at allocation; the request's first step lists it for
    # the worker to wait on, and its metadata carries no load.
    scheduler = _join_scheduler()
    _warm(scheduler._load_model)
    request = SimpleNamespace(request_id="req-0")
    blocks = _blocks(recent_index=None)

    assert scheduler.join_load_in_step(request, blocks, 32, headroom_s=1.0)
    scheduler._issue_load_early("req-0", blocks)
    (sent,) = scheduler._early_loads.sent
    assert sent.join and sent.load_spec is not None

    meta = scheduler.build_connector_meta(_make_new_scheduler_output())

    assert meta.join_wait == ["req-0"]
    assert all(r.load_spec is None for r in meta.requests)
    worker = _bare_worker()
    worker.start_load_kv(meta)
    assert worker.recv_request_queue.empty()


@pytest.mark.parametrize(
    "warm, headroom_s, recent, outcome",
    [
        (False, 1.0, None, "cold"),
        (True, 0.01, None, "no_headroom"),
        (True, 1.0, 1, "recent"),
        (True, 1.0, None, "joined"),
    ],
)
def test_join_decision(warm, headroom_s, recent, outcome):
    scheduler = _join_scheduler()
    if warm:
        _warm(scheduler._load_model)
    joined = scheduler.join_load_in_step(
        SimpleNamespace(request_id="req-0"),
        _blocks(recent_index=recent),
        32,
        headroom_s=headroom_s,
    )
    assert joined is (outcome == "joined")
    assert scheduler.join_stats[outcome] == 1
    assert ("req-0" in scheduler._joined) is joined


def test_load_time_model_is_conservative():
    model = LoadTimeModel(min_samples=4)
    assert model.predict(100) is None
    samples = [(100, 0.011), (100, 0.013), (200, 0.021), (200, 0.025), (400, 0.041)]
    for tokens, seconds in samples:
        model.add(tokens, seconds)
    for tokens, seconds in samples:
        assert model.predict(tokens) >= seconds - 1e-9
    assert model.predict(400) > model.predict(100)


def test_inflight_step_clock():
    clock = InflightStepClock(min_samples=2, quantile=0.0)
    assert clock.headroom(0.0) == 0.0
    t = 0.0
    for key in range(3):  # back-to-back 0.1 s steps
        clock.dispatched(key, t)
        t += 0.1
        clock.done(key, t)
    assert clock.headroom(t) == 0.0  # nothing in flight
    clock.dispatched(10, t)
    assert clock.headroom(t + 0.03) == pytest.approx(0.07)
    clock.dispatched(11, t + 0.04)  # queued behind it
    assert clock.headroom(t + 0.05) == pytest.approx(0.15)
    clock.done(10, t + 0.1)
    assert clock.headroom(t + 0.1) == pytest.approx(0.1)
