# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mooncake store loads issued when the step's metadata is bound.

MooncakeStoreConnector issues a step's async loads in bind_connector_metadata
(right before the model runner launches the forward) instead of in
start_load_kv (after that launch). These tests pin the properties that change
relies on:

- each load is issued exactly once per step, on every runner path (forward,
  no forward, disabled connector for dummy/DP padding runs, MultiConnector);
- a load writes its blocks only after the CUDA event recorded at bind, i.e.
  after every GPU op queued before it (earlier forwards, this step's block
  zeroing and copy-on-write copies);
- a failure while waiting for that event reports the load failed, once;
- a request finished (aborted) while its load is in flight is still reported
  exactly once, and its blocks are not handed to another load meanwhile.
"""

import queue
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import vllm.v1.worker.gpu.kv_connector as runner_kv_connector
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
    connector as mooncake_store_connector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    MooncakeStoreConnectorMetadata,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.metrics import (
    MooncakeStoreConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
    MultiKVConnectorMetadata,
)
from vllm.v1.request import RequestStatus
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector

from .test_mooncake_store_worker import (
    _make_bare_worker,
    _make_load_req,
    _make_store_recving_thread,
    _make_store_req,
)
from .utils import (
    assert_scheduler_empty,
    create_mock_connector,
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)

pytestmark = pytest.mark.cpu_test


class _FakeEvent:
    """Stands in for torch.cuda.Event; logs record/synchronize in order."""

    def __init__(self, log: list[str], gate: threading.Event | None = None):
        self._log = log
        self._gate = gate
        self.fail: BaseException | None = None

    def record(self):
        self._log.append("record")

    def synchronize(self):
        if self._gate is not None:
            assert self._gate.wait(timeout=10)
        if self.fail is not None:
            raise self.fail
        self._log.append("synchronize")


class _EventLog(list):
    """record/synchronize calls, plus the events torch.cuda.Event created."""

    def __init__(self):
        super().__init__()
        self.created: list[_FakeEvent] = []


@pytest.fixture
def events(monkeypatch) -> _EventLog:
    log = _EventLog()

    def make_event(*_args, **_kwargs):
        log.created.append(_FakeEvent(log))
        return log.created[-1]

    monkeypatch.setattr(torch.cuda, "Event", make_event)
    return log


def _meta(*reqs: ReqMeta) -> MooncakeStoreConnectorMetadata:
    meta = MooncakeStoreConnectorMetadata(set(), set())
    for req in reqs:
        meta.add_request(req)
    return meta


def _load(req_id: str) -> ReqMeta:
    req = _make_load_req(req_id, [b"h0", b"h1"], token_len=32)
    assert req.load_spec is not None
    req.load_spec.token_len = 0
    return req


def _connector(kv_role: str = "kv_consumer"):
    """A worker-role MooncakeStoreConnector around a bare store worker."""
    worker = _make_bare_worker(kv_role=kv_role)
    connector = object.__new__(mooncake_store_connector.MooncakeStoreConnector)
    connector._connector_metadata = None
    connector._loads_issued_for = None
    connector.connector_scheduler = None
    connector.connector_worker = worker
    connector._capacity_only = False
    connector._kv_cache_events = None
    return connector, worker


def _queued(worker) -> list[ReqMeta]:
    items = []
    while True:
        try:
            items.append(worker.recv_request_queue.get_nowait())
        except queue.Empty:
            return items


# ---------------------------------------------------------------------------
# Worker: issue at bind
# ---------------------------------------------------------------------------


def test_issue_orders_every_load_after_one_event_recorded_at_issue(events):
    worker = _make_bare_worker(kv_role="kv_consumer")
    a, b = _load("a"), _load("b")
    save_only = _make_store_req("s", [b"h0", b"h1"])

    worker.start_load_kv(_meta(a, save_only, b))

    assert events == ["record"]
    (ready,) = events.created
    assert _queued(worker) == [a, b]
    for req in (a, b):
        assert req.load_ready_event is ready
        assert req.load_issue_time is not None
        assert req.load_spec is not None and req.load_spec.token_len == 32
    assert save_only.load_ready_event is None


@pytest.mark.parametrize("capacity_only", [False, True])
def test_issue_without_loads_records_nothing(events, capacity_only):
    worker = _make_bare_worker(kv_role="kv_consumer")
    worker._capacity_only = capacity_only
    loads = () if not capacity_only else (_load("a"),)

    worker.start_load_kv(_meta(*loads, _make_store_req("s", [b"h0", b"h1"])))

    assert events == []
    assert _queued(worker) == []


# ---------------------------------------------------------------------------
# Recv thread: writes only after the bind event
# ---------------------------------------------------------------------------


def _issued_load(thread, req_id: str, log: list[str], gate=None) -> ReqMeta:
    req = _make_load_req(req_id, [b"h0", b"h1"], token_len=32)
    req.load_ready_event = _FakeEvent(log, gate)
    req.load_issue_time = time.perf_counter()
    return req


def test_recv_thread_writes_blocks_only_after_the_bind_event():
    log: list[str] = []
    gate = threading.Event()
    store = MagicMock()
    store.batch_get_into_multi_buffers.side_effect = lambda keys, *_a, **_k: (
        log.append("get") or [1] * len(keys)
    )
    thread = _make_store_recving_thread(store)
    recorded: list[str] = []
    thread._record_operation_cb = lambda operation, **_kw: recorded.append(operation)
    req = _issued_load(thread, "a", log, gate)

    runner = threading.Thread(target=thread._handle_request, args=(req,))
    runner.start()
    time.sleep(0.05)
    # The previous owner's / this step's GPU writes have not completed yet.
    assert log == []
    store.batch_get_into_multi_buffers.assert_not_called()

    gate.set()
    runner.join(timeout=10)
    assert not runner.is_alive()
    assert log == ["synchronize", "get"]
    assert recorded == ["load_wait_ready", "load_get"]
    assert thread.take_clean_finished_requests() == {"a"}
    assert thread.get_and_clear_finished_requests() == set()


@pytest.mark.parametrize("is_hma_required", [False, True])
def test_bind_event_failure_reports_the_load_failed_once(is_hma_required):
    log: list[str] = []
    store = MagicMock()
    thread = _make_store_recving_thread(store, is_hma_required=is_hma_required)
    req = _issued_load(thread, "a", log)
    req.load_ready_event.fail = RuntimeError("cuda error")  # type: ignore[union-attr]

    thread._handle_request(req)

    store.batch_get_into_multi_buffers.assert_not_called()
    # Failed loads are never reported early; they go in-band with the error.
    assert thread.take_clean_finished_requests() == set()
    assert thread.get_and_clear_finished_requests() == {"a"}
    assert thread.get_and_clear_finished_requests() == set()
    if is_hma_required:
        assert thread.get_and_clear_failed_requests() == {"a"}
    else:
        assert thread.get_and_clear_block_ids_with_load_errors() == {0, 1}


def test_load_without_bind_event_is_not_delayed():
    """Loads built outside start_load_kv (e.g. older tests) carry no event."""
    store = MagicMock()
    store.batch_get_into_multi_buffers.side_effect = lambda keys, *_a, **_k: (
        [1] * len(keys)
    )
    thread = _make_store_recving_thread(store)
    thread._handle_request(_make_load_req("a", [b"h0", b"h1"], token_len=32))
    assert thread.take_clean_finished_requests() == {"a"}


# ---------------------------------------------------------------------------
# Connector: bind issues, start_load_kv does not re-issue
# ---------------------------------------------------------------------------


def test_bind_issues_once_per_step(events):
    connector, worker = _connector()
    meta = _meta(_load("a"))

    connector.bind_connector_metadata(meta)
    assert len(_queued(worker)) == 1
    # Re-binding the same step's metadata and the post-launch hook add nothing.
    connector.bind_connector_metadata(meta)
    connector.start_load_kv(MagicMock())
    assert _queued(worker) == []
    connector.clear_connector_metadata()

    connector.bind_connector_metadata(_meta(_load("b")))
    assert [r.req_id for r in _queued(worker)] == ["b"]
    assert events.count("record") == 2


def test_start_load_kv_refuses_metadata_not_issued_at_bind(events):
    connector, worker = _connector()
    # Bind that bypasses the connector's own hook: loads were never issued.
    KVConnectorBase_V1.bind_connector_metadata(connector, _meta(_load("a")))
    with pytest.raises(AssertionError, match="issued when the metadata is bound"):
        connector.start_load_kv(MagicMock())
    assert _queued(worker) == []


# ---------------------------------------------------------------------------
# Model runner (stock ActiveKVConnector, unchanged) drives the connector
# ---------------------------------------------------------------------------


def _runner(monkeypatch, backend) -> ActiveKVConnector:
    monkeypatch.setattr(
        runner_kv_connector, "is_forward_context_available", lambda: True
    )
    monkeypatch.setattr(runner_kv_connector, "get_forward_context", object)
    runner = object.__new__(ActiveKVConnector)
    runner.vllm_config = SimpleNamespace()
    runner.kv_connector = backend
    runner._pending_load_kwargs = None
    runner._disabled = False
    return runner


def _sched_out(metadata, has_sync_kv_loads=False):
    return SimpleNamespace(
        kv_connector_metadata=metadata,
        finished_req_ids=set(),
        has_sync_kv_loads=has_sync_kv_loads,
    )


@pytest.mark.parametrize("has_sync_kv_loads", [False, True])
def test_runner_forward_step_issues_before_the_forward_launch(
    monkeypatch, events, has_sync_kv_loads
):
    connector, worker = _connector()
    runner = _runner(monkeypatch, connector)
    load = _load("a")

    runner.pre_forward(_sched_out(_meta(load), has_sync_kv_loads))
    # Queued before the forward is launched (pre_forward returns).
    assert _queued(worker) == [load]
    runner.finish_forward()
    output = runner.post_forward(set())
    assert _queued(worker) == []
    assert output is not None and output.finished_recving is None
    assert events == ["record"]


def test_runner_step_without_forward_issues_once(monkeypatch, events):
    """A DP rank with no tokens this step (no_forward) still issues its loads."""
    connector, worker = _connector()
    runner = _runner(monkeypatch, connector)

    runner.no_forward(_sched_out(_meta(_load("a"))))

    assert [r.req_id for r in _queued(worker)] == ["a"]
    assert events == ["record"]


def test_runner_disabled_for_dummy_runs_issues_nothing(monkeypatch, events):
    """Dummy runs (DP padding, graph capture) disable the connector."""
    connector, worker = _connector()
    runner = _runner(monkeypatch, connector)
    runner._disabled = True

    runner.pre_forward(_sched_out(_meta(_load("a"))))
    runner.post_forward(set())

    assert _queued(worker) == []
    assert events == []


def test_multi_connector_binds_and_issues_mooncake_loads(monkeypatch, events):
    """Production shape: MultiConnector(NIXL producer, Mooncake store)."""
    connector, worker = _connector(kv_role="kv_both")
    nixl = MagicMock()
    multi = object.__new__(MultiConnector)
    multi._connectors = [nixl, connector]
    multi._extra_async_saves = {}
    multi._connector_metadata = None
    nixl_meta = object()

    multi.handle_preemptions(MultiKVConnectorMetadata((nixl_meta, _meta())))
    multi.bind_connector_metadata(
        MultiKVConnectorMetadata((nixl_meta, _meta(_load("a"))))
    )
    assert [r.req_id for r in _queued(worker)] == ["a"]
    nixl.bind_connector_metadata.assert_called_once_with(nixl_meta)

    multi.start_load_kv(MagicMock())
    assert _queued(worker) == []
    nixl.start_load_kv.assert_called_once()
    assert events == ["record"]


# ---------------------------------------------------------------------------
# Finish / abort while a load issued at bind is in flight
# ---------------------------------------------------------------------------


def test_request_finished_during_load_is_still_reported_once(events):
    """The engine aborts the request after the load was issued: a later step's
    finished_req_ids carries it while the load waits / runs. The load still
    reports exactly once, so the scheduler can release the held blocks."""
    connector, worker = _connector()
    gate = threading.Event()
    store = MagicMock()
    store.batch_get_into_multi_buffers.side_effect = lambda keys, *_a, **_k: (
        [1] * len(keys)
    )
    thread = _make_store_recving_thread(store)
    worker.kv_recv_threads = [thread]
    worker.kv_connector_stats = MooncakeStoreConnectorStats()

    connector.bind_connector_metadata(_meta(_load("a")))
    (req,) = _queued(worker)
    req.load_ready_event = _FakeEvent([], gate)
    runner = threading.Thread(target=thread._handle_request, args=(req,))
    runner.start()
    connector.clear_connector_metadata()

    # Next step: the request is finished while its load is still in flight.
    next_meta = _meta()
    connector.bind_connector_metadata(next_meta)
    assert connector.poll_finished_recving() == set()
    assert connector.get_transfer_results({"a"}).finished_recving == set()
    connector.clear_connector_metadata()

    gate.set()
    runner.join(timeout=10)
    connector.bind_connector_metadata(_meta())
    reported = connector.poll_finished_recving() | (
        connector.get_transfer_results(set()).finished_recving
    )
    assert reported == {"a"}
    assert connector.poll_finished_recving() == set()
    assert connector.get_transfer_results(set()).finished_recving == set()


def _make_load_scheduler(num_blocks: int, num_external_tokens: int, load_ids):
    scheduler = create_scheduler(create_vllm_config(), num_blocks=num_blocks)
    scheduler.connector = create_mock_connector()
    scheduler.connector.get_loaded_kv_cache_group_ids.return_value = (0,)
    scheduler.connector.get_num_new_matched_tokens.side_effect = lambda request, _: (
        (num_external_tokens, True) if request.request_id in load_ids else (0, False)
    )
    scheduler.connector.request_finished.return_value = (False, None)
    scheduler.connector.take_events.return_value = ()
    scheduler.connector.has_pending_block_frees.return_value = False
    return scheduler


def _block_ids(scheduler, req_id: str) -> set[int]:
    return {
        b.block_id
        for group in scheduler.kv_cache_manager.get_blocks(req_id).blocks
        for b in group
    }


def test_aborted_loading_request_keeps_blocks_until_load_reports():
    """A load issued at bind writes the blocks of the step that allocated them.
    If that request is aborted mid-load, a later load (also issued at its own
    bind) must not be given those blocks before the first load reports."""
    block_size = 16
    num_tokens = 4 * block_size
    first = create_request(num_tokens=num_tokens)
    second = create_request(num_tokens=num_tokens)
    # 1 null block + room for exactly one request's blocks.
    scheduler = _make_load_scheduler(
        1 + 4, num_tokens - block_size, {first.request_id, second.request_id}
    )

    scheduler.add_request(first)
    out1 = scheduler.schedule()
    assert first.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert _block_ids(scheduler, first.request_id)

    scheduler.finish_requests(first.request_id, RequestStatus.FINISHED_ABORTED)
    scheduler.add_request(second)
    out2 = scheduler.schedule()
    # No room: the aborted request's blocks stay held for its in-flight load.
    assert second.status == RequestStatus.WAITING
    scheduler.update_from_output(out1, create_model_runner_output([]))
    out3 = scheduler.schedule()
    assert second.status == RequestStatus.WAITING

    scheduler.update_from_output(
        out2, create_model_runner_output([], finished_recving={first.request_id})
    )
    assert first.request_id not in scheduler.requests
    out4 = scheduler.schedule()
    assert second.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert _block_ids(scheduler, second.request_id)

    scheduler.update_from_output(
        out3, create_model_runner_output([], finished_recving={second.request_id})
    )
    scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)
    out5 = scheduler.schedule()
    for out in (out4, out5):
        scheduler.update_from_output(out, create_model_runner_output([]))
    assert_scheduler_empty(scheduler)


def test_loading_request_is_never_a_preemption_victim():
    """Preemption frees running requests only: a request whose load was issued
    at bind keeps its blocks while a running request is preempted."""
    block_size = 16
    running = create_request(num_tokens=2 * block_size, max_tokens=64)
    loading = create_request(num_tokens=4 * block_size)
    scheduler = _make_load_scheduler(1 + 7, 3 * block_size, {loading.request_id})

    scheduler.add_request(running)
    out = scheduler.schedule()
    scheduler.update_from_output(out, create_model_runner_output([running]))
    scheduler.add_request(loading)
    out = scheduler.schedule()
    assert loading.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    loading_blocks = _block_ids(scheduler, loading.request_id)

    # Decode the running request until it needs a block the pool lacks.
    for _ in range(4 * block_size):
        scheduler.update_from_output(out, create_model_runner_output([running]))
        out = scheduler.schedule()
        assert loading.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        assert _block_ids(scheduler, loading.request_id) == loading_blocks
        if running.status == RequestStatus.PREEMPTED:
            break
    assert running.status == RequestStatus.PREEMPTED, (
        running.num_computed_tokens,
        running.num_tokens,
        scheduler.kv_cache_manager.block_pool.get_num_free_blocks(),
        _block_ids(scheduler, running.request_id),
        loading_blocks,
        type(scheduler).__name__,
    )

    scheduler.update_from_output(
        out, create_model_runner_output([], finished_recving={loading.request_id})
    )
    scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)
    scheduler.update_from_output(scheduler.schedule(), create_model_runner_output([]))
    assert_scheduler_empty(scheduler)
