# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mooncake store failover: a store get / put that fails or never returns must
become a cache miss (recompute) / a dropped put within a bounded time."""

import threading
import time
from unittest.mock import MagicMock

import pytest

from tests.v1.kv_connector.unit.test_mooncake_store_worker import (
    _make_load_req,
    _make_store_recving_thread,
    _make_store_req,
    _make_store_sending_thread,
    _run_store_req,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
    worker as mooncake_store_worker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.bounded import (
    StoreCallGuard,
    StoreCallTimeout,
)

TIMEOUT_S = 0.2


class _Stuck:
    """A store call that blocks until released (a transfer on a paused rail)."""

    def __init__(self, result=None, error: BaseException | None = None):
        self.release = threading.Event()
        self.entered = threading.Event()
        self.result = result
        self.error = error
        self.calls = 0

    def __call__(self, *args):
        self.calls += 1
        self.entered.set()
        assert self.release.wait(10), "test never released the stuck call"
        if self.error is not None:
            raise self.error
        return self.result(*args) if callable(self.result) else self.result


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition never became true"
        time.sleep(0.01)


# ---------------------------------------------------------------- guard ------


def test_guard_returns_results_and_propagates_errors():
    guard = StoreCallGuard("t", TIMEOUT_S)
    assert guard.call("load_get", lambda a, b: a + b, 1, 2) == 3
    with pytest.raises(KeyError):
        guard.call("load_get", MagicMock(side_effect=KeyError("x")))
    assert guard.counters.timeouts == 0 and not guard.degraded
    guard.shutdown()


def test_guard_rejects_non_positive_timeout():
    with pytest.raises(ValueError):
        StoreCallGuard("t", 0)


def test_guard_times_out_degrades_and_recovers():
    late = []
    guard = StoreCallGuard("t", TIMEOUT_S, on_late=lambda *a: late.append(a))
    stuck = _Stuck(result=[1])
    start = time.monotonic()
    with pytest.raises(StoreCallTimeout):
        guard.call("load_get", stuck, keys=["k0", "k1"])
    assert time.monotonic() - start < TIMEOUT_S + 1.0
    assert guard.counters.timeouts == 1 and guard.degraded

    # While the stuck call is out, further calls fail fast without queueing.
    other = MagicMock(return_value=[1])
    start = time.monotonic()
    with pytest.raises(StoreCallTimeout, match="degraded"):
        guard.call("load_get", other)
    assert time.monotonic() - start < TIMEOUT_S
    other.assert_not_called()
    assert guard.counters.degraded_skips == 1

    stuck.release.set()
    _wait_for(lambda: late)
    assert late == [("load_get", ["k0", "k1"], None)]
    assert guard.counters.late_completions == 1 and guard.counters.late_gets == 1
    _wait_for(lambda: not guard.degraded)
    assert guard.call("load_get", other) == [1]
    guard.shutdown()


def test_guard_late_error_reaches_callback():
    late = []
    guard = StoreCallGuard("t", TIMEOUT_S, on_late=lambda *a: late.append(a))
    stuck = _Stuck(error=RuntimeError("rail paused"))
    with pytest.raises(StoreCallTimeout):
        guard.call("save_put", stuck, keys=["k"])
    stuck.release.set()
    _wait_for(lambda: late)
    op, keys, error = late[0]
    assert (op, keys) == ("save_put", ["k"]) and isinstance(error, RuntimeError)
    assert guard.counters.late_gets == 0
    guard.shutdown()


# ------------------------------------------------------------- load path -----


def test_load_without_guard_calls_store_directly():
    store = MagicMock()
    store.batch_get_into_multi_buffers.return_value = [256, 256]
    thread = _make_store_recving_thread(store)
    assert thread.store_guard is None
    thread._handle_request(_make_load_req("r", [b"h0", b"h1"], token_len=32))
    assert thread.get_and_clear_finished_requests() == {"r"}
    assert thread.get_and_clear_block_ids_with_load_errors() == set()


def test_stuck_load_becomes_a_cache_miss_within_the_timeout():
    store = MagicMock()
    stuck = _Stuck(result=lambda keys, *_: [256] * len(keys))
    store.batch_get_into_multi_buffers.side_effect = stuck
    thread = _make_store_recving_thread(store)
    thread.store_guard = StoreCallGuard("recv", TIMEOUT_S)

    start = time.monotonic()
    thread._handle_request(_make_load_req("r1", [b"h0", b"h1"], token_len=32))
    assert time.monotonic() - start < TIMEOUT_S + 1.0
    # Finished (the request stops waiting) with every block a load error, so
    # the scheduler recomputes those tokens.
    assert thread.get_and_clear_finished_requests() == {"r1"}
    assert thread.get_and_clear_block_ids_with_load_errors() == {0, 1}

    # Degraded: the next load is a miss at once, without touching the store.
    thread._handle_request(_make_load_req("r2", [b"h2", b"h3"], token_len=32))
    assert thread.get_and_clear_finished_requests() == {"r2"}
    assert thread.get_and_clear_block_ids_with_load_errors() == {0, 1}
    assert stuck.calls == 1

    stuck.release.set()
    _wait_for(lambda: not thread.store_guard.degraded)
    assert thread.store_guard.counters.late_gets == 1
    store.batch_get_into_multi_buffers.side_effect = None
    store.batch_get_into_multi_buffers.return_value = [256, 256]
    thread._handle_request(_make_load_req("r3", [b"h4", b"h5"], token_len=32))
    assert thread.get_and_clear_finished_requests() == {"r3"}
    assert thread.get_and_clear_block_ids_with_load_errors() == set()
    thread.store_guard.shutdown()


def test_failed_load_is_a_cache_miss():
    store = MagicMock()
    store.batch_get_into_multi_buffers.side_effect = RuntimeError("rdma")
    thread = _make_store_recving_thread(store)
    thread.store_guard = StoreCallGuard("recv", TIMEOUT_S)
    thread._handle_request(_make_load_req("r", [b"h0", b"h1"], token_len=32))
    assert thread.get_and_clear_finished_requests() == {"r"}
    assert thread.get_and_clear_block_ids_with_load_errors() == {0, 1}
    thread.store_guard.shutdown()


# ------------------------------------------------------------- save path -----


def test_stuck_put_is_dropped_and_cleaned_up_when_it_returns():
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    stuck = _Stuck(result=lambda keys, *_: [256] * len(keys))
    store.batch_put_from_multi_buffers.side_effect = stuck
    thread = _make_store_sending_thread(store)
    worker = object.__new__(mooncake_store_worker.MooncakeStoreWorker)
    worker._store_op_failover = True
    worker._store_op_timeout_s = TIMEOUT_S
    thread.store_guard = worker._make_store_guard("send", thread)

    req = _make_store_req("req-a", [b"a0", b"a1"])
    start = time.monotonic()
    _run_store_req(thread, req)
    assert time.monotonic() - start < TIMEOUT_S + 1.0
    # The job finished (no hang) and the save offset did not advance, so the
    # range is retried by a later save rather than counted as stored.
    thread.request_queue.task_done.assert_called()
    assert thread.take_completed_saves() == {req.store_job_id: 1}
    assert "req-a" not in thread._saved_offset
    put_keys = list(store.batch_put_from_multi_buffers.call_args.args[0])

    stuck.release.set()
    _wait_for(lambda: store.batch_remove.called)
    assert store.batch_remove.call_args.args[0] == put_keys
    thread.store_guard.shutdown()


def test_failed_put_is_non_fatal():
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.side_effect = RuntimeError("rdma")
    thread = _make_store_sending_thread(store)
    thread.store_guard = StoreCallGuard("send", TIMEOUT_S)
    _run_store_req(thread, _make_store_req("req-a", [b"a0", b"a1"]))
    thread.request_queue.task_done.assert_called()
    assert "req-a" not in thread._saved_offset
    thread.store_guard.shutdown()


# ---------------------------------------------------------------- config -----


def test_failover_off_by_default_and_counters_sum():
    worker = object.__new__(mooncake_store_worker.MooncakeStoreWorker)
    assert worker._make_store_guard("x", MagicMock()) is None
    worker._store_op_failover = True
    worker._store_op_timeout_s = TIMEOUT_S
    send, recv = MagicMock(), MagicMock()
    send.store_guard = worker._make_store_guard("send", send)
    recv.store_guard = worker._make_store_guard("recv", recv)
    worker.kv_send_thread, worker.kv_recv_threads = send, [recv]
    send.store_guard.counters.timeouts = 2
    recv.store_guard.counters.timeouts = 1
    recv.store_guard.counters.late_gets = 1
    totals = worker.store_failover_counters()
    assert totals["timeouts"] == 3 and totals["late_gets"] == 1
    send.store_guard.shutdown()
    recv.store_guard.shutdown()


def test_load_that_raises_before_the_store_call_still_reports():
    """A load that fails while building its key list must still finish (with
    a load error) instead of leaving its request waiting forever."""
    store = MagicMock()
    thread = _make_store_recving_thread(store)
    thread.coord = MagicMock()
    thread.coord.load_mask.side_effect = IndexError("bad geometry")
    thread._handle_request(_make_load_req("r", [b"h0", b"h1"], token_len=32))
    assert thread.get_and_clear_finished_requests() == {"r"}
    assert thread.get_and_clear_block_ids_with_load_errors() == {0, 1}
    store.batch_get_into_multi_buffers.assert_not_called()
