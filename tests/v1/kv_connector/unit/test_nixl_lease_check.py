# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV lease check for pull-mode (P->D) NIXL reads.

D (the puller) reads P's blocks with one-sided RDMA, so P cannot see a read in
flight: it frees the blocks when the lease expires and they may be reused by a
new prefill while D still reads them. D therefore must not use a read that may
have finished after P released the blocks:

* fast path: the read finished before P's exported lease deadline (minus a
  margin), so P still held the blocks -> no extra round trip;
* slow path: D asks P's scheduler (side channel) whether the blocks are still
  held, before sending the completion notification that would release them;
  held -> use the KV, released / no answer -> fail the load (recompute or
  abort per kv_load_failure_policy), never decode from it.
"""

import socket
import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.stats import (
    NixlKVConnectorStats,
)

from .test_nixl_connector import recv_worker  # noqa: F401  (fixture)
from .utils import (
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)

_REMOTE_ENGINE = "prefill-engine"


def _lease_worker(worker, *, deadline, query_result=None):
    """Arm ``recv_worker`` with one finished read of a lease-checked request."""
    worker._lease_check_enabled = True
    worker._lease_check_timeout_s = 30.0
    worker._lease_checks = {}
    worker._engine_clock_offset = {_REMOTE_ENGINE: 0.0}
    meta = worker._recving_metadata["request"]
    meta.remote = SimpleNamespace(
        engine_id=_REMOTE_ENGINE,
        request_id="prefill-request",
        host="127.0.0.1",
        port=1,
        lease_check=True,
        lease_deadline=deadline,
        blocks_expiry_time=None,
        num_tokens=None,
    )
    # One READ, already complete; its completion notification is deferred.
    worker._recving_transfers["request"] = [101]
    worker._pending_recv_notifs = {"request": [("prefill", b"prefill-request:1")]}
    worker.nixl_wrapper.check_xfer_state.return_value = "DONE"
    fut: Future = Future()
    if query_result is not None:
        fut.set_result(query_result)
    worker._submit_lease_check = MagicMock(return_value=fut)
    return fut


def test_read_within_lease_is_used_without_round_trip(recv_worker):  # noqa: F811
    worker = recv_worker
    _lease_worker(worker, deadline=time.perf_counter() + 25.0)

    results = worker.get_transfer_results()

    assert results.finished_recving == {"request"}
    assert not results.failed_recving
    worker._submit_lease_check.assert_not_called()
    worker.nixl_wrapper.send_notif.assert_called_once_with(
        "prefill", notif_msg=b"prefill-request:1"
    )


def test_read_after_lease_expiry_is_never_used(recv_worker):  # noqa: F811
    """The m1-lite-7 interleaving: P's lease ran out (P logged "Releasing
    expired KV blocks"), then D's READ completed. D must fail the load."""
    worker = recv_worker
    fut = _lease_worker(worker, deadline=time.perf_counter() - 1.0)

    # The read is complete but the lease state is unknown: not reported yet,
    # and the completion notification is held back (it would release P's
    # blocks and make a later "held" answer meaningless).
    results = worker.get_transfer_results()
    assert "request" not in results.finished_recving
    worker._submit_lease_check.assert_called_once()
    worker.nixl_wrapper.send_notif.assert_not_called()

    # P answers: blocks already released.
    fut.set_result(False)
    results = worker.get_transfer_results()
    assert results.failed_recving == {"request"}
    assert "request" in results.finished_recving
    worker.nixl_wrapper.send_notif.assert_not_called()
    assert isinstance(worker.xfer_stats, NixlKVConnectorStats)
    assert len(worker.xfer_stats.data["num_kv_lease_lost"]) == 1


def test_slow_read_with_held_lease_is_used(recv_worker):  # noqa: F811
    worker = recv_worker
    fut = _lease_worker(worker, deadline=time.perf_counter() - 1.0)

    assert "request" not in worker.get_transfer_results().finished_recving
    fut.set_result(True)
    results = worker.get_transfer_results()

    assert results.finished_recving == {"request"}
    assert not results.failed_recving
    worker.nixl_wrapper.send_notif.assert_called_once_with(
        "prefill", notif_msg=b"prefill-request:1"
    )


def test_unanswered_lease_check_fails_the_load(recv_worker):  # noqa: F811
    worker = recv_worker
    _lease_worker(worker, deadline=None)
    worker._lease_check_timeout_s = 0.2

    assert "request" not in worker.get_transfer_results().finished_recving
    worker.nixl_wrapper.send_notif.assert_not_called()
    time.sleep(0.25)
    results = worker.get_transfer_results()

    assert results.failed_recving == {"request"}
    # P may still hold the blocks: release them now rather than at lease expiry.
    worker.nixl_wrapper.send_notif.assert_called_once_with(
        "prefill", notif_msg=b"prefill-request:1"
    )
    assert len(worker.xfer_stats.data["num_kv_lease_lost"]) == 1


def test_lease_check_query_error_fails_the_load(recv_worker):  # noqa: F811
    worker = recv_worker
    fut = _lease_worker(worker, deadline=time.perf_counter() - 1.0)
    fut.set_exception(TimeoutError("side channel"))

    results = worker.get_transfer_results()
    assert results.failed_recving == {"request"}


def test_peer_without_lease_check_keeps_legacy_behaviour(recv_worker):  # noqa: F811
    worker = recv_worker
    _lease_worker(worker, deadline=None)
    worker._recving_metadata["request"].remote.lease_check = False

    results = worker.get_transfer_results()
    assert results.finished_recving == {"request"}
    worker._submit_lease_check.assert_not_called()


# ---------------------------------------------------------------------------
# P side: exported lease, hold tracking, side-channel query
# ---------------------------------------------------------------------------


def _finish_prefill(scheduler, request):
    scheduler.add_request(request)
    out = scheduler.schedule()
    return scheduler.update_from_output(out, create_model_runner_output([request]))


def test_prefill_exports_lease_and_answers_until_blocks_freed():
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)
    nixl_sched = scheduler.connector.connector_scheduler
    block_size = vllm_config.cache_config.block_size
    request = create_request(
        request_id=1,
        num_tokens=block_size * 3,
        block_size=block_size,
        do_remote_decode=True,
    )
    before = time.perf_counter()
    engine_core_outputs = _finish_prefill(scheduler, request)
    params = engine_core_outputs[0].outputs[0].kv_transfer_params

    assert params["remote_lease_check"] is True
    ttl = nixl_sched._kv_lease_duration
    assert before + ttl <= params["remote_lease_deadline"] <= time.perf_counter() + ttl
    rid = request.request_id
    assert nixl_sched._query_send_leases([rid, "unknown"]) == [True, False]

    # P's worker reports the send finished (D notified or lease expired):
    # the hold must be dropped before the blocks are freed.
    freed: list[str] = []
    orig_free = scheduler._free_blocks

    def _free_blocks(req):
        freed.append(nixl_sched._query_send_leases([req.request_id])[0])
        orig_free(req)

    scheduler._free_blocks = _free_blocks
    out = scheduler.schedule()
    scheduler.update_from_output(
        out, create_model_runner_output([], finished_sending={rid})
    )
    assert freed == [False]
    assert nixl_sched._query_send_leases([rid]) == [False]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_side_channel_lease_query_round_trip():
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import (
        NixlBaseConnectorScheduler,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
        query_remote_leases,
    )

    port = _free_port()
    ready, stop = threading.Event(), threading.Event()
    held = {"a"}
    t = threading.Thread(
        target=NixlBaseConnectorScheduler._nixl_handshake_listener,
        args=({}, ready, stop, "127.0.0.1", port),
        kwargs={"lease_query": lambda ids: [i in held for i in ids]},
        daemon=True,
    )
    t.start()
    assert ready.wait(5)
    try:
        assert query_remote_leases("127.0.0.1", port, ["a", "b"], 5.0) == [
            True,
            False,
        ]
    finally:
        stop.set()
        t.join(5)


def test_side_channel_lease_query_times_out_on_silent_peer():
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
        query_remote_leases,
    )

    port = _free_port()
    with socket.socket() as srv:  # accepts TCP, never answers ZMQ
        srv.bind(("127.0.0.1", port))
        srv.listen()
        with pytest.raises(TimeoutError):
            query_remote_leases("127.0.0.1", port, ["a"], 0.3)


def test_heartbeat_extends_known_and_ignores_unknown_requests():
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
        NixlConnectorWorker,
    )

    w = object.__new__(NixlConnectorWorker)
    w._lease_extension = 20
    old = time.perf_counter() + 1.0
    w._reqs_to_send = {"req-a": old}
    w._handle_heartbeat("req-a,req-gone")

    assert w._reqs_to_send["req-a"] > old + 15
    assert set(w._reqs_to_send) == {"req-a"}


# ---------------------------------------------------------------------------
# Lease heartbeats over the side channel (off D's step loop and the UCX path)
# ---------------------------------------------------------------------------


def _start_listener(port, **callbacks):
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import (
        NixlBaseConnectorScheduler,
    )

    ready, stop = threading.Event(), threading.Event()
    t = threading.Thread(
        target=NixlBaseConnectorScheduler._nixl_handshake_listener,
        args=({}, ready, stop, "127.0.0.1", port),
        kwargs=callbacks,
        daemon=True,
    )
    t.start()
    assert ready.wait(5)
    return stop, t


def test_d_sends_side_channel_heartbeats_only_to_lease_check_peers():
    from .utils import make_nixl_scheduler

    port = _free_port()
    got: list[list[str]] = []
    stop, t = _start_listener(
        port,
        lease_renew=lambda ids: (got.append(ids), [True] * len(ids))[1],  # type: ignore[func-returns-value]
    )
    try:
        s = make_nixl_scheduler(heartbeat=True)
        s._side_hb_enabled = True
        s._side_hb_req_ids = set()
        s._heartbeat_lock = threading.Lock()
        s._side_hb_thread = MagicMock()  # rounds driven by hand below
        new = create_request(request_id=1, do_remote_prefill=True)
        old = create_request(request_id=2, do_remote_prefill=True)
        for r in (new, old):
            r.kv_transfer_params.update(remote_host="127.0.0.1", remote_port=port)
        new.kv_transfer_params["remote_lease_check"] = True
        s.on_new_request(new)
        s.on_new_request(old)  # P without the side-channel listener: NIXL HB only

        s._send_side_heartbeats()
        assert got == [["prefill-1"]]

        s._stop_heartbeat(new.request_id)
        s._send_side_heartbeats()
        assert got == [["prefill-1"]]
    finally:
        stop.set()
        t.join(5)


def test_p_side_channel_renewal_reaches_worker_metadata():
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import (
        send_lease_heartbeat,
    )

    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)
    nixl_sched = scheduler.connector.connector_scheduler
    block_size = vllm_config.cache_config.block_size
    request = create_request(
        request_id=1,
        num_tokens=block_size * 3,
        block_size=block_size,
        do_remote_decode=True,
    )
    _finish_prefill(scheduler, request)
    rid = request.request_id

    port = _free_port()
    stop, t = _start_listener(port, lease_renew=nixl_sched._queue_lease_renewals)
    try:
        assert send_lease_heartbeat("127.0.0.1", port, [rid, "gone"], 5.0) == [
            True,
            False,
        ]
    finally:
        stop.set()
        t.join(5)

    out = scheduler.schedule()
    assert out.kv_connector_metadata.lease_renewals == {rid}
    # Drained: renewed once per heartbeat.
    assert scheduler.schedule().kv_connector_metadata.lease_renewals == set()


def test_p_worker_side_channel_renewal_prevents_expiry(
    default_vllm_config, dist_init, monkeypatch
):
    """NIXL heartbeats lost (m1-lite-7: leases expired 29-35 s after routing)
    but the side-channel renewal arrives: the lease must not be reaped."""
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
        NixlConnector,
        NixlConnectorMetadata,
    )
    from vllm.forward_context import ForwardContext

    from .test_nixl_connector import FakeNixlConnectorWorker, FakeNixlWrapper
    from .utils import make_kv_cache_config

    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker.NixlWrapper",
        FakeNixlWrapper,
    )

    vllm_config = create_vllm_config(block_size=16)
    connector = NixlConnector(
        vllm_config, KVConnectorRole.WORKER, make_kv_cache_config(block_size=16)
    )
    connector.connector_worker = FakeNixlConnectorWorker(
        vllm_config, connector.engine_id, hand_shake_latency=0
    )
    worker = connector.connector_worker
    rid = "req-renewed"
    worker._reqs_to_process.add(rid)
    worker._reqs_to_send[rid] = time.perf_counter() - 0.1  # due, not yet reaped

    metadata = NixlConnectorMetadata()
    metadata.lease_renewals = {rid}
    connector.bind_connector_metadata(metadata)
    connector.start_load_kv(
        ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={})
    )

    done_sending, _ = worker.get_finished()
    assert rid not in done_sending
    assert worker._reqs_to_send[rid] > time.perf_counter() + 10
