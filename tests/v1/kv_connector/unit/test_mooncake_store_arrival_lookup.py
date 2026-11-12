# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mooncake store lookups started when a request arrives at the engine core.

The arrival lookup is the lookup the request's first schedule would start (same
inputs), started earlier on the engine core's input thread. These tests pin
that it is then used exactly like that lookup: read by the scheduler's query,
never served for other inputs, dropped at admission, finish, abort and store
reset, never adopted for a request the scheduler did not take in, and safe
against the scheduler thread running concurrently.
"""

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
    scheduler as store_scheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import worker
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    MooncakeLookupResult,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
)
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.request import RequestStatus

from .utils import (
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)

_SOCKET = "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker." + (
    "make_zmq_socket"
)


class FakeStore:
    """Stands in for the LookupKey RPC (``LookupKeyClient._lookup``).

    Each call blocks until ``gate`` is set, then answers ``hits`` for the
    request's last block hash (0 when unknown), so a test controls when a
    lookup finishes and what the store held at that moment.
    """

    def __init__(self, gate_open: bool = True):
        self.gate = threading.Event()
        if gate_open:
            self.gate.set()
        self.calls: list[tuple[int, int, int]] = []
        self.hits: dict[bytes, int] = {}
        self.fail = False
        self._lock = threading.Lock()

    def __call__(self, num_tokens, block_hashes, known_prefix_tokens=0):
        with self._lock:
            self.calls.append((num_tokens, len(block_hashes), known_prefix_tokens))
        assert self.gate.wait(10)
        if self.fail:
            raise RuntimeError("store down")
        tip = bytes(block_hashes[-1]) if block_hashes else b""
        return MooncakeLookupResult(self.hits.get(tip, 0))


def _make_client(store: FakeStore, lookup_timeout_s: float = 10.0):
    vllm_config = create_vllm_config(
        kv_connector="MooncakeStoreConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"lookup_timeout_s": lookup_timeout_s},
    )
    with patch(_SOCKET):
        client = worker.LookupKeyClient(vllm_config)
    client._lookup = store
    return client


def _wait_done(client, req_id, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        future = client.futures.get(req_id)
        if future is not None and future.done():
            return
        time.sleep(0.002)
    raise AssertionError(f"lookup of {req_id} did not finish")


def _assert_empty(client):
    assert client.futures == {}
    assert client._inputs == {}
    assert client._arrival_owner == {}
    assert client._arrival_unread == set()


H = [b"h0" * 8, b"h1" * 8, b"h2" * 8]
H2 = [b"x0" * 8, b"x1" * 8, b"x2" * 8]


# --------------------------------------------------------------------------
# LookupKeyClient
# --------------------------------------------------------------------------


def test_arrival_lookup_is_read_at_first_query_without_a_second_rpc():
    store = FakeStore()
    store.hits[H[-1]] = 32
    client = _make_client(store)
    owner = object()
    assert client.prefetch("r", 48, H, owner)
    _wait_done(client, "r")
    client.adopt("r", owner)

    result = client.lookup("r", 48, H, non_block=True)
    assert result is not None and result.hit_length == 32
    assert store.calls == [(48, 3, 0)]
    assert client.stats["ready_at_first_use"] == 1
    assert client.stats["pending_at_first_use"] == 0
    # Kept for the rest of the admission attempt, like any lookup.
    assert client.lookup("r", 48, H, non_block=True).hit_length == 32
    assert len(store.calls) == 1
    client.discard("r")  # admission
    _assert_empty(client)


def test_arrival_lookup_still_running_defers_like_todays_lookup():
    store = FakeStore(gate_open=False)
    client = _make_client(store)
    owner = object()
    client.prefetch("r", 48, H, owner)
    client.adopt("r", owner)
    # Not done at the first schedule: the request waits, as it does today.
    assert client.lookup("r", 48, H, non_block=True) is None
    assert client.stats["pending_at_first_use"] == 1
    store.gate.set()
    _wait_done(client, "r")
    assert client.lookup("r", 48, H, non_block=True).hit_length == 0
    assert len(store.calls) == 1
    # The first-use counter counts once per arrival lookup.
    assert client.stats["ready_at_first_use"] == 0


def test_arrival_lookup_never_replaces_an_existing_lookup():
    store = FakeStore(gate_open=False)
    client = _make_client(store)
    assert client.lookup("r", 48, H, non_block=True) is None
    assert not client.prefetch("r", 48, H, object())
    assert client.stats["arrival_lookups"] == 0
    assert "r" not in client._arrival_owner
    store.gate.set()
    _wait_done(client, "r")
    assert len(store.calls) == 1


def test_lookup_for_other_inputs_never_serves_the_arrival_result():
    """Stale inputs: the request's tokens/hashes changed after it arrived."""
    store = FakeStore()
    store.hits[H[-1]] = 32  # what the arrival lookup would report
    store.hits[H2[-1]] = 16
    client = _make_client(store)
    owner = object()
    client.prefetch("r", 48, H, owner)
    _wait_done(client, "r")
    client.adopt("r", owner)

    # Other hashes (rebuilt) -> a new lookup, never the arrival result.
    client.lookup("r", 48, H2, non_block=True)
    _wait_done(client, "r")
    assert client.lookup("r", 48, H2, non_block=True).hit_length == 16
    assert client.stats["refreshed"] == 1
    assert [c[:2] for c in store.calls] == [(48, 3), (48, 3)]
    # Other token count with the same full blocks -> new lookup too.
    client.lookup("r", 50, H2, non_block=False)
    assert client.stats["refreshed"] == 2
    assert len(store.calls) == 3


def test_admission_then_preemption_looks_up_again():
    """Admission drops the arrival lookup; the re-admission after a preemption
    starts its own lookup (the arrival lookup is never reused)."""
    store = FakeStore()
    store.hits[H[-1]] = 32
    client = _make_client(store)
    owner = object()
    client.prefetch("r", 48, H, owner)
    _wait_done(client, "r")
    client.adopt("r", owner)
    assert client.lookup("r", 48, H).hit_length == 32
    client.discard("r")  # update_state_after_alloc
    _assert_empty(client)
    store.hits[H[-1]] = 16  # the store changed meanwhile
    assert client.lookup("r", 48, H).hit_length == 16
    assert len(store.calls) == 2
    assert client.stats["arrival_lookups"] == 1


def test_abort_while_arrival_lookup_in_flight_never_serves_it():
    store = FakeStore(gate_open=False)
    store.hits[H[-1]] = 32
    client = _make_client(store)
    owner = object()
    client.prefetch("r", 48, H, owner)
    client.adopt("r", owner)
    # The request finishes (abort) while the RPC is still running.
    client.discard("r")
    _assert_empty(client)
    store.hits[H[-1]] = 0
    store.gate.set()
    # A request reusing the id gets a fresh lookup, not the aborted one.
    assert client.lookup("r", 48, H).hit_length == 0
    assert len(store.calls) == 2


def test_release_unadopted_drops_only_the_owners_lookup():
    store = FakeStore()
    client = _make_client(store)
    owner, other = object(), object()
    client.prefetch("r", 48, H, owner)
    # Another request object with the same id (e.g. a duplicate add) does not
    # own it.
    client.release_unadopted("r", other)
    assert "r" in client.futures
    # Never taken in by the scheduler: dropped when the add was handled.
    client.release_unadopted("r", owner)
    _assert_empty(client)
    assert client.stats["dropped"] == 1

    # Adopted lookups are not dropped by the handled hook.
    client.prefetch("s", 48, H, owner)
    client.adopt("s", owner)
    client.release_unadopted("s", owner)
    assert "s" in client.futures
    assert client.stats["dropped"] == 1


def test_failed_arrival_lookup_is_a_miss_and_retried_on_the_next_admission():
    store = FakeStore()
    store.fail = True
    client = _make_client(store)
    owner = object()
    client.prefetch("r", 48, H, owner)
    _wait_done(client, "r")
    client.adopt("r", owner)
    assert client.lookup("r", 48, H, non_block=True).hit_length == 0
    _assert_empty(client)
    store.fail = False
    store.hits[H[-1]] = 32
    assert client.lookup("r", 48, H).hit_length == 32


def test_blocking_lookup_times_out_on_a_stuck_arrival_lookup():
    store = FakeStore(gate_open=False)
    client = _make_client(store, lookup_timeout_s=0.05)
    owner = object()
    client.prefetch("r", 48, H, owner)
    client.adopt("r", owner)
    assert client.lookup("r", 48, H, non_block=False).hit_length == 0
    assert "r" not in client.futures
    store.gate.set()


def test_timeout_does_not_drop_a_newer_lookup():
    """A late failure of an old future must not remove the lookup that
    replaced it (identity-checked drop)."""
    store = FakeStore()
    client = _make_client(store)
    client.lookup("r", 48, H)
    old = client.futures["r"]
    client.discard("r")
    client.lookup("r", 48, H)
    client._drop_if_current("r", old)
    assert "r" in client.futures


def test_concurrent_arrivals_and_scheduler_queries_keep_state_consistent():
    store = FakeStore()
    client = _make_client(store)
    n = 400
    ids = [f"r{i}" for i in range(n)]
    owners = {i: object() for i in ids}
    errors: list[BaseException] = []

    def input_thread():
        try:
            for i in ids:
                client.prefetch(i, 48, H, owners[i])
        except BaseException as e:  # pragma: no cover - reported below
            errors.append(e)

    t = threading.Thread(target=input_thread)
    t.start()
    done: set[str] = set()
    deadline = time.monotonic() + 30
    while len(done) < n and time.monotonic() < deadline:
        for k, i in enumerate(ids):
            if i in done or i not in client.futures:
                continue
            if k % 3 == 0:
                client.release_unadopted(i, owners[i])  # rejected add
            else:
                client.adopt(i, owners[i])
                client.lookup(i, 48, H, non_block=k % 2 == 0)
                client.discard(i)  # admitted or aborted
            done.add(i)
    t.join()
    assert not errors
    assert len(done) == n
    _assert_empty(client)
    st = client.stats
    assert st["arrival_lookups"] == n
    assert st["dropped"] == len(ids[::3])


def test_invalidate_all_drops_every_lookup():
    store = FakeStore()
    client = _make_client(store)
    owner = object()
    client.prefetch("a", 48, H, owner)
    client.lookup("b", 48, H2)
    client.invalidate_all()
    _assert_empty(client)


def test_dp_ranks_use_their_own_lookup_server():
    paths = set()
    for dp in (0, 1):
        cfg = create_vllm_config(
            kv_connector="MooncakeStoreConnector", kv_role="kv_both"
        )
        cfg.parallel_config.data_parallel_index = dp
        paths.add(worker.get_zmq_rpc_path_lookup(cfg))
    assert len(paths) == 2


# --------------------------------------------------------------------------
# MooncakeStoreScheduler hooks
# --------------------------------------------------------------------------


def _bare_store_scheduler(**overrides):
    sched = object.__new__(store_scheduler.MooncakeStoreScheduler)
    sched.enable_lookup = True
    sched.enable_partial_hash_hits = False
    sched._block_size = 16
    sched._hash_block_size = 16
    sched.lookup_from_local_hit = False
    sched.lookup_async = True
    sched.load_async = True
    sched.load_specs = {}
    sched.client = MagicMock()
    for k, v in overrides.items():
        setattr(sched, k, v)
    return sched


def _req(**kw):
    base = dict(
        request_id="r",
        num_tokens=48,
        block_hashes=H,
        resumable=False,
        abort_immediately=False,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_store_scheduler_starts_arrival_lookup_with_first_schedule_inputs():
    sched = _bare_store_scheduler()
    req = _req()
    sched.on_request_arrival(req)
    sched.client.prefetch.assert_called_once_with("r", 48, H, req)
    sched.on_new_request(req)
    sched.client.adopt.assert_called_once_with("r", req)
    sched.on_request_arrival_handled(req)
    sched.client.release_unadopted.assert_called_once_with("r", req)


@pytest.mark.parametrize(
    "overrides, req_kw",
    [
        ({"enable_lookup": False}, {}),
        ({"lookup_from_local_hit": True}, {}),
        ({}, {"resumable": True}),
        ({}, {"abort_immediately": True}),
        ({}, {"num_tokens": 15}),
    ],
)
def test_store_scheduler_skips_arrival_lookup_when_inputs_are_schedule_time(
    overrides, req_kw
):
    sched = _bare_store_scheduler(**overrides)
    sched.on_request_arrival(_req(**req_kw))
    sched.client.prefetch.assert_not_called()


def test_reset_store_invalidates_arrival_lookups_even_on_rpc_failure():
    sched = _bare_store_scheduler()
    sched.client.reset.return_value = True
    assert sched.reset_store() is True
    sched.client.invalidate_all.assert_called_once()
    sched.client.reset.side_effect = RuntimeError("down")
    assert sched.reset_store() is False
    assert sched.client.invalidate_all.call_count == 2


def test_multi_connector_forwards_arrival_hooks():
    mc = object.__new__(MultiConnector)
    a, b = MagicMock(), MagicMock()
    mc._connectors = [a, b]
    req = object()
    mc.on_request_arrival(req)
    mc.on_request_arrival_handled(req)
    for c in (a, b):
        c.on_request_arrival.assert_called_once_with(req)
        c.on_request_arrival_handled.assert_called_once_with(req)


# --------------------------------------------------------------------------
# Core scheduler + real MooncakeStoreConnector (lookup RPC faked)
# --------------------------------------------------------------------------


def _core_scheduler(store: FakeStore, dp_index: int = 0):
    vllm_config = create_vllm_config(
        kv_connector="MooncakeStoreConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"lookup_async": True, "load_async": True},
        max_num_batched_tokens=4096,
    )
    vllm_config.parallel_config.data_parallel_index = dp_index
    with patch(_SOCKET):
        sched = create_scheduler(vllm_config, num_blocks=1000)
    store_sched = sched.connector.connector_scheduler
    store_sched.client._lookup = store
    return sched, store_sched.client


def _arrive(sched, req):
    """What the engine core does for an ADD: arrival hook (input thread),
    add_request, then the handled hook (scheduler thread)."""
    sched.connector.on_request_arrival(req)
    try:
        sched.add_request(req)
    finally:
        sched.connector.on_request_arrival_handled(req)


def _new_req(sched, rid, nblocks=8):
    bs = sched.block_size
    return create_request(
        request_id=rid, block_size=bs, num_tokens=bs * nblocks, max_tokens=4
    )


def test_core_arrival_lookup_admits_at_the_first_schedule():
    store = FakeStore()
    sched, client = _core_scheduler(store)
    req = _new_req(sched, 1)
    _arrive(sched, req)
    _wait_done(client, req.request_id)
    out = sched.schedule()
    assert out.num_scheduled_tokens.get(req.request_id)
    assert len(store.calls) == 1
    # Admission dropped it, exactly like a lookup started by the schedule.
    assert req.request_id not in client.futures
    assert client.stats["ready_at_first_use"] == 1


def test_core_without_arrival_lookup_waits_one_schedule():
    """Reference: today's path (lookup started by the first schedule). The RPC
    takes longer than the rest of that schedule (gate), as it does on the
    deployment (~30 ms vs a few ms)."""
    store = FakeStore(gate_open=False)
    sched, client = _core_scheduler(store)
    req = _new_req(sched, 1)
    sched.add_request(req)
    out = sched.schedule()
    assert not out.num_scheduled_tokens.get(req.request_id)
    store.gate.set()
    _wait_done(client, req.request_id)
    sched.update_from_output(out, create_model_runner_output([]))
    out = sched.schedule()
    assert out.num_scheduled_tokens.get(req.request_id)
    assert len(store.calls) == 1


def test_core_arrival_lookup_not_ready_defers_to_next_schedule():
    store = FakeStore(gate_open=False)
    sched, client = _core_scheduler(store)
    req = _new_req(sched, 1)
    _arrive(sched, req)
    out = sched.schedule()
    assert not out.num_scheduled_tokens.get(req.request_id)
    store.gate.set()
    _wait_done(client, req.request_id)
    sched.update_from_output(out, create_model_runner_output([]))
    out = sched.schedule()
    assert out.num_scheduled_tokens.get(req.request_id)
    assert len(store.calls) == 1


def test_core_hit_allocates_only_at_the_admission_decision():
    """No early KV allocation: the arrival lookup reserves nothing; blocks are
    allocated by the schedule that admits the request (load then pending)."""
    store = FakeStore()
    sched, client = _core_scheduler(store)
    bs = sched.block_size
    req = _new_req(sched, 1)
    store.hits[bytes(req.block_hashes[-1])] = 4 * bs  # store holds 4 blocks
    free_before = sched.kv_cache_manager.block_pool.get_num_free_blocks()
    _arrive(sched, req)
    _wait_done(client, req.request_id)
    assert sched.kv_cache_manager.block_pool.get_num_free_blocks() == free_before
    out = sched.schedule()
    assert req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    (meta,) = [
        m
        for m in out.kv_connector_metadata.requests
        if m.req_id == req.request_id and m.load_spec is not None
    ]
    assert meta.load_spec.can_load
    assert meta.load_spec.kvpool_cached_tokens == 4 * bs
    assert sched.kv_cache_manager.block_pool.get_num_free_blocks() < free_before


def test_core_abort_while_arrival_lookup_in_flight_leaves_no_state():
    store = FakeStore(gate_open=False)
    sched, client = _core_scheduler(store)
    req = _new_req(sched, 1)
    _arrive(sched, req)
    out = sched.schedule()  # deferred: lookup pending
    sched.update_from_output(out, create_model_runner_output([]))
    sched.finish_requests(req.request_id, RequestStatus.FINISHED_ABORTED)
    sched.schedule()  # build_connector_meta drops finished requests' lookups
    assert req.request_id not in client.futures
    store.gate.set()
    _assert_empty(client)


def test_core_abort_before_first_schedule_leaves_no_state():
    store = FakeStore()
    sched, client = _core_scheduler(store)
    req = _new_req(sched, 1)
    _arrive(sched, req)
    sched.finish_requests(req.request_id, RequestStatus.FINISHED_ABORTED)
    sched.schedule()
    _assert_empty(client)
    assert len(store.calls) == 1


def test_core_preempted_request_looks_up_again_on_readmission():
    store = FakeStore()
    sched, client = _core_scheduler(store)
    req = _new_req(sched, 1, nblocks=2)
    _arrive(sched, req)
    _wait_done(client, req.request_id)
    out = sched.schedule()
    assert out.num_scheduled_tokens.get(req.request_id)
    sched.update_from_output(out, create_model_runner_output([req]))
    # Preempt it (as the scheduler does under KV pressure).
    sched.running.remove(req)
    sched._preempt_request(req, time.monotonic(), drop_stale_output=True)
    assert req.status == RequestStatus.PREEMPTED
    sched.schedule()  # re-admission starts a new lookup
    _wait_done(client, req.request_id)
    # Its own inputs (prompt + the sampled token), not the arrival lookup's.
    assert store.calls == [(32, 2, 0), (33, 2, 0)]
    assert client.stats["arrival_lookups"] == 1
    assert client.stats["ready_at_first_use"] == 1


def test_core_rejected_add_drops_the_arrival_lookup():
    store = FakeStore()
    sched, client = _core_scheduler(store)
    req = _new_req(sched, 1)
    sched.connector.on_request_arrival(req)
    # The engine core rejected the ADD (e.g. paused / shutting down): the
    # scheduler never saw it.
    sched.connector.on_request_arrival_handled(req)
    _assert_empty(client)
    assert client.stats["dropped"] == 1


def test_core_dp_ranks_keep_separate_arrival_lookups():
    store0, store1 = FakeStore(), FakeStore()
    s0, c0 = _core_scheduler(store0, dp_index=0)
    s1, c1 = _core_scheduler(store1, dp_index=1)
    r0, r1 = _new_req(s0, 1), _new_req(s1, 2)
    _arrive(s0, r0)
    _arrive(s1, r1)
    _wait_done(c0, r0.request_id)
    _wait_done(c1, r1.request_id)
    assert list(c0.futures) == [r0.request_id]
    assert list(c1.futures) == [r1.request_id]
    assert s0.schedule().num_scheduled_tokens.get(r0.request_id)
    assert s1.schedule().num_scheduled_tokens.get(r1.request_id)
    assert len(store0.calls) == len(store1.calls) == 1


# --------------------------------------------------------------------------
# Engine core wiring
# --------------------------------------------------------------------------


def _stub_core(cls):
    from vllm.v1.engine import core as core_mod

    core = object.__new__(getattr(core_mod, cls))
    connector = MagicMock()
    core.scheduler = MagicMock()
    core.scheduler.get_kv_connector.return_value = connector
    core.mm_receiver_cache = None
    core.request_block_hasher = None
    core.structured_output_manager = MagicMock()
    return core, connector


def test_preprocess_add_request_starts_the_arrival_hook():
    core, connector = _stub_core("EngineCoreProc")
    fake_req = SimpleNamespace(use_structured_output=False)
    ecr = SimpleNamespace(mm_features=None, current_wave=3)
    with patch(
        "vllm.v1.engine.core.Request.from_engine_core_request", return_value=fake_req
    ):
        req, wave = core.preprocess_add_request(ecr)
    assert req is fake_req and wave == 3
    connector.on_request_arrival.assert_called_once_with(fake_req)


@pytest.mark.parametrize("cls", ["EngineCoreProc", "DPEngineCoreProc"])
@pytest.mark.parametrize("outcome", ["added", "rejected", "raises"])
def test_add_handling_always_reports_the_arrival_handled(cls, outcome):
    core, connector = _stub_core(cls)
    req = SimpleNamespace(request_id="r")
    core._reject_add_in_shutdown = MagicMock(return_value=False)
    core._reject_add_while_paused = MagicMock(return_value=outcome == "rejected")
    core.add_request = MagicMock(
        side_effect=RuntimeError("bad") if outcome == "raises" else None
    )
    if outcome == "raises":
        with pytest.raises(RuntimeError):
            core._handle_client_request(EngineCoreRequestType.ADD, (req, 0))
    else:
        core._handle_client_request(EngineCoreRequestType.ADD, (req, 0))
    connector.on_request_arrival_handled.assert_called_once_with(req)
    assert core.add_request.call_count == (0 if outcome == "rejected" else 1)


def test_inproc_client_reports_the_arrival_handled():
    from vllm.v1.engine.core_client import InprocClient

    client = object.__new__(InprocClient)
    client.engine_core = MagicMock()
    req = object()
    client.engine_core.preprocess_add_request.return_value = (req, 0)
    client.engine_core.add_request.side_effect = RuntimeError("bad")
    with pytest.raises(RuntimeError):
        client.add_request(MagicMock())
    client.engine_core._kv_connector_arrival_handled.assert_called_once_with(req)
