# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mooncake store side of DP late binding: a queued request is sized for a
cross-rank move by its prompt minus the store hit of its COMPLETED arrival
lookup (peeked, never started, waited for or consumed), and a move drops the
lookup on the source rank without finishing the request."""

from types import SimpleNamespace

import pytest

from .test_mooncake_store_arrival_lookup import (
    H,
    H2,
    FakeStore,
    _assert_empty,
    _make_client,
    _wait_done,
)

pytestmark = pytest.mark.cpu_test


def test_peek_reads_a_completed_lookup_without_consuming_it():
    store = FakeStore()
    store.hits[H[-1]] = 32
    client = _make_client(store)
    owner = object()
    client.prefetch("r", 48, H, owner)
    _wait_done(client, "r")
    assert client.peek_hit("r", 48, H) == 32
    assert client.peek_hit("r", 48, H) == 32
    # No RPC of its own, and the scheduler's first query still reads it as the
    # arrival lookup.
    assert store.calls == [(48, 3, 0)]
    client.adopt("r", owner)
    assert client.lookup("r", 48, H, non_block=True).hit_length == 32
    assert client.stats["ready_at_first_use"] == 1
    assert len(store.calls) == 1


def test_peek_is_none_while_in_flight_absent_failed_or_for_other_inputs():
    store = FakeStore(gate_open=False)
    store.hits[H[-1]] = 32
    client = _make_client(store)
    assert client.peek_hit("r", 48, H) is None  # no lookup at all
    client.prefetch("r", 48, H, object())
    assert client.peek_hit("r", 48, H) is None  # in flight
    store.gate.set()
    _wait_done(client, "r")
    assert client.peek_hit("r", 48, H2) is None  # other inputs
    assert client.peek_hit("r", 32, H) is None
    assert "r" in client.futures  # a peek never drops or replaces it
    store.fail = True
    client.prefetch("s", 48, H, object())
    _wait_done(client, "s")
    assert client.peek_hit("s", 48, H) is None  # failed lookup


def test_move_away_drops_the_arrival_lookup():
    store = FakeStore()
    client = _make_client(store)
    owner = object()
    client.prefetch("r", 48, H, owner)
    _wait_done(client, "r")
    client.adopt("r", owner)
    client.discard("r")  # MooncakeStoreScheduler.on_request_moved_away
    _assert_empty(client)


def test_scheduler_dp_move_tokens_is_prompt_minus_hit():
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler import (
        MooncakeStoreScheduler,
    )

    store = FakeStore()
    store.hits[H[-1]] = 32
    client = _make_client(store)
    sched = object.__new__(MooncakeStoreScheduler)
    sched.client = client
    req = SimpleNamespace(
        request_id="r", num_tokens=48, num_prompt_tokens=48, block_hashes=H
    )
    assert MooncakeStoreScheduler.dp_move_tokens(sched, req) is None
    client.prefetch("r", 48, H, req)
    _wait_done(client, "r")
    assert MooncakeStoreScheduler.dp_move_tokens(sched, req) == 16
    MooncakeStoreScheduler.on_request_moved_away(sched, req)
    _assert_empty(client)
    assert MooncakeStoreScheduler.dp_move_tokens(sched, req) is None

