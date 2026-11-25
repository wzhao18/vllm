# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCP ShardQ rank-major prefetch: a repeated attention of a DSA layer over the
same gathered context (no new gather), as the DSA-layer ShardQ tests' replicated
query recompute does (``DcpShardQContext.before_attend`` / ``after_attend``):

* the first attention waits for its copies, consumes them and issues the next
  layer's prefetch (or releases on the last layer);
* a repeat on the last layer (nothing in flight, no prefetch) reads the workspace
  as it is and issues nothing;
* a repeat while the next layer's prefetch is in flight (it overwrites the
  workspace on the side stream) raises instead of racing;
* any other state still asserts. CPU, no GPU work (stand-in copies)."""

import types

import pytest

from vllm.v1.attention.ops.dcp_shardq import context as ctx_mod
from vllm.v1.attention.ops.dcp_shardq import rank_major


class Cache:
    def __init__(self, ptr):
        self.ptr = ptr

    def data_ptr(self):
        return self.ptr


class InFlight:
    def __init__(self):
        self.waited = False
        self.flags, self.rank = object(), 0

    def wait(self):
        self.waited = True


def _context(layers):
    ns = types.SimpleNamespace(
        mla_in_flight=InFlight(), mla_prefetch=None, mla_attended=None,
        rank_major=object(), prefetched=[],
    )  # fmt: skip
    for name in ("before_attend", "after_attend", "wait_mla"):
        setattr(ns, name, types.MethodType(getattr(ctx_mod.DcpShardQContext, name), ns))
    ns._dsa_layers = lambda: [(c, None) for c in layers]

    def _prefetch(cache, table, after_layer):
        ns.mla_prefetch = (cache.data_ptr(), InFlight())
        ns.prefetched.append(cache.data_ptr())

    ns._prefetch = _prefetch
    return ns


@pytest.fixture(autouse=True)
def _no_release(monkeypatch):
    monkeypatch.setattr(rank_major, "release", lambda flags, rank: None)


def test_repeat_on_last_layer_reads_as_is():
    a = Cache(1)
    c = _context([a])
    c.before_attend(a)
    assert c.mla_in_flight.waited
    c.after_attend(a)
    assert c.mla_in_flight is None and c.mla_attended == 1 and c.prefetched == []
    c.before_attend(a)  # repeat: nothing in flight, nothing overwriting
    c.after_attend(a)
    assert c.prefetched == []


def test_repeat_with_next_layer_prefetch_in_flight_raises():
    a, b = Cache(1), Cache(2)
    c = _context([a, b])
    c.before_attend(a)
    c.after_attend(a)
    assert c.prefetched == [2] and c.mla_prefetch is not None
    with pytest.raises(RuntimeError, match="repeated attention"):
        c.before_attend(a)


def test_other_states_still_assert():
    a, b = Cache(1), Cache(2)
    c = _context([a, b])
    c.mla_in_flight = None
    with pytest.raises(AssertionError):
        c.after_attend(a)  # never attended, nothing in flight
    c = _context([a, b])
    c.before_attend(a)
    c.after_attend(a)
    with pytest.raises(AssertionError):
        c.after_attend(b)  # another layer without its gather
