# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rubin MegaMoE dedup comm path: previous-call guard (BUG-E2E-3, VENDOR.md S8). CPU only.

1. An exhaustive interleaving model of one rank's next-call router against a peer's
   previous-call tail: without the guard some interleaving erases the router's remote
   ``sizes_ready`` add (the peer then waits forever); with the guard none does, and the
   guarded schedule never deadlocks.
2. The vendored sources keep the guard where it matters: the router waits for the
   previous call's done signals before any remote write, and ``reset_tail`` signals done
   after the publish barrier; the dedup tail state never aliases the GenPhase kernel's.
"""

import inspect
import itertools

import pytest


# -- 1. protocol model ---------------------------------------------------------------
# Peer B finishing call k: "reset" (zero its tail_reset counters), then "done" (signal).
# Rank A starting call k+1: optional "wait" (blocks until B's done), then "add" (remote
# sizes_ready += 1 into B). B's next router later checks that A's add is present.
def _run(order, guarded):
    ready = 1  # stale count from call k, still in B's counter before B's reset
    done = False
    added = False
    for ev in order:
        if ev == "reset":
            ready = 0
        elif ev == "done":
            done = True
        elif ev == "wait":
            if not done:
                return None  # blocked here: this order is not executable
        elif ev == "add":
            ready += 1
            added = True
    assert added
    return ready


def _orders(guarded):
    b = ["reset", "done"]
    a = ["wait", "add"] if guarded else ["add"]
    n = len(a) + len(b)
    for pos in itertools.combinations(range(n), len(a)):
        order, ia, ib = [], 0, 0
        for i in range(n):
            if i in pos:
                order.append(a[ia]); ia += 1
            else:
                order.append(b[ib]); ib += 1
        yield order


def test_unguarded_router_add_can_be_erased_by_peer_reset():
    finals = [_run(o, False) for o in _orders(False)]
    assert 0 in finals  # add before reset: erased, B's router would wait forever


def test_guarded_router_add_always_survives_and_never_deadlocks():
    finals = [r for r in (_run(o, True) for o in _orders(True)) if r is not None]
    assert finals and all(r == 1 for r in finals)
    # B's tail never waits on A, so the guarded schedule is always completable.
    assert _run(["reset", "done", "wait", "add"], True) == 1


# -- 2. source structure ---------------------------------------------------------------
cutlass = pytest.importorskip("cutlass")


def _mods():
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.src_gp.sources.communication.nvlink_domain import (
        token_comm_dedup as D,
    )
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.src_gp.sources.kernel_src.rubin.inference.mega import (
        genphase_tail as G,
    )

    return D, G


def test_dedup_tail_regions_do_not_alias_genphase():
    D, G = _mods()
    assert issubclass(D.DedupCallTail, G.GenphaseTail)
    assert D.DedupCallTail.epoch_region != G.GenphaseTail.epoch_region
    assert D.DedupCallTail.signal_region != G.GenphaseTail.signal_region


def test_router_waits_before_any_remote_write():
    D, _ = _mods()
    src = inspect.getsource(D._DedupMetadataPushRouter._router_kernel)
    wait = src.index("wait_previous_done")
    for call in ("_router_single_cta(", "_router_multiple_ctas(", "_router_push_metadata("):
        assert wait < src.index(call), call


def test_reset_tail_signals_done_after_the_publish_barrier():
    D, _ = _mods()
    src = inspect.getsource(D.TokenCommDedup.reset_tail)
    publish = src.index('"tail.nvlink_publish"')
    reset = src.index('reset_tail_space("shared"')
    done = src.index("signal_done(")
    assert reset < publish < done
