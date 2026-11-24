# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mooncake store same-step loads (load_same_step).

An early load can land while the previous step still runs (batch queue). Its
in-band finished/failed report must reach the scheduler in the output of the
step that computes the request, never an earlier one (the scheduler would see
a finished load for a RUNNING request)."""

import queue
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import worker
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    LoadSpec,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import KVCacheBlock

from .test_mooncake_store_scheduler import _make_bare_scheduler, _StubLookupClient


def _bare_worker():
    w = object.__new__(worker.MooncakeStoreWorker)
    w._capacity_only = False
    w.can_put = False
    w.load_async = True
    w._recv_poller_active = False
    w.tp_rank = 0
    w._same_step_cond = threading.Condition()
    w._same_step_landed = {}
    w._same_step_failed = set()
    w._same_step_unwaited = set()
    w._held_finished_recving = set()
    w._held_failed_recving = set()
    w.recv_request_queue = queue.Queue()
    w._increment_kv_connector_counter = lambda *a, **k: None
    w._record_kv_connector_operation = lambda *a, **k: None

    t = object.__new__(worker.KVCacheStoreRecvingThread)
    t.done_task_lock = threading.Lock()
    t.finished_requests = set()
    t._clean_finished = set()
    t._failed_requests = set()
    t._is_hma_required = False
    t._same_step = True
    t.report_same_step_done = w._report_same_step_done
    w.kv_recv_threads = [t]
    return w, t


def _issue(w, req_id: str, load_spec: LoadSpec) -> None:
    """An early load the scheduler issued when it allocated the blocks."""
    w._issue_early_load(
        worker.ReqMeta(
            req_id=req_id,
            token_len_chunk=0,
            block_ids=([1, 2],),
            block_hashes=[b"h" * 16],
            can_save=False,
            load_spec=load_spec,
        )
    )


def _step_output(w):
    res = w.get_transfer_results(set(), None)
    return res.finished_recving, res.failed_recving


def test_same_step_load_landing_during_previous_step_is_reported_with_its_step():
    w, t = _bare_worker()
    # Step N runs; the scheduler admitted R for step N+1 and issued its load,
    # which lands before step N's output is built.
    _issue(w, "R", LoadSpec(0, 32, True, same_step=True))
    t._finish_load("R", failed=False)
    assert _step_output(w) == (set(), set())  # step N: held
    # Step N+1 waits for R before its forward: its output carries R.
    w.wait_same_step_loads(["R"])
    assert w._same_step_failed == set()
    assert _step_output(w) == ({"R"}, set())
    assert _step_output(w) == (set(), set())


def test_failed_same_step_load_is_reported_with_its_step():
    w, t = _bare_worker()
    _issue(w, "R", LoadSpec(0, 32, True, same_step=True))
    t._report_load_error("R", [1, 2])
    t._finish_load("R", failed=True)
    assert _step_output(w) == (set(), set())
    w.wait_same_step_loads(["R"])
    assert w._same_step_failed == {"R"}
    assert _step_output(w) == ({"R"}, {"R"})


def test_same_step_load_landing_after_the_wait_is_reported_in_that_step():
    """The usual order: the step waits, then the load lands."""
    w, t = _bare_worker()
    _issue(w, "R", LoadSpec(0, 32, True, same_step=True))
    waiter = threading.Thread(target=w.wait_same_step_loads, args=(["R"],))
    waiter.start()
    t._finish_load("R", failed=False)
    waiter.join(5)
    assert not waiter.is_alive()
    assert _step_output(w) == ({"R"}, set())


def test_async_loads_are_not_held():
    w, t = _bare_worker()
    t._same_step = False
    _issue(w, "A", LoadSpec(0, 16, True))
    t._finish_load("A", failed=False)
    assert _step_output(w) == ({"A"}, set())


class _Channel:
    def __init__(self):
        self.sent: list[worker.ReqMeta] = []

    def send(self, req_meta: worker.ReqMeta) -> bool:
        self.sent.append(req_meta)
        return True


@pytest.mark.parametrize("confirmed", [True, False])
def test_same_step_load_is_flagged_only_when_confirmed(confirmed):
    """With load_same_step a hit is reported synchronous; the early load and
    the step metadata mark it same-step only once the core confirmed it (else
    the core scheduled it as an async load)."""
    scheduler = _make_bare_scheduler()
    scheduler.load_async = True
    scheduler.load_same_step = True
    scheduler._same_step_confirmed = set()
    scheduler._same_step_load_ids = []
    scheduler._early_loads = channel = _Channel()
    scheduler._group_block_sizes = [16]
    scheduler.client = _StubLookupClient(hit_tokens=32)
    request = SimpleNamespace(
        request_id="req-0",
        num_tokens=48,
        num_prompt_tokens=48,
        block_hashes=[b"h0", b"h1", b"h2"],
    )

    assert scheduler.get_num_new_matched_tokens(request, 0) == (32, False)
    assert scheduler.load_lands_before_step(request)
    scheduler.client = MagicMock()
    if confirmed:
        scheduler.confirm_load_before_step(request)
    blocks = [KVCacheBlock(i + 1) for i in range(3)]
    for block in blocks:
        block.ref_cnt = 1
    scheduler.update_state_after_alloc(request, KVCacheBlocks((blocks,)), 32)

    (req_meta,) = channel.sent
    assert req_meta.load_spec is not None
    assert req_meta.load_spec.same_step is confirmed
    assert scheduler._same_step_load_ids == (["req-0"] if confirmed else [])
    assert not scheduler._same_step_confirmed
