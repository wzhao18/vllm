# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DP late binding (scheduler side): which waiting requests another DP rank may
take over, and handing one over leaves nothing of it behind."""

import pytest

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


def test_lists_only_uncommitted_waiting_requests_in_queue_order():
    # Budget for one prompt per step: the rest stay queued and uncommitted.
    scheduler = create_scheduler(max_num_batched_tokens=100, enable_chunked_prefill=False)
    reqs = create_requests(num_requests=3, num_tokens=100)
    for r in reqs:
        scheduler.add_request(r)
    # Before any step all three are movable.
    assert scheduler.dp_movable_requests(8) == [(r.request_id, 100) for r in reqs]
    assert scheduler.dp_movable_requests(2) == [(r.request_id, 100) for r in reqs[:2]]
    assert scheduler.dp_movable_requests(0) == []
    out: SchedulerOutput = scheduler.schedule()
    assert list(out.num_scheduled_tokens) == [reqs[0].request_id]
    # The scheduled one holds blocks and is RUNNING: no longer movable.
    assert scheduler.dp_movable_requests(8) == [(r.request_id, 100) for r in reqs[1:]]


def test_preempted_or_partially_computed_requests_stay():
    scheduler = create_scheduler()
    reqs = create_requests(num_requests=2, num_tokens=64)
    for r in reqs:
        scheduler.add_request(r)
    reqs[0].num_preemptions = 1
    reqs[1].num_computed_tokens = 16
    assert scheduler.dp_movable_requests(8) == []
    assert scheduler.dp_export_request(reqs[0].request_id) is None
    assert reqs[0].request_id in scheduler.requests


def test_export_removes_the_request_without_finishing_it():
    scheduler = create_scheduler()
    reqs = create_requests(num_requests=2, num_tokens=64)
    for r in reqs:
        scheduler.add_request(r)
    moved = scheduler.dp_export_request(reqs[0].request_id)
    assert moved is reqs[0]
    assert moved.status == RequestStatus.WAITING
    assert reqs[0].request_id not in scheduler.requests
    assert [r.request_id for r in scheduler.waiting] == [reqs[1].request_id]
    # Not finished here: no finished id is reported for it.
    assert reqs[0].request_id not in scheduler.finished_req_ids
    # Gone: a second export (or a stale plan) is a no-op.
    assert scheduler.dp_export_request(reqs[0].request_id) is None
    # The rest schedules as usual.
    out = scheduler.schedule()
    assert list(out.num_scheduled_tokens) == [reqs[1].request_id]
    assert scheduler.get_num_unfinished_requests() == 1


def test_scheduled_request_is_not_exported():
    scheduler = create_scheduler()
    (req,) = create_requests(num_requests=1, num_tokens=64)
    scheduler.add_request(req)
    scheduler.schedule()
    assert scheduler.dp_export_request(req.request_id) is None
    assert req.request_id in scheduler.requests
