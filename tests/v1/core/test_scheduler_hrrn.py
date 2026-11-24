# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HRRN admission (SchedulerConfig.prefill_hrrn): the waiting queue is served
by response ratio, so a short prefill is not stuck behind longer ones that
arrived just before it, while a long prefill's wait stays bounded."""

import time

import pytest

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

from .utils import create_requests, create_scheduler, mock_kv

MNBT = 1024


def _scheduler(hrrn: bool, max_wait: float = 30.0, **kw) -> Scheduler:
    return create_scheduler(
        max_num_batched_tokens=MNBT,
        max_model_len=2048,
        block_size=16,
        prefill_hrrn=hrrn,
        prefill_hrrn_max_wait_s=max_wait,
        **kw,
    )


def _req(rid: str, num_tokens: int, waited_s: float) -> Request:
    (r,) = create_requests(
        num_requests=1, num_tokens=num_tokens, max_tokens=1, req_ids=[rid]
    )
    r.arrival_time = time.time() - waited_s
    return r


@pytest.mark.parametrize("hrrn", [False, True])
def test_short_prefill_is_not_stuck_behind_an_earlier_long_one(hrrn):
    scheduler = _scheduler(hrrn)
    scheduler.add_request(_req("long", 1800, waited_s=1.0))
    scheduler.add_request(_req("short", 600, waited_s=0.9))
    out = scheduler.schedule()
    if hrrn:
        assert out.num_scheduled_tokens == {"short": 600, "long": MNBT - 600}
    else:
        assert out.num_scheduled_tokens == {"long": MNBT}


def test_long_prefill_past_max_wait_goes_first():
    scheduler = _scheduler(True, max_wait=30.0)
    scheduler.add_request(_req("short", 600, waited_s=0.5))
    scheduler.add_request(_req("long", 1800, waited_s=31.0))
    out = scheduler.schedule()
    assert out.num_scheduled_tokens == {"long": MNBT}


def test_service_counts_only_uncached_work():
    """A long prompt with a large external hit is short work."""
    scheduler = _scheduler(
        True, use_kv_connector=mock_kv(matched_tokens=0, is_async=False)
    )
    hits = {"warm": 1800 - 512}
    scheduler.connector.peek_external_hit = lambda r: hits.get(r.request_id)
    scheduler.add_request(_req("cold", 1024, waited_s=1.0))
    scheduler.add_request(_req("warm", 1800, waited_s=1.0))
    scheduler.schedule()
    assert [r.request_id for r in scheduler.running][:1] == ["warm"]


def test_hit_seen_once_is_kept_when_the_lookup_is_dropped():
    scheduler = _scheduler(
        True, use_kv_connector=mock_kv(matched_tokens=0, is_async=False)
    )
    hits = {"warm": 1800 - 512}
    scheduler.connector.peek_external_hit = lambda r: hits.get(r.request_id)
    warm, cold = _req("warm", 1800, waited_s=1.0), _req("cold", 1024, waited_s=1.0)
    scheduler.waiting.add_request(cold)
    scheduler.waiting.add_request(warm)
    scheduler._hrrn_order_waiting()
    hits.clear()
    scheduler._hrrn_order_waiting()
    assert [r.request_id for r in scheduler.waiting] == ["warm", "cold"]
