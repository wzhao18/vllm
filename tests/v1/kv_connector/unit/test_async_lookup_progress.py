# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Waiting requests with an async KV connector lookup (Mooncake lookup_async)
must always make progress, including when the lookup never returns.

``AsyncLookup`` emulates LookupKeyClient: the first query submits a lookup and
returns (None, False); the result is ready ``latency_steps`` schedule() calls
later. With ``retain`` it is kept until the request is admitted (the fixed
client); without it, reading it drops it, so the next query submits a new
lookup (the old client).
"""

from unittest.mock import patch

import pytest

import vllm.v1.core.sched.scheduler as scheduler_mod
from vllm.v1.request import RequestStatus

from .utils import (
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)


class AsyncLookup:
    def __init__(self, latency_steps: int | None, retain: bool = False):
        self.latency_steps = latency_steps  # None: never returns
        self.retain = retain
        self.step = 0
        self.ready_at: dict[str, int] = {}
        self.submits = 0

    def __call__(self, request, num_computed_tokens):
        ready_at = self.ready_at.get(request.request_id)
        if ready_at is None:
            self.ready_at[request.request_id] = (
                self.step + self.latency_steps
                if self.latency_steps is not None
                else 1 << 62
            )
            self.submits += 1
            return None, False
        if self.step < ready_at:
            return None, False
        if not self.retain:
            del self.ready_at[request.request_id]
        return 0, False  # store miss

    def admitted(self, request, blocks, num_external_tokens):
        self.ready_at.pop(request.request_id, None)


def _run(lookup: AsyncLookup, max_steps: int = 200):
    vllm_config = create_vllm_config(max_num_batched_tokens=4096)
    block_size = vllm_config.cache_config.block_size
    scheduler = create_scheduler(vllm_config, num_blocks=1000)
    req = create_request(
        request_id=1, block_size=block_size, num_tokens=block_size * 64, max_tokens=1
    )
    scheduler.add_request(req)
    with (
        patch.object(
            scheduler.connector, "get_num_new_matched_tokens", side_effect=lookup
        ),
        patch.object(
            scheduler.connector, "update_state_after_alloc", side_effect=lookup.admitted
        ),
    ):
        for step in range(max_steps):
            lookup.step = step
            out = scheduler.schedule()
            if out.num_scheduled_tokens.get(req.request_id):
                return scheduler, req, step
            scheduler.update_from_output(out, create_model_runner_output([]))
    return scheduler, req, None


@pytest.mark.parametrize("latency_steps", [1, 2, 5])
def test_retained_lookup_admits_prefill_in_its_first_step(latency_steps):
    """With the result kept until admission, the request is admitted in the
    step its lookup resolves."""
    lookup = AsyncLookup(latency_steps, retain=True)
    _, req, step = _run(lookup)
    assert step == latency_steps
    assert lookup.submits == 1
    assert req.status == RequestStatus.RUNNING


def test_stalled_lookup_falls_back_to_recompute(monkeypatch):
    monkeypatch.setattr(scheduler_mod, "KV_CONNECTOR_LOOKUP_TIMEOUT_S", 0.0)
    lookup = AsyncLookup(None)
    scheduler, req, step = _run(lookup)
    assert step is not None
    assert req.status == RequestStatus.RUNNING
