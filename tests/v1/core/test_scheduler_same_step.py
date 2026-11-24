# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same-step KV loads: a connector load that lands before the step's forward
(load_lands_before_step) lets a prefill-only request compute the rest of its
prompt in its admission step instead of a step later."""

import pytest

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from .utils import create_requests, create_scheduler, mock_kv

BLOCK = 16
# 96-token prompt, 64 tokens loaded, 32 left to compute.
PROMPT, HIT = 96, 64


def _scheduler(mnbt: int = 8192, **kw) -> Scheduler:
    scheduler = create_scheduler(
        max_num_batched_tokens=mnbt,
        max_model_len=1024,
        block_size=BLOCK,
        use_kv_connector=mock_kv(matched_tokens=HIT, is_async=False),
        **kw,
    )
    scheduler.connector.confirmed = []
    scheduler.connector.load_lands_before_step = lambda request: True
    scheduler.connector.confirm_load_before_step = (
        lambda request: scheduler.connector.confirmed.append(request.request_id)
    )
    return scheduler


def _requests(ids: list[str], max_tokens: int = 1) -> list[Request]:
    return create_requests(
        num_requests=len(ids),
        num_tokens=PROMPT,
        max_tokens=max_tokens,
        block_size=BLOCK,
        req_ids=ids,
    )


def _step(scheduler: Scheduler, out: SchedulerOutput, **kv_output) -> dict:
    ids = list(out.num_scheduled_tokens)
    return scheduler.update_from_output(
        out,
        ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={r: i for i, r in enumerate(ids)},
            sampled_token_ids=[[1] for _ in ids],
            kv_connector_output=KVConnectorOutput(**kv_output),
        ),
    )


def test_same_step_load_computes_in_its_admission_step():
    """The workers' finished_recving for the load is not a remote-KV
    completion of a RUNNING request."""
    scheduler = _scheduler()
    (r0,) = _requests(["r0"])
    scheduler.add_request(r0)
    out = scheduler.schedule()
    assert out.num_scheduled_tokens == {"r0": PROMPT - HIT}
    assert out.has_sync_kv_loads
    assert r0.status == RequestStatus.RUNNING
    assert scheduler.connector.confirmed == ["r0"]

    _step(scheduler, out, finished_recving={"r0"})
    assert r0.is_finished()
    assert not scheduler._same_step_loads and not scheduler._same_step_landed


@pytest.mark.parametrize(
    "max_tokens, mnbt", [(2, 8192), (1, BLOCK)], ids=["decode", "chunk"]
)
def test_same_step_load_falls_back_to_async(max_tokens, mnbt):
    """A request that would not finish its prompt in its admission step (it
    decodes, or its prompt is chunked) loads async instead."""
    scheduler = _scheduler(mnbt)
    (r0,) = _requests(["r0"], max_tokens=max_tokens)
    scheduler.add_request(r0)
    out = scheduler.schedule()
    assert "r0" not in out.num_scheduled_tokens
    assert r0.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert scheduler.connector.confirmed == []


def test_same_step_load_without_budget_loads_ahead():
    """A same-step candidate admitted after the step's token budget is spent
    starts as an async load (load-ahead)."""
    scheduler = _scheduler(PROMPT - HIT)
    r0, r1 = _requests(["r0", "r1"])
    scheduler.add_request(r0)
    scheduler.add_request(r1)
    out = scheduler.schedule()
    assert out.num_scheduled_tokens == {"r0": PROMPT - HIT}
    assert r1.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert scheduler.connector.confirmed == ["r0"]


def test_aborted_same_step_load_skips_its_finished_report():
    """A same-step request aborted before its step output frees its blocks
    once: the workers' finished_recving for its load is skipped."""
    scheduler = _scheduler()
    num_free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    (r0,) = _requests(["r0"])
    scheduler.add_request(r0)
    out = scheduler.schedule()
    scheduler.finish_requests("r0", RequestStatus.FINISHED_ABORTED)

    _step(scheduler, out, finished_recving={"r0"})
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == num_free
    assert not scheduler._same_step_landed


def test_failed_same_step_load_recomputes():
    """A failed same-step load drops its step's output; with the recompute
    policy the request computes its whole prompt again."""
    scheduler = _scheduler()
    scheduler.recompute_kv_load_failures = True
    (r0,) = _requests(["r0"])
    scheduler.add_request(r0)
    out = scheduler.schedule()

    outputs = _step(scheduler, out, finished_recving={"r0"}, failed_recving={"r0"})
    assert not any(
        o.request_id == "r0" for outs in outputs.values() for o in outs.outputs
    )
    assert r0.status == RequestStatus.RUNNING
    assert r0.num_computed_tokens == 0
    assert not scheduler._same_step_landed

    out = scheduler.schedule()
    assert out.num_scheduled_tokens == {"r0": PROMPT}
    assert not out.has_sync_kv_loads


def test_same_step_load_caches_its_blocks_when_it_finishes():
    """A prefill-only same-step request finishes in its admission step, so it
    never allocates again: its loaded and computed blocks must still be cached,
    or the next turn of the conversation loses the hit."""
    scheduler = _scheduler(enable_prefix_caching=True)
    (r0,) = _requests(["r0"])
    scheduler.add_request(r0)
    out = scheduler.schedule()
    assert out.num_scheduled_tokens == {"r0": PROMPT - HIT}
    _step(scheduler, out, finished_recving={"r0"})
    assert r0.is_finished()
    # A later request with the same prompt hits every full block r0 held but
    # the one holding the last token (always recomputed).
    (r1,) = _requests(["r1"])
    _, num_local_hit = scheduler.kv_cache_manager.get_computed_blocks(r1)[:2]
    assert num_local_hit == PROMPT - BLOCK
