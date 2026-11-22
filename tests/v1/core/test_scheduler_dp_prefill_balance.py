# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DP prefill pull-up (scheduler side).

With an exchange set for the step, schedule() books the stock schedule, reports
its token count once, and lets the running prefill chunks that the long-prefill
threshold cut take up to the busiest DP rank's count. Without one the schedule
is stock.
"""

import pytest

from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _token_weighted_exchange(monkeypatch):
    """These tests pin the pull-up's token arithmetic: weight context 0, so the
    exchanged cost is the token count. The context weight is tested in
    test_pull_up_is_capped_by_context_weighted_cost."""
    from vllm.v1.core.sched import scheduler as sched_mod

    monkeypatch.setattr(sched_mod, "DP_PREFILL_CTX_WEIGHT", 0.0)


class _Exchange:
    """Stands in for the DP all-reduce: returns max(own, peer)."""

    def __init__(self, peer: int):
        self.peer = peer
        self.calls: list[int] = []
        self.demands: list[int] = []

    def __call__(self, num_tokens: int, demand: int = 0) -> int:
        self.calls.append(num_tokens)
        self.demands.append(demand)
        return max(num_tokens, self.peer)


def _scheduler(budget=1024, threshold=200, num_blocks=10000, **kw):
    return create_scheduler(
        model="facebook/opt-125m",
        max_num_batched_tokens=budget,
        long_prefill_token_threshold=threshold,
        num_blocks=num_blocks,
        max_model_len=2048,
        **kw,
    )


def _finish_step(scheduler, output: SchedulerOutput, requests):
    reqs = [r for r in requests if r.request_id in output.num_scheduled_tokens]
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=[r.request_id for r in reqs],
            req_id_to_index={r.request_id: i for i, r in enumerate(reqs)},
            sampled_token_ids=[
                [0]
                if r.num_computed_tokens >= r.num_prompt_tokens
                and r.request_id in output.num_scheduled_tokens
                else []
                for r in reqs
            ],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


def _running_chunked(scheduler, num_requests=3, num_tokens=1000, prefix="r"):
    """Admit long prompts and run one stock step: they become RUNNING prefill
    continuations whose stock chunk is the threshold."""
    requests = create_requests(
        num_requests=num_requests,
        num_tokens=num_tokens,
        req_ids=[f"{prefix}{i}" for i in range(num_requests)],
    )
    for r in requests:
        scheduler.add_request(r)
    out = scheduler.schedule()
    _finish_step(scheduler, out, requests)
    return requests


def _tokens(out, requests):
    return [out.num_scheduled_tokens.get(r.request_id, 0) for r in requests]


def _num_blocks(scheduler, request):
    (blocks,) = scheduler.kv_cache_manager.get_block_ids(request.request_id)
    return len(blocks)


def test_no_exchange_is_stock():
    a, b = _scheduler(), _scheduler()
    ra, rb = _running_chunked(a), _running_chunked(b)
    oa, ob = a.schedule(), b.schedule()
    assert _tokens(oa, ra) == _tokens(ob, rb) == [200, 200, 200]


def test_exchange_with_no_busier_peer_matches_stock():
    a, b = _scheduler(), _scheduler()
    ra, rb = _running_chunked(a), _running_chunked(b)
    # A new waiting prompt too: admissions see the same budget as stock.
    for s in (a, b):
        s.add_request(create_requests(1, num_tokens=150, req_ids=["new"])[0])
    ex = _Exchange(peer=0)
    b.set_dp_prefill_exchange(ex)
    oa, ob = a.schedule(), b.schedule()
    assert ex.calls == [oa.total_num_scheduled_tokens]
    assert dict(oa.num_scheduled_tokens) == dict(ob.num_scheduled_tokens)
    for ra_i, rb_i in zip(ra, rb):
        assert _num_blocks(a, ra_i) == _num_blocks(b, rb_i)


def test_pull_up_to_peer_in_order():
    scheduler = _scheduler()
    requests = _running_chunked(scheduler)
    ex = _Exchange(peer=900)
    scheduler.set_dp_prefill_exchange(ex)
    out = scheduler.schedule()
    assert ex.calls == [600]
    # The first chunked prefill takes the gap; nobody gets less than stock.
    assert _tokens(out, requests) == [500, 200, 200]
    assert out.total_num_scheduled_tokens == 900
    # Its KV slots cover the pulled chunk.
    assert _num_blocks(scheduler, requests[0]) == cdiv(200 + 500, 16)


def test_pull_spreads_and_stops_at_prompt_end():
    scheduler = _scheduler()
    requests = _running_chunked(scheduler, num_tokens=450)  # 250 left each
    scheduler.set_dp_prefill_exchange(_Exchange(peer=1000))
    out = scheduler.schedule()
    assert _tokens(out, requests) == [250, 250, 250]


def test_pull_never_exceeds_budget():
    scheduler = _scheduler(budget=1024)
    requests = _running_chunked(scheduler)
    scheduler.set_dp_prefill_exchange(_Exchange(peer=10_000))
    out = scheduler.schedule()
    assert out.total_num_scheduled_tokens == 1024
    assert _tokens(out, requests) == [624, 200, 200]


def test_pull_keeps_waiting_admissions():
    """Pulled tokens come only from the step's leftover: a new prompt admitted
    by the stock schedule keeps its chunk."""
    a, b = _scheduler(), _scheduler()
    _running_chunked(a)
    rb = _running_chunked(b)
    for s in (a, b):
        s.add_request(create_requests(1, num_tokens=300, req_ids=["new"])[0])
    b.set_dp_prefill_exchange(_Exchange(peer=1024))
    oa, ob = a.schedule(), b.schedule()
    assert oa.num_scheduled_tokens["new"] == ob.num_scheduled_tokens["new"] == 200
    # stock 3 x 200 + 200 = 800; pulled 224 into the first running prompt.
    assert _tokens(ob, rb) == [424, 200, 200]
    assert ob.total_num_scheduled_tokens == 1024


def test_exchange_consumed_by_one_schedule():
    scheduler = _scheduler()
    requests = _running_chunked(scheduler, num_tokens=2000)
    ex = _Exchange(peer=900)
    scheduler.set_dp_prefill_exchange(ex)
    out = scheduler.schedule()
    _finish_step(scheduler, out, requests)
    out2 = scheduler.schedule()
    assert ex.calls == [600]
    assert _tokens(out2, requests) == [200, 200, 200]


def test_exchange_called_once_when_nothing_to_schedule():
    scheduler = _scheduler()
    ex = _Exchange(peer=900)
    scheduler.set_dp_prefill_exchange(ex)
    out = scheduler.schedule()
    assert ex.calls == [0]
    assert out.total_num_scheduled_tokens == 0


def test_exchange_called_once_without_chunked_prefills():
    scheduler = _scheduler()
    requests = create_requests(2, num_tokens=100, req_ids=["a", "b"])
    for r in requests:
        scheduler.add_request(r)
    ex = _Exchange(peer=900)
    scheduler.set_dp_prefill_exchange(ex)
    out = scheduler.schedule()
    assert ex.calls == [200]
    assert _tokens(out, requests) == [100, 100]


def test_no_deferral_without_kv_headroom():
    """When the free KV blocks do not cover a whole step, the running chunks are
    allocated in stock order; the exchange still happens once."""
    # 1024-token budget needs 64 blocks + per-request slack: 150 blocks is short.
    a, b = _scheduler(num_blocks=150), _scheduler(num_blocks=150)
    ra, rb = _running_chunked(a), _running_chunked(b)
    ex = _Exchange(peer=900)
    b.set_dp_prefill_exchange(ex)
    oa, ob = a.schedule(), b.schedule()
    assert ex.calls == [600]
    assert _tokens(oa, ra) == _tokens(ob, rb) == [200, 200, 200]


def test_every_prompt_token_scheduled_exactly_once():
    """Drive steps to completion with a busier peer: every request runs each
    prompt token exactly once and its KV slots always cover what it ran."""
    scheduler = _scheduler(budget=1024, threshold=128)
    lengths = [2000, 1700, 900, 1500]
    requests = [
        create_requests(1, num_tokens=n, req_ids=[f"p{i}"], max_tokens=1)[0]
        for i, n in enumerate(lengths)
    ]
    for r in requests:
        scheduler.add_request(r)
    ran = {r.request_id: 0 for r in requests}
    for step in range(100):
        if not scheduler.has_unfinished_requests():
            break
        scheduler.set_dp_prefill_exchange(_Exchange(peer=700 if step % 3 else 0))
        out = scheduler.schedule()
        assert out.total_num_scheduled_tokens <= 1024
        for r in requests:
            n = out.num_scheduled_tokens.get(r.request_id, 0)
            ran[r.request_id] += n
            if n:
                assert _num_blocks(scheduler, r) >= cdiv(r.num_computed_tokens, 16)
        _finish_step(scheduler, out, requests)
    assert not scheduler.has_unfinished_requests()
    assert ran == {r.request_id: n for r, n in zip(requests, lengths)}


def test_pull_respects_mamba_block_alignment():
    """A pulled non-final chunk ends on a Mamba block boundary like a stock one."""
    block = 96
    scheduler = _scheduler(
        budget=4096, threshold=256, enable_prefix_caching=True, block_size=block
    )
    scheduler.need_mamba_block_aligned_split = True
    requests = create_requests(
        3, num_tokens=2000, req_ids=["m0", "m1", "m2"], block_size=block
    )
    for r in requests:
        scheduler.add_request(r)
    out = scheduler.schedule()
    # Stock: the 256-token chunk is cut back to the block boundary.
    assert _tokens(out, requests) == [192, 192, 192]
    _finish_step(scheduler, out, requests)
    starts = [r.num_computed_tokens for r in requests]
    scheduler.set_dp_prefill_exchange(_Exchange(peer=1500))
    out = scheduler.schedule()
    for r, start in zip(requests, starts):
        n = out.num_scheduled_tokens[r.request_id]
        assert n >= 192
        assert (start + n) % block == 0
    # Stock 3 x 192 = 576; the gap of 924 goes to m0, cut to whole blocks.
    assert _tokens(out, requests) == [192 + 864, 192, 192]
    assert out.total_num_scheduled_tokens <= 1500


def test_async_scheduler_pull_up():
    scheduler = _scheduler(async_scheduling=True)
    requests = _running_chunked(scheduler)
    scheduler.set_dp_prefill_exchange(_Exchange(peer=900))
    out = scheduler.schedule()
    assert _tokens(out, requests) == [500, 200, 200]


def test_exchange_reports_prompt_tokens_left():
    """The exchange carries the prompt tokens this rank still has to compute
    after the step (the DP lockstep step cap's input)."""
    scheduler = _scheduler(budget=1024, threshold=200)
    requests = _running_chunked(scheduler, num_requests=3, num_tokens=1000)
    left = sum(r.num_prompt_tokens - r.num_computed_tokens for r in requests)
    ex = _Exchange(peer=0)
    scheduler.set_dp_prefill_exchange(ex)
    out = scheduler.schedule()
    assert ex.demands == [left - sum(out.num_scheduled_tokens.values())]


def test_step_token_cap_bounds_the_step():
    """A DP lockstep step cap below the budget bounds the step's tokens; None
    restores the stock budget."""
    stock = _scheduler(budget=1024, threshold=0)
    capped = _scheduler(budget=1024, threshold=0)
    a = create_requests(num_requests=2, num_tokens=1000, req_ids=["a0", "a1"])
    b = create_requests(num_requests=2, num_tokens=1000, req_ids=["a0", "a1"])
    for r in a:
        stock.add_request(r)
    for r in b:
        capped.add_request(r)
    capped.set_dp_step_token_cap(256)
    assert sum(stock.schedule().num_scheduled_tokens.values()) == 1024
    out = capped.schedule()
    assert sum(out.num_scheduled_tokens.values()) == 256
    _finish_step(capped, out, b)
    capped.set_dp_step_token_cap(None)
    assert sum(capped.schedule().num_scheduled_tokens.values()) == 1024


def test_demand_counts_a_connector_load_once_it_lands():
    """A request whose KV connector load is in flight cannot run in any step
    yet: it adds to this rank's demand only once the load has finished."""
    from tests.v1.kv_connector.unit.utils import create_model_runner_output
    from vllm.v1.request import RequestStatus

    from .utils import mock_kv

    block_size = 16
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        use_kv_connector=mock_kv(matched_tokens=4 * block_size, is_async=True),
        block_size=block_size,
    )
    (request,) = create_requests(
        num_requests=1, num_tokens=10 * block_size, block_size=block_size
    )
    scheduler.add_request(request)
    output = scheduler.schedule()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert scheduler.dp_prefill_demand({}) == 0
    scheduler.update_from_output(
        output,
        create_model_runner_output([], finished_recving={request.request_id}),
    )
    left = request.num_prompt_tokens - request.num_computed_tokens
    assert left > 0
    assert scheduler.dp_prefill_demand({}) == left


def test_pull_up_is_capped_by_context_weighted_cost():
    """The exchange carries the step's context-weighted cost, and a pulled-up
    chunk stops where its added cost fills the gap to the busiest rank: with
    long context a token costs more, so fewer tokens are pulled than the
    token gap would allow."""
    from vllm.v1.core.sched import scheduler as sched_mod

    scheduler = _scheduler()
    requests = _running_chunked(scheduler)
    ex = _Exchange(peer=900)
    scheduler.set_dp_prefill_exchange(ex)
    sched_mod_w = sched_mod.DP_PREFILL_CTX_WEIGHT
    try:
        sched_mod.DP_PREFILL_CTX_WEIGHT = 1e-3  # make 200 tokens of context visible
        out = scheduler.schedule()
    finally:
        sched_mod.DP_PREFILL_CTX_WEIGHT = sched_mod_w
    w, ctx = 1e-3, 200
    own = 3 * 200 * (1 + w * (ctx + 199 / 2))
    assert ex.calls == [round(own)]
    first = out.num_scheduled_tokens[requests[0].request_id]
    gap = 900 - own
    cost = lambda n: n * (1 + w * (ctx + (n - 1) / 2))  # noqa: E731
    assert cost(first) - cost(200) <= gap < cost(first + 1) - cost(200)
    assert first < 500  # the token gap alone would have pulled to 500
