# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prompt chunks sized by predicted cost (DEP prefill lockstep): a chunk at long
context takes fewer tokens of the step budget, short-context chunks are
unchanged, and the Kimi K3 cost table is the measured PSOL step time."""

import pytest

from vllm.models.kimi_k3.nvidia.specialized_prefill import step_cost

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


class _Cost:
    """Test cost model: a token at context ctx costs 1 + ctx / 64."""

    @staticmethod
    def chunk_cost(n, ctx):
        return n * (1 + ctx / 64) if n > 0 else 0.0

    @staticmethod
    def prompt_cost(n, ctx):
        return n * (1 + (ctx + n // 2) / 64) if n > 0 else 0.0

    @staticmethod
    def max_tokens_for_cost(cost, ctx):
        return int(cost / (1 + ctx / 64) + 1e-9) if cost > 0 else 0


def test_step_cost_weight():
    assert step_cost.weight(0) == 1.0
    assert step_cost.weight(130_000) == pytest.approx(1.195)
    assert step_cost.weight(600_000) == pytest.approx(1.9)
    ws = [step_cost.weight(c) for c in range(0, 1_000_000, 10_000)]
    assert ws == sorted(ws)
    # A 16K budget at 600K context buys 16384 / 1.9 tokens.
    n = step_cost.max_tokens_for_cost(16384, 600_000)
    assert n == int(16384 / 1.9)
    assert step_cost.chunk_cost(n, 600_000) <= 16384 < step_cost.chunk_cost(
        n + 1, 600_000
    )


def test_no_cost_model_keeps_token_chunks():
    scheduler = create_scheduler(max_num_batched_tokens=64, long_prefill_token_threshold=0)
    assert scheduler._prompt_cost is None
    (req,) = create_requests(num_requests=1, num_tokens=256)
    scheduler.add_request(req)
    sizes = [scheduler.schedule().num_scheduled_tokens[req.request_id] for _ in range(3)]
    assert sizes == [64, 64, 64]


def test_long_context_chunk_takes_fewer_tokens():
    scheduler = create_scheduler(max_num_batched_tokens=64, long_prefill_token_threshold=0)
    scheduler._prompt_cost = _Cost
    (req,) = create_requests(num_requests=1, num_tokens=256)
    scheduler.add_request(req)
    sizes = [scheduler.schedule().num_scheduled_tokens[req.request_id] for _ in range(3)]
    # ctx 0: weight 1 -> 64; ctx 64: weight 2 -> 32; ctx 96: weight 2.5 -> 25,
    # all rounded down to the block size (16) by the scheduler's chunk rules
    # where they apply; at most the budget's cost in every step.
    assert sizes[0] == 64
    assert sizes[1] == 32
    assert sizes[2] <= 25
    assert _Cost.chunk_cost(sizes[2], 96) <= 64


def test_cut_and_cost_helpers():
    scheduler = create_scheduler()
    scheduler._prompt_cost = _Cost
    (req,) = create_requests(num_requests=1, num_tokens=200)
    # Fits: unchanged.
    assert scheduler._cut_prompt_chunk_to_cost(req, 0, 50, 64.0) == 50
    # ctx 64 (weight 2): 64 cost buys 32 tokens.
    assert scheduler._cut_prompt_chunk_to_cost(req, 64, 50, 64.0) == 32
    # Past the prompt (decode): never cut, costs 1 per token.
    assert scheduler._cut_prompt_chunk_to_cost(req, 200, 4, 1.0) == 4
    assert scheduler._prompt_chunk_cost(req, 200, 4) == 4.0
    # Chunk ending past the prompt: prompt part by cost, the rest 1 each.
    assert scheduler._prompt_chunk_cost(req, 192, 10) == pytest.approx(8 * 4.0 + 2)
    assert scheduler._cut_prompt_chunk_to_cost(req, 0, 0, 0.0) == 0


def test_planning_units_are_predicted_cost():
    scheduler = create_scheduler(max_num_batched_tokens=64, long_prefill_token_threshold=0)
    reqs = create_requests(num_requests=2, num_tokens=256)
    # Token units without a cost model.
    for r in reqs:
        scheduler.add_request(r)
    assert scheduler.dp_movable_requests(4) == [(r.request_id, 256) for r in reqs]
    assert scheduler.dp_move_demand() == 0
    scheduler._prompt_cost = _Cost
    # A queued prompt: all 256 tokens from context 0, midpoint weight 1 + 128/64.
    assert scheduler.dp_movable_requests(4) == [(r.request_id, 768) for r in reqs]
    scheduler.schedule()  # reqs[0] runs its first 64 tokens
    assert [rid for rid, _ in scheduler.dp_movable_requests(4)] == [reqs[1].request_id]
    # Committed: reqs[0]'s 192 tokens left from context 64: 192 * (1 + 160/64).
    assert scheduler.dp_move_demand() == 672
    # During the step's exchange the step's own tokens are already counted.
    scheduler._dp_scheduled_now = {reqs[0].request_id: 64}
    assert scheduler.dp_move_demand() == int(128 * (1 + (128 + 64) / 64) + 0.5)
    scheduler._dp_scheduled_now = {}
    assert step_cost.prompt_cost(1000, 600_000) == pytest.approx(
        1000 * step_cost.weight(600_500)
    )


def test_moved_in_requests_stay_and_count_in_demand():
    scheduler = create_scheduler(max_num_batched_tokens=64, long_prefill_token_threshold=0)
    reqs = create_requests(num_requests=3, num_tokens=256)
    for r in reqs:
        scheduler.add_request(r)
    # reqs[0] was moved here (pinned with its planned size): never offered
    # again, but its size counts in this rank's demand.
    reqs[0].dp_pinned_size = 500
    listed = scheduler.dp_movable_requests(1)
    assert listed == [(reqs[1].request_id, 256)]
    # Demand: the pinned request + reqs[2] (past the movable slots); not the
    # listed one (the planner adds it).
    assert scheduler.dp_move_demand() == 500 + 256


def test_cost_cut_never_below_one_mamba_block():
    """Rhea livelock (E62p prt, 2026-11-22): with a 16384-token Mamba alignment
    block, a cost cut of a running prompt's next chunk to slightly under 16384
    aligned to 0 tokens every step, so 32 running prompts held every slot and
    never progressed. The cut keeps at least one whole Mamba block."""
    scheduler = create_scheduler()
    scheduler._prompt_cost = _Cost
    (req,) = create_requests(num_requests=1, num_tokens=200)
    scheduler.need_mamba_block_aligned_split = True
    scheduler.mamba_block_size = 48
    # ctx 64 (weight 2): 64 cost buys 32 tokens < one 48-token block -> 48.
    assert scheduler._cut_prompt_chunk_to_cost(req, 64, 100, 64.0) == 48
    # A smaller request remainder is never exceeded.
    assert scheduler._cut_prompt_chunk_to_cost(req, 64, 40, 10.0) == 40
    # Above one block the cost cut applies as before.
    assert scheduler._cut_prompt_chunk_to_cost(req, 64, 100, 128.0) == 64
