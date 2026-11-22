# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""plan_dp_moves: token balancing of uncommitted prompts across DP ranks."""

import pytest

from vllm.v1.core.sched.dp_rebalance import DPMove, plan_dp_moves

pytestmark = pytest.mark.cpu_test


def _loads(demand, movable, moves):
    load = list(demand)
    for r, m in enumerate(movable):
        load[r] += sum(m)
    for mv in moves:
        load[mv.src] -= mv.num_tokens
        load[mv.dst] += mv.num_tokens
    return load


def test_idle_rank_takes_queued_work_from_busy_rank():
    # Rank 0 runs a full step and queues two more prompts; rank 1 is idle.
    moves = plan_dp_moves([16384, 0], [[6000, 5000], []], step_budget=16384)
    assert moves and all(m.src == 0 and m.dst == 1 for m in moves)
    assert sum(m.num_tokens for m in moves) == 11000


def test_balanced_group_moves_nothing():
    assert plan_dp_moves([4000, 4000], [[1000], [1000]], step_budget=16384) == []


def test_source_keeps_queue_head_when_split_evenly():
    # Two equal prompts on rank 0, rank 1 idle: one moves (the later one
    # placed second goes to the lighter rank), one stays home.
    moves = plan_dp_moves([0, 0], [[3000, 3000], []], step_budget=16384)
    assert len(moves) == 1 and moves[0].src == 0 and moves[0].dst == 1


def test_committed_work_never_moves_and_plan_is_deterministic():
    demand = [12000, 0, 2000, 0]
    movable = [[4000, 4000, 2000], [], [1000], []]
    a = plan_dp_moves(demand, movable, step_budget=16384)
    b = plan_dp_moves(list(demand), [list(m) for m in movable], step_budget=16384)
    assert a == b
    for mv in a:
        assert 0 <= mv.index < len(movable[mv.src])
        assert movable[mv.src][mv.index] == mv.num_tokens
    # Imbalance (max load / mean load) does not get worse.
    before = _loads(demand, movable, [])
    after = _loads(demand, movable, a)
    assert max(after) <= max(before)


def test_moves_per_rank_bounded():
    moves = plan_dp_moves(
        [16384, 0, 0, 0], [[100] * 20, [], [], []], step_budget=16384,
        max_moves_per_rank=2,
    )
    assert sum(1 for m in moves if m.src == 0) <= 2
    for r in (1, 2, 3):
        assert sum(1 for m in moves if m.dst == r) <= 2


def test_queued_prompt_moves_to_idle_rank():
    # Rank 0 has 8K committed and a queued 12K prompt; rank 1 is idle: the
    # 12K prompt goes to rank 1 (larger end load 20K -> 12K).
    moves = plan_dp_moves([8000, 0], [[12000], []], step_budget=16384)
    assert moves == [DPMove(src=0, index=0, dst=1, num_tokens=12000)]


def test_one_huge_prompt_alone_stays():
    # A lone 60K prompt sets the makespan wherever it runs: no move.
    assert plan_dp_moves([1000, 0], [[60000], []], step_budget=16384) == []


def test_small_gain_stays_home():
    # Moving 1K to save little is not worth a migration.
    assert plan_dp_moves([5000, 4800], [[1000], []], step_budget=16384) == []
