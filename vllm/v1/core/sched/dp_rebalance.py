# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token balancing of uncommitted prompt work across the DP ranks of one engine.

DP ranks of an EP group step in lockstep, and a step costs the busiest rank's
tokens. Requests are bound to a rank when they arrive, so one rank can hold
queued prompts while another has nothing to compute. ``plan_dp_moves`` decides,
from what every rank reported at the step's DP exchange, which not-yet-committed
requests (see ``Scheduler.dp_movable_requests``) to hand to which rank so that
the ranks' computable tokens approach the same per-step target.

The plan is a pure function of the exchanged numbers, so every rank computes the
same plan without further communication. It never moves a request a rank needs
to reach the target itself and never moves committed work.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DPMove:
    src: int
    # Index into the source rank's reported movable list (queue order).
    index: int
    dst: int
    num_tokens: int


def plan_dp_moves(
    demand: list[int],
    movable: list[list[int]],
    step_budget: int,
    max_moves_per_rank: int = 2,
    min_gain: int = 2048,
    penalties: list[list[int]] | None = None,
) -> list[DPMove]:
    """Plan cross-rank moves of uncommitted requests.

    Args:
        demand: per rank, prompt tokens it can already compute (committed:
            running prefills and waiting requests whose prefix is in place).
        movable: per rank, prompt-token sizes of its uncommitted waiting
            requests in queue order (the order the rank would admit them).
        step_budget: per-rank token budget of one step (the unit in which a
            rank's queued work turns into steps; kept for the bound below).
        max_moves_per_rank: at most this many requests leave or enter a rank
            per plan (bounds per-step migration work).
        min_gain: a move must lower the larger of the two ranks' outstanding
            loads by at least this many tokens.

    Places all movable work by longest-processing-time-first on top of each
    rank's committed demand (largest request to the least-loaded rank), which
    keeps the ranks' outstanding work, and so the lockstep steps' busiest rank,
    as even as whole requests allow. A request stays home unless moving it
    lowers the larger of the two ranks' loads by at least ``min_gain``, and
    only the moves (home != placed rank) are returned, in a deterministic order.
    """
    n = len(demand)
    assert len(movable) == n
    if n < 2 or step_budget <= 0 or not any(movable):
        return []
    load = list(demand)
    items = sorted(
        ((size, r, i) for r in range(n) for i, size in enumerate(movable[r])),
        key=lambda t: (-t[0], t[1], t[2]),
    )

    def penalty(r: int, i: int) -> int:
        # Extra gain a move must clear (e.g. reloading a prefix the home
        # rank's GPU cache already holds).
        if penalties is None or i >= len(penalties[r]):
            return 0
        return penalties[r][i]

    left = [0] * n
    entered = [0] * n
    moves: list[DPMove] = []
    for size, home, index in items:
        best = min(range(n), key=lambda r: (load[r], r))
        move_ok = (
            best != home
            and left[home] < max_moves_per_rank
            and entered[best] < max_moves_per_rank
            # Worth it: the larger of the two end loads drops by at least
            # min_gain tokens (one CUDA-graph step's worth by default).
            and max(load[home] + size, load[best])
            - max(load[home], load[best] + size)
            >= max(min_gain, penalty(home, index))
        )
        if move_ok:
            load[best] += size
            left[home] += 1
            entered[best] += 1
            moves.append(DPMove(home, index, best, size))
        else:
            load[home] += size
    moves.sort(key=lambda m: (m.src, m.index))
    return moves
