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
    node_of: list[int] | None = None,
    cross_min_gain: int | None = None,
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
        node_of: per rank, its node (client); None = all on one node.
        cross_min_gain: the gain a move to another node must reach (default
            min_gain); a cross-node move also wins over the best same-node
            move only by more than the difference.

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
    left = [0] * n
    entered = [0] * n
    moves: list[DPMove] = []
    node = node_of if node_of is not None else [0] * n
    cross_gain = min_gain if cross_min_gain is None else max(cross_min_gain, min_gain)

    def gain(home: int, dst: int, size: int) -> int:
        # The larger of the two end loads drops by this much.
        return max(load[home] + size, load[dst]) - max(load[home], load[dst] + size)

    for size, home, index in items:
        best = None
        if left[home] < max_moves_per_rank:
            # The least-loaded rank on the home node, and overall. A move off the
            # home node must clear cross_min_gain (it costs an output hop back to
            # the home node's client) and wins only by more than that premium.
            local = min(
                (r for r in range(n) if node[r] == node[home]), key=lambda r: (load[r], r)
            )
            anywhere = min(range(n), key=lambda r: (load[r], r))
            score = None
            for dst in (local, anywhere):
                if dst == home or entered[dst] >= max_moves_per_rank:
                    continue
                cross = node[dst] != node[home]
                g = gain(home, dst, size)
                if g < (cross_gain if cross else min_gain):
                    continue
                sc = g - (cross_gain - min_gain if cross else 0)
                if score is None or sc > score:
                    best, score = dst, sc
        move_ok = best is not None
        if move_ok:
            load[best] += size
            left[home] += 1
            entered[best] += 1
            moves.append(DPMove(home, index, best, size))
        else:
            load[home] += size
    moves.sort(key=lambda m: (m.src, m.index))
    return moves
