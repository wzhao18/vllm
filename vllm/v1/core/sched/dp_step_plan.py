# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DP lockstep step cap for prefill engines (DEP prefill role).

The DP ranks of an expert-parallel group run their steps in lockstep: every
MoE layer waits for the rank with the most tokens, so a step costs the busiest
rank's chunk. A prompt waits for whole steps (its admission, then one step per
chunk), so while most ranks have little schedulable work, a full-budget step
paced by one rank's long chunk makes every request on every rank wait for it.

Each step the DP ranks exchange their schedulable prompt tokens (see
``Scheduler.dp_prefill_demand``) and every rank sets the same token cap for
its next steps: the smallest ladder value whose step still runs at least
``STEP_CAP_COVERAGE`` of the tokens a full-budget step would run over the
group. Light or uneven groups get shorter steps at a bounded loss of work per
step; a saturated group (every rank with a full budget of work) stays at the
configured budget.
"""

from collections.abc import Sequence

# Share of a full-budget step's tokens (summed over the DP group) that a
# capped step must still run.
STEP_CAP_COVERAGE = 0.8


def step_cap_ladder(floor: int, max_tokens: int) -> list[int]:
    """Candidate caps: ``floor`` (the largest CUDA-graph batch) doubled up to
    the budget, three quarters of the budget, and the budget (= no cap)."""
    floor = max(min(floor, max_tokens), 1)
    ladder = {max_tokens}
    x = floor
    while x < max_tokens:
        ladder.add(x)
        x *= 2
    if max_tokens * 3 // 4 > floor:
        ladder.add(max_tokens * 3 // 4)
    return sorted(ladder)


def choose_step_cap(
    demands: Sequence[int],
    ladder: Sequence[int],
    coverage: float = STEP_CAP_COVERAGE,
) -> int | None:
    """Token cap of the next steps for every DP rank of the group, None for
    the configured budget (``ladder[-1]``).

    ``demands``: each rank's schedulable prompt tokens. An idle group gets the
    smallest cap: the next arrivals start with a short step, after which the
    cap follows their demand.
    """
    budget = ladder[-1]
    full = sum(min(d, budget) for d in demands)
    for cap in ladder[:-1]:
        if sum(min(d, cap) for d in demands) >= coverage * full:
            return cap
    return None
