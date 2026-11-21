# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DP lockstep step cap policy (vllm/v1/core/sched/dp_step_plan.py)."""

import pytest

from vllm.v1.core.sched.dp_step_plan import (
    STEP_CAP_COVERAGE,
    choose_step_cap,
    step_cap_ladder,
)

pytestmark = pytest.mark.cpu_test

LADDER = [2048, 4096, 8192, 12288, 16384]


def test_ladder():
    assert step_cap_ladder(2048, 16384) == LADDER
    assert step_cap_ladder(2048, 8192) == [2048, 4096, 6144, 8192]
    assert step_cap_ladder(2048, 2048) == [2048]
    assert step_cap_ladder(4096, 1024) == [1024]


def test_idle_group_takes_the_smallest_cap():
    assert choose_step_cap([0] * 16, LADDER) == 2048


def test_saturated_group_is_uncapped():
    assert choose_step_cap([16384] * 16, LADDER) is None
    assert choose_step_cap([200_000] * 16, LADDER) is None
    # 15 of 16 ranks saturated: a 12K cap keeps only 75% of the work.
    assert choose_step_cap([200_000] * 15 + [0], LADDER) is None


def test_light_group_gets_short_steps():
    # Every rank's work fits a short step.
    assert choose_step_cap([1500] * 16, LADDER) == 2048
    assert choose_step_cap([3000] * 16, LADDER) == 4096


def test_one_heavy_rank_does_not_pace_the_group():
    # One rank with a full chunk and 15 with a little work: the 4K step still
    # runs >= 80% of what the 16K step would (15 x 3500 + 4096 vs + 16384).
    assert choose_step_cap([16384] + [3500] * 15, LADDER) == 4096
    # Two heavy ranks among light ones: 8K.
    assert choose_step_cap([30000, 30000] + [1000] * 14, LADDER) == 8192


def test_cap_is_the_smallest_ladder_value_with_the_coverage():
    for demands in (
        [12000, 0, 5000, 7000, 300, 0, 16000, 2000],
        [9000, 7000, 4000, 2500] * 4,
        [40000, 0, 0, 0],
    ):
        full = sum(min(d, 16384) for d in demands)
        cap = choose_step_cap(demands, LADDER)
        ok = [
            c
            for c in LADDER[:-1]
            if sum(min(d, c) for d in demands) >= STEP_CAP_COVERAGE * full
        ]
        assert cap == (ok[0] if ok else None)
