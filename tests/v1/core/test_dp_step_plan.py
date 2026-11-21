# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DP lockstep step cap policy (vllm/v1/core/sched/dp_step_plan.py)."""

import pytest

from vllm.v1.core.sched.dp_step_plan import choose_step_cap, step_cap_ladder

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
    # Most busy ranks saturated: no cap even with a few light ones.
    assert choose_step_cap([200_000] * 9 + [1000] * 7, LADDER) is None


def test_cap_holds_the_typical_busy_rank():
    assert choose_step_cap([1500] * 16, LADDER) == 2048
    assert choose_step_cap([3000] * 16, LADDER) == 4096
    assert choose_step_cap([6000] * 16, LADDER) == 8192
    assert choose_step_cap([10000] * 16, LADDER) == 12288
    # Idle ranks do not pull the cap down.
    assert choose_step_cap([0] * 10 + [7000] * 6, LADDER) == 8192


def test_outlier_backlog_does_not_set_every_step():
    # A long cold prompt on one rank (the mean would be ~8.4K per rank).
    assert choose_step_cap([100_000] + [3000] * 15, LADDER) == 4096
    # A few outliers: still the typical rank.
    assert choose_step_cap([100_000] * 3 + [2000] * 13, LADDER) == 2048


def test_cap_is_the_smallest_ladder_value_holding_the_median():
    for demands in (
        [12000, 0, 5000, 7000, 300, 0, 16000, 2000],
        [9000, 7000, 4000, 2500] * 4,
        [40000, 0, 0, 0],
    ):
        busy = sorted(d for d in demands if d > 0)
        med = busy[len(busy) // 2]
        cap = choose_step_cap(demands, LADDER)
        assert (cap or 16384) == min(c for c in LADDER if c >= min(med, 16384))
