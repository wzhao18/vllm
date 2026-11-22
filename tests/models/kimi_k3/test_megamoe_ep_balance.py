# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Served EP-balance ring (megamoe_rubin/ep_balance.py): slot layout, drain order, per-rank loads."""

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned host memory needs CUDA")


def _write(ring, call, sizes, world, epr, t=123456789):
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import ep_balance as E

    i = (call >> E.SHIFT) % E.SLOTS
    base = i * E.SLOT_WORDS
    buf = ring.buffer
    buf[base + E.HEADER_WORDS : base + E.HEADER_WORDS + len(sizes)] = torch.tensor(sizes, dtype=torch.int32)
    buf[base + 1] = t & 0x7FFFFFFF
    buf[base + 2] = 0
    buf[base + 3] = len(sizes)
    buf[base + 4] = world
    buf[base + 5] = epr
    buf[base] = call  # written last by the kernel


@cuda
def test_drain_returns_completed_calls_in_order_with_rank_loads():
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import ep_balance as E

    ring = E.EpBalanceRing("test")
    assert ring.drain() == []
    # 4 ranks x 2 experts: rank loads [3, 7, 0, 6] -> mean 4, max/mean 1.75
    _write(ring, 64, [1, 2, 3, 4, 0, 0, 5, 1], world=4, epr=2)
    _write(ring, 128, [2, 2, 2, 2, 2, 2, 2, 2], world=4, epr=2)
    out = ring.drain()
    assert [r["call"] for r in out] == [64, 128]
    assert out[0]["rank_loads"] == [3, 7, 0, 6] and out[0]["routes"] == 16
    assert out[0]["max_over_mean"] == pytest.approx(1.75) and out[0]["expert_max"] == 5
    assert out[1]["max_over_mean"] == pytest.approx(1.0)
    assert ring.drain() == []  # nothing new


@cuda
def test_slot_being_rewritten_waits_for_the_next_drain():
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import ep_balance as E

    ring = E.EpBalanceRing("test")
    _write(ring, 64, [1, 1], world=2, epr=1)
    assert [r["call"] for r in ring.drain()] == [64]
    # call 128's slot: payload written, call word not yet (the kernel writes it last)
    i = (128 >> E.SHIFT) % E.SLOTS
    ring.buffer[i * E.SLOT_WORDS + E.HEADER_WORDS] = 9
    assert ring.drain() == []
    _write(ring, 128, [9, 1], world=2, epr=1)
    out = ring.drain()
    assert [r["call"] for r in out] == [128] and out[0]["rank_loads"] == [9, 1]
