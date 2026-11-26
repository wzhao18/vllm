# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the initial EPLB logical-to-physical map: its width follows the
physical capacity (not MAX_EXPERT_REDUNDANCY), its contents and its memory."""

import pytest
import torch

from vllm.distributed.eplb.eplb_state import MAX_EXPERT_REDUNDANCY, EplbState


def _maps(num_logical: int, num_redundant: int, capacity: int | None = None):
    p2l = torch.tensor(
        EplbState.build_initial_global_physical_to_logical_map(
            num_logical, num_redundant
        )
    )
    num_physical = p2l.shape[0]
    capacity = num_physical if capacity is None else capacity
    l2p, count = EplbState.build_initial_logical_maps(p2l, num_logical, capacity)
    return p2l, l2p, count


def _check_contents(p2l, l2p, count):
    num_logical = l2p.shape[0]
    assert int(count.sum()) == int((p2l >= 0).sum())
    for e in range(num_logical):
        slots = l2p[e, : count[e]].tolist()
        assert sorted(slots) == torch.nonzero(p2l == e).flatten().tolist()
        assert bool((l2p[e, count[e] :] == -1).all())


@pytest.mark.parametrize("num_logical", [64, 896])
def test_no_redundancy_width_one(num_logical):
    """R=0: one slot per logical expert; Kimi K3.1 (92 MoE layers x 896 experts)
    keeps 92 x 896 x 8 B = 0.63 MiB instead of 0.63 GiB."""
    p2l, l2p, count = _maps(num_logical, 0)
    assert l2p.shape == (num_logical, 1)
    assert torch.equal(l2p[:, 0], torch.arange(num_logical))
    assert bool((count == 1).all())
    _check_contents(p2l, l2p, count)
    layers = 92
    per_layer = l2p.numel() * l2p.element_size()
    assert per_layer * layers == layers * num_logical * 8
    old = layers * num_logical * (MAX_EXPERT_REDUNDANCY + 1) * 8
    assert per_layer * layers * 1024 == old


@pytest.mark.parametrize("num_logical,num_redundant", [(64, 8), (896, 16), (896, 4)])
def test_redundancy_width_is_capacity_bound(num_logical, num_redundant):
    """R>0: width = R + 1 (one logical expert may take every redundant slot)."""
    p2l, l2p, count = _maps(num_logical, num_redundant)
    assert l2p.shape == (num_logical, num_redundant + 1)
    assert l2p.numel() * l2p.element_size() == num_logical * (num_redundant + 1) * 8
    _check_contents(p2l, l2p, count)


def test_elastic_capacity_and_cap():
    """Capacity above the physical count (elastic EP: padded -1 slots) widens the
    map to capacity - logical + 1; the width never exceeds the old 1024."""
    num_logical, num_redundant = 64, 8
    p2l = torch.tensor(
        EplbState.build_initial_global_physical_to_logical_map(
            num_logical, num_redundant
        )
    )
    padded = torch.nn.functional.pad(p2l, (0, 56), value=-1)  # capacity 128
    l2p, count = EplbState.build_initial_logical_maps(padded, num_logical, 128)
    assert l2p.shape == (num_logical, 128 - num_logical + 1)
    _check_contents(padded, l2p, count)
    l2p, _ = EplbState.build_initial_logical_maps(p2l, num_logical, num_logical + 5000)
    assert l2p.shape[1] == MAX_EXPERT_REDUNDANCY + 1
