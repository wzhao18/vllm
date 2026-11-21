# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for grouped async EPLB transfers and the move-budget policy."""

import numpy as np
import pytest
import torch

from vllm.distributed.eplb import rebalance_execute as rex
from vllm.distributed.eplb.async_worker import plan_layer_groups
from vllm.distributed.eplb.policy.default import DefaultEplbPolicy

NUM_LOGICAL = 64


class FakePushCommunicator:
    """Emulates SymmMemEplbCommunicator: senders write the receivers' buffer
    rows at execute()."""

    needs_recv_rows = True

    def __init__(self, buffers):
        self.buffers = buffers
        self.sends = []
        self.recv_rows = None

    def set_transfer_context(self, old, layer_idx, recv_rows=None):
        self.recv_rows = recv_rows

    def add_send(self, tensors, dst, expert_id):
        row = self.recv_rows[dst][expert_id]
        self.sends.append(([t.clone() for t in tensors], dst, row))

    def add_recv(self, tensors, src, expert_id):
        pass

    def execute(self):
        for tensors, dst, row in self.sends:
            for t, buf in zip(tensors, self.buffers[dst]):
                buf[row].copy_(t)
        self.sends.clear()


def _expert_rows(expert: int) -> list[torch.Tensor]:
    return [torch.full((w,), float(expert * 10 + i)) for i, w in enumerate((5, 3))]


def _balancedness(placement: np.ndarray, load: np.ndarray, ep: int) -> float:
    replicas = np.bincount(placement, minlength=len(load))
    per_rank = (load[placement] / replicas[placement]).reshape(ep, -1).sum(1)
    return per_rank.mean() / per_rank.max()


@pytest.mark.parametrize("budget", [None, 8])
@pytest.mark.parametrize("ep,redundant", [(4, 4), (8, 8), (4, 0)])
def test_grouped_transfer_reaches_target_placement(ep, redundant, budget, monkeypatch):
    """Several layers staged in one receive buffer with a deferred execute
    must leave every rank's weights equal to the new placement."""
    layers = 6
    rng = np.random.default_rng(0)
    physical = NUM_LOGICAL + redundant
    per_rank = physical // ep
    old = np.stack(
        [
            np.concatenate(
                [np.arange(NUM_LOGICAL), rng.integers(0, NUM_LOGICAL, redundant)]
            )[rng.permutation(physical)]
            for _ in range(layers)
        ]
    )
    load = rng.lognormal(0, 1, (layers, NUM_LOGICAL))
    new = DefaultEplbPolicy.rebalance_experts(
        torch.from_numpy(load),
        physical,
        1,
        1,
        ep,
        torch.from_numpy(old),
        move_budget=budget or 0,
    ).numpy()

    weights = [
        [
            [
                torch.stack(
                    [
                        _expert_rows(old[layer, r * per_rank + i])[k]
                        for i in range(per_rank)
                    ]
                )
                for k in range(2)
            ]
            for layer in range(layers)
        ]
        for r in range(ep)
    ]
    buffers = [
        [torch.full_like(weights[r][0][k], -1.0) for k in range(2)] for r in range(ep)
    ]
    comms = [FakePushCommunicator(buffers) for _ in range(ep)]

    for group_layers, buffer_maps in plan_layer_groups(old, new, per_rank):
        metas = {
            r: [
                rex.move_to_buffer(
                    per_rank,
                    old[layer],
                    new[layer],
                    weights[r][layer],
                    buffers[r],
                    None,
                    r,
                    comms[r],
                    layer,
                    buffer_maps=m,
                    defer_execute=True,
                )
                for layer, m in zip(group_layers, buffer_maps)
            ]
            for r in range(ep)
        }
        for r in range(ep):
            comms[r].execute()
        for r in range(ep):
            for layer, meta in zip(group_layers, metas[r]):
                rex.move_from_buffer(weights[r][layer], buffers[r], meta, new[layer], r)
        # Buffer contents must not leak across groups.
        for r in range(ep):
            for buf in buffers[r]:
                buf.fill_(-1.0)

    for r in range(ep):
        for layer in range(layers):
            for i in range(per_rank):
                expected = _expert_rows(new[layer, r * per_rank + i])
                for k in range(2):
                    assert torch.equal(weights[r][layer][k][i], expected[k])


@pytest.mark.parametrize(
    "ep,redundant,min_balancedness",
    [(4, 4, 0.99), (8, 16, 0.99), (12, 4, 0.99), (20, 4, 0.97), (24, 16, 0.97)],
)
def test_move_budget_policy(ep, redundant, min_balancedness, monkeypatch):
    """With move_budget=32 on K3.1-sized layers (896 experts),
    including EP sizes that need redundant slots to divide: valid placement,
    at most 32 changed slots per layer, and near-balanced load."""
    num_logical, layers = 896, 8
    physical = num_logical + redundant
    rng = np.random.default_rng(ep)
    load = rng.lognormal(0, 1, (layers, num_logical))
    old = np.tile(np.arange(physical) % num_logical, (layers, 1))
    new = DefaultEplbPolicy.rebalance_experts(
        torch.from_numpy(load), physical, 1, 1, ep, torch.from_numpy(old), move_budget=32
    ).numpy()

    assert new.shape == old.shape
    for layer in range(layers):
        assert len(np.unique(new[layer])) == num_logical
        assert new[layer].min() >= 0 and new[layer].max() < num_logical
    assert (new != old).sum(1).max() <= 32
    after = np.mean(
        [_balancedness(new[layer], load[layer], ep) for layer in range(layers)]
    )
    assert after >= min_balancedness


def test_move_budget_defaults_per_communicator():
    """The move budget is a validated default for the symm_mem async path and
    unbounded for the other communicators unless set explicitly."""
    from vllm.config.parallel import EPLBConfig

    assert EPLBConfig(communicator="symm_mem").resolved_move_budget(8) == 32
    assert EPLBConfig(communicator="symm_mem").resolved_move_budget(24) == 64
    assert EPLBConfig(communicator="nixl").resolved_move_budget(24) == 0
    assert EPLBConfig(communicator=None).resolved_move_budget(8) == 0
    assert EPLBConfig(communicator="nixl", move_budget=16).resolved_move_budget(8) == 16
    with pytest.raises(ValueError):
        EPLBConfig(move_budget=-1)
