# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for sync EPLB through a receive buffer smaller than a layer
(the borrowed-scratch symm_mem path): staging passes and the rows they move."""

import numpy as np
import pytest
import torch

from vllm.distributed.eplb import rebalance_execute as rex
from vllm.distributed.eplb.eplb_communicator import _scratch_rows
from vllm.distributed.eplb.policy.default import DefaultEplbPolicy

from .test_eplb_layer_groups import FakePushCommunicator, _expert_rows


def _placements(ep, per_rank, layers, budget, seed):
    rng = np.random.default_rng(seed)
    physical = ep * per_rank
    old = np.stack([rng.permutation(physical) for _ in range(layers)])
    load = rng.lognormal(0, 1, (layers, physical))
    new = DefaultEplbPolicy.rebalance_experts(
        torch.from_numpy(load),
        physical,
        1,
        1,
        ep,
        torch.from_numpy(old),
        move_budget=budget,
    ).numpy()
    return old, new


@pytest.mark.parametrize("budget", [0, 32])
@pytest.mark.parametrize("ep,rows", [(8, 17), (16, 9), (8, 2), (16, 2)])
def test_staging_passes_respect_buffer_rows(ep, rows, budget):
    """K3.1-sized layers (896 experts): every pass stages at most `rows` rows
    per rank, every intermediate is a permutation, the last is the target."""
    per_rank = 896 // ep
    old, new = _placements(ep, per_rank, 4, budget, ep * 100 + rows)
    for layer in range(old.shape[0]):
        steps = rex.staging_passes(old[layer], new[layer], per_rank, rows)
        assert np.array_equal(steps[-1], new[layer])
        prev = old[layer]
        for step in steps:
            assert sorted(step.tolist()) == list(range(896))
            need = rex.buffer_rows_needed(prev, step, per_rank)
            assert max(len(r) for r in need) <= rows
            prev = step
        if budget:  # bounded swaps fit one pass when the buffer holds them
            need = rex.buffer_rows_needed(old[layer], new[layer], per_rank)
            if max(len(r) for r in need) <= rows:
                assert len(steps) == 1


def test_staging_passes_refuse_replicas_that_do_not_fit():
    old = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    new = np.array([3, 2, 1, 0, 3, 2, 1, 0])
    with pytest.raises(ValueError, match="permutation"):
        rex.staging_passes(old, new, 4, 2)


@pytest.mark.parametrize("budget", [0, 8])
@pytest.mark.parametrize("ep,rows", [(4, 2), (4, 3), (8, 5)])
def test_staged_transfer_reaches_target_placement(ep, rows, budget):
    """The sync rearrangement's per-pass flow (staging_passes + buffer_row_maps
    + move_to_buffer / move_from_buffer) over a `rows`-row push buffer leaves
    every rank's weights equal to the new placement."""
    per_rank, layers = 16, 5
    old, new = _placements(ep, per_rank, layers, budget, ep * 10 + rows)
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
        [torch.full((rows, w.shape[1]), -1.0) for w in weights[r][0]] for r in range(ep)
    ]
    comms = [FakePushCommunicator(buffers) for _ in range(ep)]
    for layer in range(layers):
        prev = old[layer]
        for step in rex.staging_passes(prev, new[layer], per_rank, rows):
            maps = rex.buffer_row_maps(
                rex.buffer_rows_needed(prev, step, per_rank), [0] * ep, per_rank
            )
            metas = [
                rex.move_to_buffer(
                    per_rank,
                    prev,
                    step,
                    weights[r][layer],
                    buffers[r],
                    None,
                    r,
                    comms[r],
                    layer,
                    buffer_maps=maps,
                    defer_execute=True,
                )
                for r in range(ep)
            ]
            for r in range(ep):
                comms[r].execute()
            for r in range(ep):
                rex.move_from_buffer(weights[r][layer], buffers[r], metas[r], step, r)
            for r in range(ep):  # nothing may rely on buffer contents across passes
                for buf in buffers[r]:
                    buf.fill_(-1.0)
            prev = step
    for r in range(ep):
        for layer in range(layers):
            for i in range(per_rank):
                expected = _expert_rows(new[layer, r * per_rank + i])
                for k in range(2):
                    assert torch.equal(weights[r][layer][k][i], expected[k])


def test_scratch_rows_fit():
    """Rows carved from a scratch span: every tensor's rows 256-B aligned and
    inside the span; never more rows than a layer has."""
    layer = [
        torch.empty((112, 1000), dtype=torch.uint8),
        torch.empty((112, 77), dtype=torch.float32),
    ]
    row = 1000 + 77 * 4
    for nbytes in (row - 1, row + 512, 10 * row + 600, 500 * row):
        rows, offsets = _scratch_rows(layer, nbytes)
        assert offsets[-1] <= nbytes or rows == 0
        assert all(o % 256 == 0 for o in offsets)
        assert rows <= 112
        if rows < 112:
            more = [0]
            for w in layer:
                more.append(more[-1] + -(-(rows + 1) * w[0].nbytes // 256) * 256)
            assert more[-1] > nbytes
