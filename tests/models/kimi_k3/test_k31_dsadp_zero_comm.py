# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSA-DP SP decode's DEP-ordered residual stream (``sp_attention.DepLayout``):
each card's shard of the DEP-ordered rows is its own DEP segment (its requests, then
padding requests), the final gather's inverse index puts the rows back into the batch
order, the KDA cores' per-request metadata and the MegaMoE padding mask follow the same
order, and a captured step reads the order from the runner's DEP_PERM buffers."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.models.common.ops import sequence_parallel as sp
from vllm.models.kimi_k3.nvidia.ops import dsa_dp
from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import sp_attention
from vllm.models.kimi_k3.nvidia.specialized_decode.shapes import Unsupported

TP, Q = 4, 6
OWNERS = np.array([2, 0, 2, 3, 2, 1, 0])  # 7 requests; card 2 owns three
ROWS = TP * Q * 3  # three requests per card: the busiest owner's count


def _perm(owners=OWNERS, rows=ROWS, capacity=ROWS + 2 * TP * Q):
    rows_, reqs, inv = dsa_dp.dep_order(owners, rows, TP, Q)
    perm = dsa_dp.DepPerm()
    buf = torch.arange(capacity, dtype=torch.int64).repeat(3)
    buf[:rows] = torch.from_numpy(rows_)
    buf[capacity : capacity + reqs.size] = torch.from_numpy(reqs)
    buf[2 * capacity : 2 * capacity + rows] = torch.from_numpy(inv)
    perm.buf, perm.q = buf, Q
    return perm


def _layout(monkeypatch, rank, positions=None):
    monkeypatch.setattr(dsa_dp, "DEP_PERM", _perm())
    if positions is None:
        positions = torch.arange(ROWS) + 1000
    return sp_attention.DepLayout(ROWS, Q, TP, rank, positions)


@pytest.mark.parametrize("rank", range(TP))
def test_shard_is_the_cards_segment_and_inverse_restores_order(monkeypatch, rank):
    layout = _layout(monkeypatch, rank)
    x = torch.arange(ROWS)  # a batch-order row id per row
    dep = x.index_select(0, layout.rows)  # the residual stream in DEP order
    shard = dep[rank * ROWS // TP : (rank + 1) * ROWS // TP]
    assert torch.equal(shard, layout.segment)
    owned = [int(r) // Q for r in shard[::Q] if int(r) // Q < OWNERS.size]
    assert owned == list(np.flatnonzero(rank == OWNERS))  # its requests, then padding
    assert all(int(r) // Q >= OWNERS.size for r in shard[::Q][len(owned) :])
    assert torch.equal(dep.index_select(0, layout.inverse), x)  # final gather
    assert torch.equal(layout.positions, layout.segment + 1000)


def test_kda_metadata_follows_the_dep_request_order(monkeypatch):
    layout = _layout(monkeypatch, 0)
    reqs = ROWS // Q
    flushed = []
    deferred = SimpleNamespace(
        record_rows=torch.arange(reqs) * 10,
        pending=torch.zeros(3),
        flush=lambda: flushed.append(True),
    )
    md = SimpleNamespace(
        spec_state_indices_tensor=torch.arange(reqs)[:, None] + 100,
        num_accepted_tokens=torch.arange(reqs) + 1,
        recoverssm_deferred=deferred,
    )
    dep_md = layout.kda_metadata(md)
    assert layout.kda_metadata(md) is dep_md  # once per metadata object
    order = layout.requests
    assert torch.equal(dep_md.spec_state_indices_tensor[:, 0], order + 100)
    assert torch.equal(dep_md.num_accepted_tokens, order + 1)
    assert torch.equal(dep_md.recoverssm_deferred.record_rows, order * 10)
    assert dep_md.recoverssm_deferred.pending is deferred.pending
    dep_md.recoverssm_deferred.flush()
    assert flushed == [True]


@pytest.mark.parametrize("rank", range(TP))
def test_padding_mask_follows_the_shard(monkeypatch, rank):
    layout = _layout(monkeypatch, rank)
    monkeypatch.setattr(sp, "get_tensor_model_parallel_world_size", lambda: TP)
    monkeypatch.setattr(sp, "get_tensor_model_parallel_rank", lambda: rank)
    is_padding = torch.arange(ROWS) >= OWNERS.size * Q  # padding requests last
    mask = sp.sp_padding_mask(
        is_padding.index_select(0, layout.rows), torch.empty(ROWS, 1)
    )
    assert torch.equal(mask, layout.segment >= OWNERS.size * Q)


def test_refuses_steps_that_are_not_whole_segments(monkeypatch):
    monkeypatch.setattr(dsa_dp, "DEP_PERM", _perm())
    with pytest.raises(Unsupported, match="DEP segments"):
        sp_attention.DepLayout(ROWS - Q, Q, TP, 0, torch.arange(ROWS - Q))
    with pytest.raises(Unsupported, match="no DEP order"):
        sp_attention.DepLayout(ROWS, Q + 1, TP, 0, torch.arange(ROWS))
