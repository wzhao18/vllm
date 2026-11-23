# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 selective prefill indexer row grouping (indexer_step.pack_groups): every
row lands in exactly one group in token order with its own key count, groups stay in
the scratch budget, and groups hold whole waves of work items, so a layer call
costs the fewest waves its rows allow. CPU only."""

import pytest

pytest.importorskip("cutlass")

from vllm.models.kimi_k3.nvidia.specialized_prefill import (  # noqa: E402
    indexer as pidx,
)
from vllm.models.kimi_k3.nvidia.specialized_prefill import (  # noqa: E402
    indexer_step as pstep,
)

SMS = 212


def _segs(reqs, chunk_elems=(512 << 20) // 2):
    """Stock-chunker segments (chunk, request in chunk, rows, last row's keys, key
    count, request) of requests (rows, ctx), in token order."""
    segs, ci = [], 0
    for r, (rows, ctx) in enumerate(reqs):
        cr = max(1, chunk_elems // ctx)
        for o in range(0, rows, cr):
            n = min(cr, rows - o)
            segs.append((ci, 0, n, ctx - rows + o + n, ctx, r))
            ci += 1
    return segs


def _row_lens(segs):
    return [(s[5], k) for s in segs for k in range(s[3] - s[2] + 1, s[3] + 1)]


def _items(group):
    return sum(-(-s[2] // pidx.BQ) for s in group)


@pytest.mark.parametrize(
    "reqs",
    [
        [(8192, 100_000), (8192, 100_000)],
        [(8192, 200_000), (8192, 200_000)],
        [(8192, 400_000), (8192, 400_000)],
        [(8192, 600_000), (8192, 600_000)],
        [(3000, 650_000), (500, 30_000), (12000, 120_000)],
        [(37, 5_000), (8192, 300_000), (1, 900_000)],
    ],
)
def test_pack_groups(reqs):
    segs = _segs(reqs)
    groups = pstep.pack_groups(segs, SMS)
    flat = [s for g in groups for s in g]
    # every row once, in token order, with its own key count
    assert _row_lens(flat) == _row_lens(segs)
    total_items = 0
    for g in groups:
        pitch = max(-(-s[4] // pidx.TILE) * pidx.TILE for s in g)
        rows = sum(s[2] for s in g)
        assert pstep.group_bytes(rows, pitch) <= pstep.SCRATCH_BUDGET
        cap = pstep.wave_cap_items(pitch, SMS)
        if cap:
            assert _items(g) <= cap
        total_items += _items(g)
    # whole-wave groups: only the last group may hold a partial wave
    if all(pstep.wave_cap_items(-(-max(s[4] for s in g) // pidx.TILE) * pidx.TILE, SMS)
           for g in groups):
        waves = sum(-(-_items(g) // SMS) for g in groups)
        assert waves <= -(-total_items // SMS) + 1


def test_wave_count_vs_budget_only_grouping():
    """At 400K and 600K, wave packing takes 2 x 8192 rows to the minimum of 10 waves
    (budget-only grouping at chunk granularity: 12 and 11)."""
    for ctx in (400_000, 600_000):
        groups = pstep.pack_groups(_segs([(8192, ctx), (8192, ctx)]), SMS)
        assert sum(-(-_items(g) // SMS) for g in groups) == 10
