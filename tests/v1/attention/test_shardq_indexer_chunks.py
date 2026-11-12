# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ShardQ indexer prefill chunking (dcp_shardq.context._rank_chunk_specs):
chunks bound the logits a rank holds (its fold rows x the row width), not the
chunk's query tokens; requests pack while they fit, and an over-budget request
is cut into equal parts of every rank's own rows. CPU only."""

import numpy as np
import pytest

from vllm.v1.attention.ops.dcp_shardq.context import (
    _LOGITS_ROW_ALIGN,
    _logits_width,
    _rank_chunk_specs,
    _rank_query_rows,
)
from vllm.v1.attention.ops.shardq import fold_split


def _all_ranks(seq_lens, query_lens, world, max_keys, max_elems):
    return [
        _rank_chunk_specs(seq_lens, query_lens, max_keys, max_elems, world, rank)
        for rank in range(world)
    ]


def _check(per_rank, seq_lens, query_lens, world, max_keys, max_elems):
    qsl = np.concatenate(([0], np.cumsum(query_lens)))
    folds = fold_split(qsl, world)
    for rank, chunks in enumerate(per_rank):
        own = folds[rank]
        scored = []
        for req, qs in chunks:
            assert qs.stop > qs.start
            lo, hi = qsl[req.start] + qs.start, qsl[req.start] + qs.stop
            rows = own[(own >= lo) & (own < hi)]
            assert rows.size
            width = max(_logits_width(int(s)) for s in seq_lens[req])
            padded = -(-rows.size // _LOGITS_ROW_ALIGN) * _LOGITS_ROW_ALIGN
            if req.stop - req.start > 1:
                assert int(seq_lens[req].sum()) <= max_keys
            # A part may exceed the budget only at one row block.
            assert padded * width <= max(max_elems, _LOGITS_ROW_ALIGN * width)
            scored.append(rows)
        # Every one of this rank's rows once, in order.
        np.testing.assert_array_equal(
            np.concatenate(scored) if scored else own[:0], own
        )


@pytest.mark.parametrize("q,world", [(3480, 4), (8192, 8), (1, 4), (7, 4), (5461, 4)])
def test_rank_query_rows_match_fold_split(q, world):
    per_rank = fold_split(np.array([0, q]), world)
    np.testing.assert_array_equal(
        _rank_query_rows(q, world), [len(x) for x in per_rank]
    )


def test_c512_step_packs_requests():
    """C512 n17k (5 x 3480 new tokens over 132504 cached, DCP4): the 16-bit
    budget (384M logits) holds three requests' rank rows, so 2 chunks
    replace the 10 query sub-chunks of the token-count budget."""
    seq_lens = np.full(5, 132504 + 3480)
    query_lens = np.full(5, 3480)
    per_rank = _all_ranks(seq_lens, query_lens, 4, 1 << 30, 384 << 20)
    for chunks in per_rank:
        assert [(c[0].start, c[0].stop) for c in chunks] == [(0, 3), (3, 5)]
    _check(per_rank, seq_lens, query_lens, 4, 1 << 30, 384 << 20)


def test_c512_n32k_equal_parts():
    """C512 n32k (6 x 5461 over 129024 cached, DCP4) at the FP32 budget: each
    request is cut into 2 parts of each rank's own rows (683 + 683), the same
    calls on every rank."""
    seq_lens = np.full(6, 129024 + 5461)
    query_lens = np.full(6, 5461)
    per_rank = _all_ranks(seq_lens, query_lens, 4, 1 << 30, 128 << 20)
    qsl = np.concatenate(([0], np.cumsum(query_lens)))
    folds = fold_split(qsl, 4)
    for rank, chunks in enumerate(per_rank):
        assert len(chunks) == 12
        for req, qs in chunks:
            lo, hi = qsl[req.start] + qs.start, qsl[req.start] + qs.stop
            n = int(((folds[rank] >= lo) & (folds[rank] < hi)).sum())
            assert n in (682, 683)
    _check(per_rank, seq_lens, query_lens, 4, 1 << 30, 128 << 20)


@pytest.mark.parametrize(
    "seq_lens,query_lens,world,max_keys,max_elems",
    [
        ([10000, 20000, 300], [2000, 4000, 300], 4, 1 << 20, 1 << 22),
        ([136000], [3480], 4, 1 << 20, 60 * 136192),
        ([136000], [3480], 4, 1 << 20, 100 * 136192),
        ([50000, 50000], [8192, 8192], 8, 60000, 1 << 30),  # key bound
        ([5, 6, 7], [5, 6, 7], 4, 1 << 20, 1),  # tiny budget: one row block
        ([129, 4096], [1, 4096], 2, 1 << 20, 3 << 20),
    ],
)
def test_chunks_cover_and_fit(seq_lens, query_lens, world, max_keys, max_elems):
    seq_lens, query_lens = np.array(seq_lens), np.array(query_lens)
    per_rank = _all_ranks(seq_lens, query_lens, world, max_keys, max_elems)
    _check(per_rank, seq_lens, query_lens, world, max_keys, max_elems)
    # One plan: every rank owning rows of every request makes the same calls.
    if all(q >= 2 * world for q in query_lens):
        assert len({len(c) for c in per_rank}) == 1


@pytest.mark.parametrize(
    "seq_lens,query_lens,world",
    [
        ([5], [5], 16),  # the engine's 5-token warm-up step at DCP16
        ([1], [1], 4),
        ([2], [2], 4),
        ([3], [3], 4),
        ([67], [3], 4),  # with cached context
        ([1, 9], [1, 2], 8),
    ],
)
@pytest.mark.parametrize("max_elems", [1 << 30, 1])  # packed / cut path
def test_tiny_prefill_leaves_ranks_without_chunks(
    seq_lens, query_lens, world, max_elems
):
    """A prefill of fewer tokens than 2 x the DCP size leaves some ranks no
    fold rows: those ranks get no chunks (the indexer then scores nothing on
    them and only joins the gathers), the others cover their rows once."""
    seq_lens, query_lens = np.array(seq_lens), np.array(query_lens)
    per_rank = _all_ranks(seq_lens, query_lens, world, 1 << 20, max_elems)
    _check(per_rank, seq_lens, query_lens, world, 1 << 20, max_elems)
    folds = fold_split(np.concatenate(([0], np.cumsum(query_lens))), world)
    empty = [rank for rank in range(world) if not len(folds[rank])]
    assert empty
    assert [rank for rank in range(world) if not per_rank[rank]] == empty
