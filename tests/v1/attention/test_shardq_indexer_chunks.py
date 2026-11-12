# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ShardQ indexer prefill chunking (dcp_shardq.context.split_shardq_indexer_chunks):
chunks bound the logits a rank holds (its fold rows x the row width), not the
chunk's query tokens. CPU only."""

import numpy as np
import pytest

from vllm.v1.attention.ops.dcp_shardq.context import (
    _LOGITS_ROW_ALIGN,
    _logits_width,
    _rank_query_rows,
    split_shardq_indexer_chunks,
)
from vllm.v1.attention.ops.shardq import fold_split


def _rank_rows_in(chunk, seq_lens, query_lens, world):
    """Rows each rank scores in one chunk, from fold_split itself."""
    req, qs = chunk
    qsl = np.concatenate(([0], np.cumsum(query_lens)))
    tokens = np.arange(qsl[req.start], qsl[req.stop])[qs]
    per_rank = fold_split(qsl, world)
    return np.array([np.isin(per_rank[r], tokens).sum() for r in range(world)])


def _check(chunks, seq_lens, query_lens, world, max_keys, max_elems):
    qsl = np.concatenate(([0], np.cumsum(query_lens)))
    covered = []
    for req, qs in chunks:
        assert qs.stop > qs.start
        covered.append(np.arange(qsl[req.start], qsl[req.stop])[qs])
        rows = _rank_rows_in((req, qs), seq_lens, query_lens, world)
        width = max(_logits_width(int(s)) for s in seq_lens[req])
        padded = -(-int(rows.max()) // _LOGITS_ROW_ALIGN) * _LOGITS_ROW_ALIGN
        if req.stop - req.start > 1:
            assert int(seq_lens[req].sum()) <= max_keys
        # A sub-chunk may exceed the budget only at one row block.
        assert padded * width <= max(max_elems, _LOGITS_ROW_ALIGN * width)
    # Every query token once, in order.
    np.testing.assert_array_equal(np.concatenate(covered), np.arange(qsl[-1]))


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
    chunks = split_shardq_indexer_chunks(seq_lens, query_lens, 4, 1 << 30, 384 << 20)
    assert [(c[0].start, c[0].stop) for c in chunks] == [(0, 3), (3, 5)]
    _check(chunks, seq_lens, query_lens, 4, 1 << 30, 384 << 20)


@pytest.mark.parametrize(
    "seq_lens,query_lens,world,max_keys,max_elems",
    [
        ([10000, 20000, 300], [2000, 4000, 300], 4, 1 << 20, 1 << 22),
        ([136000], [3480], 4, 1 << 20, 60 * 136192),  # cut at pieces
        ([136000], [3480], 4, 1 << 20, 100 * 136192),  # cut inside pieces
        ([50000, 50000], [8192, 8192], 8, 60000, 1 << 30),  # key bound
        ([5, 6, 7], [5, 6, 7], 4, 1 << 20, 1),  # tiny budget: one row block
        ([129, 4096], [1, 4096], 2, 1 << 20, 3 << 20),
    ],
)
def test_chunks_cover_and_fit(seq_lens, query_lens, world, max_keys, max_elems):
    seq_lens, query_lens = np.array(seq_lens), np.array(query_lens)
    chunks = split_shardq_indexer_chunks(
        seq_lens, query_lens, world, max_keys, max_elems
    )
    _check(chunks, seq_lens, query_lens, world, max_keys, max_elems)
