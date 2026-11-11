# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DcpLayout index math against the attention kernels' reference."""

import itertools

import numpy as np
import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.dcp_layout import DcpLayout
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens

pytestmark = pytest.mark.cpu_test

DCP_SIZES = (1, 2, 3, 4, 8, 16)
BLOCK_SIZES = (64, 128, 640, 1280)


def _layouts():
    for dcp_size, block_size in itertools.product(DCP_SIZES, BLOCK_SIZES):
        for interleave in sorted({1, 2, 4, 64, block_size}):
            yield DcpLayout(dcp_size, interleave, block_size)


LAYOUTS = list(_layouts())


@pytest.mark.parametrize("layout", LAYOUTS, ids=str)
def test_positions_partition_over_ranks(layout: DcpLayout):
    """Every position has exactly one (rank, row), rows of a rank are
    consecutive from 0, and `global_pos` inverts `local_row`."""
    n = 3 * layout.dcp_size * layout.block_size
    pos = np.arange(n)
    owner = layout.owner(pos)
    row = layout.local_row(pos)
    for rank in range(layout.dcp_size):
        rows = row[owner == rank]
        np.testing.assert_array_equal(rows, np.arange(rows.size))
        np.testing.assert_array_equal(layout.global_pos(rank, rows), pos[owner == rank])
    # Pages cut each rank's rows into block_size runs.
    page, in_page = layout.page_row(row)
    np.testing.assert_array_equal(page * layout.block_size + in_page, row)
    assert in_page.max() < layout.block_size


@pytest.mark.parametrize("layout", LAYOUTS, ids=str)
def test_num_local_tokens_matches_attention(layout: DcpLayout):
    """Same rows per rank as the attention kernels' `get_dcp_local_seq_lens`."""
    n = 3 * layout.dcp_size * layout.block_size
    seq_lens = torch.arange(n + 1)
    expected = get_dcp_local_seq_lens(
        seq_lens, layout.dcp_size, None, layout.interleave
    ).numpy()
    owner = layout.owner(np.arange(n))
    for rank in range(layout.dcp_size):
        got = layout.num_local_tokens(seq_lens.numpy(), rank)
        np.testing.assert_array_equal(got, expected[:, rank])
        # Rows below n are exactly the positions below n the rank owns.
        np.testing.assert_array_equal(got[1:], np.cumsum(owner == rank, dtype=np.int64))


@pytest.mark.parametrize("layout", LAYOUTS[::7], ids=str)
def test_rank_rows(layout: DcpLayout):
    n = 2 * layout.dcp_size * layout.block_size + 5
    for start, end in ((0, n), (3, n - 2), (n // 2, n // 2), (n // 3, n)):
        for rank in range(layout.dcp_size):
            rows, positions = layout.rank_rows(rank, start, end)
            expected = np.arange(start, end)
            expected = expected[layout.owner(expected) == rank]
            np.testing.assert_array_equal(positions, expected)
            np.testing.assert_array_equal(rows, layout.local_row(expected))


def test_pages_hold_contiguous_positions():
    assert DcpLayout(1, 1, 64).pages_hold_contiguous_positions
    assert DcpLayout(4, 64, 64).pages_hold_contiguous_positions
    assert not DcpLayout(4, 1, 64).pages_hold_contiguous_positions


@pytest.mark.parametrize(
    "args",
    [
        (0, 1, 64),
        (2, 0, 64),
        (2, 1, 0),
        # With DCP, a page holds whole interleave chunks.
        (2, 128, 64),
        (2, 3, 64),
    ],
)
def test_rejects_invalid_layouts(args):
    with pytest.raises(ValueError):
        DcpLayout(*args)


def test_interleave_is_free_without_dcp():
    assert DcpLayout(1, 128, 64).owner(1000) == 0
