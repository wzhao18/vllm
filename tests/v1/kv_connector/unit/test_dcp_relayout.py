# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU oracle for the DCP row relayout: a request's KV laid out by the source
ranks, pulled page-wise into staging and gathered row-wise, must equal the
same tokens laid out directly in the destination layout."""

import numpy as np
import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.dcp_layout import DcpLayout
from vllm.distributed.kv_transfer.kv_connector.dcp_relayout import (
    RowRelayout,
    copy_byte_ranges_reference,
    plan_dcp_rows,
)
from vllm.v1.kv_cache_interface import KVRowLayout

TOKEN_MAJOR = KVRowLayout.token_major(12)
# An indexer-like page: each 4-row kernel page stores its rows' 8-byte values,
# then their 4-byte scales.
SPLIT = KVRowLayout(4, ((0, 8), (32, 4)))


def _token_bytes(num_tokens: int, row_bytes: int) -> np.ndarray:
    """A distinct byte pattern per (token, byte)."""
    t = np.arange(num_tokens, dtype=np.int64)[:, None]
    b = np.arange(row_bytes, dtype=np.int64)[None, :]
    return ((t * 131 + b * 7 + 1) % 251).astype(np.uint8)


def _rank_pages(
    layout: DcpLayout, rank: int, rows: KVRowLayout, kv: np.ndarray
) -> np.ndarray:
    """`rank`'s pages (flat bytes, page after page) holding `kv` (one row per
    global position), unused rows zero."""
    local, positions = layout.rank_rows(rank, 0, kv.shape[0])
    num_pages = -(-len(local) // layout.block_size) if len(local) else 0
    page_bytes = layout.block_size * rows.bytes_per_row
    out = np.zeros(num_pages * page_bytes, dtype=np.uint8)
    for row, pos in zip(local.tolist(), positions.tolist()):
        page, r = layout.page_row(row)
        col = 0
        for off, size in rows.row_byte_ranges(r):
            base = page * page_bytes + off
            out[base : base + size] = kv[pos, col : col + size]
            col += size
    return out


CASES = [
    # (src dcp, src interleave, src block, dst dcp, dst interleave, dst block)
    (8, 1, 16, 16, 1, 8),  # K3.1 P TP8=DCP8 -> D TP16=DCP16, equal logical blocks
    (16, 1, 8, 8, 1, 16),  # the reverse
    (8, 1, 16, 16, 1, 16),  # unequal logical blocks (pinned block sizes)
    (4, 1, 8, 4, 1, 16),  # same DCP, different block sizes
    (2, 4, 8, 4, 2, 8),  # different interleaves
    (1, 1, 8, 4, 1, 8),  # no DCP -> DCP
    (6, 1, 8, 2, 1, 12),  # non power of two
]


@pytest.mark.cpu_test
@pytest.mark.parametrize("src_n,src_i,src_b,dst_n,dst_i,dst_b", CASES)
@pytest.mark.parametrize("rows", [TOKEN_MAJOR, SPLIT], ids=["token_major", "split"])
@pytest.mark.parametrize("start", [0, 8, 37], ids=["no_hit", "aligned_hit", "mid_hit"])
def test_relayout_matches_direct_layout(
    src_n, src_i, src_b, dst_n, dst_i, dst_b, rows, start
):
    if rows.kernel_page_rows > 1 and (
        src_b % rows.kernel_page_rows or dst_b % rows.kernel_page_rows
    ):
        pytest.skip("pages must hold whole kernel pages")
    num_tokens = 301
    kv = _token_bytes(num_tokens, rows.bytes_per_row)
    src = DcpLayout(src_n, src_i, src_b)
    dst = DcpLayout(dst_n, dst_i, dst_b)
    relayout = RowRelayout(rows, rows, src_b, dst_b)
    for dst_rank in range(dst_n):
        expected = torch.from_numpy(_rank_pages(dst, dst_rank, rows, kv))
        # The destination already holds the prefix hit [0, start).
        dst_pages = expected.clone()
        hit_rows = dst.num_local_tokens(start, dst_rank)
        _, keep = dst.rank_rows(dst_rank, 0, start)
        assert len(keep) == hit_rows
        mask = torch.zeros_like(dst_pages, dtype=torch.bool)
        if hit_rows < dst.num_local_tokens(num_tokens, dst_rank):
            # Clear exactly the rows the transfer must fill.
            filled = _rank_pages(
                dst, dst_rank, rows,
                np.where(np.arange(num_tokens)[:, None] >= start, 255, 0).astype(np.uint8)
                * np.ones((1, rows.bytes_per_row), dtype=np.uint8),
            )
            mask = torch.from_numpy(filled == 255)
        dst_pages[mask] = 0

        plan = plan_dcp_rows(src, dst, dst_rank, start, num_tokens)
        assert np.array_equal(plan.dst_rows, np.arange(hit_rows, len(plan.dst_rows) + hit_rows))
        # Pull every source page the plan reads into consecutive staging slots.
        staging, slot_of = [], {}
        for src_rank, pages in plan.source_pages(src).items():
            rank_bytes = _rank_pages(src, src_rank, rows, kv)
            for page in pages.tolist():
                slot_of[(src_rank, page)] = len(staging)
                lo = page * relayout.src_page_bytes
                staging.append(rank_bytes[lo : lo + relayout.src_page_bytes])
        if not staging:
            assert len(plan.dst_rows) == 0
            continue
        staging_buf = torch.from_numpy(np.concatenate(staging))
        src_page, src_row = src.page_row(plan.src_rows)
        src_slots = np.array(
            [slot_of[(r, p)] for r, p in zip(plan.src_ranks.tolist(), src_page.tolist())],
            dtype=np.int64,
        )
        dst_page, dst_row = dst.page_row(plan.dst_rows)
        copies = relayout.byte_copies(src_slots, src_row, dst_page, dst_row)
        copy_byte_ranges_reference(staging_buf, dst_pages, *copies)
        assert torch.equal(dst_pages, expected), (dst_rank, start)


@pytest.mark.cpu_test
def test_k31_ratio_two_reads_one_source_page_per_destination_page():
    """With equal logical blocks (per-rank block x DCP), a D page comes from
    exactly one P page of one P rank: the TP-scaled packed block case."""
    src = DcpLayout(8, 1, 16)
    dst = DcpLayout(16, 1, 8)
    for dst_rank in range(16):
        plan = plan_dcp_rows(src, dst, dst_rank, 0, 16 * 8 * 5)
        assert set(plan.src_ranks.tolist()) == {dst_rank % 8}
        dst_page, _ = dst.page_row(plan.dst_rows)
        src_page, src_row = src.page_row(plan.src_rows)
        assert np.array_equal(dst_page, src_page)
        # Every other row of the source page, offset by the partner index.
        assert np.all(src_row % 2 == dst_rank // 8)


@pytest.mark.cpu_test
def test_row_layouts_must_split_rows_the_same_way():
    with pytest.raises(ValueError, match="row pieces"):
        RowRelayout(TOKEN_MAJOR, SPLIT, 8, 8)
