# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU oracle for the DCP row relayout: a request's KV laid out by the source
ranks, pulled page-wise into staging and gathered row-wise, must equal the
same tokens laid out directly in the destination layout."""

import math

import numpy as np
import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.dcp_layout import DcpLayout
from vllm.distributed.kv_transfer.kv_connector.dcp_relayout import (
    RelayoutRegion,
    RowRelayout,
    copy_byte_ranges_reference,
    num_staging_blocks,
    plan_dcp_rows,
    plan_staged_relayout,
    relayout_read_amplification,
    relayout_rows_reference,
)
from vllm.utils.math_utils import cdiv
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
                dst,
                dst_rank,
                rows,
                np.where(np.arange(num_tokens)[:, None] >= start, 255, 0).astype(
                    np.uint8
                )
                * np.ones((1, rows.bytes_per_row), dtype=np.uint8),
            )
            mask = torch.from_numpy(filled == 255)
        dst_pages[mask] = 0

        plan = plan_dcp_rows(src, dst, dst_rank, start, num_tokens)
        assert np.array_equal(
            plan.dst_rows, np.arange(hit_rows, len(plan.dst_rows) + hit_rows)
        )
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
            [
                slot_of[(r, p)]
                for r, p in zip(plan.src_ranks.tolist(), src_page.tolist())
            ],
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


# ----------------------------------------------------------------------
# Staged relayout: reads into staging blocks, then the row gather
# ----------------------------------------------------------------------

# Two caches of a packed block row, e.g. MLA latent and indexer keys of one
# layer: each region is one cache's page, strided by the whole (padded) row.
REGION_ROWS = [TOKEN_MAJOR, SPLIT]
ROW_PADDING = 8


def _kernel_pages(
    layout: DcpLayout,
    rank: int,
    rows: KVRowLayout,
    kv: np.ndarray,
    block_ids: list[int],
    kernel_rows: int,
) -> np.ndarray:
    """`rank`'s kernel blocks `(block id x kernel blocks per block + k, page
    bytes)` holding `kv` (one row per position), logical page `p` in block
    `block_ids[p]`; unused rows zero."""
    per_block = layout.block_size // kernel_rows
    pages = np.zeros(
        ((max(block_ids) + 1) * per_block, kernel_rows * rows.bytes_per_row),
        dtype=np.uint8,
    )
    local, positions = layout.rank_rows(rank, 0, kv.shape[0])
    for row, pos in zip(local.tolist(), positions.tolist()):
        page, r = layout.page_row(row)
        kernel_block = block_ids[page] * per_block + r // kernel_rows
        col = 0
        for off, size in rows.row_byte_ranges(r % kernel_rows):
            pages[kernel_block, off : off + size] = kv[pos, col : col + size]
            col += size
    return pages


STAGED_CASES = [
    # (src dcp, interleave, block, kernel block, dst dcp, interleave, block,
    #  kernel block)
    (8, 1, 16, 16, 16, 1, 8, 8),  # K3.1 P TP8 -> D TP16: TP-scaled blocks
    (16, 1, 8, 8, 8, 1, 16, 16),  # the reverse
    (8, 1, 16, 4, 16, 1, 8, 4),  # logical blocks of several kernel blocks
    (8, 1, 16, 16, 16, 1, 16, 8),  # unequal logical blocks
    (4, 1, 8, 8, 4, 1, 16, 16),  # same DCP, different blocks
    (2, 4, 8, 8, 4, 4, 8, 8),  # interleave 4 on both sides
    (1, 1, 8, 8, 4, 1, 8, 4),  # no DCP -> DCP
]


@pytest.mark.cpu_test
@pytest.mark.parametrize("case", STAGED_CASES, ids=str)
@pytest.mark.parametrize("hit", ["none", "aligned", "mid_page"])
def test_staged_relayout_matches_direct_layout(case, hit):
    """Reads of the planned source units into the staging blocks, then
    `relayout_rows_reference`, leave every destination rank's blocks as if the
    request had been computed in the destination layout: rows past the request
    zeroed, rows of a prefix hit untouched, other blocks untouched."""
    src_n, src_i, src_b, src_kb, dst_n, dst_i, dst_b, dst_kb = case
    src = DcpLayout(src_n, src_i, src_b)
    dst = DcpLayout(dst_n, dst_i, dst_b)
    num_tokens = 301
    start = {"none": 0, "aligned": 2 * dst_n * dst_b, "mid_page": 37}[hit]
    unit = math.gcd(src_kb, dst_kb)
    kv = [
        (_token_bytes(num_tokens, rows.bytes_per_row) + 17 * r).astype(np.uint8)
        for r, rows in enumerate(REGION_ROWS)
    ]
    # Source pages under shuffled block ids.
    num_src_pages = cdiv(num_tokens, src_n * src_b)
    src_block_ids = [num_src_pages + 1 - p for p in range(num_src_pages)]
    src_pages = {
        rank: [
            _kernel_pages(src, rank, rows, kv[r], src_block_ids, src_kb)
            for r, rows in enumerate(REGION_ROWS)
        ]
        for rank in range(src_n)
    }
    # Destination cache: 200 blocks of packed, padded rows.
    dst_kernel_per_block = dst_b // dst_kb
    page_bytes = [dst_kb * rows.bytes_per_row for rows in REGION_ROWS]
    stride = sum(page_bytes) + ROW_PADDING
    regions = [
        RelayoutRegion(sum(page_bytes[:r]), stride, dst_kb, rows)
        for r, rows in enumerate(REGION_ROWS)
    ]
    num_staging = num_staging_blocks(src, dst, start, num_tokens)
    for dst_rank in range(dst_n):
        rng = np.random.default_rng(dst_rank)
        buf = torch.from_numpy(
            rng.integers(0, 256, 200 * dst_kernel_per_block * stride, dtype=np.uint8)
        )
        before = buf.clone()
        lo = dst.num_local_tokens(start, dst_rank)
        first_page = lo // dst_b
        # The request's pages from the hit on, plus one past its end.
        num_pages = cdiv(dst.num_local_tokens(num_tokens, dst_rank), dst_b)
        num_pages = max(num_pages, first_page) - first_page + 1
        dst_block_ids = [199 - 3 * i for i in range(num_pages)]
        staging_block_ids = list(range(num_staging))
        job = plan_staged_relayout(
            src,
            dst,
            dst_rank,
            start,
            num_tokens,
            unit_rows=unit,
            src_block_ids=src_block_ids,
            dst_block_ids=dst_block_ids,
            staging_block_ids=staging_block_ids,
            dst_kernel_block_size=dst_kb,
        )
        # The NIXL reads: source units into staging units, region by region.
        for rank, (src_units, staging_units) in job.reads.items():
            for u_src, u_dst in zip(src_units, staging_units):
                kernel_block, sub = divmod(u_src, src_kb // unit)
                dst_kernel_block, dst_sub = divmod(u_dst, dst_kb // unit)
                for r, rows in enumerate(REGION_ROWS):
                    n = unit * rows.bytes_per_row
                    data = src_pages[rank][r][kernel_block, sub * n : (sub + 1) * n]
                    at = regions[r].offset + dst_kernel_block * stride + dst_sub * n
                    buf[at : at + n] = torch.from_numpy(data.copy())
        relayout_rows_reference(
            buf, regions, job.src_blocks, job.src_rows, job.dst_blocks, job.dst_rows
        )

        written = torch.zeros_like(buf, dtype=torch.bool)
        for block in staging_block_ids:
            lo_b = block * dst_kernel_per_block * stride
            written[lo_b : lo_b + dst_kernel_per_block * stride] = True
        rows_end = (first_page + num_pages) * dst_b
        for row in range(dst.num_local_tokens(start, dst_rank), rows_end):
            pos = int(dst.global_pos(dst_rank, row))
            page, r_in_page = dst.page_row(row)
            kernel_block = (
                dst_block_ids[page - first_page] * dst_kernel_per_block
                + r_in_page // dst_kb
            )
            for r, rows in enumerate(REGION_ROWS):
                col = 0
                for off, size in rows.row_byte_ranges(r_in_page % dst_kb):
                    at = kernel_block * stride + regions[r].offset + off
                    expected = (
                        kv[r][pos, col : col + size]
                        if pos < num_tokens
                        else np.zeros(size, np.uint8)
                    )
                    assert np.array_equal(buf[at : at + size].numpy(), expected), (
                        dst_rank,
                        row,
                        r,
                    )
                    written[at : at + size] = True
                    col += size
        # Rows below the hit, padding and other blocks are untouched.
        assert torch.equal(buf[~written], before[~written]), dst_rank


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("src", "dst"),
    [
        (DcpLayout(8, 1, 1280), DcpLayout(16, 1, 640)),
        (DcpLayout(16, 1, 640), DcpLayout(8, 1, 1280)),
        (DcpLayout(4, 1, 2560), DcpLayout(16, 1, 640)),
    ],
    ids=str,
)
@pytest.mark.parametrize(("start", "end"), [(0, 130_000), (10240, 10241), (0, 0)])
def test_num_staging_blocks_covers_every_rank(src, dst, start, end):
    """A block id is a page on every rank, so staging takes the most units any
    rank reads; every rank's plan fits in that many blocks."""
    need = num_staging_blocks(src, dst, start, end)
    unit = math.gcd(src.block_size, dst.block_size)
    most = 0
    for rank in range(dst.dcp_size):
        plan = plan_dcp_rows(src, dst, rank, start, end)
        units = {
            (int(s), int(r) // unit) for s, r in zip(plan.src_ranks, plan.src_rows)
        }
        most = max(most, len(units))
    assert need == cdiv(most * unit, dst.block_size)
    if end > start:
        assert need > 0


@pytest.mark.cpu_test
def test_relayout_read_amplification():
    # Every D rank of a twice larger DCP keeps every other row it reads.
    assert relayout_read_amplification(
        DcpLayout(8, 1, 1280), DcpLayout(16, 1, 640)
    ) == pytest.approx(2.0)
    # The reverse reads whole source pages it keeps.
    assert relayout_read_amplification(
        DcpLayout(16, 1, 640), DcpLayout(8, 1, 1280)
    ) == pytest.approx(1.0)


@pytest.mark.cpu_test
def test_staged_relayout_rejects_short_staging():
    src, dst = DcpLayout(8, 1, 16), DcpLayout(16, 1, 8)
    with pytest.raises(ValueError, match="staging"):
        plan_staged_relayout(
            src,
            dst,
            0,
            0,
            1000,
            unit_rows=8,
            src_block_ids=list(range(8)),
            dst_block_ids=list(range(8)),
            staging_block_ids=[0],
            dst_kernel_block_size=8,
        )
