# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Triton row gather of the DCP relayout matches its host reference on
the page layouts of Kimi K3.1's caches."""

import numpy as np
import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.dcp_layout import DcpLayout
from vllm.distributed.kv_transfer.kv_connector.dcp_relayout import (
    RelayoutRegion,
    RowGather,
    plan_staged_relayout,
    row_table,
)
from vllm.v1.kv_cache_interface import KVRowLayout

from .dcp_relayout_reference import relayout_rows_reference

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="The relayout kernel needs a GPU"
)


def relayout_rows(buf, regions, src_blocks, src_rows, dst_blocks, dst_rows):
    """The device gather of the worker (`RowGather` over a `row_table`)."""
    RowGather(buf, regions)(
        row_table(src_blocks, src_rows, dst_blocks, dst_rows, buf.device)
    )


# Rows of one 64-row kernel page: MLA latent (fp8_ds_mla 656 B, nvfp4_ds_mla
# 352 B), and indexer keys storing the page's values before its scales (FP8:
# 128 B + 4 B, MXFP4: 64 B + 4 B).
MLA_FP8_DS = KVRowLayout.token_major(656)
MLA_NVFP4_DS = KVRowLayout.token_major(352)
INDEXER_FP8 = KVRowLayout(64, ((0, 128), (64 * 128, 4)))
INDEXER_MXFP4 = KVRowLayout(64, ((0, 64), (64 * 64, 4)))


@pytest.mark.parametrize(
    "rows",
    [
        [MLA_FP8_DS, INDEXER_FP8],
        [MLA_NVFP4_DS, INDEXER_MXFP4],
        # Target and drafter MLA pages plus indexer keys in one packed row.
        [MLA_FP8_DS, INDEXER_FP8, MLA_FP8_DS],
    ],
    ids=["fp8", "nvfp4_mxfp4", "with_drafter"],
)
@pytest.mark.parametrize(
    ("src", "dst"),
    [
        (DcpLayout(8, 1, 1280), DcpLayout(16, 1, 640)),
        (DcpLayout(16, 1, 640), DcpLayout(8, 1, 1280)),
    ],
    ids=["p8_to_d16", "p16_to_d8"],
)
@pytest.mark.parametrize("start", [0, 10240])
def test_relayout_kernel_matches_reference(rows, src, dst, start):
    kernel_rows = 64
    page_bytes = [kernel_rows * r.bytes_per_row for r in rows]
    stride = sum(page_bytes) + 512  # padded packed row
    regions = [
        RelayoutRegion(sum(page_bytes[:i]), stride, kernel_rows, r)
        for i, r in enumerate(rows)
    ]
    blocks_per_page = dst.block_size // kernel_rows
    num_blocks = 64
    job = plan_staged_relayout(
        src,
        dst,
        dst.dcp_size - 1,
        start,
        start + 20_000,
        unit_rows=kernel_rows,
        src_block_ids=list(range(64)),
        dst_block_ids=list(range(10, 10 + 20)),
        staging_block_ids=list(range(40, 64)),
        dst_kernel_block_size=kernel_rows,
    )
    assert job.dst_blocks.max() < num_blocks * blocks_per_page
    generator = torch.Generator().manual_seed(0)
    host = torch.randint(
        0,
        256,
        (num_blocks * blocks_per_page * stride,),
        dtype=torch.uint8,
        generator=generator,
    )
    device = host.cuda()
    relayout_rows_reference(
        host, regions, job.src_blocks, job.src_rows, job.dst_blocks, job.dst_rows
    )
    relayout_rows(
        device, regions, job.src_blocks, job.src_rows, job.dst_blocks, job.dst_rows
    )
    torch.cuda.synchronize()
    assert torch.equal(device.cpu(), host)


def test_relayout_kernel_random_rows():
    """Random disjoint source and destination rows, odd row sizes (byte
    elements)."""
    rows = [KVRowLayout.token_major(13), KVRowLayout(4, ((0, 6), (24, 3)))]
    kernel_rows = 8
    page_bytes = [kernel_rows * r.bytes_per_row for r in rows]
    stride = sum(page_bytes) + 5
    regions = [
        RelayoutRegion(sum(page_bytes[:i]), stride, kernel_rows, r)
        for i, r in enumerate(rows)
    ]
    rng = np.random.default_rng(0)
    n = 200  # at most the 32 x 8 destination rows
    src_blocks = rng.integers(0, 32, n)
    src_blocks[::7] = -1  # zero these rows
    dst = rng.permutation(32 * kernel_rows)[:n] + 32 * kernel_rows
    dst_blocks, dst_rows = np.divmod(dst, kernel_rows)
    src_rows = rng.integers(0, kernel_rows, n)
    host = torch.from_numpy(rng.integers(0, 256, 64 * stride, dtype=np.uint8))
    device = host.cuda()
    relayout_rows_reference(host, regions, src_blocks, src_rows, dst_blocks, dst_rows)
    relayout_rows(device, regions, src_blocks, src_rows, dst_blocks, dst_rows)
    torch.cuda.synchronize()
    assert torch.equal(device.cpu(), host)


@pytest.mark.parametrize("offset", [0, 4, 3], ids=["aligned", "int32", "byte"])
def test_relayout_kernel_long_pieces_and_offset_views(offset):
    """Row pieces longer than one copy step of the kernel, in a buffer view
    that starts `offset` bytes into its storage (narrower elements when the
    view is not 8-byte aligned)."""
    rows = [KVRowLayout.token_major(4096), KVRowLayout(8, ((0, 1000), (8000, 20)))]
    kernel_rows = 8
    page_bytes = [kernel_rows * r.bytes_per_row for r in rows]
    stride = sum(page_bytes) + 8
    regions = [
        RelayoutRegion(sum(page_bytes[:i]), stride, kernel_rows, r)
        for i, r in enumerate(rows)
    ]
    rng = np.random.default_rng(1)
    n = 40
    src_blocks = rng.integers(0, 8, n)
    src_blocks[::5] = -1
    dst = rng.permutation(8 * kernel_rows)[:n] + 8 * kernel_rows
    dst_blocks, dst_rows = np.divmod(dst, kernel_rows)
    src_rows = rng.integers(0, kernel_rows, n)
    storage = torch.from_numpy(
        rng.integers(0, 256, offset + 16 * stride, dtype=np.uint8)
    )
    host = storage[offset:]
    device = storage.cuda()[offset:]
    relayout_rows_reference(host, regions, src_blocks, src_rows, dst_blocks, dst_rows)
    relayout_rows(device, regions, src_blocks, src_rows, dst_blocks, dst_rows)
    torch.cuda.synchronize()
    assert torch.equal(device.cpu(), host)


def test_relayout_kernel_no_rows_is_a_no_op():
    rows = KVRowLayout.token_major(16)
    buf = torch.arange(256, dtype=torch.int64).to(torch.uint8).cuda()
    before = buf.clone()
    empty = np.zeros(0, dtype=np.int64)
    relayout_rows(buf, [RelayoutRegion(0, 64, 4, rows)], empty, empty, empty, empty)
    torch.cuda.synchronize()
    assert torch.equal(buf, before)
