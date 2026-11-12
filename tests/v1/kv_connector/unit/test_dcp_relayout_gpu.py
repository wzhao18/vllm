# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Triton row gather of the DCP relayout matches its host reference on
the page layouts of Kimi K3.1's caches."""

import math

import numpy as np
import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.dcp_layout import DcpLayout
from vllm.distributed.kv_transfer.kv_connector.dcp_relayout import (
    RelayoutRegion,
    RowGather,
    StagedRelayout,
    plan_staged_relayout,
    row_table,
)
from vllm.v1.kv_cache_interface import KVRowLayout

from .dcp_relayout_reference import relayout_rows_reference

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="The relayout kernel needs a GPU"
)


def relayout_rows(buf, regions, src_rows, unit_rows, job):
    """The device gather of the worker (`RowGather` over a `row_table`)."""
    RowGather(buf, regions, src_rows, unit_rows)(
        row_table(job, buf.device), len(job.src_slots)
    )


def _check(host, regions, src_rows, unit_rows, job):
    device = host.cuda()
    relayout_rows_reference(host, regions, src_rows, unit_rows, job)
    relayout_rows(device, regions, src_rows, unit_rows, job)
    torch.cuda.synchronize()
    assert torch.equal(device.cpu(), host)


# Rows of the MLA latent (fp8_ds_mla 656 B, nvfp4_ds_mla 352 B), and of
# indexer keys storing a kernel page's values before its scales (FP8: 128 B
# + 4 B, MXFP4: 64 B + 4 B), whose kernel page is 64 rows or a whole page.
MLA_FP8_DS = KVRowLayout.token_major(656)
MLA_NVFP4_DS = KVRowLayout.token_major(352)


def indexer_fp8(rows: int) -> KVRowLayout:
    return KVRowLayout(rows, ((0, 128), (rows * 128, 4)))


def indexer_mxfp4(rows: int) -> KVRowLayout:
    return KVRowLayout(rows, ((0, 64), (rows * 64, 4)))


@pytest.mark.parametrize(
    "rows",
    [
        lambda k: [MLA_FP8_DS, indexer_fp8(k)],
        lambda k: [MLA_NVFP4_DS, indexer_mxfp4(k)],
        # Target and drafter MLA pages plus indexer keys in one packed row.
        lambda k: [MLA_FP8_DS, indexer_fp8(k), MLA_FP8_DS],
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
@pytest.mark.parametrize(
    "kernel", ["64", "page"], ids=["64_row_kernel_pages", "page_kernel_pages"]
)
@pytest.mark.parametrize("start", [0, 10240])
def test_relayout_kernel_matches_reference(rows, src, dst, kernel, start):
    """With page-long kernel pages (Kimi K3.1's indexer), each side keeps
    its scales at another offset: the gather reads the source's."""
    src_kb = 64 if kernel == "64" else src.block_size
    dst_kb = 64 if kernel == "64" else dst.block_size
    unit = math.gcd(src_kb, dst_kb)
    src_rows, dst_rows = rows(min(64, src_kb)), rows(min(64, dst_kb))
    if kernel == "page":
        src_rows, dst_rows = rows(src_kb), rows(dst_kb)
    page_bytes = [dst_kb * r.bytes_per_row for r in dst_rows]
    stride = sum(page_bytes) + 512  # padded packed row
    regions = [
        RelayoutRegion(sum(page_bytes[:i]), stride, dst_kb, r)
        for i, r in enumerate(dst_rows)
    ]
    blocks_per_page = dst.block_size // dst_kb
    num_blocks = 64
    job = plan_staged_relayout(
        src,
        dst,
        dst.dcp_size - 1,
        start,
        start + 20_000,
        unit_rows=unit,
        src_kernel_block_size=src_kb,
        src_block_ids=list(range(64)),
        dst_block_ids=list(range(10, 10 + 20)),
        staging_block_ids=list(range(40, 64)),
        dst_kernel_block_size=dst_kb,
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
    _check(host, regions, src_rows, unit, job)


def _random_job(rng, n, src_kernel_rows, unit_rows, num_staging, dst_rows_total):
    """`n` random rows of source kernel blocks of `src_kernel_rows` rows
    staged in random units (two or more per block), into disjoint random
    destination rows past the staging."""
    units_per_block = src_kernel_rows // unit_rows
    staging_units = rng.permutation(num_staging)[: n * units_per_block]
    src_slots = np.arange(n, dtype=np.int64) * units_per_block
    src_slots[::7] = -1  # zero these rows
    src_rows = rng.integers(0, src_kernel_rows, n)
    dst = rng.permutation(dst_rows_total)[:n]
    return src_slots, src_rows, dst, staging_units


@pytest.mark.parametrize("src_pages", ["same", "longer"])
def test_relayout_kernel_random_rows(src_pages):
    """Random disjoint source and destination rows, odd row sizes (byte
    elements), source kernel blocks spanning two staging units, and source
    pages twice as long as the destination's (scales elsewhere)."""
    kernel_rows = 8
    dst_rows = [KVRowLayout.token_major(13), KVRowLayout(4, ((0, 6), (24, 3)))]
    src_rows = (
        dst_rows
        if src_pages == "same"
        else [KVRowLayout.token_major(13), KVRowLayout(16, ((0, 6), (96, 3)))]
    )
    page_bytes = [kernel_rows * r.bytes_per_row for r in dst_rows]
    stride = sum(page_bytes) + 5
    regions = [
        RelayoutRegion(sum(page_bytes[:i]), stride, kernel_rows, r)
        for i, r in enumerate(dst_rows)
    ]
    rng = np.random.default_rng(0)
    n = 16  # two staging units each, of the 32 staging blocks
    src_slots, src_rows_in_block, dst, staging_units = _random_job(
        rng, n, 16, kernel_rows, 32, 32 * kernel_rows
    )
    dst_blocks, dst_r = np.divmod(dst + 32 * kernel_rows, kernel_rows)
    job = StagedRelayout(
        {}, src_slots, src_rows_in_block, dst_blocks, dst_r, staging_units
    )
    host = torch.from_numpy(rng.integers(0, 256, 64 * stride, dtype=np.uint8))
    _check(host, regions, src_rows, kernel_rows, job)


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
    n = 8
    src_slots, src_rows, dst, staging_units = _random_job(
        rng, n, kernel_rows, kernel_rows, 8, 8 * kernel_rows
    )
    dst_blocks, dst_rows = np.divmod(dst + 8 * kernel_rows, kernel_rows)
    job = StagedRelayout({}, src_slots, src_rows, dst_blocks, dst_rows, staging_units)
    storage = torch.from_numpy(
        rng.integers(0, 256, offset + 16 * stride, dtype=np.uint8)
    )
    host = storage[offset:]
    device = storage.cuda()[offset:]
    relayout_rows_reference(host, regions, rows, kernel_rows, job)
    relayout_rows(device, regions, rows, kernel_rows, job)
    torch.cuda.synchronize()
    assert torch.equal(device.cpu(), host)


def test_relayout_kernel_no_rows_is_a_no_op():
    rows = KVRowLayout.token_major(16)
    buf = torch.arange(256, dtype=torch.int64).to(torch.uint8).cuda()
    before = buf.clone()
    empty = np.zeros(0, dtype=np.int64)
    job = StagedRelayout({}, empty, empty, empty, empty, empty)
    relayout_rows(buf, [RelayoutRegion(0, 64, 4, rows)], [rows], 4, job)
    torch.cuda.synchronize()
    assert torch.equal(buf, before)
