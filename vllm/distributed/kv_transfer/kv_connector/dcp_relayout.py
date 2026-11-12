# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row relayout of DCP-sharded KV between two DCP layouts.

When a KV transfer's two sides shard a sequence differently, a destination
rank's rows are not a run of whole source pages: at interleave 1 a decode rank
of a DCP twice the prefill's takes every other row of one prefill rank. Page
RDMA cannot express that, so the destination pulls the source pages that hold
its rows into staging pages and moves the rows into its own pages with a row
gather.

The planning here is general in both layouts (any DCP sizes, interleaves and
block sizes, either direction; `DcpLayout`). What a connector supports is
narrower and decided by its mode selection (for NIXL,
`select_dcp_transfer_mode`: e.g. both sides sharding need the same
interleave).

`plan_dcp_rows` maps each destination row to its source (rank, page, row) with
the two sides' `DcpLayout`s; the gather moves each row piece by piece from
each cache's `KVRowLayout`, so packed pages (e.g. a DSA indexer page that
stores a kernel page's values before its scales) move row by row like plain
token-major pages.

A transfer executes this in three steps, all derived from the same plan:

1. Staging (`num_staging_blocks`, scheduler side): the destination reserves
   blocks of its own cache to receive the source rows it needs, in units of
   `gcd(source block, destination block)` rows so that a unit of a source page
   fits inside a destination block.
2. Reads (`plan_staged_relayout`, worker side): which source units to read
   into which staging units.
3. Row gather (`RowGather`, on the device): every destination row is copied
   from its staged row, region by region, and the destination rows past the
   transferred positions are zeroed.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch

from vllm.distributed.kv_transfer.kv_connector.dcp_layout import DcpLayout
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.kv_cache_interface import KVRowLayout


@dataclass(frozen=True)
class DcpRowPlan:
    """Source of each destination row of one request, in destination order.

    All arrays have one entry per destination row the plan covers.
    """

    dst_rows: np.ndarray
    """Destination local rows."""
    src_ranks: np.ndarray
    """Source DCP rank holding each row."""
    src_rows: np.ndarray
    """Row on that source rank."""

    def source_pages(self, src: DcpLayout) -> dict[int, np.ndarray]:
        """Source rank -> the (sorted, unique) source pages the plan reads."""
        return {
            int(rank): np.unique(src.page_row(self.src_rows[self.src_ranks == rank])[0])
            for rank in np.unique(self.src_ranks)
        }


def plan_dcp_rows(
    src: DcpLayout, dst: DcpLayout, dst_rank: int, start: int, end: int
) -> DcpRowPlan:
    """Rows of destination rank `dst_rank` holding global positions
    `[start, end)`, and where the source holds each of them.

    General in both layouts: any DCP sizes, interleaves and block sizes, in
    either direction. `start` may fall anywhere (e.g. after a prefix hit that
    ends inside a page); rows below it are not in the plan.
    """
    dst_rows, positions = dst.rank_rows(dst_rank, start, end)
    return DcpRowPlan(
        dst_rows=dst_rows,
        src_ranks=np.asarray(src.owner(positions), dtype=np.int64),
        src_rows=np.asarray(src.local_row(positions), dtype=np.int64),
    )


# ----------------------------------------------------------------------
# Staging: where source rows wait until the row gather
# ----------------------------------------------------------------------


def staging_unit_rows(src: DcpLayout, dst: DcpLayout) -> int:
    """Rows of a staging unit: the largest run of rows that tiles the pages of
    both sides, so that a unit of a source page fits in a destination page."""
    return math.gcd(src.block_size, dst.block_size)


def _source_units(
    plan: DcpRowPlan, unit_rows: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The source units (runs of `unit_rows` rows of one source rank) the
    plan reads as `(ranks, units)`, sorted by rank then unit, and the index
    of each planned row's unit in them."""
    units = plan.src_rows // unit_rows
    if not len(units):
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    keys = plan.src_ranks * (int(units.max()) + 1) + units
    unique_keys, slots = np.unique(keys, return_inverse=True)
    ranks, unit_ids = np.divmod(unique_keys, int(units.max()) + 1)
    return ranks, unit_ids, slots.reshape(-1)


def num_staging_blocks(src: DcpLayout, dst: DcpLayout, start: int, end: int) -> int:
    """Destination blocks to reserve for staging positions `[start, end)`.

    A block id names a page on every destination rank, so this is the most
    that any destination rank needs. Units of `staging_unit_rows` rows bound
    the need of a worker that stages finer units (e.g. kernel blocks): every
    finer unit it reads lies in a unit counted here.
    """
    unit_rows = staging_unit_rows(src, dst)
    most = max(
        len(_source_units(plan_dcp_rows(src, dst, rank, start, end), unit_rows)[0])
        for rank in range(dst.dcp_size)
    )
    return cdiv(most * unit_rows, dst.block_size)


def relayout_read_amplification(src: DcpLayout, dst: DcpLayout) -> float:
    """Source rows the destination ranks read per row they keep, over a long
    sequence: units of whole source pages carry rows of other destination
    ranks (e.g. 2.0 when every destination rank keeps every other row)."""
    # Pages of both sides tile one period of positions whole (a page of a
    # rank spans `dcp_size * block_size` positions), so the units read repeat
    # with it and one period gives the exact ratio.
    period = math.lcm(src.dcp_size * src.block_size, dst.dcp_size * dst.block_size)
    unit_rows = staging_unit_rows(src, dst)
    read = sum(
        len(_source_units(plan_dcp_rows(src, dst, rank, 0, period), unit_rows)[0])
        for rank in range(dst.dcp_size)
    )
    return read * unit_rows / period


@dataclass(frozen=True)
class StagedRelayout:
    """One request's relayout on one destination rank: the reads into
    staging and the row gather after them.

    Rows of the gather are rows of the destination cache's physical (kernel)
    blocks; a negative source block zeroes the destination row.
    """

    reads: dict[int, tuple[list[int], list[int]]]
    """Source rank -> `(source units, staging units)` to read, each named as
    `block id * units per block + unit in block` on its side."""
    src_blocks: np.ndarray
    src_rows: np.ndarray
    dst_blocks: np.ndarray
    dst_rows: np.ndarray


def plan_staged_relayout(
    src: DcpLayout,
    dst: DcpLayout,
    dst_rank: int,
    start: int,
    end: int,
    *,
    unit_rows: int,
    src_block_ids: list[int],
    dst_block_ids: list[int],
    staging_block_ids: list[int],
    dst_kernel_block_size: int,
) -> StagedRelayout:
    """Plan the relayout of positions `[start, end)` into destination rank
    `dst_rank`.

    - `src_block_ids[j]` is page `j` of every source rank (DCP ranks share a
      request's block ids).
    - `dst_block_ids[i]` is page `first + i` of `dst_rank`, `first` being the
      page of its first row at or after `start`. Rows of that page below
      `start` (a prefix hit ending inside the page) are left untouched; rows
      of the given pages past `end` are zeroed.
    - Source units are read into consecutive units of `staging_block_ids`.
      `unit_rows` must tile both sides' pages and the destination's kernel
      blocks must tile its pages.
    """
    if src.block_size % unit_rows or dst.block_size % unit_rows:
        raise ValueError(f"{unit_rows}-row units do not tile pages of {src}, {dst}")
    if dst.block_size % dst_kernel_block_size:
        raise ValueError(
            f"{dst_kernel_block_size}-row kernel blocks do not tile pages of {dst}"
        )
    plan = plan_dcp_rows(src, dst, dst_rank, start, end)
    ranks, units, slots = _source_units(plan, unit_rows)

    # Source units, as unit-granular ids of the source blocks.
    src_units_per_page = src.block_size // unit_rows
    src_pages, src_sub = np.divmod(units, src_units_per_page)
    if len(src_pages) and src_pages.max() >= len(src_block_ids):
        raise ValueError(
            f"Source pages {len(src_block_ids)} do not cover positions "
            f"[{start}, {end}) of {src}"
        )
    src_unit_ids = (
        np.asarray(src_block_ids, dtype=np.int64)[src_pages] * src_units_per_page
        + src_sub
    )
    # Staging units, consecutive over the staging blocks.
    dst_units_per_page = dst.block_size // unit_rows
    staging_pages, staging_sub = np.divmod(
        np.arange(len(units), dtype=np.int64), dst_units_per_page
    )
    if len(units) > len(staging_block_ids) * dst_units_per_page:
        raise ValueError(
            f"{len(units)} staging units need more than the "
            f"{len(staging_block_ids)} staging blocks"
        )
    staging_ids = np.asarray(staging_block_ids, dtype=np.int64)
    staging_unit_ids = staging_ids[staging_pages] * dst_units_per_page + staging_sub
    reads = {
        int(rank): (
            src_unit_ids[ranks == rank].tolist(),
            staging_unit_ids[ranks == rank].tolist(),
        )
        for rank in np.unique(ranks)
    }

    # Gather: planned rows from their staged rows, then zero the tail.
    kernel_blocks_per_page = dst.block_size // dst_kernel_block_size
    lo = int(dst.num_local_tokens(start, dst_rank))
    hi = int(dst.num_local_tokens(end, dst_rank))
    first_page = lo // dst.block_size
    num_rows = (first_page + len(dst_block_ids)) * dst.block_size - hi
    if num_rows < 0:
        raise ValueError(
            f"Destination pages {len(dst_block_ids)} do not cover positions "
            f"[{start}, {end}) of rank {dst_rank} of {dst}"
        )
    # A unit's rows keep their order, at the unit's place in its staging page.
    src_blocks, src_rows = _kernel_block_rows(
        staging_ids[staging_pages[slots]],
        staging_sub[slots] * unit_rows + plan.src_rows % unit_rows,
        kernel_blocks_per_page,
        dst_kernel_block_size,
    )
    dst_local_rows = np.arange(lo, (first_page + len(dst_block_ids)) * dst.block_size)
    dst_pages, dst_page_rows = dst.page_row(dst_local_rows)
    dst_blocks, dst_rows = _kernel_block_rows(
        np.asarray(dst_block_ids, dtype=np.int64)[dst_pages - first_page],
        dst_page_rows,
        kernel_blocks_per_page,
        dst_kernel_block_size,
    )
    num_zero = len(dst_local_rows) - len(src_blocks)
    return StagedRelayout(
        reads=reads,
        src_blocks=np.concatenate([src_blocks, np.full(num_zero, -1, np.int64)]),
        src_rows=np.concatenate([src_rows, np.zeros(num_zero, np.int64)]),
        dst_blocks=dst_blocks,
        dst_rows=dst_rows,
    )


def _kernel_block_rows(
    blocks: np.ndarray,
    rows: np.ndarray,
    kernel_blocks_per_page: int,
    kernel_block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """`(kernel block, row in it)` of row `rows[i]` of logical block
    `blocks[i]` (see `_logical_to_kernel_block_ids`)."""
    sub, row = np.divmod(np.asarray(rows, dtype=np.int64), kernel_block_size)
    return np.asarray(blocks, dtype=np.int64) * kernel_blocks_per_page + sub, row


# ----------------------------------------------------------------------
# Row gather
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class RelayoutRegion:
    """A cache whose rows the gather moves: pages of `page_rows` rows laid
    out by `rows`, every `page_stride` bytes from byte `offset` of the
    buffer (e.g. one layer's view of a packed hybrid cache, whose pages are
    strided by the whole block row)."""

    offset: int
    page_stride: int
    page_rows: int
    rows: KVRowLayout

    def __post_init__(self) -> None:
        if (
            self.page_rows % self.rows.kernel_page_rows
            or self.page_rows * self.rows.bytes_per_row > self.page_stride
        ):
            raise ValueError(f"{self.rows} does not fit pages of {self}")


_ELEMENT_DTYPES = {8: torch.int64, 4: torch.int32, 2: torch.int16, 1: torch.uint8}
# Elements one loop step of the gather kernel copies of a row piece; longer
# pieces take several steps.
_COPY_BLOCK = 256


def _region_table(
    regions: list[RelayoutRegion], element_size: int, max_pieces: int
) -> np.ndarray:
    """One row per region, in `element_size` units: offset, page stride,
    kernel page rows, row size, then (offset, size) of each row piece, padded
    with empty pieces to `max_pieces`."""
    table = np.zeros((len(regions), 4 + 2 * max_pieces), dtype=np.int64)
    for i, region in enumerate(regions):
        table[i, :4] = (
            region.offset // element_size,
            region.page_stride // element_size,
            region.rows.kernel_page_rows,
            region.rows.bytes_per_row // element_size,
        )
        for p, (offset, size) in enumerate(region.rows.pieces):
            table[i, 4 + 2 * p : 6 + 2 * p] = (
                offset // element_size,
                size // element_size,
            )
    return table


def _element_size(buf: torch.Tensor, regions: list[RelayoutRegion]) -> int:
    """Widest element (up to 8 bytes) dividing every address the gather
    forms, so it moves whole elements."""
    values = [buf.data_ptr(), buf.numel()]
    for region in regions:
        values += [region.offset, region.page_stride, region.rows.bytes_per_row]
        values += [v for piece in region.rows.pieces for v in piece]
    common = math.gcd(*values)
    return next(size for size in (8, 4, 2, 1) if common % size == 0)


def row_table(
    src_blocks: np.ndarray,
    src_rows: np.ndarray,
    dst_blocks: np.ndarray,
    dst_rows: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """The `(4, rows)` table a `RowGather` reads, on `device`: source page,
    source row, destination page and destination row of each row (a negative
    source page zeroes the destination row). Copied asynchronously from
    pinned memory, ordered on the current stream."""
    rows = np.stack([src_blocks, src_rows, dst_blocks, dst_rows]).astype(np.int64)
    return async_tensor_h2d(rows, device, torch.int64)


class RowGather:
    """The row gather of the regions of one flat uint8 buffer on the device:
    for every region and row `i` of a row table (`row_table`), move row
    `src_rows[i]` of page `src_blocks[i]` to row `dst_rows[i]` of page
    `dst_blocks[i]`, or zero it if the source page is negative. Source and
    destination rows must not overlap.

    The region table is built on the device once."""

    def __init__(self, buf: torch.Tensor, regions: list[RelayoutRegion]) -> None:
        assert buf.dtype == torch.uint8 and buf.dim() == 1
        self.device = buf.device
        self.regions = regions
        self._max_pieces = max((len(r.rows.pieces) for r in regions), default=0)
        element_size = _element_size(buf, regions)
        self._elements = buf.view(_ELEMENT_DTYPES[element_size])
        self._table = torch.from_numpy(
            _region_table(regions, element_size, self._max_pieces)
        ).to(buf.device)

    def __call__(self, rows: torch.Tensor) -> None:
        """Run the gather of `rows` (`row_table`) on the current stream."""
        num_rows = rows.shape[1]
        if num_rows == 0 or not self.regions:
            return
        _relayout_rows_kernel[(num_rows, len(self.regions))](
            self._elements,
            rows,
            num_rows,
            self._table,
            self._table.shape[1],
            MAX_PIECES=self._max_pieces,
            BLOCK=_COPY_BLOCK,
        )


@triton.jit
def _relayout_rows_kernel(
    buf_ptr,
    rows_ptr,
    num_rows,
    table_ptr,
    table_width,
    MAX_PIECES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per (row, region): the `KVRowLayout` address arithmetic
    (`KVRowLayout.row_byte_ranges`), in elements."""
    row = tl.program_id(0).to(tl.int64)
    region = tl.program_id(1).to(tl.int64)
    src_block = tl.load(rows_ptr + row)
    src_row = tl.load(rows_ptr + num_rows + row)
    dst_block = tl.load(rows_ptr + 2 * num_rows + row)
    dst_row = tl.load(rows_ptr + 3 * num_rows + row)
    entry = table_ptr + region * table_width
    offset = tl.load(entry)
    page_stride = tl.load(entry + 1)
    kernel_page_rows = tl.load(entry + 2)
    row_size = tl.load(entry + 3)
    has_src = src_block >= 0
    src_base = (
        offset
        + tl.maximum(src_block, 0) * page_stride
        + src_row // kernel_page_rows * kernel_page_rows * row_size
    )
    dst_base = (
        offset
        + dst_block * page_stride
        + dst_row // kernel_page_rows * kernel_page_rows * row_size
    )
    src_r = src_row % kernel_page_rows
    dst_r = dst_row % kernel_page_rows
    for p in tl.static_range(MAX_PIECES):
        piece_offset = tl.load(entry + 4 + 2 * p)
        size = tl.load(entry + 5 + 2 * p)
        src = src_base + piece_offset + src_r * size
        dst = dst_base + piece_offset + dst_r * size
        for start in range(0, size, BLOCK):
            idx = start + tl.arange(0, BLOCK)
            mask = idx < size
            values = tl.load(buf_ptr + src + idx, mask=mask & has_src, other=0)
            tl.store(buf_ptr + dst + idx, values, mask=mask)
