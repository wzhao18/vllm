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
   blocks of its own cache to receive the source pages that hold the rows it
   needs.
2. Reads (`plan_staged_relayout`, worker side): which source units to read
   into which staging units. A unit is a run of `unit_rows` rows' worth of
   bytes of a source kernel block (pages split for the transfer, e.g.
   `gcd(source block, destination block)` rows), read as it is: its bytes
   are in the source's page layout, which another block size can change
   (an indexer page keeps its values before its scales, so a unit of a
   longer page is not a run of rows). Every unit of a source kernel block
   that holds a planned row is read, into consecutive staging units.
3. Row gather (`RowGather`, on the device): every destination row is copied
   piece by piece from its staged source kernel block, addressed with the
   source's `KVRowLayout`, into the destination's layout, region by region;
   the destination rows past the transferred positions are zeroed.
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


def num_staging_blocks(src: DcpLayout, dst: DcpLayout, start: int, end: int) -> int:
    """Destination blocks to reserve for staging positions `[start, end)`.

    A block id names a page on every destination rank, so this is the most
    that any destination rank needs: the whole source pages holding its rows.
    A worker staging finer units (whole source kernel blocks of units that
    tile both sides' pages) needs no more.
    """
    most = max(
        _num_source_pages(plan_dcp_rows(src, dst, rank, start, end), src)
        for rank in range(dst.dcp_size)
    )
    return cdiv(most * src.block_size, dst.block_size)


def _num_source_pages(plan: DcpRowPlan, src: DcpLayout) -> int:
    return sum(len(pages) for pages in plan.source_pages(src).values())


def relayout_read_amplification(src: DcpLayout, dst: DcpLayout) -> float:
    """Source rows the destination ranks read per row they keep, over a long
    sequence, reading whole source pages (e.g. 2.0 when every destination
    rank keeps every other row of the pages it reads)."""
    # Pages of both sides tile one period of positions whole (a page of a
    # rank spans `dcp_size * block_size` positions), so the pages read repeat
    # with it and one period gives the exact ratio.
    period = math.lcm(src.dcp_size * src.block_size, dst.dcp_size * dst.block_size)
    read = sum(
        _num_source_pages(plan_dcp_rows(src, dst, rank, 0, period), src)
        for rank in range(dst.dcp_size)
    )
    return read * src.block_size / period


@dataclass(frozen=True)
class StagedRelayout:
    """One request's relayout on one destination rank: the reads into
    staging and the row gather after them.

    The gather names each source row by the staged copy of its source kernel
    block: `src_slots[i]` is the slot (index into `staging_units`) of the
    block's first unit, its other units following in the next slots, and
    `src_rows[i]` the row in that block. A negative slot zeroes the
    destination row. Destination rows are rows of the destination cache's
    physical (kernel) blocks.
    """

    reads: dict[int, tuple[list[int], list[int]]]
    """Source rank -> `(source units, staging units)` to read, each named as
    `block id * units per block + unit in block` on its side."""
    src_slots: np.ndarray
    src_rows: np.ndarray
    dst_blocks: np.ndarray
    dst_rows: np.ndarray
    staging_units: np.ndarray
    """The staging unit of each slot, in units of the destination's kernel
    blocks (`kernel block * units per kernel block + unit in it`)."""


def plan_staged_relayout(
    src: DcpLayout,
    dst: DcpLayout,
    dst_rank: int,
    start: int,
    end: int,
    *,
    unit_rows: int,
    src_kernel_block_size: int,
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
    - Every unit (`unit_rows` rows' worth of bytes) of each source kernel
      block holding a planned row is read, into consecutive units of
      `staging_block_ids`. Units must tile both sides' kernel blocks, and
      kernel blocks their pages.
    """
    for rows, of in (
        (src_kernel_block_size, src.block_size),
        (unit_rows, src_kernel_block_size),
        (dst_kernel_block_size, dst.block_size),
        (unit_rows, dst_kernel_block_size),
    ):
        if of % rows:
            raise ValueError(
                f"{rows}-row units do not tile {of}-row blocks ({src} -> {dst}, "
                f"{src_kernel_block_size}/{dst_kernel_block_size}-row kernel "
                f"blocks, {unit_rows}-row read units)"
            )
    plan = plan_dcp_rows(src, dst, dst_rank, start, end)

    # Source kernel blocks holding planned rows, each read whole.
    src_pages, src_page_rows = src.page_row(plan.src_rows)
    src_kernel, src_rows = np.divmod(
        np.asarray(src_page_rows, dtype=np.int64), src_kernel_block_size
    )
    kernel_per_page = src.block_size // src_kernel_block_size
    keys = np.asarray(src_pages, dtype=np.int64) * kernel_per_page + src_kernel
    stride = (int(keys.max()) + 1) if len(keys) else 1
    blocks, block_of_row = np.unique(
        plan.src_ranks * stride + keys, return_inverse=True
    )
    ranks, kernel_keys = np.divmod(blocks, stride)
    pages, kernel = np.divmod(kernel_keys, kernel_per_page)
    if len(pages) and pages.max() >= len(src_block_ids):
        raise ValueError(
            f"Source pages {len(src_block_ids)} do not cover positions "
            f"[{start}, {end}) of {src}"
        )
    units_per_kernel = src_kernel_block_size // unit_rows
    sub = np.tile(np.arange(units_per_kernel, dtype=np.int64), len(blocks))
    src_unit_ids = (
        np.repeat(
            np.asarray(src_block_ids, dtype=np.int64)[pages]
            * (src.block_size // unit_rows)
            + kernel * units_per_kernel,
            units_per_kernel,
        )
        + sub
    )
    unit_ranks = np.repeat(ranks, units_per_kernel)
    # Staging units, consecutive over the staging blocks.
    dst_units_per_page = dst.block_size // unit_rows
    num_units = len(src_unit_ids)
    if num_units > len(staging_block_ids) * dst_units_per_page:
        raise ValueError(
            f"{num_units} staging units need more than the "
            f"{len(staging_block_ids)} staging blocks"
        )
    staging_pages, staging_sub = np.divmod(
        np.arange(num_units, dtype=np.int64), dst_units_per_page
    )
    # Unit ids of the staging blocks; also those of their kernel blocks, as
    # kernel blocks tile the pages and units tile the kernel blocks.
    staging_units = (
        np.asarray(staging_block_ids, dtype=np.int64)[staging_pages]
        * dst_units_per_page
        + staging_sub
    )
    reads = {
        int(rank): (
            src_unit_ids[unit_ranks == rank].tolist(),
            staging_units[unit_ranks == rank].tolist(),
        )
        for rank in np.unique(ranks)
    }

    # Gather: planned rows from their staged kernel blocks, then zero the tail.
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
    dst_local_rows = np.arange(lo, (first_page + len(dst_block_ids)) * dst.block_size)
    dst_pages, dst_page_rows = dst.page_row(dst_local_rows)
    dst_blocks, dst_rows = _kernel_block_rows(
        np.asarray(dst_block_ids, dtype=np.int64)[dst_pages - first_page],
        dst_page_rows,
        kernel_blocks_per_page,
        dst_kernel_block_size,
    )
    num_zero = len(dst_local_rows) - len(src_rows)
    return StagedRelayout(
        reads=reads,
        src_slots=np.concatenate(
            [block_of_row.reshape(-1) * units_per_kernel, np.full(num_zero, -1)]
        ).astype(np.int64),
        src_rows=np.concatenate([src_rows, np.zeros(num_zero, np.int64)]),
        dst_blocks=dst_blocks,
        dst_rows=dst_rows,
        staging_units=staging_units,
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


def check_staged_rows(src_rows: KVRowLayout, dst_rows: KVRowLayout, unit_rows: int):
    """Raise ValueError unless rows staged in `src_rows`' layout, in units of
    `unit_rows` rows' worth of bytes, can be gathered into `dst_rows`' layout:
    the rows must have the same pieces (what one token's KV is), and no piece
    of a row may straddle two units."""
    sizes = [size for _, size in src_rows.pieces]
    if sizes != [size for _, size in dst_rows.pieces]:
        raise ValueError(
            f"KV row layouts differ in their row pieces: source {src_rows}, "
            f"destination {dst_rows}"
        )
    if unit_rows % src_rows.kernel_page_rows == 0:
        return  # Units hold whole kernel pages: they end between rows.
    unit_bytes = unit_rows * src_rows.bytes_per_row
    kernel_page_bytes = src_rows.kernel_page_rows * src_rows.bytes_per_row
    for offset, size in src_rows.pieces:
        if unit_bytes % size or offset % size or kernel_page_bytes % size:
            raise ValueError(
                f"{unit_rows}-row units of pages laid out as {src_rows} split "
                f"the {size}-byte row pieces"
            )


_ELEMENT_DTYPES = {8: torch.int64, 4: torch.int32, 2: torch.int16, 1: torch.uint8}
# Elements one loop step of the gather kernel copies of a row piece; longer
# pieces take several steps.
_COPY_BLOCK = 256
# Region table columns before the pieces, and per piece.
_REGION_COLUMNS = 7
_PIECE_COLUMNS = 3


def _region_table(
    regions: list[RelayoutRegion],
    src_rows: list[KVRowLayout],
    unit_rows: int,
    element_size: int,
    max_pieces: int,
) -> np.ndarray:
    """One row per region, in `element_size` units: offset, page stride,
    kernel page rows, row size, source kernel page rows, staging units per
    page and unit size, then (offset, size, source offset) of each row piece,
    padded with empty pieces to `max_pieces`."""
    table = np.zeros(
        (len(regions), _REGION_COLUMNS + _PIECE_COLUMNS * max_pieces), dtype=np.int64
    )
    for i, (region, src) in enumerate(zip(regions, src_rows)):
        row_size = region.rows.bytes_per_row
        table[i, :_REGION_COLUMNS] = (
            region.offset // element_size,
            region.page_stride // element_size,
            region.rows.kernel_page_rows,
            row_size // element_size,
            src.kernel_page_rows,
            region.page_rows // unit_rows,
            unit_rows * row_size // element_size,
        )
        for p, ((offset, size), (src_offset, _)) in enumerate(
            zip(region.rows.pieces, src.pieces)
        ):
            at = _REGION_COLUMNS + _PIECE_COLUMNS * p
            table[i, at : at + _PIECE_COLUMNS] = (
                offset // element_size,
                size // element_size,
                src_offset // element_size,
            )
    return table


def _element_size(
    buf: torch.Tensor, regions: list[RelayoutRegion], src_rows: list[KVRowLayout]
) -> int:
    """Widest element (up to 8 bytes) dividing every address the gather
    forms, so it moves whole elements."""
    values = [buf.data_ptr(), buf.numel()]
    for region, src in zip(regions, src_rows):
        values += [region.offset, region.page_stride, region.rows.bytes_per_row]
        values += [v for piece in region.rows.pieces for v in piece]
        values += [offset for offset, _ in src.pieces]
    common = math.gcd(*values)
    return next(size for size in (8, 4, 2, 1) if common % size == 0)


def row_table(job: StagedRelayout, device: torch.device) -> torch.Tensor:
    """The table a `RowGather` reads, on `device`: source slot, source row,
    destination page and destination row of each row (`StagedRelayout`), then
    the staging unit of each slot. Copied asynchronously from pinned memory,
    ordered on the current stream."""
    rows = np.concatenate(
        [job.src_slots, job.src_rows, job.dst_blocks, job.dst_rows, job.staging_units]
    ).astype(np.int64)
    return async_tensor_h2d(rows, device, torch.int64)


def concat_staged_relayouts(jobs: list[StagedRelayout]) -> StagedRelayout:
    """The gather of several jobs as one: slots renumbered past the slots of
    the jobs before (reads are not merged)."""
    first_slot = np.cumsum([0] + [len(job.staging_units) for job in jobs[:-1]])
    return StagedRelayout(
        reads={},
        src_slots=np.concatenate(
            [
                np.where(job.src_slots < 0, -1, job.src_slots + first)
                for job, first in zip(jobs, first_slot)
            ]
        ),
        src_rows=np.concatenate([job.src_rows for job in jobs]),
        dst_blocks=np.concatenate([job.dst_blocks for job in jobs]),
        dst_rows=np.concatenate([job.dst_rows for job in jobs]),
        staging_units=np.concatenate([job.staging_units for job in jobs]),
    )


class RowGather:
    """The row gather of the regions of one flat uint8 buffer on the device,
    from one source's staged pages: for every region and row `i` of a row
    table (`row_table`), copy row `src_rows[i]` of the source kernel block
    staged from slot `src_slots[i]` on, laid out as that region's
    `src_rows` layout, into row `dst_rows[i]` of page `dst_blocks[i]`, or
    zero it if the slot is negative. Staged units hold `unit_rows` rows'
    worth of bytes. Source and destination rows must not overlap.

    The region table is built on the device once."""

    def __init__(
        self,
        buf: torch.Tensor,
        regions: list[RelayoutRegion],
        src_rows: list[KVRowLayout],
        unit_rows: int,
    ) -> None:
        assert buf.dtype == torch.uint8 and buf.dim() == 1
        if len(src_rows) != len(regions):
            raise ValueError("Every region needs its source row layout")
        for region, src in zip(regions, src_rows):
            check_staged_rows(src, region.rows, unit_rows)
            if region.page_rows % unit_rows:
                raise ValueError(
                    f"{unit_rows}-row units do not tile the pages of {region}"
                )
        self.device = buf.device
        self.regions = regions
        self._max_pieces = max((len(r.rows.pieces) for r in regions), default=0)
        element_size = _element_size(buf, regions, src_rows)
        self._elements = buf.view(_ELEMENT_DTYPES[element_size])
        self._table = torch.from_numpy(
            _region_table(regions, src_rows, unit_rows, element_size, self._max_pieces)
        ).to(buf.device)

    def __call__(self, rows: torch.Tensor, num_rows: int) -> None:
        """Run the gather of the first `num_rows` rows of `rows`
        (`row_table`) on the current stream."""
        if num_rows == 0 or not self.regions:
            return
        _relayout_rows_kernel[(num_rows, len(self.regions))](
            self._elements,
            rows,
            num_rows,
            self._table,
            self._table.shape[1],
            MAX_PIECES=self._max_pieces,
            REGION_COLUMNS=_REGION_COLUMNS,
            PIECE_COLUMNS=_PIECE_COLUMNS,
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
    REGION_COLUMNS: tl.constexpr,
    PIECE_COLUMNS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per (row, region): the `KVRowLayout` address arithmetic
    (`KVRowLayout.row_byte_ranges`) of both sides, in elements, the source
    side spread over the staging units of its kernel block."""
    row = tl.program_id(0).to(tl.int64)
    region = tl.program_id(1).to(tl.int64)
    src_slot = tl.load(rows_ptr + row)
    src_row = tl.load(rows_ptr + num_rows + row)
    dst_block = tl.load(rows_ptr + 2 * num_rows + row)
    dst_row = tl.load(rows_ptr + 3 * num_rows + row)
    slots_ptr = rows_ptr + 4 * num_rows
    entry = table_ptr + region * table_width
    offset = tl.load(entry)
    page_stride = tl.load(entry + 1)
    kernel_page_rows = tl.load(entry + 2)
    row_size = tl.load(entry + 3)
    src_kernel_page_rows = tl.load(entry + 4)
    units_per_page = tl.load(entry + 5)
    unit_size = tl.load(entry + 6)
    has_src = src_slot >= 0
    dst_base = (
        offset
        + dst_block * page_stride
        + dst_row // kernel_page_rows * kernel_page_rows * row_size
    )
    dst_r = dst_row % kernel_page_rows
    # Offset of the source row's kernel page in its staged kernel block.
    src_page = src_row // src_kernel_page_rows * src_kernel_page_rows * row_size
    src_r = src_row % src_kernel_page_rows
    for p in tl.static_range(MAX_PIECES):
        piece = entry + REGION_COLUMNS + PIECE_COLUMNS * p
        piece_offset = tl.load(piece)
        size = tl.load(piece + 1)
        src_offset = tl.load(piece + 2)
        dst = dst_base + piece_offset + dst_r * size
        # The piece in the staged block, and the staging unit holding it.
        at = src_page + src_offset + src_r * size
        unit = tl.load(
            slots_ptr + tl.maximum(src_slot, 0) + at // unit_size,
            mask=has_src & (size > 0),
            other=0,
        )
        src = (
            offset
            + unit // units_per_page * page_stride
            + unit % units_per_page * unit_size
            + at % unit_size
        )
        for start in range(0, size, BLOCK):
            idx = start + tl.arange(0, BLOCK)
            mask = idx < size
            values = tl.load(buf_ptr + src + idx, mask=mask & has_src, other=0)
            tl.store(buf_ptr + dst + idx, values, mask=mask)
