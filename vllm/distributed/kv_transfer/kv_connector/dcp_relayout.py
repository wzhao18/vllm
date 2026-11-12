# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row relayout of DCP-sharded KV between two DCP layouts.

When a KV transfer's two sides shard a sequence differently (different DCP
sizes or interleaves), a destination rank's rows are not a run of whole source
pages: at interleave 1 a decode rank of a DCP twice the prefill's takes every
other row of one prefill rank. Page RDMA cannot express that, so the
destination pulls the source pages that hold its rows into staging pages and
moves the rows into its own pages with a row gather.

`plan_dcp_rows` maps each destination row to its source (rank, page, row) with
the two sides' `DcpLayout`s; `RowRelayout` turns that into the gather of one
cache: row byte ranges from each side's `KVRowLayout`, so packed pages (e.g. a
DSA indexer page that stores a kernel page's values before its scales) move
row by row like plain token-major pages.

A transfer executes this in three steps, all derived from the same plan:

1. Staging (`num_staging_blocks`, scheduler side): the destination reserves
   blocks of its own cache to receive the source rows it needs, in units of
   `gcd(source block, destination block)` rows so that a unit of a source page
   fits inside a destination block.
2. Reads (`plan_staged_relayout`, worker side): which source units to read
   into which staging units.
3. Row gather (`relayout_rows`, device side; `relayout_rows_reference` on the
   host): every destination row is copied from its staged row, region by
   region, and the destination rows past the transferred positions are
   zeroed.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch

from vllm.distributed.kv_transfer.kv_connector.dcp_layout import DcpLayout
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
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


@dataclass(frozen=True)
class RowRelayout:
    """The row gather of one cache from staged source pages into destination
    pages.

    Byte copies are expressed per row piece (see `KVRowLayout`), so one
    relayout handles any page layout both sides share row-wise: the two
    layouts must give each row the same pieces in the same order and sizes,
    which is what makes the KV of one token identical on both sides.
    """

    src_layout: KVRowLayout
    dst_layout: KVRowLayout
    src_block_size: int
    dst_block_size: int

    def __post_init__(self) -> None:
        if [s for _, s in self.src_layout.pieces] != [
            s for _, s in self.dst_layout.pieces
        ]:
            raise ValueError(
                f"KV row layouts differ in their row pieces: source "
                f"{self.src_layout}, destination {self.dst_layout}"
            )

    @property
    def src_page_bytes(self) -> int:
        return self.src_block_size * self.src_layout.bytes_per_row

    @property
    def dst_page_bytes(self) -> int:
        return self.dst_block_size * self.dst_layout.bytes_per_row

    def byte_copies(
        self,
        src_slots: np.ndarray,
        src_rows: np.ndarray,
        dst_slots: np.ndarray,
        dst_rows: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`(src offset, dst offset, length)` of every byte copy that moves
        rows `src_rows` of staging slots `src_slots` into rows `dst_rows` of
        destination slots `dst_slots` (one entry per row; slots index pages
        of the respective buffers)."""
        return _row_byte_copies(
            self.src_layout,
            self.dst_layout,
            np.asarray(src_slots, dtype=np.int64) * self.src_page_bytes,
            src_rows,
            np.asarray(dst_slots, dtype=np.int64) * self.dst_page_bytes,
            dst_rows,
        )


def _row_byte_copies(
    src_layout: KVRowLayout,
    dst_layout: KVRowLayout,
    src_page_offsets: np.ndarray,
    src_rows: np.ndarray,
    dst_page_offsets: np.ndarray,
    dst_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(src offset, dst offset, length)` of every byte copy moving row
    `src_rows[i]` of the page at byte `src_page_offsets[i]` to row
    `dst_rows[i]` of the page at `dst_page_offsets[i]`, piece by piece."""
    src_off = []
    dst_off = []
    lengths = []
    for piece in range(len(src_layout.pieces)):
        s = _piece_offsets(src_layout, piece, src_rows)
        d = _piece_offsets(dst_layout, piece, dst_rows)
        src_off.append(src_page_offsets + s)
        dst_off.append(dst_page_offsets + d)
        lengths.append(np.full_like(src_rows, src_layout.pieces[piece][1]))
    return (
        np.concatenate(src_off).astype(np.int64),
        np.concatenate(dst_off).astype(np.int64),
        np.concatenate(lengths).astype(np.int64),
    )


def _piece_offsets(layout: KVRowLayout, piece: int, rows: np.ndarray) -> np.ndarray:
    """Offset in its page of piece `piece` of each row in `rows` (see
    `KVRowLayout.row_byte_ranges`, vectorized)."""
    kernel_page, r = np.divmod(rows, layout.kernel_page_rows)
    offset, size = layout.pieces[piece]
    return (
        kernel_page * layout.kernel_page_rows * layout.bytes_per_row + offset + r * size
    )


def copy_byte_ranges_reference(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_off: np.ndarray,
    dst_off: np.ndarray,
    lengths: np.ndarray,
) -> None:
    """Reference for the relayout gather on flat uint8 buffers: the semantics
    the device kernel must match, used by the CPU tests."""
    assert src.dtype == dst.dtype == torch.uint8 and src.dim() == dst.dim() == 1
    for s, d, n in zip(src_off.tolist(), dst_off.tolist(), lengths.tolist()):
        dst[d : d + n] = src[s : s + n]


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
    period = math.lcm(src.dcp_size * src.block_size, dst.dcp_size * dst.block_size) * 4
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


def relayout_byte_copies(
    regions: list[RelayoutRegion],
    src_blocks: np.ndarray,
    src_rows: np.ndarray,
    dst_blocks: np.ndarray,
    dst_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(src offset, dst offset, length)` in the buffer of every byte copy of
    the row gather; a negative source offset zeroes the destination range."""
    zero = np.asarray(src_blocks) < 0
    src_blocks = np.where(zero, 0, src_blocks)
    copies = []
    for region in regions:
        src_off, dst_off, lengths = _row_byte_copies(
            region.rows,
            region.rows,
            region.offset + src_blocks * region.page_stride,
            src_rows,
            region.offset + np.asarray(dst_blocks, dtype=np.int64) * region.page_stride,
            dst_rows,
        )
        src_off[np.tile(zero, len(region.rows.pieces))] = -1
        copies.append((src_off, dst_off, lengths))
    if not copies:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    return tuple(np.concatenate(c) for c in zip(*copies))  # type: ignore[return-value]


def relayout_rows_reference(
    buf: torch.Tensor,
    regions: list[RelayoutRegion],
    src_blocks: np.ndarray,
    src_rows: np.ndarray,
    dst_blocks: np.ndarray,
    dst_rows: np.ndarray,
) -> None:
    """Host reference of `relayout_rows` on a flat uint8 buffer: the
    semantics the device kernel must match, used by the CPU tests."""
    assert buf.dtype == torch.uint8 and buf.dim() == 1
    src_off, dst_off, lengths = relayout_byte_copies(
        regions, src_blocks, src_rows, dst_blocks, dst_rows
    )
    zero = src_off < 0
    buf_zero = torch.zeros(1, dtype=torch.uint8)
    for d, n in zip(dst_off[zero].tolist(), lengths[zero].tolist()):
        buf[d : d + n] = buf_zero
    copy_byte_ranges_reference(
        buf.clone(), buf, src_off[~zero], dst_off[~zero], lengths[~zero]
    )


_MAX_ROW_PIECES = 4
_ELEMENT_DTYPES = {8: torch.int64, 4: torch.int32, 2: torch.int16, 1: torch.uint8}


def _region_table(regions: list[RelayoutRegion], element_size: int) -> np.ndarray:
    """One row per region, in `element_size` units: offset, page stride,
    kernel page rows, row size, then (offset, size) of each row piece, padded
    with empty pieces."""
    width = 4 + 2 * _MAX_ROW_PIECES
    table = np.zeros((len(regions), width), dtype=np.int64)
    for i, region in enumerate(regions):
        if len(region.rows.pieces) > _MAX_ROW_PIECES:
            raise ValueError(f"{region.rows} has more than {_MAX_ROW_PIECES} pieces")
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


def relayout_rows(
    buf: torch.Tensor,
    regions: list[RelayoutRegion],
    src_blocks: np.ndarray,
    src_rows: np.ndarray,
    dst_blocks: np.ndarray,
    dst_rows: np.ndarray,
) -> None:
    """Row gather on the device, on the current stream: for every region and
    row `i`, move row `src_rows[i]` of page `src_blocks[i]` to row
    `dst_rows[i]` of page `dst_blocks[i]` (zero it if the source page is
    negative). Same semantics as `relayout_rows_reference`; source and
    destination rows must not overlap."""
    assert buf.dtype == torch.uint8 and buf.dim() == 1
    num_rows = len(src_blocks)
    if num_rows == 0 or not regions:
        return
    element_size = _element_size(buf, regions)
    elements = buf.view(_ELEMENT_DTYPES[element_size])
    table = torch.from_numpy(_region_table(regions, element_size)).to(buf.device)
    rows = torch.from_numpy(
        np.stack([src_blocks, src_rows, dst_blocks, dst_rows]).astype(np.int64)
    ).to(buf.device)
    _relayout_rows_kernel[(num_rows, len(regions))](
        elements,
        rows,
        num_rows,
        table,
        table.shape[1],
        MAX_PIECES=_MAX_ROW_PIECES,
        BLOCK=256,
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
    """One program per (row, region): the `KVRowLayout` address arithmetic of
    `_piece_offsets`, in elements."""
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
