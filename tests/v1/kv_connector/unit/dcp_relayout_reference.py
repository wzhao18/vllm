# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host oracle of the DCP row relayout (`kv_connector.dcp_relayout`): the
byte copies of the row gather, computed from each side's `KVRowLayout`, and
reference copies the device kernel (`RowGather`) must match."""

from dataclasses import dataclass

import numpy as np
import torch

from vllm.distributed.kv_transfer.kv_connector.dcp_relayout import (
    RelayoutRegion,
    StagedRelayout,
)
from vllm.v1.kv_cache_interface import KVRowLayout


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


def relayout_byte_copies(
    regions: list[RelayoutRegion],
    src_rows: list[KVRowLayout],
    unit_rows: int,
    job: StagedRelayout,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(src offset, dst offset, length)` in the buffer of every byte copy of
    the row gather of `job` (`RowGather`); a negative source offset zeroes
    the destination range. Region `r`'s staged kernel blocks are laid out as
    `src_rows[r]`, in staging units of `unit_rows` rows' worth of bytes."""
    zero = job.src_slots < 0
    slots = np.where(zero, 0, job.src_slots)
    dst_blocks = np.asarray(job.dst_blocks, dtype=np.int64)
    copies = []
    for region, src in zip(regions, src_rows):
        unit_bytes = unit_rows * src.bytes_per_row
        units_per_page = region.page_rows // unit_rows
        for piece, (_, size) in enumerate(src.pieces):
            unit_index, within = np.divmod(
                _piece_offsets(src, piece, job.src_rows), unit_bytes
            )
            unit = (
                job.staging_units[slots + unit_index]
                if len(job.staging_units)
                else np.zeros_like(slots)
            )
            src_off = (
                region.offset
                + unit // units_per_page * region.page_stride
                + unit % units_per_page * unit_bytes
                + within
            )
            dst_off = (
                region.offset
                + dst_blocks * region.page_stride
                + _piece_offsets(region.rows, piece, job.dst_rows)
            )
            copies.append(
                (
                    np.where(zero, -1, src_off).astype(np.int64),
                    dst_off.astype(np.int64),
                    np.full(len(dst_off), size, dtype=np.int64),
                )
            )
    if not copies:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    return tuple(np.concatenate(c) for c in zip(*copies))  # type: ignore[return-value]


def relayout_rows_reference(
    buf: torch.Tensor,
    regions: list[RelayoutRegion],
    src_rows: list[KVRowLayout],
    unit_rows: int,
    job: StagedRelayout,
) -> None:
    """Host reference of `RowGather` on a flat uint8 buffer: the semantics
    the device kernel must match."""
    assert buf.dtype == torch.uint8 and buf.dim() == 1
    src_off, dst_off, lengths = relayout_byte_copies(regions, src_rows, unit_rows, job)
    zero = src_off < 0
    buf_zero = torch.zeros(1, dtype=torch.uint8)
    for d, n in zip(dst_off[zero].tolist(), lengths[zero].tolist()):
        buf[d : d + n] = buf_zero
    copy_byte_ranges_reference(
        buf.clone(), buf, src_off[~zero], dst_off[~zero], lengths[~zero]
    )
