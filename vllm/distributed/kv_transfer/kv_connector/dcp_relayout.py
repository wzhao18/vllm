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
"""

from dataclasses import dataclass

import numpy as np
import torch

from vllm.distributed.kv_transfer.kv_connector.dcp_layout import DcpLayout
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
        src_off = []
        dst_off = []
        lengths = []
        for piece in range(len(self.src_layout.pieces)):
            s = _piece_offsets(self.src_layout, piece, src_rows)
            d = _piece_offsets(self.dst_layout, piece, dst_rows)
            src_off.append(src_slots * self.src_page_bytes + s)
            dst_off.append(dst_slots * self.dst_page_bytes + d)
            lengths.append(
                np.full_like(src_rows, self.src_layout.pieces[piece][1])
            )
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
    return kernel_page * layout.kernel_page_rows * layout.bytes_per_row + offset + r * size


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
