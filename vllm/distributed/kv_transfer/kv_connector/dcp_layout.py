# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Where decode context parallelism (DCP) puts a request's KV positions.

The host-side source of the DCP token placement: KV connectors use it to map
a global token position to the rank, local row and page that hold it, on
either side of a transfer. The formulas are those of the attention kernels
(`dcp_global_to_local_index` / `dcp_local_to_global_index` in Triton and
`get_dcp_local_seq_lens` in `vllm.v1.attention.backends.utils`), here on the
host and vectorized with numpy.
"""

from dataclasses import dataclass

import numpy as np

IntOrArray = int | np.ndarray


@dataclass(frozen=True)
class DcpLayout:
    """One engine's KV layout over its DCP (or, for offloading, DCP x PCP)
    ranks.

    Global position `t` lives on rank `t // interleave % dcp_size`, which
    keeps its own positions as consecutive local rows cut into pages of
    `block_size` rows.
    """

    dcp_size: int
    interleave: int
    block_size: int
    """Rows (tokens) per page on one rank."""

    def __post_init__(self) -> None:
        if min(self.dcp_size, self.interleave, self.block_size) < 1:
            raise ValueError(f"Invalid DCP layout {self}")
        # A rank's interleave chunk must not straddle two of its pages: the
        # attention kernels and `adjust_dcp_kv_cache_interleave_size` require
        # whole chunks per page.
        if self.dcp_size > 1 and (
            self.interleave > self.block_size or self.block_size % self.interleave
        ):
            raise ValueError(
                f"Invalid DCP layout {self}: the interleave must divide the block size."
            )

    @property
    def pages_hold_contiguous_positions(self) -> bool:
        """Whether every page of a rank holds one run of consecutive global
        positions: there is one rank, or a page is one interleave chunk."""
        return self.dcp_size == 1 or self.interleave == self.block_size

    def owner(self, pos: IntOrArray) -> IntOrArray:
        """DCP rank holding global position `pos`."""
        return pos // self.interleave % self.dcp_size

    def local_row(self, pos: IntOrArray) -> IntOrArray:
        """Row of global position `pos` on its owner rank."""
        chunk_stride = self.interleave * self.dcp_size
        return pos // chunk_stride * self.interleave + pos % self.interleave

    def global_pos(self, rank: int, row: IntOrArray) -> IntOrArray:
        """Global position of local row `row` of `rank`."""
        return (
            row // self.interleave * (self.dcp_size * self.interleave)
            + rank * self.interleave
            + row % self.interleave
        )

    def num_local_tokens(self, num_tokens: IntOrArray, rank: int) -> IntOrArray:
        """Rows `rank` holds of a sequence of `num_tokens` positions. Its
        rows are the positions below `num_tokens` it owns, since `global_pos`
        increases with the row."""
        base = num_tokens // (self.interleave * self.dcp_size) * self.interleave
        remainder = num_tokens - base * self.dcp_size - rank * self.interleave
        return base + np.clip(remainder, 0, self.interleave)

    def page_row(self, row: IntOrArray) -> tuple[IntOrArray, IntOrArray]:
        """`(page index, row in page)` of local row `row`."""
        return row // self.block_size, row % self.block_size

    def rank_rows(
        self, rank: int, start: int, end: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Local rows of `rank` that hold global positions `[start, end)`, and
        those positions, both in increasing order."""
        rows = np.arange(
            self.num_local_tokens(start, rank),
            self.num_local_tokens(end, rank),
            dtype=np.int64,
        )
        return rows, self.global_pos(rank, rows)
