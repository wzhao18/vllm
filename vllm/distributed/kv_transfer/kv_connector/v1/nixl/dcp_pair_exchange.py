# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCP pair exchange: D-side fix-up of NIXL pulls at DCP interleave 1.

A decoder at DCP ``2W`` pulls from a prefiller at DCP ``W`` exactly as it does
at interleave = block size: D rank ``r = p + W * c`` (``p = r % W``) reads whole
blocks ``c + 2(m + i)`` of P rank ``p`` into its uncached blocks ``m + i``. At
interleave 1 its block ``j`` must instead hold the tokens of residue ``r`` mod
``2W``, which are the parity-``c`` rows of P blocks ``2j`` and ``2j + 1``::

    D_r[j][B/2 * h + y] = P_p[2j + h][2y + c]        h in {0, 1}, y < B/2

Rank ``r`` and its partner ``(r + W) % 2W`` each hold one of those P blocks.
For every D block they share (a unit), each sends the other the rows it needs
and rewrites its own block in place::

    pack:    keep[y] = row 2y + c                send[y] = row 2y + 1 - c
    unpack:  row B/2 * c + y = keep[y]           row B/2 * (1 - c) + y = recv[y]

A unit without a partner (no P block ``2j + 1``) zero-fills rows ``[B/2, B)``
on both ranks.

This module holds the plan shared by the scheduler, the worker and the tests;
the row tables addressing every exchanged row of a packed KV block; and a torch
reference of the exchange, the oracle of the tests.
"""

import enum
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec, SparseCacheRole

# KV cache dtypes whose MLA pages are token-major rows of w bytes.
_ROW_KV_CACHE_DTYPES = ("fp8", "fp8_e4m3", "fp8_ds_mla")


class NixlPairModeError(RuntimeError):
    """A P/D configuration that DCP pair mode cannot serve. Fatal on D: it is
    never turned into a per-request recompute."""


# ----------------------------------------------------------------------------
# Plan
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class PairUnit:
    """One D block whose rows partner ranks exchange; the same on every rank."""

    group: int
    block_id: int
    has_partner: bool
    """Whether P block ``2j + 1`` exists. Without it, rows ``[B/2, B)`` are 0."""


def num_pair_units(
    num_local_blocks: int, num_cached_blocks: int, num_remote_blocks: int
) -> int:
    """Units of one group: the blocks the parity-0 rank pulls.

    Rank ``c`` reads P block ``c + 2(m + i)`` into its ``i``-th uncached block,
    so the parity-0 rank pulls ``min(len(local), ceil((n_R - 2m) / 2))`` blocks.
    The parity-1 rank pulls as many, or one fewer when the last unit has no
    partner: that needs ``n_R - 2m`` odd, and D's list not to be the limit.
    """
    remote_pairs = cdiv(num_remote_blocks - 2 * num_cached_blocks, 2)
    return max(0, min(num_local_blocks, remote_pairs))


def pair_units(
    local_ids: Sequence[Sequence[int]],
    num_cached: Sequence[int],
    remote_ids: Sequence[Sequence[int]],
    groups: Iterable[int],
) -> list[PairUnit]:
    """Exchange plan of one request, in group order.

    All three sequences are indexed like the worker indexes them, by transfer
    group; pair mode requires every KV cache group to be a transfer group, in
    the same order.

    Args:
        local_ids: D's uncached blocks per group: the pull targets.
        num_cached: D's prefix-cache hit per group, in blocks.
        remote_ids: P's block list per group.
        groups: The DCP-sharded attention groups.

    Returns:
        One unit per D block the parity-0 rank pulls. Only the last unit of a
        group can lack a partner.

    """
    units = []
    for g in groups:
        local, remote, m = local_ids[g], remote_ids[g], num_cached[g]
        for i in range(num_pair_units(len(local), m, len(remote))):
            units.append(PairUnit(g, local[i], 2 * (m + i) + 1 < len(remote)))
    return units


def check_pair_coverage(
    num_tokens: int,
    remote_num_tokens: int | None,
    remote_ids: Sequence[Sequence[int]],
    groups: Iterable[int],
    remote_block_tokens: int,
) -> None:
    """Raise unless P computed every token and block D loads.

    A short P would leave D's tail blocks zeroed, which is P/D configuration
    skew rather than a transfer failure.

    Args:
        num_tokens: Tokens D loads (the prompt minus the backoff).
        remote_num_tokens: Tokens P computed; None if P did not report them,
            which leaves the coverage unproven.
        remote_ids: P's block list per group.
        groups: The DCP-sharded attention groups.
        remote_block_tokens: Tokens per P block: block size times P's DCP size.

    """
    groups = list(groups)
    num_blocks = cdiv(num_tokens, remote_block_tokens)
    if (
        remote_num_tokens is None
        or remote_num_tokens < num_tokens
        or any(len(remote_ids[g]) < num_blocks for g in groups)
    ):
        computed = "unreported" if remote_num_tokens is None else remote_num_tokens
        raise NixlPairModeError(
            f"DCP pair exchange: D loads {num_tokens} tokens ({num_blocks} P blocks "
            f"of {remote_block_tokens} tokens), but P computed {computed} "
            f"tokens in {[len(remote_ids[g]) for g in groups]} blocks per group"
        )


@dataclass(frozen=True)
class PairLayout:
    """Rows one D rank moves for every unit. The rows are the same for every
    page: a row is a token's bytes in each layer page of the block."""

    dcp_rank: int
    partner: int
    parity: int
    """``c``: the parity of the P blocks this rank pulled, and the half it keeps.
    Its ``send`` fills half ``c`` of the partner's block."""
    block_size: int

    @property
    def half(self) -> int:
        return self.block_size // 2

    def pack_rows(self) -> tuple[np.ndarray, np.ndarray]:
        """Rows gathered into ``keep`` and into ``send``, in buffer order."""
        y = np.arange(self.half)
        return 2 * y + self.parity, 2 * y + 1 - self.parity

    def unpack_rows(self) -> tuple[np.ndarray, np.ndarray]:
        """Rows ``keep`` and ``recv`` are written to: half ``c`` and half
        ``1 - c``. Together they rewrite the whole block."""
        y = np.arange(self.half)
        return self.half * self.parity + y, self.half * (1 - self.parity) + y

    def zero_rows(self) -> np.ndarray:
        """Rows a unit without a partner zero-fills on both ranks: the half
        that P block ``2j + 1`` fills."""
        return np.arange(self.half, self.block_size)


def pair_layout(
    dcp_rank: int,
    dcp_size: int,
    remote_dcp_size: int,
    block_size: int,
    interleave: int,
) -> PairLayout:
    """Row maps of D rank ``dcp_rank``; raises unless pair mode applies."""
    if interleave != 1 or dcp_size != 2 * remote_dcp_size or block_size % 2:
        raise NixlPairModeError(
            "DCP pair exchange needs interleave 1, an even block size and a D DCP "
            f"twice P's; got interleave {interleave}, block size {block_size}, "
            f"D DCP {dcp_size} and P DCP {remote_dcp_size}"
        )
    if not 0 <= dcp_rank < dcp_size:
        raise ValueError(f"DCP rank {dcp_rank} is outside DCP size {dcp_size}")
    return PairLayout(
        dcp_rank=dcp_rank,
        partner=(dcp_rank + remote_dcp_size) % dcp_size,
        parity=dcp_rank // remote_dcp_size,
        block_size=block_size,
    )


# ----------------------------------------------------------------------------
# Row tables
# ----------------------------------------------------------------------------


class PairArrayKind(enum.Enum):
    """How a layer page stores one array's rows."""

    ROW = "row"
    """Token-major rows filling the page: MLA latent or drafter KV."""
    INDEX_VALUE = "index_value"
    """Indexer values: each kernel page holds its rows' values first ..."""
    INDEX_SCALE = "index_scale"
    """... then their scales."""


@dataclass(frozen=True)
class PairRegion:
    """One NIXL region of a DCP-sharded attention layer: its page of block
    ``b`` is ``page_len`` bytes at ``base_addr + b * block_stride``."""

    layer_name: str
    spec: KVCacheSpec
    base_addr: int
    page_len: int
    block_stride: int


@dataclass(frozen=True)
class PairRowArray:
    """Rows of one layer page. Row ``y`` of block ``b`` is ``width`` bytes at
    ``page_offset + b * S + (y // R) * subpage_stride + offset + (y % R) * width``
    from the table's base, ``S`` being its block stride and ``R``
    ``rows_per_subpage``."""

    layer_name: str
    kind: PairArrayKind
    page_offset: int
    page_len: int
    width: int
    rows_per_subpage: int
    subpage_stride: int
    offset: int


@dataclass(frozen=True)
class PairRowTable:
    """Every exchanged row of one group's blocks, as byte arrays."""

    base_addr: int
    block_stride: int
    block_size: int
    arrays: tuple[PairRowArray, ...]

    @property
    def half_bytes(self) -> int:
        """``H``: one unit's bytes in ``keep``, ``send`` and ``recv``."""
        return self.block_size // 2 * sum(a.width for a in self.arrays)

    def segments(self) -> list[tuple[int, int]]:
        """Byte range of each array within a unit's ``H`` bytes, whose half
        rows are stored array by array, row-major."""
        bounds = np.cumsum([0] + [self.block_size // 2 * a.width for a in self.arrays])
        return list(zip(bounds[:-1].tolist(), bounds[1:].tolist()))

    def row_offsets(
        self, array: PairRowArray, block_ids: Sequence[int], rows: Sequence[int]
    ) -> np.ndarray:
        """int64 ``[len(block_ids), len(rows)]`` byte offsets from ``base_addr``
        of the given rows of each block."""
        b = np.asarray(block_ids, dtype=np.int64)[:, None]
        y = np.asarray(rows, dtype=np.int64)[None, :]
        subpage, row = np.divmod(y, array.rows_per_subpage)
        return (
            array.page_offset
            + b * self.block_stride
            + subpage * array.subpage_stride
            + array.offset
            + row * array.width
        )


def _rejected(region: PairRegion, reason: str) -> NixlPairModeError:
    return NixlPairModeError(
        f"DCP pair exchange cannot exchange layer {region.layer_name}: {reason}"
    )


def _region_arrays(
    region: PairRegion,
    page_offset: int,
    index_kernel_rows: int | None,
    index_value_bytes: int | None,
) -> list[PairRowArray]:
    """Arrays of one whitelisted region; raises for any other page."""
    spec, page_len = region.spec, region.page_len
    if not (
        isinstance(spec, MLAAttentionSpec)
        and spec.dcp_sharded
        and spec.num_heads == 1
        and spec.tokens_per_state == 1
    ):
        raise _rejected(region, f"not a DCP-sharded single-head MLA page: {spec}")
    block_size, width = spec.block_size, spec.state_content_size_bytes
    if page_len != block_size * width:
        raise _rejected(
            region, f"{page_len}-byte page is not {block_size} rows of {width} bytes"
        )
    if spec.cache_role != SparseCacheRole.INDEXER:
        if spec.cache_dtype_str not in _ROW_KV_CACHE_DTYPES:
            raise _rejected(region, f"KV cache dtype {spec.cache_dtype_str}")
        return [
            PairRowArray(
                region.layer_name,
                PairArrayKind.ROW,
                page_offset,
                page_len,
                width,
                rows_per_subpage=block_size,
                subpage_stride=page_len,
                offset=0,
            )
        ]
    rows, values = index_kernel_rows, index_value_bytes
    if (
        rows is None
        or values is None
        or spec.dtype != torch.uint8
        or not 0 < values < width
        or block_size % rows
    ):
        raise _rejected(
            region,
            f"{block_size} {spec.dtype} indexer rows of {width} bytes are not "
            f"{rows}-row kernel pages of {values}-byte values and their scales",
        )
    return [
        PairRowArray(
            region.layer_name,
            kind,
            page_offset,
            page_len,
            row_width,
            rows_per_subpage=rows,
            subpage_stride=rows * width,
            offset=offset,
        )
        for kind, row_width, offset in (
            (PairArrayKind.INDEX_VALUE, values, 0),
            (PairArrayKind.INDEX_SCALE, width - values, rows * values),
        )
    ]


def build_pair_row_table(
    regions: Sequence[PairRegion],
    *,
    index_kernel_rows: int | None = None,
    index_value_bytes: int | None = None,
) -> PairRowTable:
    """Row table of one DCP-sharded attention group, from its NIXL regions.

    Only these pages are exchanged; any other raises naming the layer:
    - MLA pages (latent or drafter): a DCP-sharded ``MLAAttentionSpec`` with one
      KV head and one token per state, an fp8 or fp8_ds_mla KV cache, and rows
      of ``w`` bytes filling the page (``page_len == B * w``).
    - Indexer pages: ``cache_role == INDEXER`` uint8 rows filling the page,
      stored as kernel pages of ``index_kernel_rows`` rows that hold the rows'
      ``index_value_bytes`` values, then their scales.
    All regions must share the block size and block stride, and a block's pages
    must be disjoint and fit within one block stride.

    Args:
        regions: The group's regions, in NIXL region order.
        index_kernel_rows: Rows per kernel page of the indexer cache.
        index_value_bytes: Value bytes of an indexer row; the rest are scales.

    Returns:
        The table, its arrays in region order (indexer values before scales).

    """
    if not regions:
        raise NixlPairModeError("DCP pair exchange: no attention region to exchange")
    block_size = regions[0].spec.block_size
    block_stride = regions[0].block_stride
    base_addr = min(region.base_addr for region in regions)
    arrays: list[PairRowArray] = []
    for region in regions:
        if region.spec.block_size != block_size or region.block_stride != block_stride:
            raise _rejected(
                region,
                f"block size {region.spec.block_size} and stride "
                f"{region.block_stride} differ from {block_size} and {block_stride}",
            )
        arrays += _region_arrays(
            region, region.base_addr - base_addr, index_kernel_rows, index_value_bytes
        )

    pages = sorted((r.base_addr, r.page_len, r.layer_name) for r in regions)
    for (addr, length, name), (next_addr, _, next_name) in zip(pages, pages[1:]):
        if addr + length > next_addr:
            raise NixlPairModeError(
                f"DCP pair exchange: pages of layers {name} and {next_name} overlap"
            )
    if pages[-1][0] + pages[-1][1] - pages[0][0] > block_stride:
        raise NixlPairModeError(
            f"DCP pair exchange: pages of layers {pages[0][2]} to {pages[-1][2]} "
            f"span more than their block stride {block_stride}"
        )
    return PairRowTable(base_addr, block_stride, block_size, tuple(arrays))


# ----------------------------------------------------------------------------
# Torch reference: the test oracle. Never on a serving path.
# ----------------------------------------------------------------------------


def _array_view(
    buf: torch.Tensor, table: PairRowTable, array: PairRowArray, num_blocks: int
) -> torch.Tensor:
    """``[num_blocks, B // R, R, w]`` view of one array of a flat uint8 buffer
    that holds the table's address range."""
    assert buf.dtype == torch.uint8 and buf.dim() == 1
    start = table.base_addr + array.page_offset + array.offset - buf.data_ptr()
    assert start >= 0, "the buffer does not hold the table's addresses"
    rows = array.rows_per_subpage
    return buf.as_strided(
        (num_blocks, table.block_size // rows, rows, array.width),
        (table.block_stride, array.subpage_stride, array.width, 1),
        buf.storage_offset() + start,
    )


def _row_index(
    blocks: Sequence[int],
    rows: np.ndarray,
    rows_per_subpage: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Index of ``rows`` of each block into an ``_array_view``."""
    subpage, row = np.divmod(rows, rows_per_subpage)
    return (
        torch.as_tensor(blocks, dtype=torch.long, device=device)[:, None],
        torch.as_tensor(subpage, dtype=torch.long, device=device),
        torch.as_tensor(row, dtype=torch.long, device=device),
    )


def reference_pack(
    buf: torch.Tensor,
    table: PairRowTable,
    layout: PairLayout,
    block_ids: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """``keep`` and ``send``, ``[len(block_ids), H]`` uint8 each.

    Args:
        buf: The rank's KV storage as a flat uint8 tensor.
        table: Row table addressing ``buf``.
        layout: The rank's row maps.
        block_ids: The units' blocks, in plan order.

    """
    if not block_ids:
        empty = torch.empty((0, table.half_bytes), dtype=torch.uint8, device=buf.device)
        return empty, empty.clone()
    keep: list[torch.Tensor] = []
    send: list[torch.Tensor] = []
    for array in table.arrays:
        view = _array_view(buf, table, array, max(block_ids) + 1)
        for out, rows in zip((keep, send), layout.pack_rows()):
            index = _row_index(block_ids, rows, array.rows_per_subpage, buf.device)
            out.append(view[index].flatten(1))
    return torch.cat(keep, 1), torch.cat(send, 1)


def reference_unpack(
    buf: torch.Tensor,
    table: PairRowTable,
    layout: PairLayout,
    units: Sequence[PairUnit],
    keep: torch.Tensor,
    recv: torch.Tensor,
) -> None:
    """Rewrite the units' blocks of ``buf`` in interleave-1 order, in place.

    Args:
        buf: The rank's KV storage as a flat uint8 tensor.
        table: Row table addressing ``buf``.
        layout: The rank's row maps.
        units: The units ``keep`` and ``recv`` were packed for, in order.
        keep: This rank's ``keep``.
        recv: The partner's ``send``.

    """
    if not units:
        return
    device = buf.device
    block_ids = [u.block_id for u in units]
    lone = torch.as_tensor(
        [i for i, u in enumerate(units) if not u.has_partner],
        dtype=torch.long,
        device=device,
    )
    keep_rows, recv_rows = (
        torch.as_tensor(rows, device=device) for rows in layout.unpack_rows()
    )
    zero_rows = torch.as_tensor(layout.zero_rows(), device=device)
    all_rows = np.arange(table.block_size)
    for array, (lo, hi) in zip(table.arrays, table.segments()):
        half_shape = (len(units), layout.half, array.width)
        rows = torch.empty(
            (len(units), table.block_size, array.width),
            dtype=torch.uint8,
            device=device,
        )
        rows[:, keep_rows] = keep[:, lo:hi].reshape(half_shape)
        rows[:, recv_rows] = recv[:, lo:hi].reshape(half_shape)
        rows[lone[:, None], zero_rows] = 0
        view = _array_view(buf, table, array, max(block_ids) + 1)
        view[_row_index(block_ids, all_rows, array.rows_per_subpage, device)] = rows


def reference_pair_exchange(
    buffers: Sequence[torch.Tensor],
    tables: Sequence[Mapping[int, PairRowTable]],
    units: Sequence[PairUnit],
) -> None:
    """Run the exchange of ``units`` on every D rank, in place.

    The buffers hold today's pull placement; afterwards each unit's block holds
    its rank's interleave-1 rows.

    Args:
        buffers: Each D rank's KV storage as a flat uint8 tensor, by DCP rank.
        tables: Each D rank's row table per group.
        units: The step's units. D rank ``r`` pairs with ``(r + W) % 2W``,
            ``2W`` being the number of buffers.

    """
    dcp_size = len(buffers)
    assert dcp_size % 2 == 0 and len(tables) == dcp_size
    assert len({(u.group, u.block_id) for u in units}) == len(units)
    for group in dict.fromkeys(u.group for u in units):
        group_units = [u for u in units if u.group == group]
        block_ids = [u.block_id for u in group_units]
        layouts = [
            pair_layout(r, dcp_size, dcp_size // 2, tables[r][group].block_size, 1)
            for r in range(dcp_size)
        ]
        packed = [
            reference_pack(buffers[r], tables[r][group], layouts[r], block_ids)
            for r in range(dcp_size)
        ]
        for r, layout in enumerate(layouts):
            reference_unpack(
                buffers[r],
                tables[r][group],
                layout,
                group_units,
                keep=packed[r][0],
                recv=packed[layout.partner][1],
            )


def reference_read_pages(
    dst: torch.Tensor,
    dst_table: PairRowTable,
    dst_blocks: Sequence[int],
    src: torch.Tensor,
    src_table: PairRowTable,
    src_blocks: Sequence[int],
) -> None:
    """Copy every page of ``src_blocks`` into ``dst_blocks``, as a NIXL READ
    of whole pages does."""
    assert len(dst_blocks) == len(src_blocks)
    assert dst_table.block_size == src_table.block_size
    if not dst_blocks:
        return
    for d, s in zip(dst_table.arrays, src_table.arrays, strict=True):
        assert (d.width, d.rows_per_subpage, d.subpage_stride, d.offset) == (
            s.width,
            s.rows_per_subpage,
            s.subpage_stride,
            s.offset,
        )
        dst_view = _array_view(dst, dst_table, d, max(dst_blocks) + 1)
        src_view = _array_view(src, src_table, s, max(src_blocks) + 1)
        dst_view[torch.as_tensor(dst_blocks, device=dst.device)] = src_view[
            torch.as_tensor(src_blocks, device=src.device)
        ].to(dst.device)


def reference_zero_pages(
    buf: torch.Tensor, table: PairRowTable, block_ids: Sequence[int]
) -> None:
    """Zero every page of the given blocks, as unwritten-page zeroing does."""
    if not block_ids:
        return
    for array in table.arrays:
        view = _array_view(buf, table, array, max(block_ids) + 1)
        view[torch.as_tensor(block_ids, device=buf.device)] = 0
