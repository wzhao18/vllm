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
the row tables addressing every exchanged row of a packed KV block; a torch
reference of the exchange, the oracle of the tests; and the D rank's exchanger,
which runs it with Triton kernels and one NCCL round per chunk of units.
"""

import copy
import enum
import hashlib
import os
import socket
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.distributed

from vllm.distributed.parallel_state import get_dcp_group
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import async_tensor_h2d, current_stream
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec, SparseCacheRole

logger = init_logger(__name__)

# KV cache dtypes whose MLA pages are token-major rows of w bytes.
_ROW_KV_CACHE_DTYPES = ("fp8", "fp8_e4m3", "fp8_ds_mla", "nvfp4_ds_mla")


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
    num_sources: int | None = None
    """Ratio ``R = D DCP / P DCP`` above 2 (staged exchange, see
    ``stage_units``): how many of the unit's ``R`` P blocks ``R j + s`` exist.
    None at ratio 2, where it is ``1 + has_partner``."""
    remote_dcp_size: int | None = None
    """P's DCP size ``W`` when the ratio is above 2; None: half this DCP size."""

    @property
    def sources(self) -> int:
        """P blocks of the unit that exist, out of ``R``."""
        return 1 + self.has_partner if self.num_sources is None else self.num_sources


def num_pair_units(
    num_local_blocks: int,
    num_cached_blocks: int,
    num_remote_blocks: int,
    ratio: int = 2,
) -> int:
    """Units of one group: the blocks the source-0 ranks pull.

    Rank ``r`` (source ``s = r // W``) reads P block ``s + R(m + i)`` into its
    ``i``-th uncached block, so the source-0 ranks pull
    ``min(len(local), ceil((n_R - Rm) / R))`` blocks; the others pull as many,
    or one fewer in the last unit when P's blocks run out first.
    """
    remote_units = cdiv(num_remote_blocks - ratio * num_cached_blocks, ratio)
    return max(0, min(num_local_blocks, remote_units))


def pair_units(
    local_ids: Sequence[Sequence[int]],
    num_cached: Sequence[int],
    remote_ids: Sequence[Sequence[int]],
    groups: Iterable[int],
    ratio: int = 2,
    remote_dcp_size: int | None = None,
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
        ratio: ``R``, D's DCP size over P's: a power of two.
        remote_dcp_size: P's DCP size ``W``, recorded on the units when
            ``ratio`` is above 2 (the worker then runs ``log2(R)`` stages).

    Returns:
        One unit per D block the source-0 ranks pull. Only the last unit of a
        group can lack sources.

    """
    staged = ratio > 2
    units = []
    for g in groups:
        local, remote, m = local_ids[g], remote_ids[g], num_cached[g]
        for i in range(num_pair_units(len(local), m, len(remote), ratio)):
            first = ratio * (m + i)
            units.append(
                PairUnit(
                    g,
                    local[i],
                    first + 1 < len(remote),
                    min(ratio, len(remote) - first) if staged else None,
                    remote_dcp_size if staged else None,
                )
            )
    return units


def exchange_strides(dcp_size: int, remote_dcp_size: int) -> list[int]:
    """Partner strides of the exchange stages from P DCP ``W`` to D DCP
    ``R W``: ``W, 2W, ..., R W / 2``. Raises unless ``R`` is a power of two
    of at least 2.

    Stage ``t`` is the pair exchange at stride ``W 2^t``: it swaps the stride
    bit of the rank with the lowest row bit, so after ``log2(R)`` stages rank
    ``r`` holds the tokens of residue ``r`` mod ``R W`` in interleave-1 order.
    """
    ratio, rest = divmod(dcp_size, remote_dcp_size)
    if rest or ratio < 2 or ratio & (ratio - 1):
        raise NixlPairModeError(
            f"DCP pair exchange needs a D DCP ({dcp_size}) that is P's "
            f"({remote_dcp_size}) times a power of two"
        )
    return [remote_dcp_size << t for t in range(ratio.bit_length() - 1)]


def exchange_chunks(
    units: Sequence[PairUnit], dcp_size: int, chunk_units: int
) -> int:
    """Chunk rounds an exchange of ``units`` takes on the rank that runs the
    most (``DcpStagedExchanger.exchange``: per P DCP size, per stage, units in
    chunks of ``chunk_units``; every stage counted with all the size's units,
    an upper bound of ``stage_units``). The scheduler's per-step budget."""
    by_remote: dict[int, int] = {}
    for u in units:
        remote = u.remote_dcp_size or dcp_size // 2
        by_remote[remote] = by_remote.get(remote, 0) + 1
    return sum(
        len(exchange_strides(dcp_size, remote)) * cdiv(n, chunk_units)
        for remote, n in by_remote.items()
    )


def pair_half_bytes_bound(kv_cache_spec: Mapping[str, KVCacheSpec]) -> int:
    """An upper bound of a unit half (``PairRowTable.half_bytes``) from the
    layer specs, before the caches exist: half of one block of every
    DCP-sharded attention layer (the exchanged group holds a subset of them;
    the exchanger checks its real half against this bound)."""
    from vllm.v1.kv_cache_interface import AttentionSpec

    total = sum(
        spec.page_size_bytes
        for spec in kv_cache_spec.values()
        if isinstance(spec, AttentionSpec) and getattr(spec, "dcp_sharded", False)
    )
    return cdiv(total, 2)


def reserved_device_bytes(half_bytes: int, chunk_units: int) -> int:
    """Device bytes of one rank's exchange buffers at a unit half of
    ``half_bytes``: ``keep``, ``send`` and ``recv`` in ``PAIR_RING_SLOTS``
    chunk slots. Taken from the KV cache budget at profiling
    (``NixlBaseConnector.get_reserved_device_memory``)."""
    return 3 * PAIR_RING_SLOTS * chunk_units * half_bytes


def all_exchange_strides(dcp_size: int) -> list[int]:
    """Every stride a P DCP size can need at this D DCP size (largest first):
    ``dcp_size / 2^k`` while it is an integer."""
    strides = []
    stride = dcp_size
    while stride % 2 == 0:
        stride //= 2
        strides.append(stride)
    return strides


def stage_units(
    units: Sequence[PairUnit], dcp_rank: int, stride: int, remote_dcp_size: int
) -> list[PairUnit]:
    """The units rank ``dcp_rank`` exchanges at stage ``stride`` (a staged
    exchange's ``exchange_strides``), with their stage partner flag.

    At stage ``t`` the pair holds the P blocks of sources ``[low, low + 2^t)``
    (parity 0) and ``[high, high + 2^t)`` (parity 1), ``high = low + 2^t``.
    Sources past the unit's ``sources`` do not exist and stand for zero
    blocks: without parity-1 sources the unit has no partner (rows
    ``[B/2, B)`` are zeroed, as in the one-stage exchange), and without
    parity-0 sources both ranks hold nothing yet and skip it. Partners compute
    the same list.
    """
    t = (stride // remote_dcp_size).bit_length() - 1
    low = ((dcp_rank // remote_dcp_size) >> (t + 1)) << (t + 1)
    high = low | (1 << t)
    return [
        PairUnit(u.group, u.block_id, high < u.sources, u.sources, remote_dcp_size)
        for u in units
        if low < u.sources
    ]


def expand_sub_block_ids(
    block_ids: Sequence[int],
    sub_blocks: int,
    num_tokens: int,
    sub_block_tokens: int,
) -> list[int]:
    """A prefiller's block list as ``sub_blocks`` sub-blocks per block, flat
    id ``sub_blocks * b + q``, ending at the sub-block of the last computed
    token.

    The list ends at the block holding token ``num_tokens - 1`` (P clips
    lookahead blocks, and a sliding window keeps only its tail), so the
    sub-blocks past that token are dropped from the end.

    Args:
        block_ids: P's blocks of one group.
        sub_blocks: ``X``, P's per-rank block over D's.
        num_tokens: Tokens P computed.
        sub_block_tokens: Tokens per sub-block: D's per-rank block times P's
            DCP size for a DCP-sharded group, else D's block.
    """
    num_blocks = cdiv(num_tokens, sub_block_tokens * sub_blocks)
    block_ids = list(block_ids)[:num_blocks]
    expanded = [sub_blocks * b + q for b in block_ids for q in range(sub_blocks)]
    past_end = sub_blocks * num_blocks - cdiv(num_tokens, sub_block_tokens)
    return expanded[: len(expanded) - past_end]


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
    dcp_size: int | None = None
    """The DCP size; None: twice the partner distance (one-stage exchange)."""

    @property
    def stride(self) -> int:
        return abs(self.partner - self.dcp_rank)

    @property
    def group_size(self) -> int:
        return 2 * self.stride if self.dcp_size is None else self.dcp_size

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
    stride: int | None = None,
) -> PairLayout:
    """Row maps of D rank ``dcp_rank`` at the exchange stage of partner
    ``stride`` (default: P's DCP size, the first stage); raises unless pair
    mode applies: interleave 1, an even block size and a D DCP that is P's
    times a power of two (one stage at ratio 2, ``log2`` of it above)."""
    if interleave != 1 or block_size % 2:
        raise NixlPairModeError(
            "DCP pair exchange needs interleave 1 and an even block size; got "
            f"interleave {interleave}, block size {block_size}, D DCP {dcp_size} "
            f"and P DCP {remote_dcp_size}"
        )
    strides = exchange_strides(dcp_size, remote_dcp_size)
    stride = remote_dcp_size if stride is None else stride
    if stride not in strides:
        raise NixlPairModeError(
            f"DCP pair exchange: stride {stride} is not a stage of D DCP "
            f"{dcp_size} over P DCP {remote_dcp_size} (stages {strides})"
        )
    if not 0 <= dcp_rank < dcp_size:
        raise ValueError(f"DCP rank {dcp_rank} is outside DCP size {dcp_size}")
    parity = (dcp_rank // stride) % 2
    return PairLayout(
        dcp_rank=dcp_rank,
        partner=dcp_rank - stride if parity else dcp_rank + stride,
        parity=parity,
        block_size=block_size,
        dcp_size=None if len(strides) == 1 else dcp_size,
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
      KV head and one token per state, an fp8, fp8_ds_mla or nvfp4_ds_mla KV
      cache, and rows of ``w`` bytes filling the page (``page_len == B * w``).
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
        by_remote: dict[int, list[PairUnit]] = {}
        for u in units:
            if u.group == group:
                remote = u.remote_dcp_size or dcp_size // 2
                by_remote.setdefault(remote, []).append(u)
        for remote, group_units in by_remote.items():
            for stride in exchange_strides(dcp_size, remote):
                _reference_stage(buffers, tables, group, group_units, remote, stride)


def _reference_stage(
    buffers: Sequence[torch.Tensor],
    tables: Sequence[Mapping[int, PairRowTable]],
    group: int,
    units: Sequence[PairUnit],
    remote_dcp_size: int,
    stride: int,
) -> None:
    """One exchange stage of ``units`` on every D rank, in place."""
    dcp_size = len(buffers)
    layouts = [
        pair_layout(
            r, dcp_size, remote_dcp_size, tables[r][group].block_size, 1, stride
        )
        for r in range(dcp_size)
    ]
    rank_units = [
        stage_units(units, r, stride, remote_dcp_size) for r in range(dcp_size)
    ]
    packed = [
        reference_pack(
            buffers[r],
            tables[r][group],
            layouts[r],
            [u.block_id for u in rank_units[r]],
        )
        for r in range(dcp_size)
    ]
    for r, layout in enumerate(layouts):
        reference_unpack(
            buffers[r],
            tables[r][group],
            layout,
            rank_units[r],
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


# ----------------------------------------------------------------------------
# Exchanger: Triton pack/unpack and one NCCL round with the partner per chunk
# ----------------------------------------------------------------------------

# Vectors one program moves into or out of each buffer, and its warps.
_TILE_VECTORS = 256
_NUM_WARPS = 4
# Warm-up fails startup below this much free device memory.
_MIN_FREE_BYTES = 1 << 30

# Chunk slots of the side-stream exchange (PADMIT ADMISSION STALL, 2026-11-22).
# A step's packs run on the current stream in front of its forward. With one
# slot, the pack of chunk k + 1 waited for chunk k's peer copy and unpack on the
# side stream, so a burst of C chunks serialized C copies (~0.13 ms each at 4
# units) in front of the forward: 20 units took ~4.6 ms of the main stream
# instead of ~0.8 ms of packs (Hecate c576 CUPTI). With ``PAIR_RING_SLOTS``
# slots, a pack waits only for the chunk that used its slot ``PAIR_RING_SLOTS``
# chunks earlier. The scheduler emits at most this many chunks per step
# (``exchange_chunks``), so a step's packs never wait on its own swaps.
PAIR_RING_SLOTS = 4
# One-unit NCCL swaps timed at warm-up; the fastest is logged.
_TIMED_SWAPS = 3
_STAMP_MAGIC = int.from_bytes(b"NIXLPAIR", "big")
# Settings that decide NCCL's transport and P2P channels between the partners.
_NCCL_TRANSPORT_ENV = (
    "NCCL_MNNVL_ENABLE",
    "NCCL_P2P_DISABLE",
    "NCCL_P2P_LEVEL",
    "NCCL_NET",
    "NCCL_IB_DISABLE",
    "NCCL_SHM_DISABLE",
    "NCCL_CUMEM_ENABLE",
    "NCCL_NVLS_ENABLE",
    "NCCL_MIN_P2P_NCHANNELS",
    "NCCL_MAX_P2P_NCHANNELS",
)


@triton.jit
def _row_words(
    row,
    column,
    subpage_words,
    ROW_WORDS: tl.constexpr,
    ROWS_PER_SUBPAGE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    """int32 offsets in words, from the array's page, of word ``column`` of
    ``row``."""
    if ROWS_PER_SUBPAGE >= BLOCK_ROWS:
        words = row * ROW_WORDS + column
    else:
        subpage = row // ROWS_PER_SUBPAGE
        within = row % ROWS_PER_SUBPAGE
        words = subpage * subpage_words + within * ROW_WORDS + column
    return words


@triton.jit
def _pair_rows(
    units_ptr,
    arrays_ptr,
    keep_ptr,
    other_ptr,
    unit,
    array,
    tile,
    PACK: tl.constexpr,
    PARITY: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK_STRIDE: tl.constexpr,
    HALF_WORDS: tl.constexpr,
    WIDTH: tl.constexpr,
    VEC: tl.constexpr,
    ROWS_PER_SUBPAGE: tl.constexpr,
    TILE: tl.constexpr,
):
    """Pack or unpack ``TILE`` vectors of ``VEC`` bytes of one array of one
    unit's block. Packing gathers rows ``2y + c`` into ``keep`` and ``2y + 1 -
    c`` into ``send``; unpacking writes ``keep`` to rows ``HALF * c + y``,
    ``recv`` to rows ``HALF * (1 - c) + y``, and zeros to rows ``[HALF, 2 *
    HALF)`` of a unit without partner. Both rows are loaded before either is
    stored. Page addresses are absolute int64, offsets within a page int32."""
    LANES: tl.constexpr = VEC // 4
    VECTORS_PER_ROW: tl.constexpr = WIDTH // VEC
    NUM_VECTORS: tl.constexpr = HALF * VECTORS_PER_ROW

    block = tl.load(units_ptr + unit * 2)
    page = tl.load(arrays_ptr + array * 3) + block * BLOCK_STRIDE
    page = tl.multiple_of(tl.cast(page, tl.pointer_type(tl.int32)), VEC)
    subpage_words = tl.load(arrays_ptr + array * 3 + 1).to(tl.int32)
    buffer = unit.to(tl.int64) * HALF_WORDS + tl.load(arrays_ptr + array * 3 + 2)
    keep_buffer = tl.multiple_of(keep_ptr + buffer, VEC)
    other_buffer = tl.multiple_of(other_ptr + buffer, VEC)

    vector = tile * TILE + tl.arange(0, TILE)
    y = vector // VECTORS_PER_ROW
    column = (vector - y * VECTORS_PER_ROW) * LANES
    if PACK:
        keep_row = 2 * y + PARITY
        other_row = 2 * y + 1 - PARITY
    else:
        keep_row = HALF * PARITY + y
        other_row = HALF * (1 - PARITY) + y
    lanes = tl.arange(0, LANES)[None, :]
    keep_words = _row_words(
        keep_row, column, subpage_words, WIDTH // 4, ROWS_PER_SUBPAGE, 2 * HALF
    )
    other_words = _row_words(
        other_row, column, subpage_words, WIDTH // 4, ROWS_PER_SUBPAGE, 2 * HALF
    )
    keep_rows = page + tl.multiple_of(keep_words, LANES)[:, None] + lanes
    other_rows = page + tl.multiple_of(other_words, LANES)[:, None] + lanes
    words = tl.multiple_of(vector * LANES, LANES)[:, None] + lanes
    mask = None if NUM_VECTORS % TILE == 0 else (vector < NUM_VECTORS)[:, None]

    if PACK:
        keep = tl.load(keep_rows, mask=mask)
        other = tl.load(other_rows, mask=mask)
        tl.store(keep_buffer + words, keep, mask=mask)
        tl.store(other_buffer + words, other, mask=mask)
    else:
        # Half 1 comes from P block 2j + 1, which a unit without partner lacks.
        has_partner = tl.load(units_ptr + unit * 2 + 1).to(tl.int32)
        half_1 = (vector < has_partner * NUM_VECTORS)[:, None]
        if PARITY == 0:
            keep = tl.load(keep_buffer + words, mask=mask)
            other = tl.load(other_buffer + words, mask=half_1, other=0)
        else:
            keep = tl.load(keep_buffer + words, mask=half_1, other=0)
            other = tl.load(other_buffer + words, mask=mask)
        tl.store(keep_rows, keep, mask=mask)
        tl.store(other_rows, other, mask=mask)


@triton.jit
def _pair_row_class(
    units_ptr,
    arrays_ptr,
    keep_ptr,
    other_ptr,
    unit,
    program,
    PACK: tl.constexpr,
    PARITY: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK_STRIDE: tl.constexpr,
    HALF_WORDS: tl.constexpr,
    ROW_CLASS: tl.constexpr,
):
    """Move the tile of ``program`` if it is one of ``ROW_CLASS``'s; see
    ``_LaunchPlan.constants``."""
    WIDTH: tl.constexpr = ROW_CLASS[0]
    VEC: tl.constexpr = ROW_CLASS[1]
    ROWS_PER_SUBPAGE: tl.constexpr = ROW_CLASS[2]
    TILE: tl.constexpr = ROW_CLASS[3]
    TILES: tl.constexpr = ROW_CLASS[4]
    FIRST_ARRAY: tl.constexpr = ROW_CLASS[5]
    FIRST_PROGRAM: tl.constexpr = ROW_CLASS[6]
    END_PROGRAM: tl.constexpr = ROW_CLASS[7]
    if (program >= FIRST_PROGRAM) & (program < END_PROGRAM):
        _pair_rows(
            units_ptr,
            arrays_ptr,
            keep_ptr,
            other_ptr,
            unit,
            FIRST_ARRAY + (program - FIRST_PROGRAM) // TILES,
            (program - FIRST_PROGRAM) % TILES,
            PACK,
            PARITY,
            HALF,
            BLOCK_STRIDE,
            HALF_WORDS,
            WIDTH,
            VEC,
            ROWS_PER_SUBPAGE,
            TILE,
        )


@triton.jit
def _pair_rows_kernel(
    units_ptr,  # [n, 2] int64: block id, has partner
    arrays_ptr,  # [num_arrays, 3] int64: see _LaunchPlan.arrays
    keep_ptr,  # [K, H / 4] int32
    other_ptr,  # [K, H / 4] int32: send when packing, recv when unpacking
    PACK: tl.constexpr,
    PARITY: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK_STRIDE: tl.constexpr,
    HALF_WORDS: tl.constexpr,
    CLASSES: tl.constexpr,
):
    """Pack or unpack every array of one chunk. Grid: (programs per unit,
    units); a unit's programs take each row class's arrays tile by tile."""
    unit = tl.program_id(1)
    program = tl.program_id(0)
    for i in tl.static_range(len(CLASSES)):
        _pair_row_class(
            units_ptr,
            arrays_ptr,
            keep_ptr,
            other_ptr,
            unit,
            program,
            PACK,
            PARITY,
            HALF,
            BLOCK_STRIDE,
            HALF_WORDS,
            CLASSES[i],
        )


@dataclass(frozen=True)
class _RowClass:
    """Arrays of one row width and sub-page height."""

    width: int
    rows_per_subpage: int
    vector_bytes: int
    tile: int
    num_tiles: int
    num_arrays: int


class _LaunchPlan(NamedTuple):
    """The launch that packs or unpacks a chunk: every row class in one."""

    arrays: torch.Tensor
    """``[num_arrays, 3]`` int64, class by class: address of the array in
    block 0, its sub-page stride in words, and the offset of its half rows in
    a unit's ``H`` bytes in words."""
    num_programs: int
    """Programs per unit."""
    constants: dict[str, Any]
    """The kernel's constants. ``CLASSES`` holds each row class's width, copy
    bytes, rows per sub-page, tile, tiles per array, first array, and first
    and end program."""

    def rebased(self, delta: int) -> "_LaunchPlan":
        """The plan with its arrays ``delta`` bytes further."""
        arrays = self.arrays.clone()
        arrays[:, 0] += delta
        return self._replace(arrays=arrays)


def _vector_bytes(*values: int) -> int:
    """The widest copy, of 16, 8 or 4 bytes, that every value is a multiple
    of; 0 if none."""
    return next((v for v in (16, 8, 4) if all(x % v == 0 for x in values)), 0)


def _row_classes(
    table: PairRowTable,
) -> list[tuple[_RowClass, list[tuple[int, int, int]]]]:
    """The table's arrays grouped by row width and sub-page height, each with
    its arrays' address in block 0, sub-page stride and offset in ``H``."""
    members: dict[tuple[int, int], list[tuple[PairRowArray, int]]] = {}
    for array, (start, _) in zip(table.arrays, table.segments()):
        key = (array.width, array.rows_per_subpage)
        members.setdefault(key, []).append((array, start))
    classes = []
    for (width, rows), arrays in members.items():
        entries = [
            (table.base_addr + a.page_offset + a.offset, a.subpage_stride, start)
            for a, start in arrays
        ]
        vector_bytes = _vector_bytes(
            table.block_stride,
            table.half_bytes,
            width,
            *(value for entry in entries for value in entry),
        )
        if not vector_bytes:
            raise NixlPairModeError(
                f"DCP pair exchange: the {width}-byte rows of layers "
                f"{sorted({a.layer_name for a, _ in arrays})} are not 4-byte aligned"
            )
        vectors = table.block_size // 2 * width // vector_bytes
        tile = min(_TILE_VECTORS, 1 << (vectors - 1).bit_length())
        row_class = _RowClass(
            width=width,
            rows_per_subpage=rows,
            vector_bytes=vector_bytes,
            tile=tile,
            num_tiles=cdiv(vectors, tile),
            num_arrays=len(entries),
        )
        classes.append((row_class, entries))
    return classes


def _launch_plan(
    table: PairRowTable,
    classes: Sequence[tuple[_RowClass, list[tuple[int, int, int]]]],
    parity: int,
    device: torch.device,
) -> _LaunchPlan:
    """One launch per chunk for all the row classes."""
    arrays = torch.tensor(
        [
            (addr, stride // 4, start // 4)
            for _, entries in classes
            for addr, stride, start in entries
        ],
        dtype=torch.int64,
        device=device,
    )
    constants = []
    first_array = first_program = 0
    for row_class, _ in classes:
        end_program = first_program + row_class.num_arrays * row_class.num_tiles
        constants.append(
            (
                row_class.width,
                row_class.vector_bytes,
                row_class.rows_per_subpage,
                row_class.tile,
                row_class.num_tiles,
                first_array,
                first_program,
                end_program,
            )
        )
        first_array += row_class.num_arrays
        first_program = end_program
    return _LaunchPlan(
        arrays=arrays,
        num_programs=first_program,
        constants=dict(
            PARITY=parity,
            HALF=table.block_size // 2,
            BLOCK_STRIDE=table.block_stride,
            HALF_WORDS=table.half_bytes // 4,
            CLASSES=tuple(constants),
        ),
    )


def _unit_rows(units: Sequence[PairUnit]) -> list[list[int]]:
    """(block id, has partner) of each unit, for the kernels. A list: it
    stages straight into pinned memory."""
    return [[u.block_id, int(u.has_partner)] for u in units]


def _p2p_op(op: Any, tensor: torch.Tensor, peer: int) -> torch.distributed.P2POp:
    """A P2P op for a device communicator. As in elastic_execute.py, it skips
    ``P2POp.__init__``, which needs a torch process group: PyNccl reads only
    the op, the tensor and the group peer."""
    p2p = object.__new__(torch.distributed.P2POp)
    p2p.op, p2p.tensor, p2p.group_peer = op, tensor, peer
    p2p.peer, p2p.group, p2p.tag = peer, None, 0
    return p2p


def _dcp_communicator(layout: PairLayout) -> Any:
    """The DCP group's device communicator, which must send through PyNccl
    between the layout's ranks: raw PyNccl returns silently when disabled,
    while the device communicator raises."""
    try:
        group = get_dcp_group()
    except AssertionError as e:
        raise NixlPairModeError(
            "DCP pair exchange: the DCP group is not initialized"
        ) from e
    communicator = group.device_communicator
    pynccl = getattr(communicator, "pynccl_comm", None)
    if pynccl is None or pynccl.disabled:
        state = "missing" if pynccl is None else "disabled"
        raise NixlPairModeError(
            "DCP pair exchange needs the DCP group's PyNccl communicator, which "
            f"is {state} (device communicator: {type(communicator).__name__})"
        )
    dcp_size = layout.group_size
    if (group.rank_in_group, group.world_size) != (layout.dcp_rank, dcp_size):
        raise NixlPairModeError(
            f"DCP pair exchange: DCP rank {layout.dcp_rank} of {dcp_size} is rank "
            f"{group.rank_in_group} of {group.world_size} in the DCP group"
        )
    return communicator


def _digest(text: str) -> int:
    """A 63-bit hash of ``text``, the same in every process."""
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") >> 1


def _fill_byte(dcp_rank: int) -> int:
    """The non-zero byte a rank fills its warm-up scratch block with."""
    return dcp_rank % 255 + 1


class _Stamp(NamedTuple):
    """What a rank tells its partner at warm-up, as int64 words."""

    magic: int
    dcp_rank: int
    partner: int
    parity: int
    block_size: int
    block_stride: int
    half_bytes: int
    chunk_units: int
    num_arrays: int
    signature: int
    host: int

    @property
    def geometry(self) -> tuple[int, ...]:
        """The fields partners must share."""
        return (
            self.block_size,
            self.block_stride,
            self.half_bytes,
            self.chunk_units,
            self.num_arrays,
            self.signature,
        )

    def describe(self) -> str:
        return (
            f"DCP rank {self.dcp_rank} (partner {self.partner}, parity "
            f"{self.parity}; {self.block_size}-row blocks {self.block_stride} B "
            f"apart, {self.num_arrays} arrays of {self.half_bytes} B per half "
            f"block, {self.chunk_units} units per round, signature "
            f"{self.signature:016x})"
        )


def _pair_problem(ours: _Stamp, theirs: _Stamp) -> str | None:
    """Why two warm-up stamps do not pair, or None. Symmetric, so partners
    raise together instead of one of them waiting in NCCL."""
    if ours.magic != _STAMP_MAGIC or theirs.magic != _STAMP_MAGIC:
        return "a stamp does not come from a DCP pair exchange warm-up"
    if (ours.partner, theirs.partner) != (theirs.dcp_rank, ours.dcp_rank):
        return "the ranks are not each other's partner"
    if ours.parity + theirs.parity != 1:
        return "both ranks keep the same half"
    if ours.geometry != theirs.geometry:
        return "their row tables or chunking differ"
    return None


class _Ring:
    """The chunk slots of one rank's exchanges (every stage shares them).

    ``take`` hands out slots in order and makes the current stream wait for
    the side-stream completion of the chunk that last used the slot;
    ``release`` records that completion after a chunk's swap and unpack.
    """

    def __init__(self, slots: int):
        self.slots = slots
        self.next = 0
        self._events: list[torch.cuda.Event | None] = [None] * slots
        self._swaps: "torch.cuda.Stream | None" = None

    def swap_stream(self, device: torch.device) -> "torch.cuda.Stream":
        """The stream of the swaps (copy engines and stream memory operations
        only), one per rank."""
        if self._swaps is None:
            self._swaps = torch.cuda.Stream(device=device)
        return self._swaps

    def take(self, stream: "torch.cuda.Stream") -> int:
        slot = self.next
        self.next = (slot + 1) % self.slots
        event = self._events[slot]
        if event is not None:
            stream.wait_event(event)
        return slot

    def release(self, slot: int, side: "torch.cuda.Stream") -> None:
        event = self._events[slot]
        if event is None:
            event = self._events[slot] = torch.cuda.Event()
        event.record(side)


class DcpPairExchanger:
    """The DCP pair exchange of one D rank.

    Built once the KV caches are registered. ``warm_up`` then runs once on
    every rank, and ``exchange`` in every step with units, on all ranks of the
    DCP group with the same units. Each chunk of units is packed (the rows this
    rank keeps and sends), swapped with the partner in one NCCL round, and
    unpacked (the blocks rewritten in interleave-1 order), all on the current
    stream. Packing reads a chunk's every row before unpacking writes any, so
    the blocks are rewritten in place.

    Args:
        table: Row table of the rank's DCP-sharded attention group.
        layout: The rank's row maps; its partner is the NCCL peer.
        device: The KV cache device.
        chunk_units: Units per NCCL round, ``K``: ``keep``, ``send`` and
            ``recv`` are ``[K, H]`` uint8.
        communicator: Anything with ``batch_isend_irecv(list[P2POp])``. By
            default the DCP group's device communicator, which raises if its
            PyNccl is unavailable.
        ring_slots: Chunk slots of ``keep`` and ``send`` for side-stream
            exchanges (``PAIR_RING_SLOTS`` in serving, see ``_run``); 1 keeps
            one chunk, so a pack waits for the previous chunk's swap.

    """

    def __init__(
        self,
        table: PairRowTable,
        layout: PairLayout,
        device: torch.device,
        *,
        chunk_units: int = 4,
        communicator: Any | None = None,
        ring_slots: int = 1,
    ):
        if table.block_size != layout.block_size:
            raise NixlPairModeError(
                f"DCP pair exchange: {table.block_size}-row blocks, but a layout "
                f"for {layout.block_size}-row blocks"
            )
        if chunk_units < 1:
            raise ValueError(f"chunk_units must be positive, got {chunk_units}")
        self.table = table
        self.layout = layout
        self.device = torch.device(device)
        self.chunk_units = chunk_units
        self._communicator = (
            _dcp_communicator(layout) if communicator is None else communicator
        )
        classes = _row_classes(table)
        self._classes = [row_class for row_class, _ in classes]
        self._plan = _launch_plan(table, classes, layout.parity, self.device)
        if ring_slots < 1:
            raise ValueError(f"ring_slots must be positive, got {ring_slots}")
        # ``keep``, ``send`` and ``recv`` hold ``ring_slots`` chunks (slot s:
        # rows [s K, (s + 1) K)). Inline exchanges use slot 0 only.
        ring = (ring_slots * chunk_units, table.half_bytes)
        self.keep = torch.empty(ring, dtype=torch.uint8, device=self.device)
        self.send = torch.empty_like(self.keep)
        self.recv = torch.empty_like(self.keep)
        # The kernels move 4-byte words, which every row width is made of.
        self._words = {
            name: getattr(self, name).view(torch.int32)
            for name in ("keep", "send", "recv")
        }
        # Shared by every stage (``for_layout`` copies the reference): the
        # slots and the side-stream event of each slot's last chunk.
        self._ring = _Ring(ring_slots)
        # The round of n units from slot s, built on first use.
        self._swaps: dict[tuple[int, int], tuple[Any, Any]] = {}

    def for_layout(self, layout: PairLayout) -> "DcpPairExchanger":
        """The exchanger of another stage of this rank (``layout``'s partner
        and parity), sharing the table, communicator and buffers: stages run
        one after another on the same stream."""
        if (layout.dcp_rank, layout.block_size) != (
            self.layout.dcp_rank,
            self.layout.block_size,
        ):
            raise NixlPairModeError(
                "DCP pair exchange: a stage layout of another rank or block size"
            )
        stage = copy.copy(self)
        stage.layout = layout
        stage._plan = _launch_plan(
            self.table,
            _row_classes(self.table),
            layout.parity,
            self.device,
        )
        stage._swaps = {}
        return stage

    @property
    def signature(self) -> tuple:
        """What partners must share: the row geometry, in table order (the
        buffers' layout), and the chunking. Never addresses."""
        table = self.table
        arrays = tuple(
            (
                a.layer_name,
                a.kind.value,
                a.page_offset,
                a.page_len,
                a.width,
                a.rows_per_subpage,
                a.subpage_stride,
                a.offset,
            )
            for a in table.arrays
        )
        return (
            table.block_size,
            table.block_stride,
            table.half_bytes,
            self.chunk_units,
            arrays,
        )

    def exchange(
        self,
        units: Sequence[PairUnit],
        side: "torch.cuda.Stream | None" = None,
    ) -> None:
        """Swap the units' rows with the partner and rewrite their blocks in
        interleave-1 order, queued without a host sync.

        Args:
            units: The step's units of this rank's group, distinct blocks in
                metadata order. Every rank of the DCP group passes the same
                units: each chunk of ``chunk_units`` is one round with the
                partner.
            side: None: pack, swap and unpack on the current stream. Else the
                packs run on the current stream and each chunk's swap and
                unpack on ``side`` (see ``_run``).

        """
        if units:
            rows = async_tensor_h2d(_unit_rows(units), self.device, torch.int64)
            if side is not None:
                # Partners must use the same slots (a swap reads the partner's
                # slot at this rank's offset), but stages before this call may
                # have taken different numbers of chunks on the two ranks: every
                # call starts at slot 0. Safe: ``take`` still waits for each
                # slot's last chunk.
                self._ring.next = 0
            self._run(units, rows, self._plan, side)

    def record_event(self) -> torch.cuda.Event:
        """An event on the current stream, after every exchange queued so
        far."""
        event = torch.cuda.Event()
        event.record(current_stream())
        return event

    def warm_up(self) -> None:
        """Compile the kernels and check the partner, once on every rank.

        Partners first swap fixed-size stamps of their DCP ranks and
        signatures, so skewed partners both fail here, naming both sides,
        before any message whose size depends on the geometry. A scratch block
        filled with this rank's byte then goes through pack, NCCL and unpack,
        which compiles the pack and unpack kernels; half ``c`` of each array must
        come back with this rank's byte and the other half with the partner's.
        Logs what decides the NCCL transport, the buffer bytes and the free
        device memory before and after.

        Raises:
            NixlPairModeError: If the partner does not pair, the scratch block
                comes back wrong, or less than 1 GiB of device memory is free.

        """
        free_before = torch.accelerator.get_memory_info(self.device)[0]
        stamp = self._stamp()
        peer = self._swap_stamps(stamp)
        if problem := _pair_problem(stamp, peer):
            raise NixlPairModeError(
                f"DCP pair exchange: {stamp.describe()} cannot pair with its NCCL "
                f"peer, {peer.describe()}: {problem}"
            )

        # The scratch block starts at the table base's offset from 16-byte
        # alignment, so every row keeps its copy width.
        offset = self.table.base_addr % 16
        scratch = torch.full(
            (offset + self.table.block_stride,),
            _fill_byte(self.layout.dcp_rank),
            dtype=torch.uint8,
            device=self.device,
        )
        delta = scratch.data_ptr() + offset - self.table.base_addr
        unit = [PairUnit(0, 0, True)]
        rows = async_tensor_h2d(_unit_rows(unit), self.device, torch.int64)
        self._run(unit, rows, self._plan.rebased(delta))
        swap_ms = self._time_swaps()
        self._check_scratch(scratch[offset:].cpu().numpy())
        del scratch
        free_after = torch.accelerator.get_memory_info(self.device)[0]

        pynccl = getattr(self._communicator, "pynccl_comm", None)
        available = getattr(pynccl, "available", False)
        nccl = pynccl.nccl.ncclGetVersion() if available else "unknown"
        settings = [
            f"{k}={os.environ[k]}" for k in _NCCL_TRANSPORT_ENV if k in os.environ
        ]
        half_mb = self.table.half_bytes / 1e6
        logger.info(
            "NIXL DCP pair exchange: DCP rank %d pairs with %d on %s over NCCL %s "
            "(%s). One unit, %.2f MB each way, swaps in %.3f ms (%.0f GB/s); "
            "NCCL_DEBUG=INFO shows the transport, which should be P2P over "
            "multi-node NVLink, not NET/IB. Buffers: keep + send + recv %d slots "
            "x %d units x %.2f MB = %.1f MB. Kernels: %s. Free device memory: %.2f GiB before warm-up, "
            "%.2f GiB after.",
            stamp.dcp_rank,
            peer.dcp_rank,
            "this host" if peer.host == stamp.host else "another host",
            nccl,
            ", ".join(settings) or "no NCCL transport settings",
            half_mb,
            swap_ms,
            half_mb / max(swap_ms, 1e-6),
            self._ring.slots,
            self.chunk_units,
            half_mb,
            (self.keep.numel() + self.send.numel() + self.recv.numel()) / 1e6,
            ", ".join(
                f"{c.num_arrays} x {c.width} B rows in {c.vector_bytes} B copies"
                for c in self._classes
            ),
            free_before / (1 << 30),
            free_after / (1 << 30),
        )
        if free_after < _MIN_FREE_BYTES:
            raise NixlPairModeError(
                f"DCP pair exchange: {free_after / (1 << 30):.2f} GiB of device "
                "memory is free after warm-up, under 1 GiB"
            )

    def check_round_trip(self) -> None:
        """Pack, swap and unpack a scratch block filled with this rank's byte
        through the current communicator and check the partner's rows came
        back (as in ``warm_up``, without the stamp swap and the NCCL timing).
        On every rank of the DCP group at once."""
        offset = self.table.base_addr % 16
        scratch = torch.full(
            (offset + self.table.block_stride,),
            _fill_byte(self.layout.dcp_rank),
            dtype=torch.uint8,
            device=self.device,
        )
        delta = scratch.data_ptr() + offset - self.table.base_addr
        unit = [PairUnit(0, 0, True)]
        rows = async_tensor_h2d(_unit_rows(unit), self.device, torch.int64)
        self._run(unit, rows, self._plan.rebased(delta))
        self._check_scratch(scratch[offset:].cpu().numpy())

    def use_send_buffer(self, send: torch.Tensor, communicator: Any) -> None:
        """Swap through ``communicator`` from ``send`` (same shape as the
        built buffer), e.g. a buffer the partner maps for peer copies."""
        if send.shape != self.send.shape or send.dtype != self.send.dtype:
            raise NixlPairModeError(
                f"DCP pair exchange: send buffer {tuple(send.shape)} "
                f"{send.dtype}, expected {tuple(self.send.shape)} {self.send.dtype}"
            )
        self.send = send
        self._words["send"] = send.view(torch.int32)
        self._communicator = communicator
        self._swaps = {}

    def _run(
        self,
        units: Sequence[PairUnit],
        rows: torch.Tensor,
        plan: _LaunchPlan,
        side: "torch.cuda.Stream | None" = None,
    ) -> None:
        """Pack, swap and unpack the units chunk by chunk, ``rows`` being
        their ``_unit_rows`` on the device.

        With ``side``, every pack kernel stays on the current stream and only
        the swap (stream memory operations and copy-engine copies) and the
        unpack run on ``side``. The partner's swap waits for this rank's
        "packed" counter, so the pack must never sit behind work that can wait
        for the partner: on the current stream it precedes this step's
        forward, whose collectives and MegaMoE spin on the peers, so it always
        runs, even when every SM is held by a spinning kernel or the device
        serializes kernels (CUPTI tracing). A pack on ``side`` could be
        starved by such a kernel while the partner's host blocks on its side
        stream, which closes a cycle. The unpack only rewrites this rank's
        own blocks; no peer waits for it.

        Chunks go through ``PAIR_RING_SLOTS`` slots of ``keep``, ``send`` and
        ``recv`` (``_Ring``). A pack waits (on the device, ``wait_event``) only
        for the chunk that last used its slot: its swap has consumed ``send``
        (the partner's copy is behind the swap's "consumed" wait) and its
        unpack has read ``keep`` and ``recv``.

        The swaps (stream memory operations and copy-engine copies, no SM
        work) run on their own stream (``_Ring.swap_stream``), the unpacks on
        ``side`` after their swap. A partner's swap waits for this rank's pack
        and for this rank's copy of its ``send``; neither may sit behind an
        unpack, which needs SMs that a spinning forward kernel can hold (with
        the unpacks between the copies on one stream, a spin queued after the
        packs starved unpack k and with it copy k + 1, which the partner's
        "consumed" wait needed: the cycle ``test_side_stream_peer_copy_is_exact
        _and_cannot_deadlock`` catches). Every wait the current stream takes
        (a slot's last unpack) precedes this step's forward on it, so those
        unpacks run before any of its kernels can spin. The cross-rank
        counters are written by ``PeerCopySwap.batch_isend_irecv`` on the swap
        stream itself: "packed" when the swap starts (after the pack, which
        the swap stream waits for), "consumed" right after the copy, before
        any unpack; neither waits for SM work.
        """
        if side is None:
            for start in range(0, len(units), self.chunk_units):
                chunk = units[start : start + self.chunk_units]
                chunk_rows = rows[start : start + len(chunk)]
                self._pack(chunk, chunk_rows, plan)
                self._communicate(len(chunk))
                self._unpack(chunk, chunk_rows, plan)
            return
        compute = current_stream()
        rows.record_stream(side)
        ring = self._ring
        swaps = ring.swap_stream(side.device)
        for start in range(0, len(units), self.chunk_units):
            chunk = units[start : start + self.chunk_units]
            chunk_rows = rows[start : start + len(chunk)]
            slot = ring.take(compute)
            self._pack(chunk, chunk_rows, plan, slot=slot)
            swaps.wait_stream(compute)
            with torch.cuda.stream(swaps):
                self._communicate(len(chunk), slot=slot)
            side.wait_stream(swaps)
            with torch.cuda.stream(side):
                self._unpack(chunk, chunk_rows, plan, slot=slot)
                ring.release(slot, side)

    def _slot(self, name: str, slot: int, num_units: int) -> torch.Tensor:
        """Rows of slot ``slot`` of a ring buffer."""
        first = slot * self.chunk_units
        return self._words[name][first : first + num_units]

    def _pack(
        self,
        units: Sequence[PairUnit],
        rows: torch.Tensor,
        plan: _LaunchPlan,
        slot: int = 0,
    ) -> None:
        """Fill slot ``slot`` of ``keep`` and ``send`` for one chunk."""
        n = len(units)
        self._launch(True, rows, n, self._slot("keep", slot, n),
                     self._slot("send", slot, n), plan)  # fmt: skip

    def _unpack(
        self,
        units: Sequence[PairUnit],
        rows: torch.Tensor,
        plan: _LaunchPlan,
        slot: int = 0,
    ) -> None:
        """Rewrite one chunk's blocks from slot ``slot`` of ``keep`` and
        ``recv``."""
        n = len(units)
        self._launch(False, rows, n, self._slot("keep", slot, n),
                     self._slot("recv", slot, n), plan)  # fmt: skip

    def _launch(
        self,
        pack: bool,
        rows: torch.Tensor,
        num_units: int,
        keep: torch.Tensor,
        other: torch.Tensor,
        plan: _LaunchPlan,
    ) -> None:
        _pair_rows_kernel[(plan.num_programs, num_units)](
            rows,
            plan.arrays,
            keep,
            other,
            PACK=pack,
            **plan.constants,
            num_warps=_NUM_WARPS,
        )

    def _communicate(self, num_units: int, slot: int = 0) -> None:
        """Swap the first ``num_units`` rows of slot ``slot`` of ``send`` into
        ``recv`` with the partner."""
        ops = self._swaps.get((slot, num_units))
        if ops is None:
            first = slot * self.chunk_units
            ops = self._swaps[(slot, num_units)] = (
                _p2p_op(
                    torch.distributed.isend,
                    self.send[first : first + num_units],
                    self.layout.partner,
                ),
                _p2p_op(
                    torch.distributed.irecv,
                    self.recv[first : first + num_units],
                    self.layout.partner,
                ),
            )
        self._communicator.batch_isend_irecv(list(ops))

    def _stamp(self) -> _Stamp:
        return _Stamp(
            magic=_STAMP_MAGIC,
            dcp_rank=self.layout.dcp_rank,
            partner=self.layout.partner,
            parity=self.layout.parity,
            block_size=self.table.block_size,
            block_stride=self.table.block_stride,
            half_bytes=self.table.half_bytes,
            chunk_units=self.chunk_units,
            num_arrays=len(self.table.arrays),
            signature=_digest(repr(self.signature)),
            host=_digest(socket.gethostname()),
        )

    def _swap_stamps(self, stamp: _Stamp) -> _Stamp:
        send = torch.tensor(stamp, dtype=torch.int64, device=self.device)
        recv = torch.empty_like(send)
        peer = self.layout.partner
        self._communicator.batch_isend_irecv(
            [
                _p2p_op(torch.distributed.isend, send, peer),
                _p2p_op(torch.distributed.irecv, recv, peer),
            ]
        )
        return _Stamp(*recv.tolist())

    def _time_swaps(self) -> float:
        """Milliseconds of the fastest of a few one-unit NCCL swaps."""
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        best = float("inf")
        for _ in range(_TIMED_SWAPS):
            start.record(current_stream())
            self._communicate(1)
            end.record(current_stream())
            end.synchronize()
            best = min(best, start.elapsed_time(end))
        return best

    def _check_scratch(self, scratch: np.ndarray) -> None:
        """Raise unless the exchanged scratch block holds the partner's byte
        exactly in the rows ``recv`` fills, and this rank's byte elsewhere."""
        expected = np.full_like(scratch, _fill_byte(self.layout.dcp_rank))
        _, recv_rows = self.layout.unpack_rows()
        for array in self.table.arrays:
            starts = self.table.row_offsets(array, [0], recv_rows)[0]
            expected[starts[:, None] + np.arange(array.width)] = _fill_byte(
                self.layout.partner
            )
        wrong = np.flatnonzero(scratch != expected)
        if wrong.size:
            raise NixlPairModeError(
                f"DCP pair exchange: the warm-up block of DCP rank "
                f"{self.layout.dcp_rank} came back with {wrong.size} wrong bytes, "
                f"the first at offset {wrong[0]}: {scratch[wrong[0]]} instead of "
                f"{expected[wrong[0]]}"
            )


def build_dcp_pair_exchanger(
    regions: Sequence[PairRegion],
    *,
    index_kernel_rows: int,
    index_value_bytes: int,
    layout: PairLayout,
    device: torch.device,
    chunk_units: int = 4,
    ring_slots: int = 1,
) -> DcpPairExchanger:
    """The exchanger of a D rank's DCP-sharded attention group, from its NIXL
    regions. Not warmed up: every rank calls ``warm_up`` next.

    Indexer pages must have the CUDA fp8 or MXFP4 layout: kernel pages of
    ``index_kernel_rows`` rows that hold their fp8 or E2M1 values, a multiple
    of 16 bytes per row, then their 4 scale bytes per row.

    Args:
        regions: The group's regions, in NIXL region order.
        index_kernel_rows: Rows per kernel page of the indexer cache.
        index_value_bytes: fp8 or E2M1 value bytes of an indexer row.
        layout: The rank's row maps.
        device: The KV cache device.
        chunk_units: Units per NCCL round.
        ring_slots: Chunk slots of the side-stream exchange (``_run``).

    Returns:
        The exchanger, its communicator the DCP group's.

    """
    if current_platform.is_rocm():
        raise NixlPairModeError(
            "DCP pair exchange supports only the CUDA indexer cache layout"
        )
    table = build_pair_row_table(
        regions,
        index_kernel_rows=index_kernel_rows,
        index_value_bytes=index_value_bytes,
    )
    for array in table.arrays:
        if (array.kind is PairArrayKind.INDEX_VALUE and array.width % 16) or (
            array.kind is PairArrayKind.INDEX_SCALE and array.width != 4
        ):
            raise NixlPairModeError(
                f"DCP pair exchange cannot exchange layer {array.layer_name}: "
                f"{array.width}-byte {array.kind.value} rows are not the CUDA fp8 "
                "or MXFP4 indexer layout"
            )
    return DcpPairExchanger(
        table, layout, device, chunk_units=chunk_units, ring_slots=ring_slots
    )


class DcpStagedExchanger:
    """The DCP exchange of one D rank for every P DCP size ``W`` this D DCP
    size can pair with (``dcp_size / W`` a power of two of at least 2).

    Holds one pair exchanger per partner stride (``all_exchange_strides``),
    sharing one set of buffers. A request pulled from P DCP ``W`` runs the
    stages of ``exchange_strides(dcp_size, W)`` in order: one stage at ratio 2
    (the original pair exchange), ``log2(R)`` at ratio ``R``. ``warm_up`` and
    ``exchange`` run on every rank of the DCP group with the same units.
    """

    def __init__(self, stages: Mapping[int, DcpPairExchanger], dcp_size: int):
        strides = all_exchange_strides(dcp_size)
        if sorted(stages) != sorted(strides):
            raise NixlPairModeError(
                f"DCP staged exchange at DCP {dcp_size} needs the strides "
                f"{strides}, got {sorted(stages)}"
            )
        self.stages = dict(stages)
        self.dcp_size = dcp_size
        # The one-stage exchange of the default peer (P DCP = half this one).
        self.first = self.stages[dcp_size // 2]

    @property
    def table(self) -> PairRowTable:
        return self.first.table

    @property
    def layout(self) -> PairLayout:
        return self.first.layout

    @property
    def dcp_rank(self) -> int:
        return self.first.layout.dcp_rank

    def warm_up(self) -> None:
        """Warm every stage up, largest stride first, on every rank."""
        for stride in sorted(self.stages, reverse=True):
            self.stages[stride].warm_up()

    def exchange(
        self,
        units: Sequence[PairUnit],
        side: "torch.cuda.Stream | None" = None,
    ) -> None:
        """Run the stages of every P DCP size among ``units``, in the order
        the sizes first appear (the same on every rank). ``side`` as in
        ``DcpPairExchanger.exchange``: a later stage's packs wait for the
        earlier stage's unpacks on ``side``."""
        # Without a side stream, stages are called as before (``exchange(units)``).
        kwargs = {} if side is None else {"side": side}
        by_remote: dict[int, list[PairUnit]] = {}
        for u in units:
            remote = u.remote_dcp_size or self.dcp_size // 2
            by_remote.setdefault(remote, []).append(u)
        for remote, group_units in by_remote.items():
            strides = exchange_strides(self.dcp_size, remote)
            if len(strides) == 1:
                # Ratio 2: the one-stage exchange of the units as planned.
                self.stages[strides[0]].exchange(group_units, **kwargs)
                continue
            for i, stride in enumerate(strides):
                if i and side is not None:
                    # A stage rewrites the blocks the previous stage's unpacks
                    # wrote on ``side``: its packs (current stream) read them.
                    current_stream().wait_stream(side)
                self.stages[stride].exchange(
                    stage_units(group_units, self.dcp_rank, stride, remote),
                    **kwargs,
                )

    def record_event(self) -> torch.cuda.Event:
        return self.first.record_event()

    def use_send_buffer(self, send: torch.Tensor, communicator: Any) -> None:
        """Every stage swaps through ``communicator`` from ``send``; each
        stage's round trip is checked (largest stride first, as warm_up), on
        every rank of the DCP group at once."""
        for stride in sorted(self.stages, reverse=True):
            stage = self.stages[stride]
            stage.use_send_buffer(send, communicator)
            stage.check_round_trip()


def build_dcp_staged_exchanger(
    regions: Sequence[PairRegion],
    *,
    index_kernel_rows: int,
    index_value_bytes: int,
    dcp_rank: int,
    dcp_size: int,
    block_size: int,
    interleave: int,
    device: torch.device,
    chunk_units: int = 4,
    ring_slots: int = 1,
) -> DcpStagedExchanger:
    """The staged exchanger of a D rank: ``build_dcp_pair_exchanger`` at the
    largest stride (P DCP = half this one), then the other strides' stages
    sharing its buffers. Not warmed up."""
    strides = all_exchange_strides(dcp_size)
    if not strides:
        raise NixlPairModeError(
            f"DCP pair exchange needs an even DCP size, got {dcp_size}"
        )

    def layout(stride: int) -> PairLayout:
        # The stage of ``stride`` exists for P DCP ``stride`` (its first stage).
        return pair_layout(dcp_rank, dcp_size, stride, block_size, interleave)

    first = build_dcp_pair_exchanger(
        regions,
        index_kernel_rows=index_kernel_rows,
        index_value_bytes=index_value_bytes,
        layout=layout(strides[0]),
        device=device,
        chunk_units=chunk_units,
        ring_slots=ring_slots,
    )
    stages = {strides[0]: first}
    for stride in strides[1:]:
        stages[stride] = first.for_layout(layout(stride))
    return DcpStagedExchanger(stages, dcp_size)


# ----------------------------------------------------------------------------
# Peer-copy swap (one host): copy engines + stream memory operations
# ----------------------------------------------------------------------------


class _CudaDriver:
    """The few driver calls the peer-copy swap needs, through ctypes."""

    def __init__(self):
        import ctypes

        lib = ctypes.CDLL("libcuda.so.1")

        def fn(*names: str):
            for name in names:
                f = getattr(lib, name, None)
                if f is not None:
                    f.restype = ctypes.c_int
                    return f
            raise OSError(f"libcuda has none of {names}")

        u64, ptr = ctypes.c_uint64, ctypes.c_void_p
        self._wait = fn("cuStreamWaitValue64_v2", "cuStreamWaitValue64")
        self._wait.argtypes = [ptr, u64, u64, ctypes.c_uint]
        self._write = fn("cuStreamWriteValue64_v2", "cuStreamWriteValue64")
        self._write.argtypes = [ptr, u64, u64, ctypes.c_uint]
        self._copy = fn("cuMemcpyAsync")
        self._copy.argtypes = [u64, u64, ctypes.c_size_t, ptr]

    @staticmethod
    def _check(name: str, rc: int) -> None:
        if rc != 0:
            raise RuntimeError(f"DCP peer-copy swap: {name} returned CUresult {rc}")

    def wait_geq(self, stream: int, addr: int, value: int) -> None:
        # CU_STREAM_WAIT_VALUE_GEQ
        self._check("cuStreamWaitValue64", self._wait(stream, addr, value, 0))

    def write(self, stream: int, addr: int, value: int) -> None:
        # Default flags: a memory barrier orders the stream's earlier writes.
        self._check("cuStreamWriteValue64", self._write(stream, addr, value, 0))

    def copy(self, stream: int, dst: int, src: int, nbytes: int) -> None:
        self._check("cuMemcpyAsync", self._copy(dst, src, nbytes, stream))


class _DeviceBuffer:
    """A uint8 view of raw device memory, for ``torch.as_tensor``."""

    def __init__(self, ptr: int, shape: tuple[int, ...]):
        self.__cuda_array_interface__ = {
            "shape": shape,
            "typestr": "|u1",
            "data": (ptr, False),
            "strides": None,
            "version": 3,
        }


class PeerCopySwap:
    """``batch_isend_irecv`` for the pair exchanger when one host holds the
    whole DCP group: each rank copies the partner's ``send`` rows into its
    ``recv`` with a copy engine (the partner's buffer is mapped over CUDA
    IPC), ordered by stream memory operations on per-pair counters.

    Nothing waits on an SM: a rank's stream waits for the partner's "packed"
    and "consumed" counters in front-end stream waits, so the exchange can run
    on a side stream beside a forward whose kernels spin on peers (MegaMoE,
    the DCP collectives) without being able to starve them. Per chunk ``s`` of
    the pair (both ranks issue the same chunks in the same order)::

        write partner.packed[me] = s          (after this rank's pack kernel)
        wait  mine.packed[partner] >= s
        copy  partner.send[:n] -> recv[:n]    (copy engine)
        write partner.consumed[me] = s
        wait  mine.consumed[partner] >= s     (before the next pack overwrites send)

    Args:
        dcp_rank: This rank in the DCP group.
        dcp_size: The group size.
        send_ptrs: Every rank's mapped ``send`` buffer.
        flag_ptrs: Every rank's mapped counters: ``2 * dcp_size`` int64, zeroed.
        send_bytes: Bytes of each rank's mapped ``send`` (every ring slot): a
            round sends from a slot of it, and the partner's copy reads the
            same slot of the partner's buffer.
    """

    def __init__(
        self,
        dcp_rank: int,
        dcp_size: int,
        send_ptrs: Sequence[int],
        flag_ptrs: Sequence[int],
        driver: _CudaDriver | None = None,
        send_bytes: int | None = None,
    ):
        self.dcp_rank, self.dcp_size = dcp_rank, dcp_size
        self.send_ptrs, self.flag_ptrs = list(send_ptrs), list(flag_ptrs)
        # Bytes of each mapped buffer (all ring slots); None: unchecked (tests).
        self.send_bytes = 1 << 62 if send_bytes is None else send_bytes
        self._cu = driver or _CudaDriver()
        self._seq = [0] * dcp_size

    def _packed(self, owner: int, peer: int) -> int:
        return self.flag_ptrs[owner] + 8 * peer

    def _consumed(self, owner: int, peer: int) -> int:
        return self.flag_ptrs[owner] + 8 * (self.dcp_size + peer)

    def batch_isend_irecv(self, ops: Sequence[Any]) -> None:
        send, recv = ops[0].tensor, ops[1].tensor
        partner, me = ops[0].group_peer, self.dcp_rank
        # A ring slot of the mapped buffer: the partner's copy reads the same
        # offset of the partner's buffer (both ranks run the same chunks in the
        # same order, so the same slots).
        offset = send.data_ptr() - self.send_ptrs[me]
        if (
            not 0 <= offset <= self.send_bytes - send.numel()
            or ops[1].group_peer != partner
        ):
            raise NixlPairModeError(
                "DCP peer-copy swap: the round must send from this rank's mapped "
                "buffer and receive from the same partner"
            )
        self._seq[partner] += 1
        s = self._seq[partner]
        stream = current_stream().cuda_stream
        cu = self._cu
        cu.write(stream, self._packed(partner, me), s)
        cu.wait_geq(stream, self._packed(me, partner), s)
        cu.copy(stream, recv.data_ptr(), self.send_ptrs[partner] + offset, send.numel())
        cu.write(stream, self._consumed(partner, me), s)
        cu.wait_geq(stream, self._consumed(me, partner), s)


def build_peer_copy_swap(
    exchanger: "DcpStagedExchanger", cpu_group: Any
) -> "PeerCopySwap | None":
    """Switch ``exchanger`` to peer-copy swaps when every rank of the DCP group
    (``cpu_group``) is on this host and can map the others' buffers; else
    leave it on NCCL and return None. Collective over the group."""
    import socket

    import torch.distributed as dist

    first = exchanger.first
    ok = True
    reason = ""
    try:
        from vllm import _custom_ops as ops

        driver = _CudaDriver()
        ok = hasattr(ops, "allocate_shared_buffer_and_handle")
        # Probe the stream memory operations and the copy on this device.
        probe = torch.zeros(4, dtype=torch.int64, device=first.device)
        stream = current_stream().cuda_stream
        driver.write(stream, probe.data_ptr(), 7)
        driver.wait_geq(stream, probe.data_ptr(), 7)
        driver.copy(stream, probe.data_ptr() + 8, probe.data_ptr(), 8)
        ok = ok and probe.tolist() == [7, 7, 0, 0]
        if not ok:
            reason = "stream memory operation probe failed"
    except Exception as e:  # noqa: BLE001 - capability probe
        ok, reason, driver = False, repr(e), None
    views: list[Any] = [None] * dist.get_world_size(cpu_group)
    dist.all_gather_object(views, (socket.gethostname(), ok), group=cpu_group)
    if len({h for h, _ in views}) != 1 or not all(o for _, o in views):
        logger.info(
            "NIXL DCP pair exchange: peer-copy swap off (DCP group on hosts %s, "
            "capable %s%s); NCCL swaps inline.",
            sorted({h for h, _ in views}),
            [o for _, o in views],
            f", {reason}" if reason else "",
        )
        return None
    from vllm.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce,
    )

    rank = dist.get_rank(cpu_group)
    size = dist.get_world_size(cpu_group)
    shape = tuple(first.send.shape)
    send_ptrs = CustomAllreduce.create_shared_buffer(first.send.numel(), cpu_group)
    flag_ptrs = CustomAllreduce.create_shared_buffer(16 * size, cpu_group)
    device = first.device
    send = torch.as_tensor(_DeviceBuffer(send_ptrs[rank], shape), device=device)
    flags = torch.as_tensor(_DeviceBuffer(flag_ptrs[rank], (16 * size,)), device=device)
    flags.zero_()
    torch.accelerator.synchronize()
    dist.barrier(group=cpu_group)
    swap = PeerCopySwap(rank, size, send_ptrs, flag_ptrs, driver, first.send.numel())
    exchanger.use_send_buffer(send, swap)
    torch.accelerator.synchronize()
    # The replaced send buffer goes back to the driver: the exchange's
    # reservation (``reserved_device_bytes``) covers one send buffer.
    torch.accelerator.empty_cache()
    logger.info(
        "NIXL DCP pair exchange: peer-copy swaps (copy engines, stream memory "
        "operations) between the %d DCP ranks of this host; exchanges run on a "
        "side stream off the forward's critical path.",
        size,
    )
    return swap
