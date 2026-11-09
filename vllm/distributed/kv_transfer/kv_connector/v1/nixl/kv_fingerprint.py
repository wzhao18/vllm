# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Debug fingerprints of the attention KV that NIXL moves at DCP interleave 1.

Off unless ``VLLM_NIXL_KV_FINGERPRINT=1``. Each rank then appends one JSON line
per request and KV cache group to ``kvfp-<side>-dcp<rank>-<host>-<pid>.jsonl``
in ``VLLM_NIXL_KV_FINGERPRINT_DIR`` (default ``$VLLM_ITERLOG_DIR/kv_fingerprint``),
and an offline comparator joins a job's P and D files on P's request id.

At interleave 1, row ``y`` of a request's block ``i`` on DCP rank ``r`` of ``W``
holds token ``t = (i * block_size + y) * W + r``. With
``c(row) = sum_k word_k * M_k`` over the row's little-endian 32-bit words and
``M_k = splitmix64(k) | 1``, a row adds
``splitmix64(t ^ salt(layer, field)) * (2 * c(row) + 1)`` (mod 2**64) to the
``F`` of its cell (``FingerprintGrid``): the tokens that D rank ``rho`` holds in
half ``h`` of the request's D block ``j``. Moving a row to another token changes
``F``, which a sum of separate token and content terms would not. D rank ``r``
logs ``rho = r``; P rank ``p`` logs ``rho`` in ``{p, p + W_P}``, its block ``i``
being half ``i % 2`` of D block ``i // 2``.

Rows are read through the views the attention backends bind, not the pair
exchanger's row tables: one row per token for MLA and drafter pages, and the
indexer's 64-row kernel pages, which hold the rows' fp8 values, then one
4-byte scale per row.

Call sites: the scheduler sends ``KVFingerprintReq``s (metadata.py) and each
rank logs them with views from ``layer_views``, built when its caches are
registered, and the grid ``FingerprintGrid(dcp_p, dcp_d, block_size)`` (P
passes ``dcp_d = 2 * dcp_p``):

- D, pair mode: in ``finalize_loaded_kv``, right after the step's exchanges,
  for each pull that ended (``fingerprint_pulls``) and each KV cache group
  ``g``: ``log_request(remote_request_id, views of g, table[m:], num_tokens=L,
  first_block=m)``. ``table`` is the request's block table of ``g`` from the KV
  cache manager, and ``m`` its local prefix hit in tokens divided by
  ``block_size * dcp_d``. Neither may come from the pair plan: its units and
  ``m_g`` also place the reads, so a wrong ``2m`` offset or unit list would
  label the rows it misplaced with their own tokens and pass. A pull that ended
  without its plan being emitted is logged by ``log_unexchanged``.
- P: in ``start_load_kv`` of the step after ``request_finished``
  (``fingerprint_sends``): ``log_request(request_id, views of g, block ids of
  g, num_tokens=remote_num_tokens)``, with the block ids ``request_finished``
  receives. The worker holds such a request's ``finished_sending`` until that
  step, so its blocks cannot be freed first.

Logging synchronizes with the device to read the sums back. On D that also
completes the exchange before the step posts its READs, so a fingerprinted run
cannot show the race the pair exchange's reuse guard closes.
"""

import functools
import hashlib
import json
import os
import socket
import time
from collections.abc import Container, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec, SparseCacheRole

FINGERPRINT_ENV = "VLLM_NIXL_KV_FINGERPRINT"
FINGERPRINT_DIR_ENV = "VLLM_NIXL_KV_FINGERPRINT_DIR"
RECORD_VERSION = 1
# Bytes of the per-row scale that follows an indexer kernel page's fp8 values.
INDEXER_SCALE_BYTES = 4

_U32 = (1 << 32) - 1
_U64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15
_MIX1 = 0xBF58476D1CE4E5B9
_MIX2 = 0x94D049BB133111EB
# Bounds the int64 temporaries to about 10 MB per pass at 1024-row blocks.
_BLOCKS_PER_PASS = 8


def fingerprint_dir() -> str | None:
    """Where fingerprints go, or None when they are off."""
    if os.environ.get(FINGERPRINT_ENV, "").strip().lower() not in ("1", "true"):
        return None
    out_dir = os.environ.get(FINGERPRINT_DIR_ENV)
    if not out_dir and (iterlog_dir := os.environ.get("VLLM_ITERLOG_DIR")):
        out_dir = os.path.join(iterlog_dir, "kv_fingerprint")
    if not out_dir:
        raise ValueError(
            f"{FINGERPRINT_ENV}=1 needs {FINGERPRINT_DIR_ENV} or VLLM_ITERLOG_DIR"
        )
    return out_dir


def splitmix64(x: int) -> int:
    """The splitmix64 finalizer of ``x + golden``, on unsigned 64-bit ints."""
    x = (x + _GOLDEN) & _U64
    x = ((x ^ (x >> 30)) * _MIX1) & _U64
    x = ((x ^ (x >> 27)) * _MIX2) & _U64
    return x ^ (x >> 31)


def salt(layer: str, field: str) -> int:
    """The unsigned 64-bit salt of a layer's field."""
    digest = hashlib.blake2b(f"{layer}\0{field}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def _i64(x: int) -> int:
    """An unsigned 64-bit int as the int64 with the same bits."""
    return x - (1 << 64) if x >> 63 else x


def _lsr(x: torch.Tensor, shift: int) -> torch.Tensor:
    """Logical right shift of int64 bits (``>>`` sign-extends)."""
    return (x >> shift) & ((1 << (64 - shift)) - 1)


def _splitmix64(x: torch.Tensor) -> torch.Tensor:
    # int64 add and multiply wrap mod 2**64, so this matches splitmix64().
    x = x + _i64(_GOLDEN)
    x = (x ^ _lsr(x, 30)) * _i64(_MIX1)
    x = (x ^ _lsr(x, 27)) * _i64(_MIX2)
    return x ^ _lsr(x, 31)


@functools.cache
def _multipliers(num_words: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [_i64(splitmix64(k) | 1) for k in range(num_words)],
        dtype=torch.int64,
        device=device,
    )


def _row_hash(rows: torch.Tensor) -> torch.Tensor:
    """``c(row)`` of each row of a ``[n, width]`` uint8 tensor, as int64 bits."""
    words = rows.contiguous().view(torch.int32).to(torch.int64) & _U32
    return (words * _multipliers(words.shape[1], words.device)).sum(dim=1)


@dataclass(frozen=True)
class FingerprintGrid:
    """The cells ``F`` is keyed by, for P DCP ``p_dcp_size`` feeding D DCP
    ``d_dcp_size``, both with ``block_size``-row blocks. Token ``t`` falls in
    cell ``(j, rho, h)``: D block ``j = t // (d_dcp_size * block_size)``, on D
    rank ``rho = t % d_dcp_size``, in its half (P block of ``j``)
    ``h = t // (p_dcp_size * block_size) % halves``."""

    p_dcp_size: int
    d_dcp_size: int
    block_size: int

    def __post_init__(self) -> None:
        if (
            min(self.p_dcp_size, self.d_dcp_size, self.block_size) < 1
            or self.d_dcp_size % self.p_dcp_size
        ):
            raise ValueError(f"Invalid fingerprint grid: {self}")

    @property
    def halves(self) -> int:
        return self.d_dcp_size // self.p_dcp_size

    def cell_keys(self, t: torch.Tensor) -> torch.Tensor:
        """``(j * d_dcp_size + rho) * halves + h`` of each token."""
        j = t // (self.d_dcp_size * self.block_size)
        h = t // (self.p_dcp_size * self.block_size) % self.halves
        return (j * self.d_dcp_size + t % self.d_dcp_size) * self.halves + h

    def decode(self, key: int) -> tuple[int, int, int]:
        """The ``(j, rho, h)`` of a cell key."""
        jr, h = divmod(key, self.halves)
        j, rho = divmod(jr, self.d_dcp_size)
        return j, rho, h


@dataclass(frozen=True)
class LayerView:
    """One layer's cache, as its attention backend binds it.

    ``cache`` is ``[num_pages, (1,) page_rows, width]``: manager blocks
    (``page_rows`` is the block size, ``pages_per_block`` 1) or kernel pages,
    page ``i`` of block ``b`` being ``b * pages_per_block + i``. Token rows
    (MLA, drafter) are ``width`` bytes; an indexer page holds its rows' fp8
    values, then their ``INDEXER_SCALE_BYTES``-byte scales.
    """

    name: str
    cache: torch.Tensor
    pages_per_block: int = 1
    indexer: bool = False
    draft: bool = False

    @property
    def fields(self) -> tuple[str, ...]:
        return ("idx_val", "idx_scale") if self.indexer else ("kv",)

    def rows(
        self, block_ids: Sequence[int], block_size: int
    ) -> list[tuple[str, torch.Tensor]]:
        """Each field's ``[len(block_ids) * block_size, width]`` uint8 rows."""
        cache = self.cache
        if cache.dtype != torch.uint8:
            cache = cache.view(torch.uint8)
        if cache.dim() == 4 and cache.shape[1] == 1:
            cache = cache[:, 0]
        if cache.dim() != 3:
            raise ValueError(f"{self.name}: unexpected cache shape {self.cache.shape}")
        num_pages, page_rows, width = cache.shape
        per_block = block_size // page_rows
        if (
            block_size % page_rows
            or per_block > self.pages_per_block
            or cache.stride(2) != 1
            or cache.stride(1) != width
            or width % 4
            or (self.indexer and width <= INDEXER_SCALE_BYTES)
        ):
            raise ValueError(
                f"{self.name}: {tuple(cache.shape)} pages of stride "
                f"{cache.stride()} do not hold {block_size}-row blocks "
                f"{self.pages_per_block} pages apart in 32-bit words"
            )
        if len(block_ids) and (
            min(block_ids) < 0
            or max(block_ids) * self.pages_per_block + per_block > num_pages
        ):
            raise ValueError(f"{self.name}: block ids out of {num_pages} pages")
        pages = cache.as_strided((num_pages, page_rows * width), (cache.stride(0), 1))
        ids = torch.tensor(block_ids, dtype=torch.int64, device=cache.device)
        offsets = torch.arange(per_block, dtype=torch.int64, device=cache.device)
        block_pages = pages[ids[:, None] * self.pages_per_block + offsets]
        num_rows = len(block_ids) * block_size
        if not self.indexer:
            return [("kv", block_pages.reshape(num_rows, width))]
        value_bytes = page_rows * (width - INDEXER_SCALE_BYTES)
        return [
            ("idx_val", block_pages[..., :value_bytes].reshape(num_rows, -1)),
            ("idx_scale", block_pages[..., value_bytes:].reshape(num_rows, -1)),
        ]


def layer_views(
    kv_caches: Mapping[str, torch.Tensor],
    layer_specs: Mapping[str, KVCacheSpec],
    forward_context: Mapping[str, Any],
    draft_layers: Container[str] = (),
) -> list[LayerView]:
    """The DCP-sharded layers' caches, through the views their backends bind.

    Args:
        kv_caches: The manager-block views registered with the connector.
        layer_specs: Each layer's KV cache spec.
        forward_context: The static forward context. A layer's ``kv_cache`` is
            the view its backend reads: 64-row kernel pages for the indexer.
        draft_layers: The drafter's layer names, besides specs marked
            ``is_draft``.

    Raises:
        ValueError: A DCP-sharded layer is not one token per row of an MLA
            page, or has no bound view paging its registered cache.

    """
    views = []
    for name, cache in kv_caches.items():
        spec = layer_specs.get(name)
        if spec is None or not spec.dcp_sharded:
            continue
        if not isinstance(spec, MLAAttentionSpec) or spec.tokens_per_state != 1:
            raise ValueError(f"{name}: cannot fingerprint {type(spec).__name__}")
        bound = getattr(forward_context.get(name), "kv_cache", None)
        if not isinstance(bound, torch.Tensor) or bound.numel() == 0:
            raise ValueError(f"{name}: no KV cache view is bound to the layer")
        block_stride = cache.stride(0) * cache.element_size()
        page_stride = bound.stride(0) * bound.element_size()
        if bound.data_ptr() != cache.data_ptr() or block_stride % page_stride:
            raise ValueError(f"{name}: the bound view does not page the cache")
        views.append(
            LayerView(
                name,
                bound,
                block_stride // page_stride,
                indexer=spec.cache_role == SparseCacheRole.INDEXER,
                draft=name in draft_layers or spec.is_draft,
            )
        )
    return views


def compute_fingerprint(
    views: Sequence[LayerView],
    block_ids: Sequence[int],
    num_tokens: int,
    grid: FingerprintGrid,
    dcp_size: int,
    dcp_rank: int,
    first_block: int = 0,
) -> tuple[list[tuple[int, int, int]], dict[tuple[str, str], list[int]]]:
    """``F`` of the cells of this rank's rows of a request's tokens
    ``[first_block * block_size * dcp_size, num_tokens)``.

    Args:
        views: The layers to fingerprint, all addressed by ``block_ids``.
        block_ids: The request's blocks from its ``first_block``-th on.
        num_tokens: The request's token count ``L``.
        grid: The cells ``F`` is keyed by.
        dcp_size: This side's DCP size.
        dcp_rank: This rank's DCP rank.
        first_block: The index of ``block_ids[0]`` among the request's blocks.

    Returns:
        The sorted cells ``(j, rho, h)``, and each ``(layer, field)``'s unsigned
        ``F`` per cell.

    """
    if not 0 <= dcp_rank < dcp_size or num_tokens < 0 or first_block < 0:
        raise ValueError(f"rank {dcp_rank}/{dcp_size}, {num_tokens=}, {first_block=}")
    block_size = grid.block_size
    first_row = first_block * block_size
    num_rows = max(0, cdiv(num_tokens - dcp_rank, dcp_size) - first_row)
    num_blocks = cdiv(num_rows, block_size)
    if len(block_ids) < num_blocks:
        raise ValueError(
            f"{len(block_ids)} blocks cannot hold rank {dcp_rank}'s {num_rows} "
            f"rows from block {first_block} on"
        )
    fields = [(view.name, field) for view in views for field in view.fields]
    if not views or num_rows == 0:
        return [], {key: [] for key in fields}
    device = views[0].cache.device
    tokens = (
        torch.arange(first_row, first_row + num_rows, device=device) * dcp_size
        + dcp_rank
    )
    keys, cell_of_row = torch.unique(grid.cell_keys(tokens), return_inverse=True)
    sums: list[torch.Tensor] = []
    for view in views:
        if view.cache.device != device:
            raise ValueError(f"{view.name} is not on {device}")
        view_sums = {
            field: torch.zeros(len(keys), dtype=torch.int64, device=device)
            for field in view.fields
        }
        for b0 in range(0, num_blocks, _BLOCKS_PER_PASS):
            r0 = b0 * block_size
            r1 = min(num_rows, r0 + _BLOCKS_PER_PASS * block_size)
            batch = block_ids[b0 : min(num_blocks, b0 + _BLOCKS_PER_PASS)]
            for field, rows in view.rows(batch, block_size):
                mix = _splitmix64(tokens[r0:r1] ^ _i64(salt(view.name, field)))
                terms = mix * (2 * _row_hash(rows[: r1 - r0]) + 1)
                view_sums[field].index_add_(0, cell_of_row[r0:r1], terms)
        sums.extend(view_sums.values())
    cells = [grid.decode(key) for key in keys.tolist()]
    values = torch.stack(sums).cpu().tolist()
    return cells, {
        key: [v & _U64 for v in row] for key, row in zip(fields, values, strict=True)
    }


class KVFingerprintLog:
    """Appends one rank's fingerprint records to its own JSONL file."""

    def __init__(
        self,
        out_dir: str,
        side: str,
        dcp_rank: int,
        grid: FingerprintGrid,
        engine_id: str | None = None,
    ):
        if side not in ("P", "D"):
            raise ValueError(f"side must be 'P' or 'D', not {side!r}")
        self.side = side
        self.dcp_rank = dcp_rank
        self.dcp_size = grid.p_dcp_size if side == "P" else grid.d_dcp_size
        self.grid = grid
        self.engine_id = engine_id
        os.makedirs(out_dir, exist_ok=True)
        host = socket.gethostname().split(".")[0]
        self.path = os.path.join(
            out_dir, f"kvfp-{side}-dcp{dcp_rank}-{host}-{os.getpid()}.jsonl"
        )

    @classmethod
    def maybe_create(
        cls,
        side: str,
        dcp_rank: int,
        grid: FingerprintGrid,
        engine_id: str | None = None,
    ) -> "KVFingerprintLog | None":
        out_dir = fingerprint_dir()
        if out_dir is None:
            return None
        return cls(out_dir, side, dcp_rank, grid, engine_id)

    def log_request(
        self,
        req_id: str,
        views: Sequence[LayerView],
        block_ids: Sequence[int],
        num_tokens: int,
        first_block: int = 0,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Fingerprints one request's KV cache group and appends the record.

        ``req_id`` joins the P and D logs: P's request id on both sides. The
        other arguments are ``compute_fingerprint``'s.
        """
        cells, fingerprints = compute_fingerprint(
            views,
            block_ids,
            num_tokens,
            self.grid,
            self.dcp_size,
            self.dcp_rank,
            first_block,
        )
        fp = [
            {
                "layer": view.name,
                "field": field,
                "draft": view.draft,
                "f": [f"{v:016x}" for v in fingerprints[view.name, field]],
            }
            for view in views
            for field in view.fields
        ]
        return self._append(
            req_id, num_tokens, first_block, block_ids, cells, fp, extra
        )

    def log_unexchanged(
        self,
        req_id: str,
        num_tokens: int,
        first_block: int,
        reason: str,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Appends the record of a pull that ended without its exchange.

        It has no cells: the comparator then reports every cell of the request
        missing on D, rather than listing the request as P only (aborted, or
        recomputed after a failed pull), which logs nothing on D.
        """
        extra = {"exchanged": False, "reason": reason, **(extra or {})}
        return self._append(req_id, num_tokens, first_block, [], [], [], extra)

    def _append(
        self,
        req_id: str,
        num_tokens: int,
        first_block: int,
        block_ids: Sequence[int],
        cells: list[tuple[int, int, int]],
        fp: list[dict[str, Any]],
        extra: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        record = {
            "v": RECORD_VERSION,
            "side": self.side,
            "rank": self.dcp_rank,
            "grid": [self.grid.p_dcp_size, self.grid.d_dcp_size, self.grid.block_size],
            "req": req_id,
            "engine": self.engine_id,
            "num_tokens": int(num_tokens),
            "first_block": int(first_block),
            "block_ids": [int(b) for b in block_ids],
            "cells": cells,
            "fp": fp,
            "time": time.time(),
            **({"extra": dict(extra)} if extra else {}),
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
        return record
