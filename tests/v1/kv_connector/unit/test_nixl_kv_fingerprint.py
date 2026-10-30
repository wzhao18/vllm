# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV fingerprints of NIXL pulls at DCP interleave 1 (design gate A9).

Each rank packs MLA, indexer and drafter pages into one block row (BLHNC) as
K3.1 does, P and D with different block strides, and binds the indexer as the
V2 runner does (64-row kernel pages). Tokens are written through the backends'
layouts where the offloading connector's canonical mapping (the inverse of the
V2 slot mapping) places them, never by this module's formula. P (DCP 8) and D
(DCP 16) must then agree cell by cell, also after today's pull and the pair
exchange's reference, and a corrupted row must change exactly its cell. CPU
only, except the CUDA parity check. The comparator test runs
RainHollow/k31-pd/tools/kv_fingerprint_compare.py, a local script, when it is
found.
"""

import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.dcp_pair_exchange import (
    PairRegion,
    build_pair_row_table,
    pair_units,
    reference_pair_exchange,
    reference_read_pages,
    reference_zero_pages,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.kv_fingerprint import (
    FINGERPRINT_DIR_ENV,
    FINGERPRINT_ENV,
    FingerprintGrid,
    KVFingerprintLog,
    compute_fingerprint,
    fingerprint_dir,
    layer_views,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.page_zeroer import (
    unwritten_blocks,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    NixlPullConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.canonical_mapping import (  # noqa: E501
    _local_to_canonical_token,
    _RankContext,
)
from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    SparseCacheRole,
    create_kv_cache_views,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.utils import AttentionGroup

W_P, W_D, B = 8, 16, 128
GRID = FingerprintGrid(W_P, W_D, B)
MLA = ("model.layers.3.self_attn.attn", "model.layers.7.self_attn.attn")
IDX = (
    "model.layers.3.self_attn.indexer.k_cache",
    "model.layers.7.self_attn.indexer.k_cache",
)
DRAFT = "model.layers.61.self_attn.attn"
MLA_WIDTH, IDX_WIDTH, IDX_HEAD, PAGE_ROWS = 576, 132, 128, 64
SCALE = IDX_WIDTH - IDX_HEAD
NUM_BLOCKS = 10
P_BLOCKS = [7, 2, 9, 4, 0, 5, 3, 8]
D_BLOCKS = [6, 1, 9, 3, 7]
# Two D chunks, then 1500 tokens: P block 5 is partial (even P block count).
L = 2 * W_D * B + 1500
# Every tail boundary: one row per D rank, a P block, a D chunk, and an odd P
# block count (W_D * B + W_P * B + 1, 2 * W_D * B + W_P * B - 100).
LENGTHS = [
    1,
    15,
    W_D,
    17,
    W_P * B - 1,
    W_P * B,
    W_P * B + 1,
    W_D * B - 1,
    W_D * B,
    W_D * B + 1,
    W_D * B + W_P * B + 1,
    2 * W_D * B + W_P * B - 100,
    L,
]
U64 = (1 << 64) - 1


class _IndexerBackend:
    """The two questions map_kv_cache_to_kernel_blocks asks, answered as
    KimiK3IndexerBackend does."""

    @staticmethod
    def get_name() -> str:
        return "KIMI_K3_INDEXER"

    @staticmethod
    def writes_kv_cache_from_block_table() -> bool:
        return True


def token_content(num_tokens: int, seed: int = 0) -> dict[tuple, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)

    def rows(width: int) -> torch.Tensor:
        return torch.randint(
            0, 256, (num_tokens, width), dtype=torch.uint8, generator=g
        )

    content = {(name, "kv"): rows(MLA_WIDTH) for name in (*MLA, DRAFT)}
    for name in IDX:
        content[name, "idx_val"] = rows(IDX_HEAD)
        content[name, "idx_scale"] = rows(SCALE)
    return content


class PackedRank:
    """One rank's KV cache: block rows of two MLA pages, two indexer pages and
    a drafter page, then padding (KDA state in K3.1) that differs between P
    and D, as their block strides do."""

    def __init__(self, dcp_size: int, dcp_rank: int, device: str = "cpu"):
        self.dcp_size, self.dcp_rank = dcp_size, dcp_rank
        mla = MLAAttentionSpec(
            block_size=B,
            num_kv_heads=1,
            head_size=MLA_WIDTH,
            dtype=torch.uint8,
            cache_dtype_str="fp8",
        )
        idx = MLAAttentionSpec(
            block_size=B,
            num_kv_heads=1,
            head_size=IDX_WIDTH,
            dtype=torch.uint8,
            cache_role=SparseCacheRole.INDEXER,
        )
        self.specs = {name: mla for name in MLA} | {name: idx for name in IDX}
        self.specs[DRAFT] = replace(mla, is_draft=True)
        mla_page, idx_page = B * MLA_WIDTH, B * IDX_WIDTH
        tensors = [(MLA, mla_page, 0), (IDX, idx_page, 2 * mla_page)]
        tensors.append(((DRAFT,), mla_page, 2 * (mla_page + idx_page)))
        self.used = 3 * mla_page + 2 * idx_page
        pad = 4096 if dcp_size == W_D else 3 * 4096
        self.row = round_up(self.used + pad, PAGE_ROWS * IDX_WIDTH)
        self.raw = torch.zeros(NUM_BLOCKS * self.row, dtype=torch.int8, device=device)
        self.buf = self.raw.view(torch.uint8)
        self.kv_caches: dict[str, torch.Tensor] = {}
        for names, page, offset in tensors:
            tensor = KVCacheTensor(
                size=self.raw.numel(),
                layers=list(names),
                layer_stride=page,
                block_stride=self.row,
                offset=offset,
            )
            views = create_kv_cache_views(
                self.raw, self.specs[names[0]], NUM_BLOCKS, KVCacheLayout.BLHNC, tensor
            )
            self.kv_caches.update(zip(names, views))
        self.bound = {name: self.kv_caches[name].squeeze(1) for name in (*MLA, DRAFT)}
        self.pages_per_block = {}
        for name in IDX:
            group = AttentionGroup(
                _IndexerBackend, [name], idx, 0, kernel_block_size=PAGE_ROWS
            )
            kernel_view = group.map_kv_cache_to_kernel_blocks(self.kv_caches[name])
            self.bound[name] = kernel_view.squeeze(1)
            self.pages_per_block[name] = group.kernel_block_stride
        self.forward_context = {
            name: SimpleNamespace(kv_cache=view) for name, view in self.bound.items()
        }
        self.views = layer_views(self.kv_caches, self.specs, self.forward_context)

    @property
    def table(self):
        """The pair exchanger's row table of this rank's pages."""
        regions = [
            PairRegion(
                name,
                self.specs[name],
                view.data_ptr(),
                self.specs[name].page_size_bytes,
                view.stride(0) * view.element_size(),
            )
            for name, view in self.kv_caches.items()
        ]
        return build_pair_row_table(
            regions, index_kernel_rows=PAGE_ROWS, index_value_bytes=IDX_HEAD
        )

    def tokens(self, first_block: int, num_blocks: int) -> np.ndarray:
        """The token that each row of the request's blocks ``[first_block,
        first_block + num_blocks)`` holds at interleave 1."""
        ctx = _RankContext(
            tp_size=self.dcp_size,
            dcp_size=self.dcp_size,
            pcp_size=1,
            interleave=1,
            total_kv_heads=1,
            rank=self.dcp_rank,
        )
        local = np.arange(first_block * B, (first_block + num_blocks) * B)
        return _local_to_canonical_token(local, ctx)

    def write(self, name: str, field: str, blocks, ys, rows) -> None:
        """Writes rows at ``(blocks, ys)`` as the backend's cache writer does."""
        blocks, ys = torch.as_tensor(blocks), torch.as_tensor(ys)
        if field == "kv":
            # concat_and_cache_mla: slot block * B + y of [blocks, B, width].
            self.bound[name][blocks, ys] = rows
            return
        # The indexer writer, at the block-table slot (block * stride + y // 64)
        # * 64 + y % 64: a kernel page holds its rows' values, then scales.
        pages = self.bound[name]
        flat = pages.view(pages.shape[0], -1)
        page = blocks * self.pages_per_block[name] + ys // PAGE_ROWS
        i = (ys % PAGE_ROWS)[:, None]
        if field == "idx_val":
            cols = i * IDX_HEAD + torch.arange(IDX_HEAD)
        else:
            cols = PAGE_ROWS * IDX_HEAD + i * SCALE + torch.arange(SCALE)
        flat[page[:, None], cols] = rows

    def fill(self, content, block_ids, num_tokens: int, first_block: int = 0):
        """Writes every token below ``num_tokens`` that the request's blocks
        from ``first_block`` on hold on this rank."""
        tokens = self.tokens(first_block, len(block_ids))
        rows = np.flatnonzero(tokens < num_tokens)
        blocks = np.asarray(block_ids)[rows // B]
        held = torch.from_numpy(tokens[rows])
        for (name, field), values in content.items():
            self.write(name, field, blocks, rows % B, values[held])

    def locate(self, t: int, block_ids, first_block: int = 0) -> tuple[int, int]:
        (row,) = np.flatnonzero(self.tokens(first_block, len(block_ids)) == t)
        return block_ids[row // B], row % B

    def write_token(self, name, field, t, row, block_ids, first_block=0) -> None:
        block, y = self.locate(t, block_ids, first_block)
        self.write(name, field, [block], [y], row[None])

    def fingerprint(self, block_ids, num_tokens: int, first_block: int = 0):
        return compute_fingerprint(
            self.views,
            block_ids,
            num_tokens,
            GRID,
            self.dcp_size,
            self.dcp_rank,
            first_block,
        )


def merged(results) -> dict[tuple, int]:
    """(layer, field, j, rho, h) -> F over ranks; no cell twice."""
    out: dict[tuple, int] = {}
    for cells, fingerprints in results:
        for (layer, field), values in fingerprints.items():
            for cell, value in zip(cells, values, strict=True):
                key = (layer, field, *cell)
                assert key not in out
                out[key] = value
    return out


def deploy(num_tokens: int, m: int, d_draft_blocks=D_BLOCKS):
    """P and D ranks as the exchange should leave them: every rank holds its
    interleave-1 tokens. D's m cached blocks hold unrelated bytes. On D, the
    drafter may be its own KV cache group, in ``d_draft_blocks``."""
    content = token_content(num_tokens)
    p_ranks = [PackedRank(W_P, p) for p in range(W_P)]
    for rank in p_ranks:
        rank.fill(content, P_BLOCKS, num_tokens)
    d_ranks = [PackedRank(W_D, r) for r in range(W_D)]
    for rank in d_ranks:
        for blocks, draft in ((D_BLOCKS, False), (d_draft_blocks, True)):
            group = {k: v for k, v in content.items() if (k[0] == DRAFT) == draft}
            rank.fill(group, blocks[m:], num_tokens, first_block=m)
        rank.raw.view(NUM_BLOCKS, rank.row)[D_BLOCKS[:m]] = 77
    return content, p_ranks, d_ranks


def fingerprints(p_ranks, d_ranks, num_tokens: int, m: int):
    p = merged(rank.fingerprint(P_BLOCKS, num_tokens) for rank in p_ranks)
    d = merged(
        rank.fingerprint(D_BLOCKS[m:], num_tokens, first_block=m) for rank in d_ranks
    )
    return p, d


def assert_d_matches_p(p: dict, d: dict, m: int) -> None:
    checked = {key: value for key, value in p.items() if key[2] >= m}
    assert d.keys() == checked.keys()
    assert d == checked


@pytest.mark.cpu_test
def test_layer_views_follow_the_bound_views():
    rank = PackedRank(W_D, 0)
    assert rank.row != PackedRank(W_P, 0).row
    views = {view.name: view for view in rank.views}
    assert views.keys() == {*MLA, *IDX, DRAFT}
    for name in IDX:
        assert views[name].indexer
        assert views[name].fields == ("idx_val", "idx_scale")
        assert views[name].pages_per_block == rank.row // (PAGE_ROWS * IDX_WIDTH)
        assert views[name].cache.shape[1:] == (PAGE_ROWS, IDX_WIDTH)
    for name in (*MLA, DRAFT):
        assert views[name].fields == ("kv",)
        assert views[name].pages_per_block == 1
        # The drafter is marked by its spec, or by name.
        assert views[name].draft == (name == DRAFT)
    named = layer_views(
        rank.kv_caches, rank.specs, rank.forward_context, draft_layers={MLA[0]}
    )
    assert {view.name for view in named if view.draft} == {MLA[0], DRAFT}
    # A replicated cache is not DCP-sharded, so not fingerprinted.
    replicated = rank.specs | {DRAFT: replace(rank.specs[DRAFT], dcp_sharded=False)}
    assert DRAFT not in {
        view.name
        for view in layer_views(rank.kv_caches, replicated, rank.forward_context)
    }


@pytest.mark.cpu_test
def test_layer_views_fail_loudly():
    """No fallback to the registered manager view, which would read the
    indexer's kernel pages as one page, and no layer is silently skipped."""
    rank = PackedRank(W_D, 0)
    unbound = rank.forward_context | {IDX[0]: SimpleNamespace(kv_cache=torch.empty(0))}
    with pytest.raises(ValueError, match="no KV cache view"):
        layer_views(rank.kv_caches, rank.specs, unbound)
    missing = {k: v for k, v in rank.forward_context.items() if k != MLA[1]}
    with pytest.raises(ValueError, match="no KV cache view"):
        layer_views(rank.kv_caches, rank.specs, missing)
    gqa = FullAttentionSpec(
        block_size=B, num_kv_heads=2, head_size=64, dtype=torch.bfloat16
    )
    with pytest.raises(ValueError, match="cannot fingerprint FullAttentionSpec"):
        layer_views(rank.kv_caches, rank.specs | {MLA[0]: gqa}, rank.forward_context)


@pytest.mark.cpu_test
@pytest.mark.parametrize("num_tokens", LENGTHS)
@pytest.mark.parametrize("m", [0, 1, 2])
def test_d_matches_p_cell_by_cell(num_tokens: int, m: int):
    _, p_ranks, d_ranks = deploy(num_tokens, m)
    p, d = fingerprints(p_ranks, d_ranks, num_tokens, m)
    assert_d_matches_p(p, d, m)
    assert {key[2] for key in p} == set(range(cdiv(num_tokens, W_D * B)))


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("num_tokens", "m"), [(L, 1), (2 * W_D * B + W_P * B - 100, 0)]
)
def test_fingerprints_match_after_pull_and_exchange(num_tokens: int, m: int):
    """Today's pull (the worker's own block matching, whole pages, unwritten
    pages zeroed) and the pair exchange's torch reference leave D's
    fingerprints equal to P's. Without the exchange every cell differs."""
    content = token_content(num_tokens)
    p_ids = P_BLOCKS[: cdiv(num_tokens, W_P * B)]
    d_ids = D_BLOCKS[: cdiv(num_tokens, W_D * B)]
    p_ranks = [PackedRank(W_P, p) for p in range(W_P)]
    for rank in p_ranks:
        rank.fill(content, p_ids, num_tokens)
    d_ranks = [PackedRank(W_D, r) for r in range(W_D)]
    worker = object.__new__(NixlPullConnectorWorker)
    for r, rank in enumerate(d_ranks):
        rank.raw.fill_(0x3C)
        dst, src = worker._apply_dcp_prefix_caching(
            d_ids[m:],
            p_ids,
            remote_rank=r % W_P,
            local_dcp_size=W_D,
            local_dcp_rank=r,
            remote_dcp_size=W_P,
            local_num_computed_blocks=m,
        )
        p_rank = p_ranks[r % W_P]
        reference_read_pages(rank.buf, rank.table, dst, p_rank.buf, p_rank.table, src)
        reference_zero_pages(
            rank.buf, rank.table, unwritten_blocks(d_ids[m:], set(dst))
        )
    p = merged(rank.fingerprint(p_ids, num_tokens) for rank in p_ranks)

    def d_fingerprints():
        return merged(rank.fingerprint(d_ids[m:], num_tokens, m) for rank in d_ranks)

    skipped = d_fingerprints()
    units = pair_units([d_ids[m:]], [m], [p_ids], groups=[0])
    reference_pair_exchange(
        [rank.buf for rank in d_ranks], [{0: rank.table} for rank in d_ranks], units
    )
    assert_d_matches_p(p, d_fingerprints(), m)
    assert skipped.keys() == d_fingerprints().keys()
    assert all(value != p[key] for key, value in skipped.items())


@pytest.mark.cpu_test
def test_corrupted_row_is_caught_and_localized():
    m = 1
    content, p_ranks, d_ranks = deploy(L, m)
    p, d = fingerprints(p_ranks, d_ranks, L, m)
    assert_d_matches_p(p, d, m)

    # D rank 11, block j=2, row B/2 + 3: half 1. One flipped indexer scale bit.
    t = W_D * (2 * B + B // 2 + 3) + 11
    row = content[IDX[1], "idx_scale"][t].clone()
    row[2] ^= 0x10
    d_ranks[11].write_token(IDX[1], "idx_scale", t, row, D_BLOCKS[m:], m)
    d_bad = merged(r.fingerprint(D_BLOCKS[m:], L, first_block=m) for r in d_ranks)
    diff = {key for key in d if d[key] != d_bad[key]}
    assert diff == {(IDX[1], "idx_scale", 2, 11, 1)}

    # The same token's MLA row on P rank 3 (block 5, odd rows).
    rank = p_ranks[11 % W_P]
    block, y = rank.locate(t, P_BLOCKS)
    assert (block, y % 2) == (P_BLOCKS[5], 1)
    row = content[MLA[0], "kv"][t].clone()
    row[-1] ^= 1
    rank.write_token(MLA[0], "kv", t, row, P_BLOCKS)
    p_bad = merged(r.fingerprint(P_BLOCKS, L) for r in p_ranks)
    diff = {key for key in p if p[key] != p_bad[key]}
    assert diff == {(MLA[0], "kv", 2, 11, 1)}


@pytest.mark.cpu_test
def test_fingerprint_reads_only_the_layer_pages():
    rank = PackedRank(W_D, 5)
    content = token_content(L)
    rank.fill(content, D_BLOCKS, L)
    before = rank.fingerprint(D_BLOCKS, L)
    rows = rank.raw.view(NUM_BLOCKS, rank.row)
    rows[:, rank.used :] = -1
    rows[[b for b in range(NUM_BLOCKS) if b not in D_BLOCKS]] = 3
    assert rank.fingerprint(D_BLOCKS, L) == before


@pytest.mark.cpu_test
def test_moving_a_row_between_tokens_changes_f():
    rank = PackedRank(W_D, 2)
    content = token_content(L)
    rank.fill(content, D_BLOCKS, L)
    cells, before = rank.fingerprint(D_BLOCKS, L)
    rows = content[MLA[1], "kv"]

    def swap(t1: int, t2: int):
        moved = {key: list(values) for key, values in before.items()}
        for src, dst in ((t1, t2), (t2, t1)):
            rank.write_token(MLA[1], "kv", dst, rows[src], D_BLOCKS)
        _, after = rank.fingerprint(D_BLOCKS, L)
        for src in (t1, t2):
            rank.write_token(MLA[1], "kv", src, rows[src], D_BLOCKS)
        changed = {
            cells[i]
            for i, (a, b) in enumerate(zip(after[MLA[1], "kv"], before[MLA[1], "kv"]))
            if a != b
        }
        del moved[MLA[1], "kv"], after[MLA[1], "kv"]
        assert after == moved
        return changed

    # Same cell: the cell's multiset of rows is unchanged, its F is not.
    t1, t2 = 2 + W_D * 5, 2 + W_D * 9
    assert swap(t1, t2) == {(0, 2, 0)}
    # Across halves and chunks: both cells change.
    t3 = 2 + W_D * (B + B // 2 + 1)
    assert swap(t1, t3) == {(0, 2, 0), (1, 2, 1)}
    assert rank.fingerprint(D_BLOCKS, L) == (cells, before)


@pytest.mark.cpu_test
def test_identical_content_in_different_slots_does_not_collide():
    rank = PackedRank(W_P, 1)
    # All rows zero: every cell, layer and field still gets its own F.
    cells, zeros = rank.fingerprint(P_BLOCKS, L)
    values = [v for vs in zeros.values() for v in vs]
    assert len(cells) == 2 * cdiv(L, W_P * B)
    assert len(set(values)) == len(values)

    # One row X at token t1 or at token t2 of the same cell.
    x = token_content(1, seed=5)[MLA[0], "kv"][0]
    t1, t2 = 1 + W_P * 6, 1 + W_P * 10
    seen = []
    for t in (t1, t2):
        rank.write_token(MLA[0], "kv", t, x, P_BLOCKS)
        seen.append(rank.fingerprint(P_BLOCKS, L)[1][MLA[0], "kv"])
        rank.write_token(MLA[0], "kv", t, torch.zeros_like(x), P_BLOCKS)
    assert seen[0] != seen[1]
    assert sum(a != b for a, b in zip(*seen)) == 1


@pytest.mark.cpu_test
def test_indexer_scale_bytes_are_covered():
    rank = PackedRank(W_D, 9)
    content = token_content(L)
    rank.fill(content, D_BLOCKS, L)
    cells, before = rank.fingerprint(D_BLOCKS, L)
    t = 9 + W_D * (B + 70)  # chunk 1, half 1, kernel page 1 of its block
    for field, byte in (("idx_scale", 3), ("idx_val", 0), ("idx_val", 127)):
        row = content[IDX[0], field][t].clone()
        row[byte] ^= 0x80
        rank.write_token(IDX[0], field, t, row, D_BLOCKS)
        _, after = rank.fingerprint(D_BLOCKS, L)
        rank.write_token(IDX[0], field, t, content[IDX[0], field][t], D_BLOCKS)
        changed = {
            (key, cells[i])
            for key in before
            for i, (a, b) in enumerate(zip(after[key], before[key]))
            if a != b
        }
        assert changed == {((IDX[0], field), (1, 9, 1))}


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & U64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & U64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & U64
    return x ^ (x >> 31)


def reference(content, rank: PackedRank, num_tokens: int, first_block: int):
    """The design's F, straight from the tokens' rows: no cache involved."""
    multipliers = [_splitmix64(k) | 1 for k in range(MLA_WIDTH // 4)]
    tokens = rank.tokens(first_block, NUM_BLOCKS)
    out: dict[tuple, int] = {}
    for (layer, field), rows in content.items():
        digest = hashlib.blake2b(f"{layer}\0{field}".encode(), digest_size=8)
        salt = int.from_bytes(digest.digest(), "little")
        for t in tokens[tokens < num_tokens].tolist():
            data = bytes(rows[t].tolist())
            words = [
                int.from_bytes(data[k : k + 4], "little")
                for k in range(0, len(data), 4)
            ]
            c = sum(w * m for w, m in zip(words, multipliers)) & U64
            cell = (t // (W_D * B), t % W_D, t // (W_P * B) % (W_D // W_P))
            key = (layer, field, *cell)
            out[key] = (out.get(key, 0) + _splitmix64(t ^ salt) * (2 * c + 1)) & U64
    return out


@pytest.mark.cpu_test
@pytest.mark.parametrize(("dcp_size", "dcp_rank", "m"), [(W_P, 6, 0), (W_D, 13, 1)])
def test_matches_the_design_definition(dcp_size: int, dcp_rank: int, m: int):
    num_tokens = W_D * B + W_P * B + 300
    content = token_content(num_tokens, seed=3)
    rank = PackedRank(dcp_size, dcp_rank)
    rank.fill(content, D_BLOCKS[m:], num_tokens, first_block=m)
    got = merged([rank.fingerprint(D_BLOCKS[m:], num_tokens, first_block=m)])
    assert got == reference(content, rank, num_tokens, m)


@pytest.mark.cpu_test
def test_rejects_block_lists_too_short_for_the_tokens():
    rank = PackedRank(W_D, 0)
    with pytest.raises(ValueError, match="blocks cannot hold"):
        rank.fingerprint(D_BLOCKS[:2], L)
    cells, values = rank.fingerprint([], 0)
    assert cells == [] and len(values) == 7
    assert all(v == [] for v in values.values())


@pytest.mark.cpu_test
def test_env_gating(monkeypatch, tmp_path):
    for env in (FINGERPRINT_ENV, FINGERPRINT_DIR_ENV, "VLLM_ITERLOG_DIR"):
        monkeypatch.delenv(env, raising=False)
    assert fingerprint_dir() is None
    assert KVFingerprintLog.maybe_create("D", 3, GRID) is None
    monkeypatch.setenv(FINGERPRINT_ENV, "0")
    assert fingerprint_dir() is None
    monkeypatch.setenv(FINGERPRINT_ENV, "1")
    with pytest.raises(ValueError, match=FINGERPRINT_DIR_ENV):
        fingerprint_dir()
    monkeypatch.setenv("VLLM_ITERLOG_DIR", str(tmp_path / "iterlog"))
    assert fingerprint_dir() == str(tmp_path / "iterlog" / "kv_fingerprint")
    monkeypatch.setenv(FINGERPRINT_DIR_ENV, str(tmp_path / "kvfp"))
    assert fingerprint_dir() == str(tmp_path / "kvfp")


@pytest.mark.cpu_test
def test_log_appends_one_record_per_call(monkeypatch, tmp_path):
    monkeypatch.setenv(FINGERPRINT_ENV, "1")
    monkeypatch.setenv(FINGERPRINT_DIR_ENV, str(tmp_path))
    rank = PackedRank(W_D, 3)
    rank.fill(token_content(L), D_BLOCKS[1:], L, first_block=1)
    log = KVFingerprintLog.maybe_create("D", 3, GRID, engine_id="d0")
    assert log is not None
    log.log_request("p-req", rank.views, np.array(D_BLOCKS[1:]), L, first_block=1)
    log.log_request("p-req2", rank.views, [], 0, extra={"group": 0})
    (path,) = tmp_path.iterdir()
    assert path.name.startswith("kvfp-D-dcp3-")
    first, second = (json.loads(line) for line in path.read_text().splitlines())
    cells, values = rank.fingerprint(D_BLOCKS[1:], L, first_block=1)
    assert first["req"] == "p-req" and first["side"] == "D" and first["rank"] == 3
    assert first["grid"] == [W_P, W_D, B] and first["engine"] == "d0"
    assert (first["num_tokens"], first["first_block"]) == (L, 1)
    assert first["block_ids"] == D_BLOCKS[1:]
    assert [tuple(c) for c in first["cells"]] == cells
    logged = {(e["layer"], e["field"]): e for e in first["fp"]}
    assert logged.keys() == values.keys()
    for key, entry in logged.items():
        assert [int(f, 16) for f in entry["f"]] == values[key]
        assert entry["draft"] == (key[0] == DRAFT)
    assert second["cells"] == [] and second["extra"] == {"group": 0}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_cuda_matches_cpu():
    content = token_content(L)
    cpu = PackedRank(W_D, 7)
    cpu.fill(content, D_BLOCKS, L)
    gpu = PackedRank(W_D, 7, device="cuda")
    gpu.raw.copy_(cpu.raw)
    assert gpu.fingerprint(D_BLOCKS, L) == cpu.fingerprint(D_BLOCKS, L)


def _comparator(monkeypatch):
    """The comparator script: $KV_FINGERPRINT_COMPARE, else the main checkout's."""
    path = os.environ.get("KV_FINGERPRINT_COMPARE") or str(
        Path(__file__).resolve().parents[4]
        / "RainHollow/k31-pd/tools/kv_fingerprint_compare.py"
    )
    if not os.path.exists(path):
        pytest.skip(f"comparator not found at {path} (set KV_FINGERPRINT_COMPARE)")
    spec = importlib.util.spec_from_file_location("kv_fingerprint_compare", path)
    module = importlib.util.module_from_spec(spec)
    # Leave no bytecode cache beside the local script.
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    spec.loader.exec_module(module)
    return module


@pytest.mark.cpu_test
def test_comparator_expects_the_cells_ranks_log(monkeypatch):
    """The cells it expects from L and m alone are those the ranks log, at
    every tail boundary."""
    compare = _comparator(monkeypatch)
    views = PackedRank(W_D, 0).views[:1]
    blocks = list(range(NUM_BLOCKS))
    for num_tokens in LENGTHS:
        for m in (0, 1, 2):
            expected = set(compare.expected_cells(num_tokens, m, (W_P, W_D, B)))
            for dcp_size, first in ((W_D, m), (W_P, 0)):
                logged = set()
                for rank in range(dcp_size):
                    cells, _ = compute_fingerprint(
                        views, blocks[first:], num_tokens, GRID, dcp_size, rank, first
                    )
                    logged.update(cell for cell in cells if cell[0] >= m)
                assert logged == expected, (num_tokens, m, dcp_size)


@pytest.mark.cpu_test
def test_comparator_localizes_a_corrupted_row(tmp_path, capsys, monkeypatch):
    compare = _comparator(monkeypatch)
    m = 1
    # D's drafter is its own KV cache group: other blocks, its own record.
    d_draft_blocks = [2, 8, 0, 4, 5]
    content, p_ranks, d_ranks = deploy(L, m, d_draft_blocks)

    def log_all(out_dir, d_drafter_flag=True):
        for rank in p_ranks:
            log = KVFingerprintLog(str(out_dir), "P", rank.dcp_rank, GRID)
            log.log_request("req-0", rank.views, P_BLOCKS, L)
        for rank in d_ranks:
            log = KVFingerprintLog(str(out_dir), "D", rank.dcp_rank, GRID)
            target = [view for view in rank.views if not view.draft]
            (draft,) = (view for view in rank.views if view.draft)
            draft = replace(draft, draft=d_drafter_flag)
            for views, blocks in ((target, D_BLOCKS), ([draft], d_draft_blocks)):
                log.log_request("req-0", views, blocks[m:], L, first_block=m)
        # Aborted before its exchange: P only, not checked.
        KVFingerprintLog(str(out_dir), "P", 0, GRID).log_request(
            "req-aborted", p_ranks[0].views, P_BLOCKS, L
        )

    def corrupt(rank, t, block_ids, first_block=0):
        row = content[DRAFT, "kv"][t].clone()
        row[100] ^= 0xFF
        rank.write_token(DRAFT, "kv", t, row, block_ids, first_block)
        return lambda: rank.write_token(
            DRAFT, "kv", t, content[DRAFT, "kv"][t], block_ids, first_block
        )

    good = tmp_path / "good"
    log_all(good)
    assert compare.main([str(good)]) == 0
    out = capsys.readouterr().out
    assert "1 checked, 1 P only" in out and "0 MISMATCH" in out
    assert "layers: 5 checked, 1 drafter" in out
    assert compare.main([str(good), "--require-d"]) == 1
    # A torn line is a record that cannot be checked.
    with open(next(good.iterdir()), "a") as f:
        f.write('{"v":1,"side":')
    assert compare.main([str(good)]) == 1
    assert "torn line" in capsys.readouterr().out
    # The drafter exemption needs both sides to agree on which layers draft.
    log_all(tmp_path / "unflagged", d_drafter_flag=False)
    assert compare.main([str(tmp_path / "unflagged")]) == 1
    assert "logged both as drafter and not" in capsys.readouterr().out

    t = W_D * (B + 5) + 4  # D rank 4, chunk 1, half 0, row 5
    restore = corrupt(d_ranks[4], t, d_draft_blocks[m:], m)
    log_all(tmp_path / "bad")
    restore()
    report_path = tmp_path / "report.json"
    assert compare.main([str(tmp_path / "bad"), "--json", str(report_path)]) == 1
    report = json.loads(report_path.read_text())
    (x,) = report["mismatches"]
    assert (x["req"], x["layer"], x["field"]) == ("req-0", DRAFT, "kv")
    assert (x["j"], x["rho"], x["h"], x["draft"]) == (1, 4, 0, True)
    assert (x["d_rank"], x["d_rows"]) == (4, [0, B // 2])
    assert x["d_block"] == d_draft_blocks[1]
    assert (x["p_rank"], x["p_block_index"], x["p_block"]) == (4, 2, P_BLOCKS[2])
    assert x["p_row_residue"] == [0, 2]
    assert report["draft_tail"] == [] and not report["passed"]

    # A drafter row P's last block lacks: the propose window, acceptance only.
    t = (L // W_P - 2) * W_P + 2  # P rank 2, last block
    corrupt(p_ranks[2], t, P_BLOCKS)
    log_all(tmp_path / "tail")
    assert compare.main([str(tmp_path / "tail")]) == 1
    assert "draft tail: req-0" in capsys.readouterr().out
    assert compare.main([str(tmp_path / "tail"), "--allow-draft-tail"]) == 0

    # P's last prefill chunk spans its last two blocks: only a window that
    # covers both passes their drafter rows.
    corrupt(p_ranks[2], 2 + W_P * (4 * B + 8), P_BLOCKS)
    log_all(tmp_path / "tail2")
    tail2 = [str(tmp_path / "tail2"), "--allow-draft-tail"]
    assert compare.main(tail2) == 1
    assert compare.main([*tail2, "--draft-tail-tokens", str(L - 4 * W_P * B)]) == 0
