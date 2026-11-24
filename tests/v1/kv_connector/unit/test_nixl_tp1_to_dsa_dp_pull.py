# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A TP1 prefiller into a TP4 DSA-DP decoder at DCP 1, on CPU.

K3.1's packed block covers the per-rank KDA state, so a TP1 prefiller (DEP
prefill) has 4x the tokens per block of a TP4 decoder (16384 vs 4096). The
decoder runs DSA-DP: a request's DSA KV lives in its owner card's region. The
real handshake and read path pull over the real planner's packed geometry with
a byte-moving NIXL stand-in (test_nixl_pair_exchange_worker.py).
"""

import numpy as np
import pytest

from tests.v1.kv_connector.unit import test_nixl_pair_exchange_worker as k31
from tests.v1.kv_connector.unit.test_nixl_pair_exchange_worker import (
    CANARY,
    DRAFT_HIT_UNIT,
    DRAFT_KV_HEADS,
    DRAFT_WINDOW,
    KDA_FILL,
    NULL_FILL,
    P_CANARY,
    STALE,
    _canonical_tokens,
    _draft_page,
    _handshake,
    _Rank,
    _recv_metadata,
    _row_tags,
)
from vllm.utils.math_utils import cdiv

pytestmark = pytest.mark.cpu_test

# The K3.1 geometry fixtures: KDA state groups and the byte-moving transport.
fabric, k31_env = k31.fabric, k31.k31_env

D_TP, D_BLOCK, SUB_BLOCKS = 4, 64, 4
P_BLOCK = SUB_BLOCKS * D_BLOCK
NUM_DSA, NUM_KDA, STATE_GROUPS = 2, 4, 2


class _DsaDpRank(_Rank):
    """A decoder rank whose DSA group lives in per-card regions: block ids
    ``[0, S)`` are shared, card ``c`` owns ``[S + c D, S + (c + 1) D)`` at its
    local pages ``[S, S + D)`` (vllm/v1/core/dsa_dp_regions.py)."""

    def __init__(self, *args, region_blocks: int, **kwargs):
        self.region_blocks = region_blocks
        super().__init__(*args, **kwargs)

    def _start_worker(self, fabric, layers, extra):
        cfg = self.kv_cache_config
        cfg.dsa_dp_group_ids = (self.dsa_group,)
        cfg.dsa_dp_region_blocks = self.region_blocks
        cfg.dsa_dp_shared_blocks = cfg.num_blocks - self.region_blocks
        cfg.dsa_dp_num_regions = self.tp
        return super()._start_worker(fabric, layers, extra)


def _geometry(draft: str | None) -> dict:
    """``draft``: None, "heads" (the SWA DSpark draft's GQA blocks follow the
    pool block, head-major pages) or "slots" (its blocks are the prefix-match
    unit, packed as slots of the pool rows)."""
    if draft is None:
        return dict(num_dsa=NUM_DSA, num_kda=NUM_KDA)
    return dict(
        num_dsa=NUM_DSA,
        num_kda=NUM_KDA,
        draft_window=DRAFT_WINDOW,
        draft_hit_unit=DRAFT_HIT_UNIT if draft == "slots" else None,
    )


def _prefiller(fabric, num_blocks: int, draft: str | None = None) -> _Rank:
    return _Rank(
        fabric, 1, 0, num_blocks, block_size=P_BLOCK, decoder=False, **_geometry(draft)
    )


def _decoders(
    fabric, num_blocks: int, region_blocks: int, draft: str | None = None
) -> list[_DsaDpRank]:
    return [
        _DsaDpRank(
            fabric,
            D_TP,
            r,
            num_blocks,
            block_size=D_BLOCK,
            decoder=True,
            dcp=1,
            region_blocks=region_blocks,
            **_geometry(draft),
        )
        for r in range(D_TP)
    ]


def _by_group(rank: _Rank, dsa, kda: list, draft: list) -> tuple:
    values = dict(zip(rank.kda_groups(), kda)) | dict(zip(rank.draft_groups, draft))
    values[rank.dsa_group] = dsa
    return tuple(values[g] for g in range(len(values)))


def test_tp1_prefiller_handshake_with_tp4_dsa_dp_decoder(k31_env, fabric):
    """The decoder's handshake with a TP1 prefiller whose packed blocks hold 4x
    its tokens. Without remote block splitting it failed in block_size_ratio
    ("Local block size 64 is not divisible by remote block size 256"), so P/D
    at these shapes either crashed the handshake or, with
    kv_load_failure_policy=recompute, silently recomputed every prompt."""
    prefiller = _prefiller(fabric, 8)
    for rank in _decoders(fabric, 24, region_blocks=12):
        _handshake(rank, {0: prefiller}, remote_tp=1)
        assert set(rank.worker._remote_agents["P"]) == {(0, 0)}


@pytest.mark.parametrize(
    "num_tokens,num_cached,owner,draft",
    [
        (1000, 0, 2, None),
        (1000, 3, 1, None),
        (700, 0, 0, None),
        (1024, 4, 3, None),
        (65, 0, 1, None),
        (1000, 0, 2, "heads"),
        (1000, 13, 1, "heads"),
        (700, 0, 3, "heads"),
        (1000, 0, 2, "slots"),
        (1000, 13, 0, "slots"),
        (777, 0, 3, "slots"),
    ],
)
def test_tp1_prefiller_pulls_into_tp4_dsa_dp_decoder(
    k31_env, fabric, num_tokens, num_cached, owner, draft
):
    """Every D rank pulls its KDA state shard and its draft heads' window from
    P's only rank; the owner card alone pulls the DSA rows (latent, indexer
    values and scales), each local block from the P sub-block holding its
    tokens, into its region pages after its prefix-cache hit. Rows past the
    prompt are zero or P's junk, and no other byte changes on any card."""
    rng = np.random.default_rng(num_tokens + owner)
    num_p = cdiv(num_tokens, P_BLOCK)
    num_d = cdiv(num_tokens, D_BLOCK)

    # Draft blocks by token position. P sends its window tail ending at the
    # last computed token; D allocates its window after its prefix-cache hit.
    p_draft_block = DRAFT_HIT_UNIT if draft == "slots" else P_BLOCK
    d_draft_block = DRAFT_HIT_UNIT if draft == "slots" else D_BLOCK
    p_num_pos = cdiv(num_tokens, p_draft_block)
    p_first = max(0, p_num_pos - (cdiv(DRAFT_WINDOW, p_draft_block) + 1))
    p_positions = range(p_first, p_num_pos)
    d_num_pos = cdiv(num_tokens, d_draft_block)
    window_first = max(0, num_tokens - DRAFT_WINDOW + 1) // d_draft_block
    d_first = max(num_cached * D_BLOCK // d_draft_block, window_first)
    d_positions = range(d_first, d_num_pos)
    num_draft_groups = 3 if draft else 0

    def draft_ids(rank: _Rank, rows: list[int], num: int) -> list[list[int]]:
        """``num`` draft block ids per draft group, each group from its own
        ``num`` rows: the rows, or shuffled ids of their slots when the group
        packs ``blocks_per_pool_block`` slots per row (row 0 is the null)."""
        ids = []
        for i, g in enumerate(rank.draft_groups):
            group_rows = rows[i * num : (i + 1) * num]
            slots = rank.kv_cache_config.kv_cache_groups[g].blocks_per_pool_block
            if slots > 1:
                slot_ids = [row * slots + k for row in group_rows for k in range(slots)]
                group_rows = rng.permutation(slot_ids).tolist()
            ids.append(group_rows[:num])
        return ids

    # P: block 0 is the null block; the request's DSA blocks, a KDA block per
    # state group, the draft rows and a spare.
    num_p_draft_rows = num_draft_groups * len(p_positions)
    num_p_pool = 1 + num_p + STATE_GROUPS + num_p_draft_rows + 1
    p_pool = (1 + rng.permutation(num_p_pool - 1)).tolist()
    p_ids, p_kda = p_pool[:num_p], p_pool[num_p : num_p + STATE_GROUPS]
    p_rest = p_pool[num_p + STATE_GROUPS :]
    prefiller = _prefiller(fabric, num_p_pool, draft)
    p_draft = draft_ids(prefiller, p_rest, len(p_positions))
    arrays = prefiller.arrays()
    num_arrays = len(arrays)
    prefiller.buf.fill_(P_CANARY)
    prefiller.buf[prefiller.page_mask(list(prefiller.specs), [0])] = NULL_FILL
    tokens = _canonical_tokens(0, 1, P_BLOCK, num_p)
    held = (tokens < num_tokens)[..., None]
    for a, array in enumerate(arrays):
        valid = _row_tags(tokens, a, array[2])
        junk = _row_tags(tokens, a + num_arrays, array[2])
        prefiller.write_rows(array, p_ids, np.where(held, valid, junk))
    for g, block in zip(prefiller.kda_groups(), p_kda):
        prefiller.buf[prefiller.page_mask(prefiller.group_layers(g), [block])] = (
            KDA_FILL
        )
    for g, blocks in zip(prefiller.draft_groups, p_draft):
        for name in prefiller.group_layers(g):
            for block, q in zip(blocks, p_positions, strict=True):
                page = _draft_page(
                    name, range(DRAFT_KV_HEADS), q * p_draft_block, prefiller
                )
                prefiller.write_draft_heads(name, block, page)

    # D: the shared region [0, S) holds the null block, the KDA blocks, the
    # draft rows and a spare; each card's DSA region has D pages at local pages
    # [S, S + D).
    num_d_draft_rows = num_draft_groups * len(d_positions)
    shared = 1 + STATE_GROUPS + num_d_draft_rows + 1
    region = num_d + 2
    d_kda = [1 + i for i in range(STATE_GROUPS)]
    d_rows = list(range(1 + STATE_GROUPS, 1 + STATE_GROUPS + num_d_draft_rows))
    d_pages = (shared + rng.permutation(region)[:num_d]).tolist()
    d_ids = [page + owner * region for page in d_pages]
    decoders = _decoders(fabric, shared + region, region, draft)
    d_draft = draft_ids(decoders[0], d_rows, len(d_positions))
    for r, rank in enumerate(decoders):
        assert rank.draft_groups == prefiller.draft_groups
        rank.buf.fill_(CANARY)
        for g, block in zip(rank.kda_groups(), d_kda):
            rank.buf[rank.page_mask(rank.group_layers(g), [block])] = STALE
        for g, blocks in zip(rank.draft_groups, d_draft):
            rank.buf[rank.page_mask(rank.group_layers(g), blocks)] = STALE
        if r == owner:
            rank.buf[rank.page_mask(rank.dsa_layers, d_pages[num_cached:])] = STALE
            cached = _canonical_tokens(0, 1, D_BLOCK, num_cached)
            for a, array in enumerate(arrays):
                rank.write_rows(
                    array, d_pages[:num_cached], _row_tags(cached, a, array[2])
                )
        _handshake(rank, {0: prefiller}, remote_tp=1)

    rank0 = decoders[0]
    hit_in_window = d_first - window_first
    local = _by_group(rank0, d_ids[num_cached:], [[b] for b in d_kda], d_draft)
    computed = _by_group(
        rank0, num_cached, [0] * STATE_GROUPS, [hit_in_window] * num_draft_groups
    )
    remote = _by_group(rank0, p_ids, [[b] for b in p_kda], p_draft)
    for rank in decoders:
        rank.worker.start_load_kv(
            _recv_metadata("req", local, computed, remote, num_tokens, remote_tp=1)
        )
        results = rank.worker.get_transfer_results()
        assert results.finished_recving == {"req"} and not results.failed_recving

    d_heads = DRAFT_KV_HEADS // D_TP
    for r, rank in enumerate(decoders):
        pulled = rank.page_mask(
            [n for g in rank.kda_groups() for n in rank.group_layers(g)], d_kda
        )
        assert (rank.buf[pulled] == KDA_FILL).all(), f"rank {r}: KDA state"
        heads = range(r * d_heads, (r + 1) * d_heads)
        for g, blocks in zip(rank.draft_groups, d_draft):
            for name in rank.group_layers(g):
                for block, q in zip(blocks, d_positions, strict=True):
                    np.testing.assert_array_equal(
                        rank.draft_heads(name, block),
                        _draft_page(name, heads, q * d_draft_block, rank),
                        err_msg=f"rank {r} {name} draft position {q}",
                    )
            pulled |= rank.page_mask(rank.group_layers(g), blocks)
        if r == owner:
            tokens = _canonical_tokens(0, 1, D_BLOCK, num_d)
            held = tokens < num_tokens
            for a, array in enumerate(arrays):
                got = rank.read_rows(array, d_pages)
                valid = _row_tags(tokens, a, array[2])
                junk = _row_tags(tokens, a + num_arrays, array[2])
                np.testing.assert_array_equal(
                    got[held], valid[held], err_msg=f"rank {r} {array}"
                )
                past = got[~held]
                assert ((past == 0).all(-1) | (past == junk[~held]).all(-1)).all()
            pulled |= rank.page_mask(rank.dsa_layers, d_pages)
        assert (rank.buf[~pulled] == CANARY).all(), f"rank {r}: padding or blocks"
