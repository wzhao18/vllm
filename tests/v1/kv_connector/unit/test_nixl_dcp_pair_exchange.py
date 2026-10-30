# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCP pair exchange for a P DCP8 -> D DCP16 NIXL pull at interleave 1, on CPU.

The oracle never restates the interleave-1 mapping: expected tokens come from
the offloading connector's canonical mapping (the inverse of the V2 slot
mapping), and the pull placement from the worker's own block matching.
"""

import threading
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm.config import CacheConfig
from vllm.distributed.kv_transfer.kv_connector.utils import TransferTopology
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.dcp_pair_exchange import (
    NixlPairModeError,
    PairArrayKind,
    PairRegion,
    PairRowArray,
    PairUnit,
    build_pair_row_table,
    check_pair_coverage,
    pair_layout,
    pair_units,
    reference_pair_exchange,
    reference_read_pages,
    reference_zero_pages,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.page_zeroer import (
    unwritten_blocks,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    NixlPullConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    compute_tp_mapping,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.canonical_mapping import (  # noqa: E501
    _local_to_canonical_token,
    _RankContext,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.core.kv_cache_planning import get_kv_cache_config_from_groups
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SparseCacheRole,
    UniformTypeKVCacheSpecs,
    get_kv_quant_mode,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.utils import (
    AttentionGroup,
    allocate_kv_cache,
    customize_attention_spec,
)

pytestmark = pytest.mark.cpu_test

P_DCP, D_DCP = 8, 16
CANARY, STALE = 0x5A, 0x3C
# Side handle of P rank p's agent is P_HANDLE + p, so reads name their P rank.
P_HANDLE = 100


@pytest.fixture(autouse=True, scope="module")
def _one_thread():
    """Thousands of small tensor ops: keep them off OpenMP, whose fork/join
    cost dominates them on a loaded host."""
    num_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(num_threads)


def _canonical_tokens(
    rank: int, dcp_size: int, block_size: int, num_blocks: int
) -> np.ndarray:
    """``[num_blocks, block_size]`` token each row of a rank's first blocks
    holds at interleave 1."""
    ctx = _RankContext(
        tp_size=dcp_size,
        dcp_size=dcp_size,
        pcp_size=1,
        interleave=1,
        total_kv_heads=1,
        rank=rank,
    )
    local = np.arange(num_blocks * block_size, dtype=np.int64)
    return _local_to_canonical_token(local, ctx).reshape(num_blocks, block_size)


def _block_tables(rng, num_d: int, num_p: int) -> tuple[list[int], list[int]]:
    """Shuffled D and P block tables drawn from pools with spare blocks."""
    return (
        rng.permutation(num_d + 2)[:num_d].tolist(),
        rng.permutation(num_p + 2)[:num_p].tolist(),
    )


def _pull_as_today(d_ranks, d_ids, num_cached, p_ranks, p_ids) -> None:
    """Every D rank reads whole P pages as the worker matches them, then zeroes
    the pages its reads leave unwritten."""
    worker = object.__new__(NixlPullConnectorWorker)
    local = d_ids[num_cached:]
    for r, d in enumerate(d_ranks):
        p = p_ranks[r % P_DCP]
        dst, src = worker._apply_dcp_prefix_caching(
            local,
            p_ids,
            remote_rank=r % P_DCP,
            local_dcp_size=D_DCP,
            local_dcp_rank=r,
            remote_dcp_size=P_DCP,
            local_num_computed_blocks=num_cached,
        )
        reference_read_pages(d.buf, d.table, dst, p.buf, p.table, src)
        reference_zero_pages(d.buf, d.table, unwritten_blocks(local, set(dst)))


def _exchange(d_ranks, d_ids, num_cached, p_ids) -> list[PairUnit]:
    units = pair_units([d_ids[num_cached:]], [num_cached], [p_ids], groups=[0])
    reference_pair_exchange(
        [d.buf for d in d_ranks], [{0: d.table} for d in d_ranks], units
    )
    return units


# ----------------------------------------------------------------------------
# A1: token-id oracle
# ----------------------------------------------------------------------------


class _TokenRank:
    """A rank's KV storage holding one 8-byte token-id row per token, in a page
    set between padding bytes of each block."""

    def __init__(self, block_size: int, num_blocks: int, lead: int, tail: int):
        self.block_size, self.lead = block_size, lead
        self.stride = lead + 8 * block_size + tail
        self.buf = torch.full((num_blocks * self.stride,), CANARY, dtype=torch.uint8)
        spec = MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.uint8,
            cache_dtype_str="fp8",
        )
        page = PairRegion(
            "tokens", spec, self.buf.data_ptr() + lead, 8 * block_size, self.stride
        )
        self.table = build_pair_row_table([page])

    def _pages(self) -> torch.Tensor:
        rows = self.buf.view(-1, self.stride)
        return rows[:, self.lead : self.lead + 8 * self.block_size]

    def read(self, block_ids: list[int]) -> np.ndarray:
        return self._pages()[block_ids].contiguous().view(torch.int64).numpy()

    def write(self, block_ids: list[int], tokens: np.ndarray) -> None:
        if block_ids:
            rows = torch.from_numpy(np.ascontiguousarray(tokens, dtype=np.int64))
            self._pages()[block_ids] = rows.view(torch.uint8)

    def untouched_outside(self, block_ids: list[int]) -> bool:
        """Whether every byte but the pages of ``block_ids`` is still canary."""
        rows = self.buf.view(-1, self.stride)
        outside = torch.ones_like(rows, dtype=torch.bool)
        outside[block_ids, self.lead : self.lead + 8 * self.block_size] = False
        return bool((rows[outside] == CANARY).all())


def _check_token_oracle(
    block_size: int,
    num_tokens: int,
    num_cached: int,
    extra_d: int,
    extra_p: int,
    junk_tail: bool,
) -> None:
    """Pull plus exchange of one request; ``extra_d``/``extra_p`` blocks past
    the ones holding tokens below ``num_tokens`` (lookahead), and P rows past
    ``num_tokens`` zero or junk (drafter rows)."""
    case = (
        f"B={block_size} L={num_tokens} m={num_cached} extra_d={extra_d} "
        f"extra_p={extra_p} junk_tail={junk_tail}"
    )
    num_d = cdiv(num_tokens, D_DCP * block_size) + extra_d
    num_p = cdiv(num_tokens, P_DCP * block_size) + extra_p
    d_ids, p_ids = _block_tables(np.random.default_rng(num_tokens), num_d, num_p)

    p_ranks = [_TokenRank(block_size, num_p + 2, 8, 40) for _ in range(P_DCP)]
    for p, rank in enumerate(p_ranks):
        tokens = _canonical_tokens(p, P_DCP, block_size, num_p)
        past = -(tokens + 1) if junk_tail else 0
        rank.write(p_ids, np.where(tokens < num_tokens, tokens + 1, past))
    d_ranks = [_TokenRank(block_size, num_d + 2, 16, 8) for _ in range(D_DCP)]
    stale = np.full((num_d - num_cached, block_size), STALE * 0x0101010101010101)
    for r, rank in enumerate(d_ranks):
        # D's prefix cache holds its own rows; the other blocks a stale owner's.
        cached = _canonical_tokens(r, D_DCP, block_size, num_cached)
        rank.write(d_ids[:num_cached], cached + 1)
        rank.write(d_ids[num_cached:], stale)

    _pull_as_today(d_ranks, d_ids, num_cached, p_ranks, p_ids)
    _exchange(d_ranks, d_ids, num_cached, p_ids)

    for r, rank in enumerate(d_ranks):
        tokens = _canonical_tokens(r, D_DCP, block_size, num_d)
        got = rank.read(d_ids)
        held = tokens < num_tokens
        np.testing.assert_array_equal(
            got[held], tokens[held] + 1, err_msg=f"{case} rank={r}"
        )
        past = got[~held]
        assert ((past == 0) | (junk_tail & (past == -(tokens[~held] + 1)))).all(), (
            f"{case} rank={r}: a row past L holds another token or stale bytes"
        )
        assert rank.untouched_outside(d_ids), f"{case} rank={r}"


@pytest.mark.parametrize("block_size", [64, 1024])
@pytest.mark.parametrize(
    "num_tokens",
    [
        *(1, 15, 16, 17),
        *(8191, 8192, 8193, 16383, 16384, 16385, 24577, 3 * 16384 + 5),
        *(200_000, 2**20 - 1),
    ],
)
def test_token_oracle(block_size: int, num_tokens: int):
    """A1: every D row holds its interleave-1 token, D's cached blocks are kept,
    rows past L are zero (or junk P wrote past L for that same token), and no
    byte outside the request's pages changes."""
    num_d_blocks = cdiv(num_tokens, D_DCP * block_size)
    for num_cached in sorted({0, 1, 2, num_d_blocks - 1} & set(range(num_d_blocks))):
        for extra_d, extra_p in ((0, 0), (1, 0), (0, 1), (1, 1)):
            _check_token_oracle(
                block_size, num_tokens, num_cached, extra_d, extra_p, junk_tail=True
            )
        _check_token_oracle(block_size, num_tokens, num_cached, 0, 0, junk_tail=False)


def test_lone_unit_zero_fills_its_second_half():
    """A unit without P block 2j+1 zeroes rows [B/2, B) on both partners, even
    when the parity-1 rank's block still holds a stale owner's bytes."""
    block_size = 8
    ranks = [_TokenRank(block_size, 2, lead=0, tail=0) for _ in range(2)]
    ranks[0].write([1], np.arange(block_size)[None] + 1)
    ranks[1].write([1], np.full((1, block_size), -7))
    reference_pair_exchange(
        [r.buf for r in ranks], [{0: r.table} for r in ranks], [PairUnit(0, 1, False)]
    )
    half = block_size // 2
    for c, rank in enumerate(ranks):
        expected = [*range(1 + c, block_size + 1, 2), *[0] * half]
        assert rank.read([1])[0].tolist() == expected
        assert rank.untouched_outside([1])


# ----------------------------------------------------------------------------
# A2: the plan matches the blocks the worker pulls
# ----------------------------------------------------------------------------


def _mla_pull_worker(dcp_rank: int, block_size: int, num_d: int, num_p: int):
    """TP16/DCP16 decoder rank with two MLA groups pulling from a TP8/DCP8
    prefiller. Real block matching and descriptors; the transport is mocked."""
    w = object.__new__(NixlPullConnectorWorker)
    w.engine_id, w.tp_rank = "D", dcp_rank
    w.block_size = block_size
    w.use_mla, w._has_mamba = True, False
    w.dcp_size, w.dcp_rank = D_DCP, dcp_rank
    w._physical_blocks_per_logical_kv_block = 1
    w._bidirectional_kv_xfer_enabled = False
    w._engine_last_active = {}
    w._handshake_lock = threading.RLock()
    w._handshake_futures = {}
    w._recving_transfers = {}
    w._invalid_remote_engines = set()
    w._remote_registration_generations = {}
    w._transfer_layer_group_ids = ()
    spec = MLAAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=8, dtype=torch.uint8
    )
    w.kv_cache_config = KVCacheConfig(
        num_blocks=num_d,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["a"], spec), KVCacheGroupSpec(["b"], spec)],
    )
    w._group_spec_types = (MLAAttentionSpec, MLAAttentionSpec)
    w.region_group_ids = [0, 1]
    w.dst_region_group_ids = {"P": [0, 1]}
    w.num_regions = 2
    w.block_len_per_layer = [8 * block_size] * 2
    w.transfer_topo = MagicMock()
    w.transfer_topo.tp_ratio.return_value = D_DCP // P_DCP
    w.transfer_topo.block_size_ratio.return_value = 1
    w.transfer_topo.get_engine_info.return_value = SimpleNamespace(
        remote_tp_size=P_DCP,
        remote_dcp_size=P_DCP,
        remote_block_size=block_size,
        remote_physical_blocks_per_logical=1,
    )
    topo = object.__new__(TransferTopology)
    topo.tp_rank, topo.tp_size, topo.dcp_size = dcp_rank, D_DCP, D_DCP
    topo.is_mla, topo.is_mamba, topo.total_num_kv_heads = True, False, 1
    w.tp_mappings = {
        "P": compute_tp_mapping(
            transfer_topology=topo,
            remote_tp_size=P_DCP,
            group_spec_types=w._group_spec_types,
            remote_dcp_size=P_DCP,
        )
    }
    w._mixed_mem_types = True
    w.src_xfer_handles_by_block_size = {block_size: 1}
    w._dram_src_handles_by_block_size = {block_size: 2}
    w.dst_xfer_side_handles = {"P": {p: P_HANDLE + p for p in range(P_DCP)}}
    w.dst_num_blocks = {"P": num_p, "D": num_d}
    w.dst_region_num_blocks = {"P": [num_p] * 2, "D": [num_d] * 2}
    w.dst_uses_region_group_mapping = {"P": True}
    w._uses_region_group_mapping = True
    w._remote_agents = {"P": {(0, r): f"P{r}" for r in range(P_DCP)}}
    w._read_blocks_mixed = MagicMock()
    w._queue_unwritten_zeroing = MagicMock()
    w.nixl_wrapper = MagicMock()
    w.use_host_buffer = False
    return w


def _posted_reads(
    dcp_rank, block_size, local_ids, num_cached, remote_ids, num_tokens, pools
) -> tuple[dict[tuple[int, int], tuple[int, int]], list[set[int]]]:
    """``{(group, D block): (P rank, P block)}`` the rank's real read path
    posts, and the blocks it reports written per group."""
    num_d, num_p = pools
    w = _mla_pull_worker(dcp_rank, block_size, num_d, num_p)
    metadata = NixlConnectorMetadata()
    metadata.add_new_req_to_recv(
        request_id="r",
        local_block_ids=local_ids,
        local_num_computed_blocks=tuple(num_cached),
        awaiting_kvs=True,
        kv_transfer_params={
            "remote_engine_id": "P",
            "remote_request_id": "r-P",
            "remote_host": "localhost",
            "remote_port": 1,
            "remote_num_tokens": num_tokens,
            "remote_block_ids": remote_ids,
        },
    )
    meta = metadata.reqs_to_recv["r"]
    meta.local_physical_block_ids = local_ids
    w._recving_metadata = {"r": meta}
    w._read_blocks_for_req("r", meta)
    reads = {}
    for call in w._read_blocks_mixed.call_args_list:
        p_rank = call.kwargs["remote_xfer_side_handle"] - P_HANDLE
        for local, remote in zip(
            call.kwargs["local_block_descs_ids"], call.kwargs["remote_block_descs_ids"]
        ):
            group, block = divmod(int(local), num_d)
            remote_group, remote_block = divmod(int(remote), num_p)
            assert group == remote_group and (group, block) not in reads
            reads[(group, block)] = (p_rank, remote_block)
    return reads, w._queue_unwritten_zeroing.call_args.args[2]


def _check_plan_matches_worker_pulls(
    num_tokens: int, m0: int, extras: tuple[tuple[int, int], ...]
) -> None:
    block_size = 1024
    num_d = cdiv(num_tokens, D_DCP * block_size)
    num_p = cdiv(num_tokens, P_DCP * block_size)
    num_cached = (m0, 0 if m0 else min(1, num_d - 1))
    rng = np.random.default_rng(num_tokens)
    pools = (2 * num_d + 4, 2 * num_p + 4)
    d_pool, p_pool = rng.permutation(pools[0]), rng.permutation(pools[1])
    # Per group, (D, P) lookahead blocks past the ones holding the tokens.
    (d0, p0), (d1, p1) = extras
    d_tables = [
        d_pool[: num_d + d0].tolist(),
        d_pool[num_d + d0 : 2 * num_d + d0 + d1].tolist(),
    ]
    p_tables = [
        p_pool[: num_p + p0].tolist(),
        p_pool[num_p + p0 : 2 * num_p + p0 + p1].tolist(),
    ]
    local_ids = [d_tables[g][num_cached[g] :] for g in (0, 1)]
    units = pair_units(local_ids, num_cached, p_tables, groups=(0, 1))

    for r in range(D_DCP):
        c = r // P_DCP
        reads, written = _posted_reads(
            r, block_size, local_ids, num_cached, p_tables, num_tokens, pools
        )
        for g in (0, 1):
            case = f"L={num_tokens} m={num_cached} extras={extras} rank={r} group={g}"
            group_units = [u for u in units if u.group == g]
            assert all(u.has_partner for u in group_units[:-1]), case
            # P rank r % W holds residue r's tokens.
            expected = {
                u.block_id: (r % P_DCP, p_tables[g][c + 2 * (num_cached[g] + i)])
                for i, u in enumerate(group_units)
                if c == 0 or u.has_partner
            }
            group_reads = {b: p for (gg, b), p in reads.items() if gg == g}
            assert group_reads == expected, case
            assert written[g] == set(expected), case
            assert len(expected) == len(group_units) - (
                c == 1 and bool(group_units) and not group_units[-1].has_partner
            ), case


@pytest.mark.parametrize(
    "extras",
    [((0, 0), (1, 1)), ((0, 1), (1, 0))],
    ids=["exact,both_lookahead", "p_lookahead,d_lookahead"],
)
@pytest.mark.parametrize("num_tokens", [1, 8192, 8193, 24577, 3 * 16384 + 5, 200_000])
def test_plan_matches_worker_pulls(num_tokens: int, extras):
    """A2: the parity-0 rank pulls exactly the plan's units, the parity-1 rank
    the same or all but a last unit without partner, each from P rank r % 8 and
    P block 2(m + i) + c. Groups differ in hit and list lengths, so a plan that
    mixed up group indices would disagree. A P lookahead block with an even P
    block count makes D's list the limit: n_R - 2m is odd, yet the parity-1
    rank pulls every unit."""
    num_d_blocks = cdiv(num_tokens, D_DCP * 1024)
    for m0 in sorted({0, 1, num_d_blocks - 1} & set(range(num_d_blocks))):
        _check_plan_matches_worker_pulls(num_tokens, m0, extras)


def test_pair_units_cover_only_the_given_groups():
    units = pair_units([[5, 6], [7]], [0, 0], [[1, 2, 3], [4]], groups=[1])
    assert units == [PairUnit(1, 7, False)]
    # P holds nothing past D's prefix cache: nothing to exchange.
    assert pair_units([[5]], [2], [[1, 2, 3, 4]], groups=[0]) == []


def test_coverage_check_rejects_a_short_prefiller():
    remote = [[0, 1, 2], [3, 4, 5]]
    check_pair_coverage(3 * 8192, 3 * 8192, remote, (0, 1), remote_block_tokens=8192)
    for num_tokens, remote_num_tokens, remote_ids in (
        (3 * 8192 + 1, 3 * 8192 + 1, remote),  # a block short
        (3 * 8192, 3 * 8192 - 1, remote),  # a token short
        (3 * 8192, 3 * 8192, [[0, 1, 2], [3, 4]]),  # one group a block short
        (3 * 8192, None, remote),  # P did not report its tokens
    ):
        with pytest.raises(NixlPairModeError):
            check_pair_coverage(num_tokens, remote_num_tokens, remote_ids, (0, 1), 8192)


@pytest.mark.parametrize(
    "dcp_size,remote_dcp_size,block_size,interleave",
    [(16, 8, 1024, 1024), (16, 8, 1024, 64), (16, 16, 1024, 1), (16, 4, 1024, 1)],
)
def test_pair_layout_rejects_other_modes(
    dcp_size, remote_dcp_size, block_size, interleave
):
    with pytest.raises(NixlPairModeError):
        pair_layout(0, dcp_size, remote_dcp_size, block_size, interleave)


# ----------------------------------------------------------------------------
# Row tables on the real planner's packed Kimi-K3.1 geometry
# ----------------------------------------------------------------------------

# Kimi-K3.1 at fp8: 576-byte latent rows (kv_lora_rank 512 + rope 64) and
# 132-byte indexer rows (index_head_dim 128 fp8 values + one fp32 scale), the
# indexer read as 64-row kernel pages that store their values, then scales.
LATENT_BYTES, INDEX_ROW_BYTES, INDEX_VALUE_BYTES, INDEX_KERNEL_ROWS = 576, 132, 128, 64


def _k31_dsa_specs(block_size: int, num_dsa: int, num_draft: int) -> dict:
    """The packed DSA group as K3.1's layers declare it and the backends
    customize it: per DSA layer a latent and an indexer page, plus the DSpark
    drafter's latent pages."""
    latent = customize_attention_spec(
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE.get_class(),
        MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            max_tp_shards=1,
            head_size=LATENT_BYTES,
            dtype=torch.uint8,
            cache_dtype_str="fp8",
            kv_quant_mode=get_kv_quant_mode("fp8"),
        ),
    )
    index = customize_attention_spec(
        DeepseekV32IndexerBackend,
        MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            max_tp_shards=1,
            head_size=INDEX_ROW_BYTES,
            dtype=torch.uint8,
            cache_role=SparseCacheRole.INDEXER,
        ),
    )
    specs = {}
    for i in range(num_dsa):
        specs[f"model.layers.{i}.self_attn.attn"] = latent
        specs[f"model.layers.{i}.self_attn.indexer.k_cache"] = index
    for i in range(num_draft):
        specs[f"draft.layers.{i}.self_attn.attn"] = replace(latent, is_draft=True)
    return specs


class _PlannedRank:
    """A rank's packed KV cache as the real planner lays it out and the worker
    allocates it, with the NIXL regions the worker registers for it and the
    attention backends' own views of its rows."""

    def __init__(self, specs: dict, num_blocks: int):
        self.specs = specs
        self.block_size = next(iter(specs.values())).block_size
        cache_config = CacheConfig(block_size=self.block_size)
        cache_config.kv_cache_layout = "BLHNC"
        cache_config.num_gpu_blocks_override = num_blocks
        config = SimpleNamespace(
            cache_config=cache_config,
            attention_config=SimpleNamespace(hisparse_config=None),
        )
        group = KVCacheGroupSpec(
            list(specs), UniformTypeKVCacheSpecs(self.block_size, specs)
        )
        self.kv_cache_config = get_kv_cache_config_from_groups(config, [group], 1)
        self.views = allocate_kv_cache(
            self.kv_cache_config, torch.device("cpu"), KVCacheLayout.BLHNC
        )
        storage = next(iter(self.views.values())).untyped_storage()
        self.buf = torch.empty(0, dtype=torch.uint8).set_(storage)
        # As register_kv_caches registers block-major layer views.
        regions = [
            PairRegion(
                name,
                specs[name],
                view.data_ptr(),
                specs[name].page_size_bytes,
                view.stride(0) * view.element_size(),
            )
            for name, view in self.views.items()
        ]
        self.table = build_pair_row_table(
            regions,
            index_kernel_rows=INDEX_KERNEL_ROWS,
            index_value_bytes=INDEX_VALUE_BYTES,
        )

    def _kernel_pages(self, name: str, block_ids: list[int]):
        """The indexer's kernel-page view and the ids of the blocks' pages."""
        group = AttentionGroup(
            backend=DeepseekV32IndexerBackend,
            layer_names=[name],
            kv_cache_spec=self.specs[name],
            kv_cache_group_id=0,
            kernel_block_size=INDEX_KERNEL_ROWS,
            maps_slot_mapping=True,
        )
        pages = group.map_kv_cache_to_kernel_blocks(self.views[name])
        per_block = self.block_size // INDEX_KERNEL_ROWS
        # As map_to_kernel_block_table: no stride when a block is one page.
        stride = group.kernel_block_stride or per_block
        ids = np.asarray(block_ids)[:, None] * stride
        return pages, torch.as_tensor((ids + np.arange(per_block)).ravel())

    @staticmethod
    def _index_part(kind: PairArrayKind) -> tuple[int, int]:
        """Offset in a kernel page, and row width, of the indexer's values or
        scales: its cache kernel stores a page's values, then their scales."""
        if kind is PairArrayKind.INDEX_VALUE:
            return 0, INDEX_VALUE_BYTES
        return (
            INDEX_KERNEL_ROWS * INDEX_VALUE_BYTES,
            INDEX_ROW_BYTES - INDEX_VALUE_BYTES,
        )

    def row_addresses(self, array: PairRowArray, block_id: int) -> np.ndarray:
        rows = np.arange(self.block_size)
        if array.kind is PairArrayKind.ROW:
            view = self.views[array.layer_name]
            row_stride = view.stride(2) * view.element_size()
            return view[block_id, 0].data_ptr() + rows * row_stride
        pages, ids = self._kernel_pages(array.layer_name, [block_id])
        starts = np.array([pages[i].data_ptr() for i in ids.tolist()])
        offset, width = self._index_part(array.kind)
        subpage, row = np.divmod(rows, INDEX_KERNEL_ROWS)
        return starts[subpage] + offset + row * width

    def read_rows(self, array: PairRowArray, block_ids: list[int]) -> np.ndarray:
        """``[n, B, w]`` rows of one array, read through the backends' views."""
        if array.kind is PairArrayKind.ROW:
            return self.views[array.layer_name][block_ids, 0].numpy()
        pages, ids = self._kernel_pages(array.layer_name, block_ids)
        offset, width = self._index_part(array.kind)
        flat = pages[ids].reshape(len(ids), -1)
        part = flat[:, offset : offset + INDEX_KERNEL_ROWS * width]
        return part.reshape(len(block_ids), self.block_size, width).numpy()

    def write_rows(
        self, array: PairRowArray, block_ids: list[int], rows: np.ndarray
    ) -> None:
        if not block_ids:
            return
        if array.kind is PairArrayKind.ROW:
            self.views[array.layer_name][block_ids, 0] = torch.from_numpy(rows)
            return
        pages, ids = self._kernel_pages(array.layer_name, block_ids)
        offset, width = self._index_part(array.kind)
        flat = pages[ids].reshape(len(ids), -1)
        part = flat[:, offset : offset + INDEX_KERNEL_ROWS * width]
        part[:] = torch.from_numpy(rows).reshape(part.shape)
        pages[ids] = flat.reshape(-1, 1, INDEX_KERNEL_ROWS, INDEX_ROW_BYTES)

    def page_mask(self, block_ids: list[int]) -> torch.Tensor:
        """Bytes of the blocks' attention pages, from the backends' views."""
        mask = torch.zeros_like(self.buf, dtype=torch.bool)
        for name, view in self.views.items():
            for b in block_ids:
                start = view[b].data_ptr() - self.buf.data_ptr()
                mask[start : start + self.specs[name].page_size_bytes] = True
        return mask


@pytest.mark.parametrize("num_draft", [0, 5])
def test_row_table_matches_k31_planner_geometry(num_draft: int):
    """Every row the table addresses is the row the backends' views read, and a
    block's arrays tile its attention pages exactly."""
    rank = _PlannedRank(_k31_dsa_specs(1024, 24, num_draft), num_blocks=2)
    table = rank.table
    assert table.block_stride == rank.kv_cache_config.kv_cache_tensors[0].block_stride
    # Design §3.5: H is 8,699,904 B for 24 DSA layers plus 294,912 B per draft.
    assert table.half_bytes == 8_699_904 + num_draft * 294_912
    assert Counter(a.kind for a in table.arrays) == {
        PairArrayKind.ROW: 24 + num_draft,
        PairArrayKind.INDEX_VALUE: 24,
        PairArrayKind.INDEX_SCALE: 24,
    }
    rows = np.arange(1024)
    page_bytes: dict[str, list[tuple[int, int]]] = {}
    for array in table.arrays:
        for b in (0, 1):
            np.testing.assert_array_equal(
                table.base_addr + table.row_offsets(array, [b], rows)[0],
                rank.row_addresses(array, b),
                err_msg=array.layer_name,
            )
        starts = table.base_addr + table.row_offsets(array, [1], rows)[0]
        page_bytes.setdefault(array.layer_name, []).extend(
            (start, start + array.width) for start in starts.tolist()
        )
        # int64 addressing past 2**31 bytes.
        far = 2**31 // table.block_stride + 7
        delta = table.row_offsets(array, [far], rows) - table.row_offsets(
            array, [0], rows
        )
        assert (delta == far * table.block_stride).all()
    # Block 1's arrays tile each of its pages, so the exchange moves every page
    # byte and nothing else.
    for name, intervals in page_bytes.items():
        intervals.sort()
        start = rank.views[name][1].data_ptr()
        ends = [start, *(end for _, end in intervals)]
        assert [s for s, _ in intervals] == ends[:-1], name
        assert ends[-1] == start + rank.specs[name].page_size_bytes, name


def _row_tags(tokens: np.ndarray, salt: int, width: int) -> np.ndarray:
    """``[..., width]`` bytes for each token's row in array ``salt``: unique per
    (token, array) in the first four bytes, and never all zero."""
    t = tokens.astype(np.uint32)[..., None]
    head = ((t + 1) | np.uint32(salt << 24)).view(np.uint8)
    k = np.arange(width - 4, dtype=np.uint32)
    tail = (t * np.uint32(2654435761) + k * np.uint32(40503) + np.uint32(salt)) >> 13
    return np.concatenate([head, tail.astype(np.uint8)], axis=-1)


@pytest.mark.parametrize(
    "block_size,num_tokens,num_cached",
    [(64, 1500, 0), (64, 2600, 1), (128, 5000, 0), (128, 6000, 1)],
)
def test_exchange_on_planner_geometry(
    block_size: int, num_tokens: int, num_cached: int
):
    """A1 on the real packed geometry, scaled to blocks of one indexer kernel
    page (which the halves split) or two (which the halves align with): every
    array's rows land on their interleave-1 token, and no byte outside the
    request's attention pages changes."""
    specs = _k31_dsa_specs(block_size, num_dsa=2, num_draft=1)
    num_d = cdiv(num_tokens, D_DCP * block_size)
    num_p = cdiv(num_tokens, P_DCP * block_size)
    d_ids, p_ids = _block_tables(np.random.default_rng(num_tokens), num_d, num_p)
    p_ranks = [_PlannedRank(specs, num_p + 2) for _ in range(P_DCP)]
    arrays = p_ranks[0].table.arrays
    num_arrays = len(arrays)
    for p, rank in enumerate(p_ranks):
        tokens = _canonical_tokens(p, P_DCP, block_size, num_p)
        held = (tokens < num_tokens)[..., None]
        for a, array in enumerate(rank.table.arrays):
            valid = _row_tags(tokens, a, array.width)
            junk = _row_tags(tokens, a + num_arrays, array.width)
            rank.write_rows(array, p_ids, np.where(held, valid, junk))
    d_ranks = [_PlannedRank(specs, num_d + 2) for _ in range(D_DCP)]
    snapshots = []
    for r, rank in enumerate(d_ranks):
        rank.buf.fill_(STALE)
        tokens = _canonical_tokens(r, D_DCP, block_size, num_cached)
        for a, array in enumerate(rank.table.arrays):
            rank.write_rows(
                array, d_ids[:num_cached], _row_tags(tokens, a, array.width)
            )
        snapshots.append(rank.buf.clone())

    _pull_as_today(d_ranks, d_ids, num_cached, p_ranks, p_ids)
    units = _exchange(d_ranks, d_ids, num_cached, p_ids)
    # One case ends on a lone unit, the other on a pair.
    assert units[-1].has_partner == ((num_p - 2 * num_cached) % 2 == 0)

    for r, rank in enumerate(d_ranks):
        tokens = _canonical_tokens(r, D_DCP, block_size, num_d)
        held = tokens < num_tokens
        for a, array in enumerate(arrays):
            got = rank.read_rows(array, d_ids)
            valid = _row_tags(tokens, a, array.width)
            junk = _row_tags(tokens, a + num_arrays, array.width)
            np.testing.assert_array_equal(got[held], valid[held], err_msg=f"{r} {a}")
            past = got[~held]
            assert ((past == 0).all(-1) | (past == junk[~held]).all(-1)).all()
        outside = ~rank.page_mask(d_ids)
        assert torch.equal(rank.buf[outside], snapshots[r][outside]), f"rank {r}"


# ----------------------------------------------------------------------------
# Row table whitelist
# ----------------------------------------------------------------------------


def _region(name: str, spec, base: int = 0, stride: int = 1 << 20) -> PairRegion:
    return PairRegion(name, spec, base, spec.page_size_bytes, stride)


def test_row_table_accepts_fp8_ds_mla_rows():
    spec = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.uint8,
        cache_dtype_str="fp8_ds_mla",
        state_content_bytes=656,
    )
    (array,) = build_pair_row_table([_region("L0", spec)]).arrays
    assert (array.kind, array.width, array.rows_per_subpage) == (
        PairArrayKind.ROW,
        656,
        64,
    )


def _latent(**kwargs) -> MLAAttentionSpec:
    fields = dict(
        block_size=64,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.uint8,
        cache_dtype_str="fp8",
    )
    return MLAAttentionSpec(**{**fields, **kwargs})


_INDEXER = MLAAttentionSpec(
    block_size=64,
    num_kv_heads=1,
    head_size=132,
    dtype=torch.uint8,
    cache_role=SparseCacheRole.INDEXER,
)
_GQA = FullAttentionSpec(block_size=64, num_kv_heads=1, head_size=64, dtype=torch.half)


@pytest.mark.parametrize(
    "regions,index_format",
    [
        # Not MLA, bf16 latent rows, two KV heads, a replicated page, a padded page.
        ([_region("L0", _GQA)], None),
        ([_region("L0", _latent(dtype=torch.bfloat16, cache_dtype_str="auto"))], None),
        ([_region("L0", _latent(num_kv_heads=2))], None),
        ([_region("L0", _latent(dcp_sharded=False))], None),
        ([_region("L0", _latent(page_size_padded=64 * 576 + 64))], None),
        # Compressed states.
        ([_region("L0", _latent(tokens_per_state=4))], None),
        # An indexer without its page format, rows that split no kernel page,
        # or rows that are not bytes.
        ([_region("L0", _INDEXER)], None),
        ([_region("L0", _INDEXER)], (48, 128)),
        ([_region("L0", _INDEXER)], (64, 132)),
        ([_region("L0", replace(_INDEXER, dtype=torch.bfloat16))], (64, 128)),
        # Overlapping pages, unequal strides or block sizes, a block wider than
        # its stride.
        ([_region("L0", _latent()), _region("L1", _latent(), base=64)], None),
        ([_region("L0", _latent()), _region("L1", _latent(), 1 << 19, 1 << 21)], None),
        (
            [_region("L0", _latent()), _region("L1", _latent(block_size=128), 1 << 19)],
            None,
        ),
        ([_region("L0", _latent(), stride=64 * 575)], None),
    ],
)
def test_row_table_rejects_unlisted_pages(regions, index_format):
    rows, values = index_format or (None, None)
    with pytest.raises(NixlPairModeError):
        build_pair_row_table(regions, index_kernel_rows=rows, index_value_bytes=values)
