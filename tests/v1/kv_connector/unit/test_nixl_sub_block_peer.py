# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sub-block peers: a DCP pair-mode decoder reading a prefiller whose per-rank
block is X times its own (DEP16 TP1 prefill, 16384-token packed blocks, into
TP4/DCP4 decode, 4096-token blocks: X = 4)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import SubBlockPeer
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.dcp_pair_exchange import (
    expand_sub_block_ids,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    NixlPullConnectorWorker,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import MambaSpec, MLAAttentionSpec


@pytest.mark.parametrize(
    "block_ids,num_tokens,expected",
    [
        # Full blocks, last one partly used: drop the sub-blocks past the end.
        ([7, 3], 5 * 8 + 1, [28, 29, 30, 31, 12, 13]),
        # Exactly on a sub-block boundary.
        ([7, 3], 6 * 8, [28, 29, 30, 31, 12, 13]),
        # One token.
        ([5], 1, [20]),
        # Lookahead blocks past the computed tokens are dropped first.
        ([7, 3, 9], 8, [28]),
        # A sliding window keeps the tail blocks only: they end at the last
        # token's block, so trimming works from the end.
        ([3], 6 * 8, [12, 13]),
    ],
)
def test_expand_sub_block_ids(block_ids, num_tokens, expected):
    assert expand_sub_block_ids(block_ids, 4, num_tokens, 8) == expected


def _decoder(block=8, dcp=4, tp=4, regions=3, ssm=(0,)):
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
        NixlConnectorWorker,
    )

    w = object.__new__(NixlConnectorWorker)
    w._sub_block_peers = {}
    w.block_size = block
    w._physical_blocks_per_logical_kv_block = 1
    w.dcp_size, w.pcp_size, w.world_size = dcp, 1, tp
    w._TRANSFER_MODE = "pull"
    w._dcp_pair_exchange = True
    w.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1)
    )
    w._transfer_layer_region_indices = ()
    w.use_host_buffer = False
    w._mixed_mem_types = False
    w.region_names = [f"layer{i}" for i in range(regions)]
    w._ssm_region_indices = list(ssm)
    w._sub_block_peers = {}
    w._region_is_mla = [i not in ssm for i in range(regions)]
    w._group_spec_types = (MambaSpec, MLAAttentionSpec)
    w.transfer_topo = SimpleNamespace(tp_size=tp)
    return w


def _prefiller_meta(block=32, regions=3, page=128, stride=400, num_blocks=5):
    return SimpleNamespace(
        engine_id="P",
        block_size=block,
        physical_blocks_per_logical_kv_block=1,
        pcp_size=1,
        dcp_size=1,
        cp_kv_cache_interleave_size=1,
        region_names=[f"layer{i}" for i in range(regions)],
        block_lens=[96] + [page] * (regions - 1),
        block_strides=[stride] * regions,
        region_num_blocks=None,
        num_blocks=num_blocks,
        kv_caches_base_addr=[1000 + 100 * i for i in range(regions)],
        device_id=0,
    )


def test_sub_block_remote_meta_rewrites_attention_regions_only():
    w = _decoder()
    w.block_len_per_layer = [24, 32, 32]
    meta = _prefiller_meta()
    assert w._sub_block_remote_meta(meta, remote_dcp_size=1) == 4
    assert meta.block_size == 8
    # The KDA state region keeps its page; attention pages split in 4.
    assert meta.block_lens == [96, 32, 32]
    assert meta.region_num_blocks == [5, 20, 20]
    assert meta.num_blocks == 5


@pytest.mark.parametrize(
    "change",
    [
        {"_dcp_pair_exchange": False},
        {"_TRANSFER_MODE": "push"},
        {"dcp_size": 1},
        {"_physical_blocks_per_logical_kv_block": 2},
        {"region_names": ["x", "y", "z"]},
    ],
)
def test_sub_block_remote_meta_needs_pair_mode(change):
    w = _decoder()
    w.block_len_per_layer = [24, 32, 32]
    for name, value in change.items():
        setattr(w, name, value)
    meta = _prefiller_meta()
    assert w._sub_block_remote_meta(meta, remote_dcp_size=1) == 1
    assert meta.block_size == 32 and meta.block_lens == [96, 128, 128]


def test_sub_block_remote_meta_equal_blocks_unchanged():
    w = _decoder(block=32)
    w.block_len_per_layer = [96, 128, 128]
    assert w._sub_block_remote_meta(_prefiller_meta(), remote_dcp_size=1) == 1


def _expand(descs: np.ndarray) -> list[tuple[int, int]]:
    """(addr, len) of every block a Nx5 strided desc array describes."""
    return [
        (int(addr) + k * int(stride), int(length))
        for addr, length, _, stride, count in descs
        for k in range(int(count))
    ]


# Token oracle: P (TP1, DCP1, 32-token blocks) -> D (TP4, DCP4, interleave 1,
# 8-token per-rank blocks, virtual block 32 = P block). Two DSA regions share a
# packed row (offsets 0 and PAGE_P) with padding, as in the Kimi K3 layout.
X, B_D, ROW = 4, 8, 4  # sub-blocks, D tokens per rank-block, bytes per token row
B_P = X * B_D
PAGE_P, PAGE_D = B_P * ROW, B_D * ROW
STRIDE_P, STRIDE_D = 2 * PAGE_P + 16, 2 * PAGE_D + 8


def _p_memory(p_blocks: list[int], num_tokens: int) -> np.ndarray:
    mem = np.full(STRIDE_P * 16, 0xEE, dtype=np.uint8)
    for j, pb in enumerate(p_blocks):
        for layer in range(2):
            for t in range(B_P):
                tok = j * B_P + t
                if tok >= num_tokens:
                    continue
                at = pb * STRIDE_P + layer * PAGE_P + t * ROW
                mem[at : at + ROW] = [tok & 0xFF, tok >> 8, layer, 0xAB]
    return mem


@pytest.mark.parametrize("num_tokens", [1, 9, 31, 32, 33, 70, 96])
@pytest.mark.parametrize("cached", [0, 1])
def test_sub_block_reads_token_oracle(num_tokens, cached):
    """Each D rank s pulls P sub-block s + 4 (m + i) into its block i: the
    contiguous tokens the staged exchange then redistributes by residue."""
    if cached and num_tokens <= B_P:
        pytest.skip("a prefix hit needs a full virtual block before the pull")
    w = _decoder(regions=2, ssm=())
    w.block_len_per_layer = [PAGE_D, PAGE_D]
    p_blocks = [9, 2, 5, 11][: cdiv(num_tokens, B_P)]
    meta = SimpleNamespace(
        engine_id="P",
        block_size=B_P,
        physical_blocks_per_logical_kv_block=1,
        pcp_size=1,
        dcp_size=1,
        cp_kv_cache_interleave_size=1,
        region_names=["layer0", "layer1"],
        block_lens=[PAGE_P, PAGE_P],
        block_strides=[STRIDE_P, STRIDE_P],
        region_num_blocks=[16, 16],
        num_blocks=16,
        kv_caches_base_addr=[0, PAGE_P],
        device_id=0,
    )
    w._sub_block_peers["P"] = SubBlockPeer(w._sub_block_remote_meta(meta, 1), [], B_P)
    remote_descs = _expand(w._build_fa_remote(None, meta, 1))
    assert len(remote_descs) == 2 * 16 * X
    p_mem = _p_memory(p_blocks, num_tokens)
    remote_ids = expand_sub_block_ids(p_blocks, X, num_tokens, B_D)
    num_virtual = cdiv(num_tokens, B_P)
    for s in range(4):
        d_mem = np.zeros(STRIDE_D * 8, dtype=np.uint8)
        local_ids = [3, 0, 6, 1][cached:num_virtual]
        local, remote = w._apply_dcp_prefix_caching(
            local_ids,
            remote_ids,
            remote_rank=0,
            local_dcp_size=4,
            local_dcp_rank=s,
            remote_dcp_size=1,
            local_num_computed_blocks=cached,
        )
        for layer in range(2):
            for lb, rb in zip(local, remote):
                addr, length = remote_descs[layer * 16 * X + rb]
                assert length == PAGE_D
                at = lb * STRIDE_D + layer * PAGE_D
                d_mem[at : at + PAGE_D] = p_mem[addr : addr + length]
        for i, lb in enumerate(local_ids):
            first = B_D * (s + 4 * (cached + i))
            for layer in range(2):
                page = d_mem[lb * STRIDE_D + layer * PAGE_D :][:PAGE_D]
                rows = page.reshape(B_D, ROW)
                for t in range(B_D):
                    tok = first + t
                    if first >= num_tokens:
                        # Past the end: no remote sub-block, left unwritten.
                        assert (rows[t] == 0).all()
                    elif tok < num_tokens:
                        assert list(rows[t]) == [tok & 0xFF, tok >> 8, layer, 0xAB]
                    else:
                        # Tail of the last sub-block: P never wrote it.
                        assert (rows[t] == 0xEE).all()


def test_draft_head_dlists_geometry():
    """TP1 P draft page [16 heads, 32 tokens, c]; TP4 D rank 2 holds heads 8-11
    as [4 heads, 8 tokens, c]: head j of sub-block q of P block b sits at
    (8 + j) * 32 c + q * 8 c of the P page."""
    c = 4  # bytes per token-head
    w = _decoder(regions=2, ssm=())
    w.tp_rank, w.engine_id, w.device_id = 2, "D", 0
    w._head_sharded_draft_kv_heads = 16
    w._head_sharded_draft_regions = {1}
    w.block_len_per_layer = [PAGE_D, 4 * B_D * c]
    w.block_stride_per_layer = [STRIDE_D, 1000]
    w.region_num_blocks = [8, 8]
    w.kv_caches_base_addr = {"D": {2: [0, 50_000]}}
    w._sub_block_peers["P"] = SubBlockPeer(X, [PAGE_P, 16 * B_P * c], B_P)
    meta = SimpleNamespace(
        engine_id="P",
        region_num_blocks=[16 * X, 16 * X],
        kv_caches_base_addr=[0, 100_000],
        block_strides=[STRIDE_P, 5000],
        device_id=7,
    )
    local, remote = w._draft_head_dlists(meta, remote_tp_rank=0, remote_tp_size=1)
    assert len(local) == len(remote) == 4
    for j in range(4):
        loc = _expand(local[j])
        rem = _expand(remote[j])
        assert len(loc) == 8 and len(rem) == 16 * X
        assert loc[3] == (50_000 + 3 * 1000 + j * B_D * c, B_D * c)
        b, q = 5, 3
        assert rem[X * b + q] == (
            100_000 + b * 5000 + (8 + j) * B_P * c + q * B_D * c,
            B_D * c,
        )
    # A P rank that does not hold this rank's heads has no draft dlists.
    assert w._draft_head_dlists(meta, remote_tp_rank=0, remote_tp_size=2) is None


@pytest.mark.parametrize(
    "remote_sub_blocks,peer_sub_blocks,local_block_ids,skewed",
    [
        # Notify-only: aborted before it was scheduled, ids never expanded.
        (1, 4, [], False),
        (1, 4, [[], []], False),
        # A read whose expansion matches the handshake.
        (4, 4, [[3, 4], []], False),
        # A read with unexpanded ids from a sub-block peer would scramble KV.
        (1, 4, [[3, 4], []], True),
        # Expanded ids, but the handshake found an equal-block peer.
        (4, 1, [[3]], True),
    ],
)
def test_sub_block_skew_only_fails_reads(
    remote_sub_blocks, peer_sub_blocks, local_block_ids, skewed
):
    """A request aborted before it was scheduled carries P's unexpanded block
    ids and no local blocks: it only frees P's blocks and must not fail on
    the sub-block skew (DP8 P/D gate, 2026-11-20: such cleanups logged
    sub_block_mismatch and never notified P)."""
    meta = SimpleNamespace(
        remote=SimpleNamespace(sub_blocks=remote_sub_blocks),
        local_block_ids=local_block_ids,
    )
    assert NixlPullConnectorWorker._sub_block_skew(meta, peer_sub_blocks) == skewed
