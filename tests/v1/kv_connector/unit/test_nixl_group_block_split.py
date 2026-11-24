# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-group remote block splits: a TP1 prefiller whose packed draft group has a
smaller block than its target groups feeds a TP4 DSA-DP decoder (K3.1 numbers:
target 16384 vs 4096 tokens, SWA DSpark draft 8192 vs 4096 tokens, 16 draft KV
heads of 256 B per token, head-major draft pages)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlAgentMetadata,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowSpec

from .utils import expand_strided_descs

TARGET = [1757184, 2359296, 278528]  # local (D) MLA / indexer region pages
HEAD_TOKEN = 256  # draft bytes per KV head per token
DRAFT_HEADS = 16
LOCAL_TP = 4
LOCAL_BLOCK = 4096
LOCAL_DRAFT = LOCAL_BLOCK * (DRAFT_HEADS // LOCAL_TP) * HEAD_TOKEN  # 4194304
ROWS = 3


def _worker(local_rank: int = 1) -> NixlBaseConnectorWorker:
    w = object.__new__(NixlBaseConnectorWorker)
    w.block_size = LOCAL_BLOCK
    w.block_len_per_layer = TARGET + [LOCAL_DRAFT, LOCAL_DRAFT]
    w._ssm_region_indices = []
    w._head_sharded_draft_regions = {3, 4}
    w._head_sharded_draft_kv_heads = DRAFT_HEADS
    w._transfer_layer_region_indices = []
    w.transfer_topo = SimpleNamespace(
        tp_size=LOCAL_TP,
        get_engine_info=lambda engine_id: SimpleNamespace(remote_tp_size=1),
    )
    w._group_spec_types = [MLAAttentionSpec, SlidingWindowSpec]
    w._group_region_indices = [[0, 1, 2], [3, 4]]
    w._remote_block_split = {}
    w._remote_region_split = {}
    w._remote_group_split = {}
    w.draft_tp_mappings = {"remote": SimpleNamespace(rank_offset_factor=local_rank)}
    w.use_mla = True
    w._physical_blocks_per_logical_kv_block = 1
    mla = MLAAttentionSpec(
        block_size=LOCAL_BLOCK, num_kv_heads=1, head_size=576, dtype=torch.uint8
    )
    swa = SlidingWindowSpec(
        block_size=LOCAL_BLOCK,
        num_kv_heads=DRAFT_HEADS // LOCAL_TP,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=8192,
    )
    w.kv_cache_config = SimpleNamespace(
        transfer_groups=[SimpleNamespace(kv_cache_spec=s) for s in (mla, swa)],
        transfer_group_ids=[0, 1],
    )
    return w


def _remote_meta(draft_block_tokens: int) -> NixlAgentMetadata:
    """P (TP1): 16384-token target pages, draft pages at all 16 heads, padded
    rows (row stride larger than each page)."""
    draft = draft_block_tokens * DRAFT_HEADS * HEAD_TOKEN
    lens = [4 * t for t in TARGET] + [draft, draft]
    row = 2 * sum(lens)
    return NixlAgentMetadata(
        engine_id="remote",
        agent_metadata=b"",
        device_id=0,
        kv_caches_base_addr=[0x1_0000_0000 * (i + 1) for i in range(len(lens))],
        num_blocks=ROWS,
        block_lens=lens,
        block_strides=[row] * len(lens),
        kv_cache_layout="HND",
        block_size=16384,
        ssm_sizes=(0, 0),
        attn_backend_name="FLASHMLA_SPARSE",
        region_num_blocks=[ROWS] * len(lens),
        physical_blocks_per_logical_kv_block=1,
    )


def _draft_expected_len(meta: NixlAgentMetadata, i: int) -> int:
    """The hybrid MLA validation's expected local length of a draft region."""
    return meta.block_lens[i] * (DRAFT_HEADS // LOCAL_TP) // DRAFT_HEADS


@pytest.mark.cpu_test
def test_packed_draft_group_keeps_its_own_ratio():
    """P's 8192-token draft blocks pair with D's 4096-token blocks two to one
    while the target groups pair four to one."""
    w = _worker()
    meta = _remote_meta(draft_block_tokens=8192)
    w._split_remote_blocks(meta, remote_tp_size=1)
    assert w._remote_group_split["remote"] == (4, 2)
    assert meta.block_lens[:3] == TARGET
    assert all(_draft_expected_len(meta, i) == LOCAL_DRAFT for i in (3, 4))
    assert w._remote_kernel_block_ids("remote", [[3], [5]], 4) == [
        [12, 13, 14, 15],
        [10, 11],
    ]


@pytest.mark.cpu_test
def test_capped_draft_block_pairs_one_to_one():
    """A 4096-token P draft block (VLLM_K3_DRAFT_BLOCK_MAX=4096) is one local
    block: the draft region is not cut, only its head slice is read."""
    w = _worker()
    meta = _remote_meta(draft_block_tokens=4096)
    w._split_remote_blocks(meta, remote_tp_size=1)
    assert w._remote_group_split["remote"] == (4, 1)
    assert all(_draft_expected_len(meta, i) == LOCAL_DRAFT for i in (3, 4))


@pytest.mark.cpu_test
def test_non_integer_group_ratio_is_not_paired():
    """A draft block of 1.5 local blocks cannot be split: the remote is left
    unsplit, so the handshake fails the block-length validation."""
    w = _worker()
    meta = _remote_meta(draft_block_tokens=6144)
    w._split_remote_blocks(meta, remote_tp_size=1)
    assert "remote" not in w._remote_block_split
    assert _draft_expected_len(meta, 3) != LOCAL_DRAFT


@pytest.mark.cpu_test
def test_uniform_split_takes_the_existing_path():
    """When every group splits alike nothing per-group is recorded."""
    w = _worker()
    meta = _remote_meta(draft_block_tokens=16384)
    w._split_remote_blocks(meta, remote_tp_size=1)
    assert w._remote_block_split["remote"] == (4, True)
    assert "remote" not in w._remote_group_split
    assert w._remote_kernel_block_ids("remote", [[3], [5]], 4) == [
        [12, 13, 14, 15],
        [20, 21, 22, 23],
    ]


@pytest.mark.cpu_test
@pytest.mark.parametrize("local_rank", [0, 3])
def test_draft_units_address_this_ranks_heads(local_rank):
    """Kernel id ``r * 2 + j`` of a draft region reads this rank's 4 heads of
    tokens ``[j * 4096, (j + 1) * 4096)`` of remote row ``r``; every region has
    ``ROWS * 4`` kernel slots so desc ids stay ``region * num_blocks + id``."""
    w = _worker(local_rank)
    meta = _remote_meta(draft_block_tokens=8192)
    w._split_remote_blocks(meta, remote_tp_size=1)
    chunks = (1, 1, 1, 4, 4)
    descs = expand_strided_descs(
        w._build_fa_remote_rows(meta, 4, chunks, w._remote_region_split["remote"])
    )
    per_region = [ROWS * 4 * c for c in chunks]
    assert len(descs) == sum(per_region)
    offsets = np.cumsum([0, *per_region[:-1]])
    head_page = 8192 * HEAD_TOKEN
    head_unit = LOCAL_BLOCK * HEAD_TOKEN
    for region in (3, 4):
        base = meta.kv_caches_base_addr[region]
        row = meta.block_strides[region]
        for r in range(ROWS):
            for j in range(2):
                kernel_id = r * 2 + j
                for h in range(4):
                    addr, length, _ = descs[offsets[region] + kernel_id * 4 + h]
                    head = local_rank * 4 + h
                    assert int(length) == head_unit
                    assert (
                        int(addr) == base + head * head_page + r * row + j * head_unit
                    )
    # Target regions keep four units per row.
    for region in (0, 1, 2):
        base = meta.kv_caches_base_addr[region]
        addr, length, _ = descs[offsets[region] + 5]
        assert int(length) == TARGET[region]
        assert int(addr) == base + 1 * meta.block_strides[region] + 1 * TARGET[region]


@pytest.mark.cpu_test
def test_sliding_window_units_align_per_group():
    """The SWA draft window pairs by token position with its own unit count:
    the remote list's last block ends at the block holding the last token."""
    w = _worker()
    meta = _remote_meta(draft_block_tokens=8192)
    w._split_remote_blocks(meta, remote_tp_size=1)
    num_tokens = 3 * LOCAL_BLOCK + 100  # last unit 3, in P draft block 1
    remote = w._remote_kernel_block_ids("remote", [[5], [7]], 4)
    local = [[1, 2, 3, 4], [21, 22]]  # D: target units 0..3, draft window 2..3
    aligned_local, aligned_remote = w._align_split_block_ids(
        local, remote, (0, 0), num_tokens, w._remote_group_split["remote"]
    )
    assert aligned_local == [[1, 2, 3, 4], [21, 22]]
    assert aligned_remote == [[20, 21, 22, 23], [14, 15]]
