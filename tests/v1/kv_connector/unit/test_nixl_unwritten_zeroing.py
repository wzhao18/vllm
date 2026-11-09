# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Zeroing of the local attention pages a NIXL pull leaves unwritten.

The scheduler skips alloc-time zeroing for blocks an async load fills, so every
page the pull does not write must be zeroed by the connector, and on a failed
pull every page. No GPU or NIXL required.
"""

import queue
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.utils import TransferTopology
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.page_zeroer import (
    PageZeroer,
    page_segments,
    unwritten_blocks,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    compute_tp_mapping,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
)

from .utils import region_pairs_by_index

POISON = 0xFF


def test_unwritten_blocks_whole_blocks():
    assert unwritten_blocks([4, 5, 6, 7], {5, 7}) == [4, 6]
    assert unwritten_blocks([4, 5], {4, 5}) == []
    assert unwritten_blocks([4, 5], set()) == [4, 5]
    # Duplicates in the request's list are zeroed once.
    assert unwritten_blocks([4, 4, 5], set()) == [4, 5]


def test_unwritten_blocks_sub_block_units():
    # ratio 4: block 2 owns units 8..11. A partly written block counts as
    # touched; the receive post-process zeroes its token tail.
    assert unwritten_blocks([2, 3, 9], {8, 9, 12}, ratio=4) == [9]
    assert unwritten_blocks([2, 3], {11}, ratio=4) == [3]


def test_page_segments_block_major():
    cache = torch.zeros(6, 4, 8, dtype=torch.bfloat16)
    ((addr, stride, page, n),) = page_segments(cache, None)
    assert (addr, stride, page, n) == (cache.data_ptr(), 64, 64, 6)


def test_page_segments_padded_page():
    # A layer view narrower than the shared block stride (hybrid page padding).
    backing = torch.zeros(6, 96, dtype=torch.uint8)
    view = backing[:, :40].view(6, 5, 8)
    ((addr, stride, page, n),) = page_segments(view, None)
    assert (addr, stride, page, n) == (backing.data_ptr(), 96, 40, 6)


def test_page_segments_outer_head_dim():
    # Heads laid out outside the block stride (LHBNC-like): one segment each.
    backing = torch.zeros(3, 6, 4, 8, dtype=torch.uint8)  # [H, B, N, C]
    view = backing.permute(1, 0, 2, 3)  # [B, H, N, C]
    segs = page_segments(view, None)
    assert [s[0] - backing.data_ptr() for s in segs] == [0, 192, 384]
    assert {(s[1], s[2], s[3]) for s in segs} == {(32, 32, 6)}


def test_page_segments_flat_view():
    flat = torch.zeros(6 * 48, dtype=torch.uint8)
    assert page_segments(flat, 48) == [(flat.data_ptr(), 48, 48, 6)]


def _poisoned(*shape):
    return torch.full(shape, POISON, dtype=torch.uint8)


def test_page_zeroer_zeroes_only_given_pages():
    a, b = _poisoned(6, 4, 8), _poisoned(6, 4, 8)
    other = _poisoned(6, 4, 8)
    zeroer = PageZeroer({0: [a, b], 2: [other]}, {0: None, 2: None})
    assert zeroer.groups == (0, 2)
    zeroer.zero({0: [1, 4], 1: [0], 2: []})
    for cache in (a, b):
        assert not cache[[1, 4]].any()
        assert (cache[[0, 2, 3, 5]] == POISON).all()
    assert (other == POISON).all()


def test_page_zeroer_keeps_padding_and_widest_overlay():
    backing = _poisoned(6, 96)
    narrow = backing[:, :16].view(6, 2, 8)
    wide = backing[:, :40].view(6, 5, 8)
    zeroer = PageZeroer({0: [narrow, wide]}, {0: None})
    zeroer.zero({0: [3]})
    assert not backing[3, :40].any()
    # Bytes past the attention page belong to no attention layer here.
    assert (backing[3, 40:] == POISON).all()
    assert (backing[[0, 1, 2, 4, 5]] == POISON).all()


def test_page_zeroer_async_on_cpu_is_inline():
    cache = _poisoned(4, 2, 8)
    zeroer = PageZeroer({0: [cache]}, {0: None})
    zeroer.zero_async({0: [2]})
    zeroer.step_fence(include_fresh=True)
    assert not cache[2].any() and (cache[[0, 1, 3]] == POISON).all()


def _mla_pull_worker(dcp_rank: int, cache: torch.Tensor):
    """Pure-MLA TP16/DCP16 decoder rank pulling from a TP8/DCP8 prefiller,
    per-rank block 896 and interleave 896 on both sides. Real block planning
    and descriptors, with only the transport mocked."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
        NixlConnectorWorker,
    )

    w = object.__new__(NixlConnectorWorker)
    w.engine_id, w.tp_rank = "D", dcp_rank
    w.block_size = 896
    w.use_mla, w._has_mamba = True, False
    w.dcp_size, w.dcp_rank = 16, dcp_rank
    w._physical_blocks_per_logical_kv_block = 1
    w.enable_permute_local_kv = False
    w.enable_heterogeneous_attn_post_process = False
    w._bidirectional_kv_xfer_enabled = False
    w._engine_last_active = {}
    w._handshake_lock = threading.RLock()
    w._handshake_futures = {}
    w._recving_transfers = {}
    w._invalid_remote_engines = set()
    w._remote_registration_generations = {}
    w._transfer_layer_group_ids = ()
    spec = MLAAttentionSpec(
        block_size=896, num_kv_heads=1, head_size=8, dtype=torch.uint8
    )
    w.kv_cache_config = KVCacheConfig(
        num_blocks=cache.shape[0],
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["mla"], spec)],
    )
    w._group_spec_types = (MLAAttentionSpec,)
    w.region_group_ids = [0]
    w.dst_region_group_ids = {"P": [0]}
    w.num_regions = 1
    w.block_len_per_layer = [cache[0].nbytes]
    w.transfer_topo = MagicMock()
    w.transfer_topo.tp_ratio.return_value = 2
    w.transfer_topo.block_size_ratio.return_value = 1
    w.transfer_topo.get_engine_info.return_value = SimpleNamespace(
        remote_tp_size=8,
        remote_dcp_size=8,
        remote_block_size=896,
        remote_physical_blocks_per_logical=1,
    )
    topo = object.__new__(TransferTopology)
    topo.tp_rank, topo.tp_size, topo.dcp_size = dcp_rank, 16, 16
    topo.is_mla, topo.is_mamba, topo.total_num_kv_heads = True, False, 1
    w.tp_mappings = {
        "P": compute_tp_mapping(
            transfer_topology=topo,
            remote_tp_size=8,
            group_spec_types=(MLAAttentionSpec,),
            remote_dcp_size=8,
        )
    }
    w._mixed_mem_types = True
    w.src_xfer_handles_by_block_size = {896: 1}
    w._dram_src_handles_by_block_size = {896: 2}
    w.dst_xfer_side_handles = {"P": {r: 3 for r in range(8)}}
    w.dst_num_blocks = {"P": 64, "D": cache.shape[0]}
    w.dst_region_num_blocks = {"P": [64], "D": [cache.shape[0]]}
    w._dst_pairs = {"P": region_pairs_by_index(1)}
    w._remote_agents = {"P": {(0, r): f"P{r}" for r in range(8)}}
    w._read_blocks_mixed = MagicMock()
    w.nixl_wrapper = MagicMock()
    w.use_host_buffer = False
    w._page_zeroer = PageZeroer({0: [cache]}, {0: None})
    w._unwritten_batch = {}
    w._fresh_zeroed_reqs = set()
    return w


def _recv_meta(local_blocks: list[int], remote_blocks: list[int], num_tokens: int):
    metadata = NixlConnectorMetadata()
    metadata.add_new_req_to_recv(
        request_id="r",
        local_block_ids=[local_blocks],
        local_num_computed_blocks=(0,),
        awaiting_kvs=True,
        kv_transfer_params={
            "remote_engine_id": "P",
            "remote_request_id": "r-P",
            "remote_host": "localhost",
            "remote_port": 1,
            "remote_num_tokens": num_tokens,
            "remote_block_ids": [remote_blocks],
        },
    )
    meta = metadata.reqs_to_recv["r"]
    meta.local_physical_block_ids = [list(local_blocks)]
    return meta


@pytest.mark.parametrize(
    "dcp_rank,zeroed",
    [
        # A 4855-token prefill fills one DCP8 block (chunks 0-7). Decoder
        # ranks 0-7 pull chunk r from it; ranks 8-15 own chunk 8 + (r - 8),
        # which no prefiller block holds, so the pull never writes their block.
        (0, False),
        (7, False),
        (8, True),
        (15, True),
    ],
)
def test_hetero_dcp_clipped_block_is_zeroed(dcp_rank, zeroed):
    cache = _poisoned(8, 896, 8)
    w = _mla_pull_worker(dcp_rank, cache)
    meta = _recv_meta([5], [11], 4855)
    w._recving_metadata = {"r": meta}
    w._read_blocks_for_req("r", meta)
    # Nothing is zeroed until the step's reads have all been posted.
    assert (cache == POISON).all()
    w._flush_unwritten_zeroing()
    assert (not cache[5].any()) if zeroed else (cache[5] == POISON).all()
    # Other blocks are never touched.
    assert (cache[[0, 1, 2, 3, 4, 6, 7]] == POISON).all()
    assert w._read_blocks_mixed.called != zeroed
    assert (w._fresh_zeroed_reqs == {"r"}) == zeroed


def test_partially_covered_request_zeroes_only_the_unwritten_blocks():
    # 20900 tokens: two DCP16 blocks per decoder rank (chunks r and 16 + r).
    # The prefill holds chunks 0..23 in three DCP8 blocks, so rank 2 gets both
    # of its chunks (2, 18) but rank 9 only its first (9; 25 is past the end).
    for dcp_rank, zeroed in ((2, []), (9, [6])):
        cache = _poisoned(8, 896, 8)
        w = _mla_pull_worker(dcp_rank, cache)
        meta = _recv_meta([5, 6], [10, 11, 12], 20900)
        w._read_blocks_for_req("r", meta)
        w._flush_unwritten_zeroing()
        for b in range(8):
            assert (not cache[b].any()) if b in zeroed else (cache[b] == POISON).all()


def _completion_harness(w, meta, failed: bool):
    w._recving_metadata = {"r": meta}
    if failed:
        # Every handle of the failed pull has retired.
        w._recving_transfers.pop("r", None)
    w._get_new_notifs = MagicMock(return_value=set())
    w._pop_done_transfers = MagicMock(
        return_value=(set(), {"r"}) if failed else ({"r"}, set())
    )
    w._replicated_pcp_done_sending = set()
    w._failed_recv_reqs = queue.Queue()
    w._recv_failures = set()
    w._pending_recv_notifs = {}
    w._is_hma_required = True
    w._send_pending_recv_notifs = MagicMock()
    w._sync_device_after_direct_recv = MagicMock()
    w._reap_expired_send_leases = MagicMock()
    w._cleanup_remote_engines = MagicMock()
    w._reqs_to_send = {}


@pytest.mark.parametrize("failed", [False, True])
def test_completion_zeroes_every_page_only_on_failure(failed):
    cache = _poisoned(8, 896, 8)
    w = _mla_pull_worker(0, cache)
    meta = _recv_meta([5, 6], [10, 11, 12], 20900)
    w._read_blocks_for_req("r", meta)  # rank 0 writes both blocks
    w._flush_unwritten_zeroing()
    assert (cache == POISON).all()
    _completion_harness(w, meta, failed)
    result = w.get_transfer_results()
    assert result.finished_recving == {"r"}
    assert result.failed_recving == ({"r"} if failed else set())
    assert (not cache[[5, 6]].any()) if failed else (cache == POISON).all()
    assert (cache[[0, 1, 2, 3, 4, 7]] == POISON).all()


def test_host_buffer_defers_zeroing_past_the_sync():
    cache = _poisoned(8, 896, 8)
    w = _mla_pull_worker(9, cache)
    w.use_host_buffer = True
    meta = _recv_meta([5, 6], [10, 11, 12], 20900)
    w._read_blocks_for_req("r", meta)
    w._flush_unwritten_zeroing()
    assert meta.unwritten_blocks == {0: [6]}
    assert (cache == POISON).all()
    _completion_harness(w, meta, failed=False)
    w.sync_recved_kv_to_device = MagicMock(
        side_effect=lambda *_: cache.fill_(POISON)  # the sync rewrites every page
    )
    w.get_transfer_results()
    assert not cache[6].any()
    assert (cache[[0, 1, 2, 3, 4, 5, 7]] == POISON).all()
