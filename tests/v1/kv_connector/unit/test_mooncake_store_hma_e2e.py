# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end save->lookup test for MooncakeStoreConnector on a hybrid
(SWA + Full) attention config, using a dict-backed mock store."""

import sys
import threading
import types
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.v1.attention.utils import dense_kv_cache_views
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
    worker as mooncake_store_worker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.coordinator import (  # noqa: E501
    MooncakeStoreCoordinator,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    ChunkedTokenDatabase,
    KeyMetadata,
    LoadSpec,
    PoolKey,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (  # noqa: E501
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
)
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KpoolTailSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_cache_layout import KVCacheLayout


class _DictStore:
    """In-memory MooncakeDistributedStore stand-in."""

    def __init__(self):
        self._data: dict[str, bytes] = {}

    def setup(self, *_args, **_kwargs):
        return 0

    def register_buffer(self, addr, length):
        return 0

    def batch_is_exist(self, keys):
        return [1 if k in self._data else 0 for k in keys]

    def batch_put_from_multi_buffers(self, keys, addrs, sizes, *_args, **_kwargs):
        for k in keys:
            self._data[k] = b"x"
        return [0] * len(keys)

    def batch_get_into_multi_buffers(self, keys, addrs, sizes, *_args, **_kwargs):
        return [0 if k in self._data else -1 for k in keys]


def _run_dcp8_aligned_tail_replay(
    prompt_tail: int,
    replay_boundary: int,
    computed_end_tokens: int | None = None,
    append_tokens: int = 0,
    block_publication: bool = False,
) -> tuple[int, dict[int, int], dict[int, int]]:
    """Save exact Kimi-K3 group geometry, then perform an identical lookup."""
    mamba = MambaSpec(
        block_size=896,
        shapes=((1, 1),),
        dtypes=(torch.bfloat16,),
        mamba_cache_mode="align",
    )
    full = FullAttentionSpec(
        block_size=7168,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.float8_e4m3fn,
    )
    groups = [KVCacheGroupSpec([f"m{i}"], mamba) for i in range(3)] + [
        KVCacheGroupSpec(["fa"], full, is_eagle_group=True)
    ]
    coord = MooncakeStoreCoordinator(
        groups,
        scheduler_block_size=7168,
        hash_block_size=128,
        use_eagle=True,
        retention_interval=0,
        dcp_world_size=8,
    )
    put_entered = threading.Event()
    release_put = threading.Event()

    class _PublicationStore(_DictStore):
        def __init__(self):
            super().__init__()
            self.discovery_calls: list[list[str]] = []
            self.lease_calls: list[list[str]] = []

        def batch_is_exist(self, keys):
            self.lease_calls.append(list(keys))
            return super().batch_is_exist(keys)

        def batch_is_exist_no_lease(self, keys):
            self.discovery_calls.append(list(keys))
            return super().batch_is_exist(keys)

        def batch_put_from_multi_buffers(self, keys, addrs, sizes, *args, **kwargs):
            if block_publication:
                put_entered.set()
                assert release_put.wait(timeout=5)
            return super().batch_put_from_multi_buffers(
                keys, addrs, sizes, *args, **kwargs
            )

    store = _PublicationStore()
    dbs = []
    for group_id, block_size in enumerate((896, 896, 896, 7168)):
        db = ChunkedTokenDatabase(
            KeyMetadata("model", 0, 0, 0, 0, group_id=group_id),
            block_size=block_size,
            hash_block_size=128,
        )
        db.set_kv_caches_base_addr([group_id * 10_000])
        db.set_block_len([512])
        dbs.append(db)

    sender = KVCacheStoreSendingThread(
        store=store,
        coord=coord,
        token_databases=dbs,
        block_size=7168,
        tp_rank=0,
        group_put_steps=[1, 1, 1, 1],
        kv_role="kv_both",
        ready_event=threading.Event(),
        replicate_config=MagicMock(),
    )
    # num_prompt_tokens is prompt_tail + 1, so prompt_tail is the final full
    # 128-token hash boundary.  Do not synthesize a future hash beyond the
    # producer's tokens: EAGLE lookup may only use proof the producer computed.
    hash_count = prompt_tail // 128
    block_hashes = [
        BlockHash(i.to_bytes(8, byteorder="little")) for i in range(1, hash_count + 1)
    ]
    req = ReqMeta(
        req_id="r0",
        token_len_chunk=0,
        block_ids=(
            [1] * 16,
            [2] * 16,
            [3] * 16,
            [4] * ((prompt_tail + 7167) // 7168 + 1),
        ),
        block_hashes=block_hashes,
        can_save=True,
        num_prompt_tokens=prompt_tail + 1,
        computed_end_tokens=computed_end_tokens,
        boundary_state_offloads=[
            (group_id, 20 + group_id, boundary)
            for boundary in (replay_boundary, prompt_tail)
            for group_id in range(3)
        ],
    )
    worker = object.__new__(mooncake_store_worker.MooncakeStoreWorker)
    worker._capacity_only = False
    worker.coord = coord
    worker.token_dbs = dbs
    worker._kv_cache_groups = groups
    worker.hash_block_size = 128
    worker.store = store
    worker._lookup_key_prefixes = tuple(
        (PoolKey.build_prefix(db.metadata),) for db in dbs
    )
    worker._record_kv_connector_operation = lambda *_args, **_kwargs: None

    def group_counts(boundary: int) -> dict[int, int]:
        suffix = "@" + block_hashes[boundary // 128 - 1].hex()
        return {
            group_id: sum(
                f"@group:{group_id}@" in key and key.endswith(suffix)
                for key in store._data
            )
            for group_id in range(4)
        }

    target_tokens = prompt_tail + 1 + append_tokens
    target_hash_count = target_tokens // 128
    target_hashes = block_hashes + [
        BlockHash(i.to_bytes(8, byteorder="little"))
        for i in range(hash_count + 1, target_hash_count + 1)
    ]
    if block_publication:
        save_result: list[bool] = []
        save_thread = threading.Thread(
            target=lambda: save_result.append(
                sender._maybe_offload_boundary_states(req)
            )
        )
        save_thread.start()
        assert put_entered.wait(timeout=5)
        assert worker.lookup(target_tokens, target_hashes).hit_length == 0
        release_put.set()
        save_thread.join(timeout=5)
        assert not save_thread.is_alive()
        assert save_result == [True]
    else:
        assert sender._maybe_offload_boundary_states(req)
    # Model the normal positional save of completed attention pages. Boundary
    # hand-offs only add the partial page that contains each retained state.
    for boundary in range(7168, prompt_tail + 1, 7168):
        store._data[dbs[3].key_for(block_hashes[boundary // 128 - 1])] = b"x"

    store.discovery_calls.clear()
    store.lease_calls.clear()
    result = worker.lookup(num_tokens=target_tokens, block_hashes=target_hashes)
    load_keys = worker._lookup_load_keys(target_hashes, result)
    lease_keys = store.lease_calls[0] if result.hit_length else []
    assert all(key in store._data for key in load_keys)
    if result.hit_length:
        assert store.lease_calls == [lease_keys]
        assert set(lease_keys) <= set(store.discovery_calls[0])
    if result.hit_length == replay_boundary and prompt_tail == replay_boundary + 128:
        tail_boundaries = {
            boundary.group_id: boundary.num_tokens
            for boundary in result.tail_key_boundaries
        }
        assert tail_boundaries == {
            0: replay_boundary,
            1: replay_boundary,
            2: replay_boundary,
            3: replay_boundary,
        }
        proof_key = dbs[3].key_for(block_hashes[prompt_tail // 128 - 1])
        assert proof_key in store.discovery_calls[0]
        assert proof_key not in load_keys
        assert proof_key in lease_keys
        proof_value = store._data.pop(proof_key)
        assert worker.lookup(target_tokens, target_hashes).hit_length < replay_boundary
        store._data[proof_key] = proof_value
    return result.hit_length, group_counts(replay_boundary), group_counts(prompt_tail)


@pytest.mark.parametrize("prompt_tail", [1792, 1920])
def test_dcp8_mamba_aligned_replay_has_attention_tail(prompt_tail: int):
    """Aligned and neighboring prompt tails both preserve the 896 replay hit."""
    hit, replay_keys, tail_keys = _run_dcp8_aligned_tail_replay(
        prompt_tail=prompt_tail, replay_boundary=896
    )

    assert replay_keys == {0: 1, 1: 1, 2: 1, 3: 1}
    assert tail_keys == {0: 1, 1: 1, 2: 1, 3: 1}
    assert hit == 896


def test_dcp8_aligned_replay_after_full_attention_page():
    """The aligned-tail fix also retains a replay inside a later FA page."""
    hit, replay_keys, tail_keys = _run_dcp8_aligned_tail_replay(
        prompt_tail=8960, replay_boundary=8064
    )

    assert replay_keys == {0: 1, 1: 1, 2: 1, 3: 1}
    assert tail_keys == {0: 1, 1: 1, 2: 1, 3: 1}
    assert hit == 8064


def _minimal_vllm_config(cache_block_size=16):
    cfg = MagicMock()
    cfg.cache_config.block_size = cache_block_size
    cfg.cache_config.num_gpu_blocks = 4
    cfg.cache_config.hash_block_size = None
    cfg.cache_config.prefix_match_unit = None
    cfg.cache_config.enable_prefix_caching = True
    cfg.parallel_config.prefill_context_parallel_size = 1
    cfg.parallel_config.decode_context_parallel_size = 1
    cfg.parallel_config.pipeline_parallel_size = 1
    cfg.parallel_config.world_size = 1
    cfg.parallel_config.rank = 0
    cfg.parallel_config.data_parallel_rank_local = 0
    cfg.parallel_config.data_parallel_size_local = 1
    cfg.kv_transfer_config.kv_role = "kv_both"
    cfg.kv_transfer_config.kv_connector_extra_config = {}
    cfg.kv_events_config = None
    cfg.model_config.model = "/tmp/m"
    cfg.model_config.use_mla = False
    cfg.model_config.get_num_layers.return_value = 2
    cfg.model_config.get_total_num_kv_heads.return_value = 8
    cfg.model_config.max_model_len = 4096
    cfg.scheduler_config.max_num_batched_tokens = 8192
    # Without this, MagicMock's truthy use_eagle() triggers the coordinator's
    # "use_eagle && nothing annotated → flag all groups" fallback.
    cfg.speculative_config = None
    return cfg


def _build_worker_with_dict_store(vllm_config, kv_cache_config, store):
    """Build a MooncakeStoreWorker patching all distributed dependencies."""
    fake_mooncake_store = types.ModuleType("mooncake.store")
    fake_mooncake_store.MooncakeDistributedStore = lambda: store  # type: ignore[attr-defined]
    fake_mooncake_store.ReplicateConfig = MagicMock  # type: ignore[attr-defined]
    with (
        patch.dict(sys.modules, {"mooncake.store": fake_mooncake_store}),
        patch.object(mooncake_store_worker, "MooncakeStoreConfig") as MCfg,
        patch.object(mooncake_store_worker, "LookupKeyServer"),
    ):
        sc = MCfg.load_from_config.return_value
        sc.metadata_server = ""
        sc.global_segment_size = 1 << 20
        sc.local_buffer_size = 1 << 20
        sc.protocol = "tcp"
        sc.device_name = ""
        sc.master_server_address = ""
        sc.mode = "embedded"
        sc.enable_offload = False
        with (
            patch(
                "vllm.distributed.kv_transfer.kv_connector.v1.mooncake"
                ".store.worker.get_tensor_model_parallel_rank",
                return_value=0,
            ),
            patch(
                "vllm.distributed.kv_transfer.kv_connector.v1.mooncake"
                ".store.worker.get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            patch(
                "vllm.distributed.kv_transfer.kv_connector.v1.mooncake"
                ".store.worker.get_pcp_group"
            ) as mock_pcp,
            patch(
                "vllm.distributed.kv_transfer.kv_connector.v1.mooncake"
                ".store.worker.get_dcp_group"
            ) as mock_dcp,
            patch(
                "vllm.distributed.kv_transfer.kv_connector.v1.mooncake"
                ".store.worker.get_ip",
                return_value="127.0.0.1",
            ),
        ):
            mock_pcp.return_value.world_size = 1
            mock_dcp.return_value.world_size = 1
            worker = mooncake_store_worker.MooncakeStoreWorker(
                vllm_config, kv_cache_config=kv_cache_config
            )
    return worker


def test_e2e_swa_plus_full_save_then_lookup_hits():
    """
    E2E: build a SWA+Full hybrid worker, save all blocks via the sending
    thread (synchronously), then verify lookup returns the full hit length.
    Also verify that evicting SWA's early blocks (outside its window) still
    allows a full hit because the window covers the tail.
    """
    full = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=None)
    swa = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=64,
        dtype=None,
        sliding_window=32,
    )
    cfg = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[
            KVCacheTensor(
                size=4 * full.page_size_bytes,
                layers=["L0"],
                layer_stride=4 * full.page_size_bytes,
                block_stride=full.page_size_bytes,
            ),
            KVCacheTensor(
                size=4 * swa.page_size_bytes,
                layers=["L1"],
                layer_stride=4 * swa.page_size_bytes,
                block_stride=swa.page_size_bytes,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["L0"], full),
            KVCacheGroupSpec(["L1"], swa),
        ],
    )
    vllm_config = _minimal_vllm_config(cache_block_size=16)
    store = _DictStore()

    worker = _build_worker_with_dict_store(vllm_config, cfg, store)
    worker.tp_size = 1
    worker.pp_size = 1
    worker.num_kv_head = 8

    # Register kv_caches using mocked thread classes so register_kv_caches
    # doesn't try to start real background threads (which set ready_event).
    raw_full = torch.zeros(4 * full.page_size_bytes, dtype=torch.int8)
    raw_swa = torch.zeros(4 * swa.page_size_bytes, dtype=torch.int8)
    kv_caches = {
        "L0": dense_kv_cache_views(raw_full, full, 4, 1, KVCacheLayout.LBNHC)[0],
        "L1": dense_kv_cache_views(raw_swa, swa, 4, 1, KVCacheLayout.LBNHC)[0],
    }

    def _fake_thread_init(*args, **kwargs):
        """Mock thread that sets all threading.Event args so waits don't block."""
        for v in list(args) + list(kwargs.values()):
            if isinstance(v, threading.Event):
                v.set()
        m = MagicMock()
        m.start = lambda: None
        return m

    with (
        patch.object(
            mooncake_store_worker,
            "KVCacheStoreSendingThread",
            side_effect=_fake_thread_init,
        ),
        patch.object(
            mooncake_store_worker,
            "KVCacheStoreRecvingThread",
            side_effect=_fake_thread_init,
        ),
    ):
        worker.register_kv_caches(kv_caches)

    # Now build a real sending thread (no .start()) over the worker's token_dbs
    # and the dict-backed store, so _handle_request runs synchronously.
    ready = threading.Event()
    send_thread = KVCacheStoreSendingThread(
        store=store,
        token_databases=worker.token_dbs,
        block_size=worker.block_size,
        coord=worker.coord,
        tp_rank=worker.tp_rank,
        group_put_steps=worker._group_tp_replication_factors,
        kv_role=worker.kv_role,
        ready_event=ready,
        enable_kv_event=False,
    )

    hs = [BlockHash(bytes([i + 1]) * 4) for i in range(4)]
    # 1-based: block 0 is the reserved null block and the save skips it.
    save_req = ReqMeta(
        req_id="r0",
        token_len_chunk=64,
        # Block 0 is reserved as NULL_BLOCK_ID by the production block pool.
        block_ids=([1, 2, 3, 4], [1, 2, 3, 4]),
        block_hashes=hs,
        can_save=True,
        store_job_id=1,
    )
    # add_request also queues the job, so task_done() doesn't underflow.
    send_thread.add_request(save_req)
    send_thread._handle_request(send_thread.request_queue.get())

    # Point worker.store at the dict store (the worker constructor captured
    # the MagicMock; replace with the real dict store for lookup).
    worker.store = store

    # Both groups stored all 4 blocks -> full hit.
    assert worker.lookup(num_tokens=65, block_hashes=hs).hit_length == 64
    # Exact-multiple prompt: the full hit is re-derived one block lower,
    # where both groups' stored blocks still cover the SWA window.
    assert worker.lookup(num_tokens=64, block_hashes=hs).hit_length == 48

    # Evict SWA's first two blocks (outside its window of 32 tokens = 2 blocks).
    swa_keys_outside_window = [
        k
        for k in list(store._data.keys())
        if "@group:1" in k and (("@" + hs[0].hex()) in k or ("@" + hs[1].hex()) in k)
    ]
    for k in swa_keys_outside_window:
        del store._data[k]

    # SWA window=32 -> only last 2 blocks must be present in SWA group.
    # Full has all 4. Coordinator should still return 64.
    assert worker.lookup(num_tokens=65, block_hashes=hs).hit_length == 64
    # Exact-multiple prompt after eviction: the boundary one block lower
    # needs SWA block 1, which is gone — no usable stored boundary remains
    # (the pre-fix arithmetic clamp would have returned 48 and livelocked
    # on load failure -> recompute -> same lookup).
    assert worker.lookup(num_tokens=64, block_hashes=hs).hit_length == 0


def test_recv_skips_swa_blocks_before_window():
    """Producer stored every block for both groups; consumer must only fetch
    SWA blocks within the sliding window, not the head."""
    full = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=None)
    # sliding_window=32, block_size=16 → 2 contiguous blocks within window.
    swa = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=64,
        dtype=None,
        sliding_window=32,
    )
    groups = [
        KVCacheGroupSpec(["L0"], full),
        KVCacheGroupSpec(["L1"], swa),
    ]
    md0 = KeyMetadata("m", 0, 0, 0, 0, group_id=0)
    md1 = KeyMetadata("m", 0, 0, 0, 0, group_id=1)
    db_full = ChunkedTokenDatabase(md0, block_size=16, hash_block_size=16)
    db_swa = ChunkedTokenDatabase(md1, block_size=16, hash_block_size=16)
    db_full.set_kv_caches_base_addr([0])
    db_full.set_block_len([1024])
    db_swa.set_kv_caches_base_addr([1 << 20])
    db_swa.set_block_len([1024])

    requested_keys: list[str] = []

    class _CapturingStore:
        def batch_get_into_multi_buffers(self, keys, addrs, sizes):
            requested_keys.extend(keys)
            return [0] * len(keys)

    ready = threading.Event()
    coord = MooncakeStoreCoordinator(
        groups, scheduler_block_size=16, hash_block_size=16
    )
    recv = KVCacheStoreRecvingThread(
        store=_CapturingStore(),
        token_databases=[db_full, db_swa],
        block_size=16,
        tp_rank=0,
        ready_event=ready,
        coord=coord,
    )

    hs = [BlockHash(bytes([i + 1]) * 4) for i in range(4)]
    req = ReqMeta(
        req_id="r0",
        token_len_chunk=64,
        block_ids=([0, 1, 2, 3], [0, 1, 2, 3]),
        block_hashes=hs,
        load_spec=LoadSpec(
            vllm_cached_tokens=0, kvpool_cached_tokens=64, can_load=True, token_len=64
        ),
    )
    recv.request_queue.put(req)
    recv._handle_request(recv.request_queue.get())

    full_keys = [k for k in requested_keys if "@group:0" in k]
    swa_keys = [k for k in requested_keys if "@group:1" in k]
    # Full attention: load every block (4).
    assert len(full_keys) == 4
    # SWA: only the 2 tail-window blocks (hashes hs[2], hs[3]).
    assert len(swa_keys) == 2
    swa_hashes = {k.rsplit("@", 1)[-1] for k in swa_keys}
    assert swa_hashes == {hs[2].hex(), hs[3].hex()}


def test_chunked_token_database_hash_block_size_smaller_than_block_size():
    """DSv4-style: hash_block_size=4, group block_size=16 — process_tokens
    keys each chunk by its ending fine hash, including a partial tail."""
    md = KeyMetadata("m", 0, 0, 0, 0, group_id=3)
    db = ChunkedTokenDatabase(md, block_size=16, hash_block_size=4)
    db.set_kv_caches_base_addr([0])
    db.set_block_len([512])
    fine_hashes = [BlockHash(bytes([i + 1]) * 4) for i in range(8)]

    # 8 fine-grained hashes (32 tokens at hash_block_size=4) → 2 group chunks.
    out = list(db.process_tokens(token_len=32, block_hashes=fine_hashes))
    assert len(out) == 2
    assert out[0][0] == 0 and out[0][1] == 16
    assert out[1][0] == 16 and out[1][1] == 32
    # Each chunk's hash is its last (4th) fine hash, which already chains the
    # prior three.
    assert out[0][2].hex() == fine_hashes[3].hex()
    assert out[1][2].hex() == fine_hashes[7].hex()

    # Sub-block hit: emit the partial chunk under its ending fine hash.
    out = list(db.process_tokens(token_len=12, block_hashes=fine_hashes[:3]))
    assert [(s, e) for s, e, _ in out] == [(0, 12)]
    assert out[0][2].hex() == fine_hashes[2].hex()

    # Cross-block hit: emit both the full chunk and its partial tail.
    out = list(db.process_tokens(token_len=28, block_hashes=fine_hashes[:7]))
    assert [(s, e) for s, e, _ in out] == [(0, 16), (16, 28)]
    assert out[0][2].hex() == fine_hashes[3].hex()
    assert out[1][2].hex() == fine_hashes[6].hex()


def test_sub_block_partial_tail_offload_reads_cow_block():
    """Sub-block prompt (the 900/128/1536 shape, scaled to 12/4/16): the
    partial tail is offloaded for both groups under the boundary sub-hash. The
    full-attention block is read from the request block table; the mamba block
    is the core-provided CoW target, not block_ids."""
    full = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=None)
    mamba = MambaSpec(
        block_size=16,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    groups = [
        KVCacheGroupSpec(["L0"], full),
        KVCacheGroupSpec(["L1"], mamba),
    ]
    coord = MooncakeStoreCoordinator(groups, scheduler_block_size=16, hash_block_size=4)
    assert coord.enable_partial_hash_hits

    class _RecordingStore(_DictStore):
        def __init__(self):
            super().__init__()
            self.puts: dict[str, list[int]] = {}

        def batch_put_from_multi_buffers(self, keys, addrs, sizes, *a, **k):
            for key, addr in zip(keys, addrs):
                self.puts[key] = addr
            return super().batch_put_from_multi_buffers(keys, addrs, sizes, *a, **k)

    store = _RecordingStore()
    token_dbs = []
    for g_idx in range(2):
        db = ChunkedTokenDatabase(
            KeyMetadata("m", 0, 0, 0, 0, group_id=g_idx),
            block_size=16,
            hash_block_size=4,
        )
        db.set_kv_caches_base_addr([g_idx * 10_000])
        db.set_block_len([512])
        token_dbs.append(db)

    send = KVCacheStoreSendingThread(
        store=store,
        coord=coord,
        token_databases=token_dbs,
        block_size=16,
        tp_rank=0,
        group_put_steps=[1, 1],
        kv_role="kv_both",
        ready_event=threading.Event(),
        replicate_config=MagicMock(),
    )

    # The surrounding metadata may describe a longer resumed replay, but the
    # handoff identifies the exact state boundary to persist.
    hs = [BlockHash(bytes([i + 1]) * 4) for i in range(5)]
    mamba_cow_block = 7
    req = ReqMeta(
        req_id="r0",
        token_len_chunk=0,
        block_ids=([1], [2]),
        block_hashes=hs,
        can_save=True,
        num_prompt_tokens=20,
        boundary_state_offloads=[(1, mamba_cow_block, 12)],
    )

    send._maybe_offload_boundary_states(req)

    # boundary = 12 // 4 * 4 = 12 -> keyed by hs[12 // 4 - 1] = hs[2].
    partial_hash = hs[2]
    fa_key = token_dbs[0].key_for(partial_hash)
    mamba_key = token_dbs[1].key_for(partial_hash)
    assert set(store.puts) == {fa_key, mamba_key}
    # FA reads block_ids[0][0] = block 1: addr = base(0) + 1 * 512.
    assert store.puts[fa_key] == [512]
    # Mamba reads the CoW block 7, not block_ids[1][0]=2.
    assert store.puts[mamba_key] == [10_000 + mamba_cow_block * 512]


def test_offload_syncs_event_before_put():
    """An offload-carrying meta synchronizes its CoW-fence event before the
    store put reads the blocks, then completes in one pass and drains the
    completion counter."""
    full = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=None)
    mamba = MambaSpec(
        block_size=16,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    groups = [
        KVCacheGroupSpec(["L0"], full),
        KVCacheGroupSpec(["L1"], mamba),
    ]
    coord = MooncakeStoreCoordinator(groups, scheduler_block_size=16, hash_block_size=4)
    event = MagicMock()

    class _FencedStore(_DictStore):
        def batch_put_from_multi_buffers(self, keys, addrs, sizes, *a, **k):
            assert event.synchronize.called, "put must run after the event sync"
            return super().batch_put_from_multi_buffers(keys, addrs, sizes, *a, **k)

    store = _FencedStore()
    token_dbs = []
    for g_idx in range(2):
        db = ChunkedTokenDatabase(
            KeyMetadata("m", 0, 0, 0, 0, group_id=g_idx),
            block_size=16,
            hash_block_size=4,
        )
        db.set_kv_caches_base_addr([g_idx * 10_000])
        db.set_block_len([512])
        token_dbs.append(db)

    send = KVCacheStoreSendingThread(
        store=store,
        coord=coord,
        token_databases=token_dbs,
        block_size=16,
        tp_rank=0,
        group_put_steps=[1, 1],
        kv_role="kv_both",
        ready_event=threading.Event(),
        replicate_config=MagicMock(),
    )

    hs = [BlockHash(bytes([i + 1]) * 4) for i in range(3)]
    req = ReqMeta(
        req_id="r1",
        token_len_chunk=0,
        block_ids=([1], [2]),
        block_hashes=hs,
        can_save=True,
        num_prompt_tokens=12,
        store_job_id=1,
        boundary_state_offloads=[(1, 7, 12)],
    )
    req.current_event = event

    send.add_request(req)
    send._handle_request(send.request_queue.get())
    assert send.request_queue.qsize() == 0
    assert store._data
    assert send.stored_requests["r1"] == set()
    event.synchronize.assert_called_once()


def test_sub_block_partial_tail_offload_covers_smaller_group_blocks():
    """The K3-shaped 900/128/1536 scenario scaled to 12/4/16, with a
    full-attention group whose block (4) is smaller than the lcm (16): the
    offload must persist every FA block up to the boundary — the normal save
    floors to the lcm, so those blocks are otherwise never written and the
    consumer's per-group lookup would miss. The mamba boundary block still
    reads the core-provided CoW target."""
    full = FullAttentionSpec(block_size=4, num_kv_heads=8, head_size=64, dtype=None)
    mamba = MambaSpec(
        block_size=16,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    groups = [
        KVCacheGroupSpec(["L0"], full),
        KVCacheGroupSpec(["L1"], mamba),
    ]
    coord = MooncakeStoreCoordinator(groups, scheduler_block_size=16, hash_block_size=4)
    assert coord.enable_partial_hash_hits

    class _RecordingStore(_DictStore):
        def __init__(self):
            super().__init__()
            self.puts: dict[str, list[int]] = {}

        def batch_put_from_multi_buffers(self, keys, addrs, sizes, *a, **k):
            for key, addr in zip(keys, addrs):
                self.puts[key] = addr
            return super().batch_put_from_multi_buffers(keys, addrs, sizes, *a, **k)

    store = _RecordingStore()
    token_dbs = []
    for g_idx, block_size in enumerate([4, 16]):
        db = ChunkedTokenDatabase(
            KeyMetadata("m", 0, 0, 0, 0, group_id=g_idx),
            block_size=block_size,
            hash_block_size=4,
        )
        db.set_kv_caches_base_addr([g_idx * 10_000])
        db.set_block_len([512])
        token_dbs.append(db)

    send = KVCacheStoreSendingThread(
        store=store,
        coord=coord,
        token_databases=token_dbs,
        block_size=16,
        tp_rank=0,
        group_put_steps=[1, 1],
        kv_role="kv_both",
        ready_event=threading.Event(),
        replicate_config=MagicMock(),
    )

    hs = [BlockHash(bytes([i + 1]) * 4) for i in range(3)]  # 3 hash units = 12 tok
    mamba_cow_block = 7
    req = ReqMeta(
        req_id="r2",
        token_len_chunk=0,
        block_ids=([1, 2, 3], [4]),
        block_hashes=hs,
        can_save=True,
        num_prompt_tokens=12,
        boundary_state_offloads=[(1, mamba_cow_block, 12)],
    )

    send._maybe_offload_boundary_states(req)

    # FA (block 4): full blocks ending at 4, 8 and 12, keyed by their normal
    # block-end hashes; mamba (block 16): the partial boundary block under
    # the boundary sub-hash, read from the CoW target.
    expected = {
        token_dbs[0].key_for(hs[0]): [1 * 512],
        token_dbs[0].key_for(hs[1]): [2 * 512],
        token_dbs[0].key_for(hs[2]): [3 * 512],
        token_dbs[1].key_for(hs[2]): [10_000 + mamba_cow_block * 512],
    }
    assert store.puts == expected


def test_worker_lookup_hits_sub_block_partial_tail():
    """worker.lookup must query sub-block keys when partial hash hits are on.

    Regression test: ``lookup`` hard-coded ``fine_grained = False``, so the
    sub-block keys persisted by ``_sub_block_tail_puts`` were never probed and
    partial prefix hits silently returned 0 while the store side kept writing
    them. Here the mamba block (16) exceeds the hash unit (4), so
    ``enable_partial_hash_hits`` is on and the lookup must find the stored
    boundary at 12.
    """
    full = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=None)
    mamba = MambaSpec(
        block_size=16,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    cfg = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[
            KVCacheTensor(
                size=4 * full.page_size_bytes,
                layers=["L0"],
                layer_stride=4 * full.page_size_bytes,
                block_stride=full.page_size_bytes,
            ),
            KVCacheTensor(
                size=4 * mamba.page_size_bytes,
                layers=["L1"],
                layer_stride=4 * mamba.page_size_bytes,
                block_stride=mamba.page_size_bytes,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["L0"], full),
            KVCacheGroupSpec(["L1"], mamba),
        ],
    )
    vllm_config = _minimal_vllm_config(cache_block_size=16)
    # Hash unit 4 < mamba block 16 -> partial hash hits are enabled.
    vllm_config.cache_config.prefix_match_unit = 4
    store = _DictStore()

    worker = _build_worker_with_dict_store(vllm_config, cfg, store)
    worker.tp_size = 1
    worker.pp_size = 1
    worker.num_kv_head = 8
    assert worker.coord.enable_partial_hash_hits

    for g_idx, db in enumerate(worker.token_dbs):
        db.set_kv_caches_base_addr([g_idx * 10_000])
        db.set_block_len([512])

    send_thread = KVCacheStoreSendingThread(
        store=store,
        token_databases=worker.token_dbs,
        block_size=worker.block_size,
        coord=worker.coord,
        tp_rank=0,
        group_put_steps=[1, 1],
        kv_role="kv_both",
        ready_event=threading.Event(),
        replicate_config=MagicMock(),
    )

    # Persist the sub-block partial tail at boundary 12 (keyed by hs[12//4-1]).
    hs = [BlockHash(bytes([i + 1]) * 4) for i in range(5)]
    req = ReqMeta(
        req_id="r0",
        token_len_chunk=0,
        block_ids=([1], [2]),
        block_hashes=hs,
        can_save=True,
        num_prompt_tokens=20,
        boundary_state_offloads=[(1, 7, 12)],
    )
    send_thread._maybe_offload_boundary_states(req)

    worker.store = store

    # A 13-token prompt sharing the prefix must hit the stored boundary at 12.
    assert worker.lookup(num_tokens=13, block_hashes=hs).hit_length == 12


def test_worker_setup_tolerates_finer_scratch_group():
    """Setup and lookup must tolerate a non-prefix-cacheable scratch group.

    Regression: GLM-5.3-Flash carries a kpool-tail scratch group whose block
    size (``index_kpool`` tokens) is finer than the hash unit, so the
    coordinator's divisibility assert and the scratch group's
    ``ChunkedTokenDatabase`` both rejected worker setup for any hash unit the
    scratch block does not divide. Scratch groups never participate in
    store/load/lookup, so setup must skip them and lookup must still hit
    stored sub-block boundaries on the participating groups.
    """
    full = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=None)
    mamba = MambaSpec(
        block_size=16,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    # Scratch block 4 is not divisible by the hash unit 8 below.
    scratch = KpoolTailSpec(
        block_size=4,
        num_kv_heads=2,
        head_size=64,
        head_size_v=0,
        dtype=torch.bfloat16,
        sliding_window=4,
    )
    cfg = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[
            KVCacheTensor(
                size=4 * full.page_size_bytes,
                layers=["L0"],
                layer_stride=4 * full.page_size_bytes,
                block_stride=full.page_size_bytes,
            ),
            KVCacheTensor(
                size=4 * mamba.page_size_bytes,
                layers=["L1"],
                layer_stride=4 * mamba.page_size_bytes,
                block_stride=mamba.page_size_bytes,
            ),
            KVCacheTensor(
                size=4 * scratch.page_size_bytes,
                layers=["L2"],
                layer_stride=4 * scratch.page_size_bytes,
                block_stride=scratch.page_size_bytes,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["L0"], full),
            KVCacheGroupSpec(["L1"], mamba),
            KVCacheGroupSpec(["L2"], scratch),
        ],
    )
    vllm_config = _minimal_vllm_config(cache_block_size=16)
    # Hash unit 8 divides the participating groups (16) but not the scratch
    # group (4); mamba block 16 > 8 keeps partial hash hits on.
    vllm_config.cache_config.prefix_match_unit = 8
    store = _DictStore()

    worker = _build_worker_with_dict_store(vllm_config, cfg, store)
    worker.tp_size = 1
    worker.pp_size = 1
    worker.num_kv_head = 8
    assert worker.coord.enable_partial_hash_hits
    # Only the participating full-attention and Mamba groups are registered.
    assert len(worker.token_dbs) == 2
    assert all(db.hash_block_size == 8 for db in worker.token_dbs)

    for g_idx, db in enumerate(worker.token_dbs):
        db.set_kv_caches_base_addr([g_idx * 10_000])
        db.set_block_len([512])

    send_thread = KVCacheStoreSendingThread(
        store=store,
        token_databases=worker.token_dbs,
        block_size=worker.block_size,
        coord=worker.coord,
        tp_rank=0,
        group_put_steps=[1, 1],
        kv_role="kv_both",
        ready_event=threading.Event(),
        replicate_config=MagicMock(),
        group_participates=[True, True],
    )

    # Persist the sub-block partial tail at boundary 12 (keyed by hs[12//8-1]).
    hs = [BlockHash(bytes([i + 1]) * 8) for i in range(3)]
    req = ReqMeta(
        req_id="r0",
        token_len_chunk=0,
        block_ids=([1], [2]),
        block_hashes=hs,
        can_save=True,
        num_prompt_tokens=20,
        boundary_state_offloads=[(1, 7, 12)],
    )
    send_thread._maybe_offload_boundary_states(req)

    worker.store = store

    # A 13-token prompt sharing the prefix must hit the first hash unit.
    assert worker.lookup(num_tokens=13, block_hashes=hs).hit_length == 8
    # The scratch group's namespace never enters the store.
    assert not any("@group:2" in key for key in store._data)


def test_ring_scratch_group_is_never_stored_and_does_not_block_hits():
    """A per-request ring group (CircularBufferSpec, capacity as block_size)
    sits beside the paged group: worker setup tolerates its block size (the
    DeepSeek-V4.1 compressor ring is 8 rows), saving a request stores paged
    blocks only, and lookup hits on the paged group alone."""
    full = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=None)
    ring = CircularBufferSpec(
        block_size=8, num_kv_heads=1, head_size=64, head_size_v=0, dtype=torch.uint8
    )
    cfg = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[
            KVCacheTensor(
                size=4 * full.page_size_bytes,
                layers=["L0"],
                layer_stride=4 * full.page_size_bytes,
                block_stride=full.page_size_bytes,
            ),
            KVCacheTensor(
                size=4 * ring.page_size_bytes,
                layers=["L1"],
                layer_stride=4 * ring.page_size_bytes,
                block_stride=ring.page_size_bytes,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["L0"], full),
            KVCacheGroupSpec(["L1"], ring),
        ],
    )
    vllm_config = _minimal_vllm_config(cache_block_size=16)
    store = _DictStore()

    worker = _build_worker_with_dict_store(vllm_config, cfg, store)
    worker.tp_size = 1
    worker.pp_size = 1
    worker.num_kv_head = 8
    assert len(worker.token_dbs) == 1
    assert worker.token_dbs[0].hash_block_size == 16

    raw_full = torch.zeros(4 * full.page_size_bytes, dtype=torch.int8)
    raw_ring = torch.zeros(4 * ring.page_size_bytes, dtype=torch.int8)
    kv_caches = {
        "L0": dense_kv_cache_views(raw_full, full, 4, 1, KVCacheLayout.LBNHC)[0],
        "L1": dense_kv_cache_views(raw_ring, ring, 4, 1, KVCacheLayout.LBNHC)[0],
    }

    def _fake_thread_init(*args, **kwargs):
        for v in list(args) + list(kwargs.values()):
            if isinstance(v, threading.Event):
                v.set()
        m = MagicMock()
        m.start = lambda: None
        return m

    with (
        patch.object(
            mooncake_store_worker,
            "KVCacheStoreSendingThread",
            side_effect=_fake_thread_init,
        ),
        patch.object(
            mooncake_store_worker,
            "KVCacheStoreRecvingThread",
            side_effect=_fake_thread_init,
        ),
    ):
        worker.register_kv_caches(kv_caches)

    send_thread = KVCacheStoreSendingThread(
        store=store,
        token_databases=worker.token_dbs,
        block_size=worker.block_size,
        coord=worker.coord,
        tp_rank=worker.tp_rank,
        group_put_steps=worker._group_tp_replication_factors,
        kv_role=worker.kv_role,
        ready_event=threading.Event(),
        enable_kv_event=False,
        group_participates=[True],
    )
    hs = [BlockHash(bytes([i + 1]) * 4) for i in range(4)]
    save_req = ReqMeta(
        req_id="r0",
        token_len_chunk=64,
        # Scheduler metadata is projected to the participating store groups.
        block_ids=([1, 2, 3, 4],),
        block_hashes=hs,
        can_save=True,
        store_job_id=1,
    )
    send_thread.add_request(save_req)
    send_thread._handle_request(send_thread.request_queue.get())
    worker.store = store

    assert worker.lookup(num_tokens=65, block_hashes=hs).hit_length == 64
    assert not any("@group:1" in key for key in store._data)
