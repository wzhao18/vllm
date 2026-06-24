# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    LoadSpec,
    MooncakeStoreConnectorMetadata,
    MooncakeStoreWorkerMetadata,
    ReqMeta,
    RequestTracker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler import (
    MooncakeStoreScheduler,
)
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
from vllm.v1.outputs import KVConnectorOutput


def _make_bare_scheduler() -> MooncakeStoreScheduler:
    scheduler = object.__new__(MooncakeStoreScheduler)
    scheduler.kv_role = "kv_both"
    scheduler.store_policy = "write_through"
    scheduler.write_back_max_blocks_per_step = 64
    scheduler.lookup_async = False
    scheduler._block_size = 16
    scheduler.load_specs = {}
    scheduler._preempted_req_ids = set()
    scheduler._unfinished_request_ids = {"req-0"}
    scheduler._unfinished_requests = {}
    scheduler._request_trackers = {}
    scheduler._expected_worker_count = 1
    scheduler._gpu_block_pool = None
    scheduler._write_back_event_counter = 0
    scheduler._write_back_events = {}
    scheduler._write_back_pending_counts = {}
    scheduler._write_back_inflight_block_ids = set()
    scheduler._write_back_completed_blocks = {}
    scheduler._pending_write_back_stores = []
    return scheduler


def _set_block_hash(block, block_hash, *, num_tokens: int = 16) -> None:
    block.set_block_hash(block_hash, num_tokens=num_tokens)


def _make_scheduler_output(*, scheduled_spec_tokens: list[int] | None):
    return SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["req-0"],
            new_block_ids=[([2],)],
            num_computed_tokens=[44],
        ),
        num_scheduled_tokens={"req-0": 4},
        scheduled_spec_decode_tokens=(
            {"req-0": scheduled_spec_tokens} if scheduled_spec_tokens else {}
        ),
    )


def _make_preemption_scheduler_output():
    return SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids={"req-0"},
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[],
            new_block_ids=[],
            num_computed_tokens=[],
        ),
        num_scheduled_tokens={},
        scheduled_spec_decode_tokens={},
    )


def _add_unfinished_request(
    scheduler: MooncakeStoreScheduler,
    *,
    token_ids: list[int],
    block_hashes: list[bytes],
    prefill_end_tokens: int,
) -> None:
    request = SimpleNamespace(
        all_token_ids=token_ids,
        block_hashes=block_hashes,
        num_output_placeholders=0,
    )
    scheduler._unfinished_requests["req-0"] = (request, ([0, 1],))
    scheduler._request_trackers["req-0"] = RequestTracker(
        req_id="req-0",
        token_len=44,
        allocated_block_ids=([0, 1],),
        num_saved_tokens=32,
        token_ids=token_ids[:44],
        prefill_end_tokens=prefill_end_tokens,
    )


def test_cached_request_with_spec_decode_does_not_save_scheduled_drafts():
    # Drafts in scheduled_spec_decode_tokens are not appended to all_token_ids
    # yet, so the tracker's token_len does not advance and num_tokens_to_save
    # stays below chunk_boundary — the save is naturally skipped.
    scheduler = _make_bare_scheduler()
    _add_unfinished_request(
        scheduler,
        token_ids=list(range(44)),
        block_hashes=[b"h0", b"h1"],
        prefill_end_tokens=48,
    )

    meta = scheduler.build_connector_meta(
        _make_scheduler_output(scheduled_spec_tokens=[101, 102, 103])
    )

    assert meta.requests == []
    tracker = scheduler._request_trackers["req-0"]
    assert tracker.token_len == 44
    assert tracker.num_saved_tokens == 32
    assert tracker.allocated_block_ids == ([0, 1, 2],)


def test_cached_request_without_spec_decode_keeps_current_step_save_overlap():
    scheduler = _make_bare_scheduler()
    _add_unfinished_request(
        scheduler,
        token_ids=list(range(48)),
        block_hashes=[b"h0", b"h1", b"h2"],
        prefill_end_tokens=48,
    )

    meta = scheduler.build_connector_meta(
        _make_scheduler_output(scheduled_spec_tokens=None)
    )

    assert len(meta.requests) == 1
    req_meta = meta.requests[0]
    assert req_meta.req_id == "req-0"
    assert req_meta.can_save is True
    assert req_meta.token_len_chunk == 48
    tracker = scheduler._request_trackers["req-0"]
    assert tracker.token_len == 48
    assert tracker.num_saved_tokens == 48


def test_write_back_policy_skips_request_save_and_finish_delay():
    scheduler = _make_bare_scheduler()
    scheduler.store_policy = "write_back"
    _add_unfinished_request(
        scheduler,
        token_ids=list(range(48)),
        block_hashes=[b"h0", b"h1", b"h2"],
        prefill_end_tokens=48,
    )

    meta = scheduler.build_connector_meta(
        _make_scheduler_output(scheduled_spec_tokens=None)
    )

    assert meta.requests == []
    assert meta.write_back_stores == []
    tracker = scheduler._request_trackers["req-0"]
    assert tracker.token_len == 48
    assert tracker.num_saved_tokens == 32
    request = SimpleNamespace(request_id="req-0")
    assert scheduler.request_finished(request, ([0, 1, 2],)) == (False, None)


def test_write_back_policy_does_not_scan_free_queue_on_metadata_build():
    scheduler = _make_bare_scheduler()
    scheduler.store_policy = "write_back"
    gpu_block_pool = BlockPool(
        num_gpu_blocks=4,
        enable_caching=True,
        hash_block_size=16,
    )
    scheduler.bind_gpu_block_pool(gpu_block_pool)
    _set_block_hash(
        gpu_block_pool.blocks[1],
        make_block_hash_with_group_id(BlockHash(b"a0"), 0),
    )

    meta = MooncakeStoreConnectorMetadata(set(), set())
    scheduler._prepare_write_back_event(meta)

    assert meta.write_back_stores == []
    assert gpu_block_pool.blocks[1].ref_cnt == 0


def test_write_back_event_pins_and_releases_gpu_blocks():
    scheduler = _make_bare_scheduler()
    scheduler.store_policy = "write_back"
    scheduler.write_back_max_blocks_per_step = 1
    gpu_block_pool = BlockPool(
        num_gpu_blocks=4,
        enable_caching=True,
        hash_block_size=16,
    )
    scheduler.bind_gpu_block_pool(gpu_block_pool)
    block_hash = make_block_hash_with_group_id(BlockHash(b"a0"), 0)
    _set_block_hash(gpu_block_pool.blocks[1], block_hash)

    assert scheduler.write_back_blocks_before_allocate(1) is True
    meta = MooncakeStoreConnectorMetadata(set(), set())
    scheduler._prepare_write_back_event(meta)

    assert len(meta.write_back_stores) == 1
    store_meta = meta.write_back_stores[0]
    assert store_meta.event_id == 0
    assert store_meta.block_ids == [1]
    assert store_meta.block_hashes == [block_hash]
    assert gpu_block_pool.blocks[1].ref_cnt == 1
    assert 1 in scheduler._write_back_inflight_block_ids

    scheduler.update_connector_output(
        KVConnectorOutput(kv_connector_worker_meta=MooncakeStoreWorkerMetadata({0: 1}))
    )

    assert gpu_block_pool.blocks[1].ref_cnt == 0
    assert gpu_block_pool.free_block_queue.get_all_free_blocks()[0].block_id == 1
    assert 1 not in scheduler._write_back_inflight_block_ids
    assert scheduler._write_back_events == {}
    assert scheduler._write_back_completed_blocks[1] == {block_hash}


def test_completed_write_back_returns_to_allocator_reuse_order():
    scheduler = _make_bare_scheduler()
    scheduler.store_policy = "write_back"
    scheduler.write_back_max_blocks_per_step = 1
    gpu_block_pool = BlockPool(
        num_gpu_blocks=5,
        enable_caching=True,
        hash_block_size=16,
    )
    scheduler.bind_gpu_block_pool(gpu_block_pool)
    block_hashes = [
        make_block_hash_with_group_id(BlockHash(b"a0"), 0),
        make_block_hash_with_group_id(BlockHash(b"a1"), 0),
        make_block_hash_with_group_id(BlockHash(b"a2"), 0),
    ]
    for block_id, block_hash in zip((1, 2, 3), block_hashes, strict=True):
        _set_block_hash(gpu_block_pool.blocks[block_id], block_hash)

    assert scheduler.write_back_blocks_before_allocate(1) is True
    first_meta = MooncakeStoreConnectorMetadata(set(), set())
    scheduler._prepare_write_back_event(first_meta)
    assert first_meta.write_back_stores[0].block_ids == [1]

    scheduler.update_connector_output(
        KVConnectorOutput(kv_connector_worker_meta=MooncakeStoreWorkerMetadata({0: 1}))
    )

    assert gpu_block_pool.free_block_queue.get_all_free_blocks()[0].block_id == 1
    assert scheduler.write_back_blocks_before_allocate(1) is False


def test_write_back_uses_one_event_for_allocator_free_prefix():
    scheduler = _make_bare_scheduler()
    scheduler.store_policy = "write_back"
    scheduler.write_back_max_blocks_per_step = 1
    gpu_block_pool = BlockPool(
        num_gpu_blocks=5,
        enable_caching=True,
        hash_block_size=16,
    )
    scheduler.bind_gpu_block_pool(gpu_block_pool)
    block_1_hash = make_block_hash_with_group_id(BlockHash(b"a0"), 0)
    block_3_hash = make_block_hash_with_group_id(BlockHash(b"a2"), 0)
    _set_block_hash(gpu_block_pool.blocks[1], block_1_hash)
    _set_block_hash(gpu_block_pool.blocks[3], block_3_hash)

    assert scheduler.write_back_blocks_before_allocate(3) is True
    meta = MooncakeStoreConnectorMetadata(set(), set())
    scheduler._prepare_write_back_event(meta)

    assert len(meta.write_back_stores) == 1
    assert meta.write_back_stores[0].block_ids == [1, 3]
    assert meta.write_back_stores[0].block_hashes == [block_1_hash, block_3_hash]
    assert gpu_block_pool.blocks[1].ref_cnt == 1
    assert gpu_block_pool.blocks[2].ref_cnt == 0
    assert gpu_block_pool.blocks[3].ref_cnt == 1


def test_write_back_waits_for_single_inflight_event():
    scheduler = _make_bare_scheduler()
    scheduler.store_policy = "write_back"
    gpu_block_pool = BlockPool(
        num_gpu_blocks=5,
        enable_caching=True,
        hash_block_size=16,
    )
    scheduler.bind_gpu_block_pool(gpu_block_pool)
    _set_block_hash(
        gpu_block_pool.blocks[1],
        make_block_hash_with_group_id(BlockHash(b"a0"), 0),
    )
    _set_block_hash(
        gpu_block_pool.blocks[2],
        make_block_hash_with_group_id(BlockHash(b"a1"), 0),
    )

    assert scheduler.write_back_blocks_before_allocate(1) is True
    meta = MooncakeStoreConnectorMetadata(set(), set())
    scheduler._prepare_write_back_event(meta)

    assert scheduler.write_back_blocks_before_allocate(2) is True
    assert len(scheduler._write_back_events) == 1
    assert scheduler._write_back_event_counter == 1


def test_write_back_includes_all_hash_aliases_for_reused_block():
    scheduler = _make_bare_scheduler()
    scheduler.store_policy = "write_back"
    scheduler.write_back_max_blocks_per_step = 1
    gpu_block_pool = BlockPool(
        num_gpu_blocks=4,
        enable_caching=True,
        hash_block_size=16,
    )
    scheduler.bind_gpu_block_pool(gpu_block_pool)
    primary_hash = make_block_hash_with_group_id(BlockHash(b"a0"), 0)
    alias_hash = make_block_hash_with_group_id(BlockHash(b"a1"), 1)
    _set_block_hash(gpu_block_pool.blocks[1], primary_hash)
    gpu_block_pool.cached_block_hashes_by_block[1] = {alias_hash}

    assert scheduler.write_back_blocks_before_allocate(1) is True
    meta = MooncakeStoreConnectorMetadata(set(), set())
    scheduler._prepare_write_back_event(meta)

    assert meta.write_back_stores[0].block_ids == [1, 1]
    assert meta.write_back_stores[0].block_hashes == [primary_hash, alias_hash]
    assert gpu_block_pool.blocks[1].ref_cnt == 1

    scheduler.update_connector_output(
        KVConnectorOutput(kv_connector_worker_meta=MooncakeStoreWorkerMetadata({0: 1}))
    )

    assert gpu_block_pool.blocks[1].ref_cnt == 0
    assert gpu_block_pool.free_block_queue.get_all_free_blocks()[0].block_id == 1
    assert scheduler._write_back_completed_blocks[1] == {
        primary_hash,
        alias_hash,
    }
    assert scheduler.write_back_blocks_before_allocate(1) is False


def test_write_back_skips_completed_same_hash_until_reused_with_new_hash():
    scheduler = _make_bare_scheduler()
    scheduler.store_policy = "write_back"
    gpu_block_pool = BlockPool(
        num_gpu_blocks=4,
        enable_caching=True,
        hash_block_size=16,
    )
    scheduler.bind_gpu_block_pool(gpu_block_pool)
    old_hash = make_block_hash_with_group_id(BlockHash(b"a0"), 0)
    _set_block_hash(gpu_block_pool.blocks[1], old_hash)

    assert scheduler.write_back_blocks_before_allocate(1) is True
    meta = MooncakeStoreConnectorMetadata(set(), set())
    scheduler._prepare_write_back_event(meta)
    scheduler.update_connector_output(
        KVConnectorOutput(kv_connector_worker_meta=MooncakeStoreWorkerMetadata({0: 1}))
    )

    assert scheduler.write_back_blocks_before_allocate(3) is False

    new_hash = make_block_hash_with_group_id(BlockHash(b"a1"), 0)
    gpu_block_pool.blocks[1].reset_hash()
    _set_block_hash(gpu_block_pool.blocks[1], new_hash)

    assert scheduler.write_back_blocks_before_allocate(3) is True
    meta = MooncakeStoreConnectorMetadata(set(), set())
    scheduler._prepare_write_back_event(meta)
    assert meta.write_back_stores[0].block_ids == [1]
    assert meta.write_back_stores[0].block_hashes == [new_hash]


def test_preemption_resets_tracker_before_request_finished():
    scheduler = _make_bare_scheduler()
    _add_unfinished_request(
        scheduler,
        token_ids=list(range(44)),
        block_hashes=[b"h0", b"h1"],
        prefill_end_tokens=48,
    )

    scheduler.build_connector_meta(_make_preemption_scheduler_output())

    tracker = scheduler._request_trackers["req-0"]
    assert tracker.token_len == 0
    assert tracker.allocated_block_ids == ()
    assert tracker.num_saved_tokens == 0
    assert tracker.token_ids is None
    assert tracker.prefill_end_tokens == 0
    request = SimpleNamespace(request_id="req-0")
    assert scheduler.request_finished(request, ([0, 1],)) == (False, None)


def test_preemption_clears_stale_load_state():
    scheduler = _make_bare_scheduler()
    _make_pending_load_unfinished_request(
        scheduler,
        num_tokens=48,
        block_hashes=[b"h0", b"h1", b"h2"],
        block_ids=([10, 11, 12],),
    )
    scheduler.load_specs["req-0"] = LoadSpec(
        vllm_cached_tokens=0,
        kvpool_cached_tokens=48,
        can_load=True,
    )

    meta = scheduler.build_connector_meta(_make_preemption_scheduler_output())

    assert meta.requests == []
    assert "req-0" not in scheduler.load_specs
    assert "req-0" not in scheduler._unfinished_requests


def _make_pending_load_unfinished_request(
    scheduler: MooncakeStoreScheduler,
    *,
    num_tokens: int,
    block_hashes: list[bytes],
    block_ids: tuple[list[int], ...] = ([0, 1, 2],),
) -> None:
    request = SimpleNamespace(
        num_tokens=num_tokens,
        block_hashes=block_hashes,
        num_output_placeholders=0,
    )
    scheduler._unfinished_requests["req-0"] = (request, block_ids)


def _make_pending_load_scheduler_output() -> SimpleNamespace:
    """scheduler_output for a step where req-0 is parked on a pending load
    (not in scheduled_new_reqs or scheduled_cached_reqs)."""
    return SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[],
            new_block_ids=[],
            num_computed_tokens=[],
        ),
        num_scheduled_tokens={},
        scheduled_spec_decode_tokens={},
    )


def test_pending_load_does_not_co_queue_save():
    # Regression: a cache-hit request waiting on an async load must not also
    # enqueue a save in the same scheduling step. Co-queuing both produces a
    # recv+send pair for the same req_id, and the scheduler's
    # _update_from_kv_xfer_finished then trips `assert req_id in self.requests`
    # when both completions land for the delay-freed request.
    scheduler = _make_bare_scheduler()
    _make_pending_load_unfinished_request(
        scheduler,
        num_tokens=48,
        block_hashes=[b"h0", b"h1", b"h2"],
    )
    scheduler.load_specs["req-0"] = LoadSpec(
        vllm_cached_tokens=0,
        kvpool_cached_tokens=48,
        can_load=True,
    )

    meta = scheduler.build_connector_meta(_make_pending_load_scheduler_output())

    assert len(meta.requests) == 1
    req_meta = meta.requests[0]
    assert req_meta.req_id == "req-0"
    # Save must be off so the worker does not call add_stored_request.
    assert req_meta.can_save is False
    # Load is still issued as planned.
    assert req_meta.load_spec is not None
    assert req_meta.load_spec.can_load is True
    # And the tracker's saved-tokens watermark stays at 0 so request_finished
    # later sees `num_saved_tokens <= 0` and frees immediately rather than
    # waiting for a finished_sending that will never come.
    tracker = scheduler._request_trackers["req-0"]
    assert tracker.num_saved_tokens == 0


def _make_resumed_unfinished_request(
    scheduler: MooncakeStoreScheduler,
    *,
    token_ids: list[int],
    block_hashes: list[bytes],
    num_computed_tokens: int,
) -> None:
    request = SimpleNamespace(
        all_token_ids=token_ids,
        block_hashes=block_hashes,
        num_computed_tokens=num_computed_tokens,
        num_output_placeholders=0,
    )
    scheduler._unfinished_requests["req-0"] = (request, ([0, 1],))


def _make_resumed_scheduler_output(*, num_scheduled_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["req-0"],
            new_block_ids=[([2],)],
            num_computed_tokens=[0],
        ),
        num_scheduled_tokens={"req-0": num_scheduled_tokens},
        scheduled_spec_decode_tokens={},
    )


def test_resumed_from_preemption_with_load_skips_save():
    # On resume-from-preemption with a cache hit, the same co-queueing race
    # applies: the resumed-from-preemption branch in build_connector_meta also
    # passes load_spec.can_load=True. Skip save in this step; subsequent
    # cached_reqs steps will save new tokens normally.
    scheduler = _make_bare_scheduler()
    scheduler._preempted_req_ids = {"req-0"}
    _make_resumed_unfinished_request(
        scheduler,
        token_ids=list(range(48)),
        block_hashes=[b"h0", b"h1", b"h2"],
        num_computed_tokens=0,
    )
    scheduler.load_specs["req-0"] = LoadSpec(
        vllm_cached_tokens=0,
        kvpool_cached_tokens=48,
        can_load=True,
    )

    meta = scheduler.build_connector_meta(
        _make_resumed_scheduler_output(num_scheduled_tokens=48)
    )

    assert len(meta.requests) == 1
    req_meta = meta.requests[0]
    assert req_meta.req_id == "req-0"
    assert req_meta.can_save is False
    assert req_meta.load_spec is not None
    assert req_meta.load_spec.can_load is True
    tracker = scheduler._request_trackers["req-0"]
    assert tracker.num_saved_tokens == 0


def test_resumed_from_preemption_without_load_still_saves():
    # No load_spec → behavior is unchanged: save proceeds.
    scheduler = _make_bare_scheduler()
    scheduler._preempted_req_ids = {"req-0"}
    _make_resumed_unfinished_request(
        scheduler,
        token_ids=list(range(48)),
        block_hashes=[b"h0", b"h1", b"h2"],
        num_computed_tokens=0,
    )

    meta = scheduler.build_connector_meta(
        _make_resumed_scheduler_output(num_scheduled_tokens=48)
    )

    assert len(meta.requests) == 1
    req_meta = meta.requests[0]
    assert req_meta.req_id == "req-0"
    assert req_meta.can_save is True
    assert req_meta.load_spec is None
    tracker = scheduler._request_trackers["req-0"]
    assert tracker.num_saved_tokens == 48


# Focused tests for ReqMeta.from_request_tracker — the centralized guard that
# enforces "a ReqMeta never carries both a save and a load".


def test_from_request_tracker_load_overrides_caller_skip_save():
    # Caller asks for skip_save=False, but load_spec.can_load=True. The
    # function must force skip_save=True to avoid producing a ReqMeta the
    # worker would enqueue on both kv_send_thread and kv_recv_thread.
    tracker = RequestTracker(
        req_id="req-0",
        token_len=48,
        allocated_block_ids=([0, 1, 2],),
        num_saved_tokens=0,
    )
    load_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=48, can_load=True)

    req_meta = ReqMeta.from_request_tracker(
        tracker,
        block_size=16,
        load_spec=load_spec,
        skip_save=False,
        block_hashes=[b"h0", b"h1", b"h2"],
    )

    assert req_meta is not None
    assert req_meta.can_save is False
    assert req_meta.load_spec is load_spec
    assert tracker.num_saved_tokens == 0


def test_from_request_tracker_load_with_can_load_false_still_saves():
    # A LoadSpec with can_load=False (e.g., no external tokens to load after
    # update_state_after_alloc) must not suppress the save.
    tracker = RequestTracker(
        req_id="req-0",
        token_len=48,
        allocated_block_ids=([0, 1, 2],),
        num_saved_tokens=0,
    )
    load_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=48, can_load=False)

    req_meta = ReqMeta.from_request_tracker(
        tracker,
        block_size=16,
        load_spec=load_spec,
        skip_save=False,
        block_hashes=[b"h0", b"h1", b"h2"],
    )

    assert req_meta is not None
    assert req_meta.can_save is True
    # from_request_tracker clears load_spec when can_load is False.
    assert req_meta.load_spec is None
    assert tracker.num_saved_tokens == 48


def test_from_request_tracker_no_load_saves_normally():
    tracker = RequestTracker(
        req_id="req-0",
        token_len=48,
        allocated_block_ids=([0, 1, 2],),
        num_saved_tokens=0,
    )

    req_meta = ReqMeta.from_request_tracker(
        tracker,
        block_size=16,
        load_spec=None,
        skip_save=False,
        block_hashes=[b"h0", b"h1", b"h2"],
    )

    assert req_meta is not None
    assert req_meta.can_save is True
    assert req_meta.load_spec is None
    assert tracker.num_saved_tokens == 48


class _StubLookupClient:
    def __init__(self, hit_tokens: int) -> None:
        self._hit_tokens = hit_tokens
        self.local_hit_tokens: int | None = None

    def lookup(
        self,
        req_id: str,
        token_len: int,
        block_hashes: list[bytes],
        local_hit_tokens: int = 0,
        non_block: bool = False,
    ) -> int:
        self.local_hit_tokens = local_hit_tokens
        return self._hit_tokens


def test_full_external_hit_keeps_kvpool_cached_tokens_block_aligned():
    # When the external store hits the entire prompt, scheduler must leave at
    # least one token uncomputed for sampling but stay on a block boundary.
    # Otherwise the recv-side load mask floors token_len to
    # (num_tokens-1)//block_size, the tail partial chunk is dropped, and -- if
    # the local cache covers the aligned prefix -- key_list ends up empty
    # (ZeroDivisionError in the recv thread's `tp_rank % len(key_list)`).
    scheduler = _make_bare_scheduler()
    scheduler.load_async = True
    scheduler.client = _StubLookupClient(hit_tokens=48)  # full hit on 48-token prompt

    request = SimpleNamespace(
        request_id="req-0",
        num_tokens=48,
        block_hashes=[b"h0", b"h1", b"h2"],
    )

    need_to_allocate, load_async = scheduler.get_num_new_matched_tokens(
        request, num_computed_tokens=16
    )

    # 47 // 16 * 16 == 32 tokens left in external store after reserving the
    # sub-block tail for sampling. 32 - 16 (local) == 16 to load.
    assert need_to_allocate == 16
    assert load_async is True
    assert scheduler.client.local_hit_tokens == 16
    load_spec = scheduler.load_specs["req-0"]
    assert load_spec.vllm_cached_tokens == 16
    assert load_spec.kvpool_cached_tokens == 32
    assert load_spec.kvpool_cached_tokens % 16 == 0


def test_full_external_hit_with_full_local_hit_skips_load():
    # When local prefix cache already covers the block-aligned external hit,
    # there is nothing for the connector to load. The pre-fix behavior would
    # have scheduled a 15-token load that the recv thread couldn't translate
    # into any block-aligned key.
    scheduler = _make_bare_scheduler()
    scheduler.load_async = True
    scheduler.client = _StubLookupClient(hit_tokens=48)

    request = SimpleNamespace(
        request_id="req-0",
        num_tokens=48,
        block_hashes=[b"h0", b"h1", b"h2"],
    )

    need_to_allocate, load_async = scheduler.get_num_new_matched_tokens(
        request, num_computed_tokens=32
    )

    assert need_to_allocate == 0
    assert load_async is False
    assert "req-0" not in scheduler.load_specs
