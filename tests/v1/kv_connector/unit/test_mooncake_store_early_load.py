# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mooncake store loads issued at schedule time (early loads).

With async scheduling and one worker per engine, the scheduler hands a load
to the worker over an IPC channel when it allocates the load's blocks, instead
of in the step's connector metadata, which the worker binds only after it
launched the previous step and then orders after that step's forward. These
tests pin the pieces that change relies on:

- BlockFreeClock: a block is "recent" when an unfenced free returned it since
  the last non-empty schedule() returned (it may be in use by the step in
  flight); fenced frees (deferred frees, frees of requests whose last step was
  processed) and frees while the in-flight step was scheduled are not; a packed
  slot is recent when its parent block is;
- the scheduler's fenced frees stamp FENCED;
- the Mooncake scheduler issues a load early only when no block it writes is
  recent, and then not again in the step's metadata; otherwise (or when the
  channel is not connected) the metadata carries it as before;
- the worker queues an early load at once, with no GPU-order wait;
- the channel round-trips a load-only ReqMeta;
- early loads are only enabled where the clock's argument holds.
"""

import queue
import shutil
import tempfile
import threading
import time
from types import SimpleNamespace

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
    scheduler as store_scheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
    worker as store_worker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    LoadSpec,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
    metrics,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.metrics import (
    MooncakeStoreConnectorStats,
)
from vllm.v1.core.block_pool import (
    BLOCK_FREE_CLOCK,
    BlockFreeClock,
    BlockPool,
    SlotBlockPool,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.request import RequestStatus

from .test_mooncake_store_worker import _make_bare_worker
from .utils import (
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)

pytestmark = pytest.mark.cpu_test

clock = BLOCK_FREE_CLOCK


# ---------------------------------------------------------------------------
# BlockFreeClock
# ---------------------------------------------------------------------------


def test_clock_marks_unfenced_frees_since_the_last_built_step_recent():
    pool = BlockPool(num_gpu_blocks=8, enable_caching=False, hash_block_size=16)
    a, b, c, d = pool.get_new_blocks(4)

    clock.tick()  # schedule() of step 1 starts
    pool.free_blocks([a])  # freed while step 1 is scheduled
    clock.step_built()  # step 1 may now be in flight
    pool.free_blocks([b])  # unfenced free after step 1 was built
    with clock.fenced():
        pool.free_blocks([c])  # its last GPU user has completed
    clock.tick()  # schedule() of step 2 starts (step 1 in flight)
    pool.free_blocks([d])  # freed while step 2 is scheduled

    assert c.free_epoch == BlockFreeClock.FENCED
    assert [clock.freed_recently(x) for x in (a, b, c, d)] == [
        False,
        True,
        False,
        True,
    ]

    clock.step_built()  # step 2 built; step 1 has completed by step 3
    clock.tick()
    assert not any(clock.freed_recently(x) for x in (a, b, c, d))


def test_clock_empty_schedule_does_not_end_the_window():
    pool = BlockPool(num_gpu_blocks=4, enable_caching=False, hash_block_size=16)
    (a,) = pool.get_new_blocks(1)
    clock.tick()
    clock.step_built()  # step 1 in flight
    pool.free_blocks([a])
    clock.tick()  # an empty schedule(): no step built
    clock.tick()  # the next schedule(); step 1 may still be in flight
    assert clock.freed_recently(a)


def test_clock_marks_a_slot_recent_when_its_parent_block_is():
    parent = BlockPool(num_gpu_blocks=4, enable_caching=False, hash_block_size=16)
    slots = SlotBlockPool(parent, slots_per_block=2)
    (row,) = parent.get_new_blocks(1)
    clock.tick()
    clock.step_built()
    parent.free_blocks([row])  # the parent block, freed under the step in flight
    clock.tick()
    (slot,) = slots.get_new_blocks(1)  # takes that block as a new row
    assert slots.row_block(slot.block_id) is row
    assert slot.free_epoch == BlockFreeClock.FENCED  # the slot itself: never freed
    assert clock.freed_recently(slot)


# ---------------------------------------------------------------------------
# Scheduler: fenced frees
# ---------------------------------------------------------------------------


def test_scheduler_frees_of_processed_or_deferred_requests_are_fenced():
    vllm_config = create_vllm_config(kv_role="kv_consumer")
    vllm_config.scheduler_config.async_scheduling = True
    scheduler = create_scheduler(vllm_config, num_blocks=64)
    assert scheduler.defer_block_free
    req = create_request(num_tokens=48, max_tokens=1)
    scheduler.add_request(req)
    out = scheduler.schedule()
    blocks = list(scheduler.kv_cache_manager.get_blocks(req.request_id).blocks[0])
    assert blocks
    # Aborted while its step is in flight: freed once that step is processed.
    scheduler.finish_requests(req.request_id, RequestStatus.FINISHED_ABORTED)
    assert all(b.ref_cnt == 1 for b in blocks)
    scheduler.update_from_output(out, create_model_runner_output([]))
    assert all(b.ref_cnt == 0 for b in blocks)
    assert all(b.free_epoch == BlockFreeClock.FENCED for b in blocks)
    assert not any(clock.freed_recently(b) for b in blocks)


# ---------------------------------------------------------------------------
# Mooncake scheduler: early issue
# ---------------------------------------------------------------------------


class _Channel:
    def __init__(self, connected: bool = True):
        self.connected = connected
        self.sent: list[ReqMeta] = []

    def send(self, req_meta: ReqMeta) -> bool:
        if self.connected:
            self.sent.append(req_meta)
        return self.connected


def _store_scheduler(channel: _Channel, vllm_cached_tokens: int = 16):
    sched = object.__new__(store_scheduler.MooncakeStoreScheduler)
    sched._early_loads = channel
    sched._store_group_ids = (0,)
    sched._group_block_sizes = [16]
    sched._block_size = 16
    sched._request_trackers = {}
    sched.num_early_loads = 0
    sched.num_metadata_loads = 0
    sched.load_specs = {
        "r": LoadSpec(
            vllm_cached_tokens=vllm_cached_tokens,
            kvpool_cached_tokens=64,
            can_load=True,
        )
    }
    request = SimpleNamespace(
        block_hashes=[bytes([i]) * 4 for i in range(5)], num_prompt_tokens=80
    )
    sched._unfinished_requests = {"r": (request, ([1, 2, 3, 4, 5],))}
    return sched


def _blocks(recent_index: int | None) -> KVCacheBlocks:
    blocks = [KVCacheBlock(i + 1) for i in range(5)]
    for block in blocks:
        block.ref_cnt = 1
    clock.tick()
    clock.step_built()
    if recent_index is not None:
        blocks[recent_index].free_epoch = clock.epoch
    clock.tick()
    return KVCacheBlocks((blocks,))


def test_load_into_unused_blocks_is_issued_at_allocation_not_in_metadata():
    channel = _Channel()
    sched = _store_scheduler(channel)
    sched._issue_load_early("r", _blocks(recent_index=None))

    (req_meta,) = channel.sent
    assert req_meta.req_id == "r"
    assert req_meta.load_spec is not None
    assert req_meta.load_spec.kvpool_cached_tokens == 64
    assert req_meta.block_ids == ([1, 2, 3, 4, 5],)
    assert sched.num_early_loads == 1
    assert "r" in sched._request_trackers
    # The step's metadata no longer carries it.
    assert "r" not in sched.load_specs
    assert sched._load_only_req_meta("r") is None


def test_local_hit_blocks_are_not_checked():
    # Block 0 holds the local hit (vllm_cached_tokens 16): the load does not
    # write it, so its recent free does not keep the load in the metadata.
    channel = _Channel()
    sched = _store_scheduler(channel)
    sched._issue_load_early("r", _blocks(recent_index=0))
    assert len(channel.sent) == 1


def test_load_into_a_recently_freed_block_stays_in_metadata():
    channel = _Channel()
    sched = _store_scheduler(channel)
    sched._issue_load_early("r", _blocks(recent_index=3))
    assert channel.sent == []
    assert sched.num_early_loads == 0
    req_meta = sched._load_only_req_meta("r")
    assert req_meta is not None and req_meta.load_spec is not None


def test_unconnected_channel_leaves_the_load_in_metadata():
    sched = _store_scheduler(_Channel(connected=False))
    sched._issue_load_early("r", _blocks(recent_index=None))
    assert sched.num_early_loads == 0
    assert sched._load_only_req_meta("r") is not None


# ---------------------------------------------------------------------------
# Worker: queue at once, no GPU-order wait
# ---------------------------------------------------------------------------


def _load_only_meta(req_id: str = "r") -> ReqMeta:
    return ReqMeta(
        req_id=req_id,
        token_len_chunk=64,
        block_ids=([1, 2, 3, 4],),
        block_hashes=[bytes([i]) * 4 for i in range(4)],
        can_save=False,
        load_spec=LoadSpec(
            vllm_cached_tokens=0, kvpool_cached_tokens=64, can_load=True
        ),
    )


def test_worker_queues_an_early_load_without_a_ready_event():
    worker = _make_bare_worker(kv_role="kv_both")
    worker._kv_connector_stats_lock = threading.Lock()
    worker.kv_connector_stats = MooncakeStoreConnectorStats()
    req_meta = _load_only_meta()

    worker._issue_early_load(req_meta)

    assert worker.recv_request_queue.get_nowait() is req_meta
    assert req_meta.load_ready_event is None
    assert req_meta.load_issue_time is not None
    assert req_meta.load_spec is not None and req_meta.load_spec.token_len == 64
    counts = worker.kv_connector_stats.data[metrics._EVENT_COUNTS]
    assert counts == {"load_issued_early": 1}


# ---------------------------------------------------------------------------
# Channel
# ---------------------------------------------------------------------------


def _channel_config(dp_rank: int = 0):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={}),
        parallel_config=SimpleNamespace(data_parallel_index=dp_rank),
    )


@pytest.fixture
def short_rpc_dir(monkeypatch):
    # IPC socket paths are limited to 107 bytes; pytest's tmp_path is too long.
    path = tempfile.mkdtemp(prefix="el", dir="/tmp")
    monkeypatch.setenv("VLLM_RPC_BASE_PATH", path)
    yield path
    shutil.rmtree(path, ignore_errors=True)


def test_channel_round_trips_a_load_only_req_meta(short_rpc_dir):
    received: queue.Queue[ReqMeta] = queue.Queue()
    server = store_worker.EarlyLoadServer(_channel_config(), received.put)
    client = store_worker.EarlyLoadClient(_channel_config())
    try:
        sent = _load_only_meta()
        deadline = time.monotonic() + 10
        while not client.send(sent):  # the IPC connect is asynchronous
            assert time.monotonic() < deadline
            time.sleep(0.01)
        got = received.get(timeout=10)
        assert got.req_id == sent.req_id
        assert got.block_ids == sent.block_ids
        assert got.block_hashes == sent.block_hashes
        assert got.load_spec == sent.load_spec
    finally:
        client.close()
        server.close()


def test_channel_send_fails_while_the_worker_is_not_listening(short_rpc_dir):
    client = store_worker.EarlyLoadClient(_channel_config())
    try:
        assert client.send(_load_only_meta()) is False
    finally:
        client.close()


def test_channel_paths_are_per_engine_and_differ_from_lookup(short_rpc_dir):
    paths = {store_worker.get_zmq_path_early_load(_channel_config(dp)) for dp in (0, 1)}
    assert len(paths) == 2
    assert store_worker.get_zmq_path_early_load(
        _channel_config()
    ) != store_worker.get_zmq_rpc_path_lookup(_channel_config())


# ---------------------------------------------------------------------------
# Where early loads are enabled
# ---------------------------------------------------------------------------


def _support_config(
    *,
    load_async=True,
    consumer=True,
    world_size=1,
    pp=1,
    max_concurrent_batches=2,
):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={"load_async": load_async},
            is_kv_consumer=consumer,
        ),
        parallel_config=SimpleNamespace(
            world_size=world_size, pipeline_parallel_size=pp
        ),
        max_concurrent_batches=max_concurrent_batches,
    )


@pytest.mark.parametrize(
    "kw, supported",
    [
        ({}, True),
        ({"load_async": False}, False),
        # Producers do not defer block frees.
        ({"consumer": False}, False),
        # The scheduler addresses a single worker.
        ({"world_size": 2}, False),
        ({"pp": 2}, False),
        # Synchronous scheduling, or more than one step in flight.
        ({"max_concurrent_batches": 1}, False),
        ({"max_concurrent_batches": 3}, False),
    ],
)
def test_early_loads_only_where_the_clock_argument_holds(kw, supported):
    assert store_worker.early_load_supported(_support_config(**kw)) is supported
