# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL wiring of the DCP row relayout (`DcpTransferMode.RELAYOUT`): staging
blocks reserved by the decode scheduler, the reads planned by the worker, the
caches the row gather moves, and receives reported only once the row gather
completed."""

import queue
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.dcp_layout import DcpLayout
from vllm.distributed.kv_transfer.kv_connector.dcp_relayout import (
    RelayoutRegion,
    StagedRelayout,
    num_staging_blocks,
    plan_staged_relayout,
)
from vllm.distributed.kv_transfer.kv_connector.utils import TransferTopology
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
    RemoteMeta,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import (
    NixlPullConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    DcpTransferMode,
    compute_tp_mapping,
    engine_dcp_layout,
    relayout_config_problem,
    select_dcp_transfer_mode,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
    NixlConnectorWorker,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVRowLayout,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import KVConnectorOutput

from .dcp_relayout_reference import relayout_rows_reference
from .utils import (
    create_mock_connector,
    create_request,
    create_scheduler,
    create_vllm_config,
    expand_strided_descs,
)

pytestmark = pytest.mark.cpu_test

# Kimi K3.1 shapes: P TP8=DCP8 with 1280-token blocks, D TP16=DCP16 with 640.
P_LAYOUT = DcpLayout(8, 1, 1280)
D_LAYOUT = DcpLayout(16, 1, 640)


def _mla(block_size: int = 640, **kwargs) -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.uint8,
        **kwargs,
    )


def _mamba(block_size: int = 640) -> MambaSpec:
    return MambaSpec(block_size=block_size, shapes=((4, 3),), dtypes=(torch.float32,))


def _kv_cache_config(*groups: KVCacheGroupSpec) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=8, kv_cache_tensors=[], kv_cache_groups=list(groups)
    )


def _params(**overrides) -> dict:
    params = dict(
        do_remote_prefill=True,
        remote_block_ids=[[1, 2]],
        remote_engine_id="P",
        remote_request_id="p",
        remote_host="h",
        remote_port=1,
        tp_size=8,
        dcp_size=8,
        block_size=1280,
        cp_kv_cache_interleave_size=1,
    )
    params.update(overrides)
    return params


class _FakeBlockPool:
    def __init__(self):
        self.next_id = 100
        self.free_count = 0

    def get_new_blocks(self, n):
        blocks = [SimpleNamespace(block_id=self.next_id + i) for i in range(n)]
        self.next_id += n
        return blocks

    def free_blocks(self, blocks):
        self.free_count += len(list(blocks))


def _scheduler(
    layout: DcpLayout = D_LAYOUT, relayout_problem: str | None = None
) -> NixlPullConnectorScheduler:
    scheduler = object.__new__(NixlPullConnectorScheduler)
    scheduler._dcp_layout = layout
    scheduler.transfer_tp_size = layout.dcp_size
    scheduler._has_mamba = True
    scheduler._relayout_problem = relayout_problem
    scheduler._staging_needs = {}
    scheduler._staging = {}
    scheduler._heartbeat_by_engine = {}
    scheduler._heartbeat_req_engine = {}
    scheduler._block_pool = None
    scheduler.bind_gpu_block_pool(_FakeBlockPool())
    return scheduler


def _request(**params) -> SimpleNamespace:
    return SimpleNamespace(request_id="req", kv_transfer_params=_params(**params))


# ----------------------------------------------------------------------
# One DCP layout and one relayout predicate for the scheduler and workers
# ----------------------------------------------------------------------


def test_engine_dcp_layout_uses_the_attention_groups():
    """The per-rank logical block is the attention groups' (unwrapped from
    `UniformTypeKVCacheSpecs`); Mamba state and circular buffers, which are
    not placed by position, do not count."""
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=16, cp_kv_cache_interleave_size=1
        ),
        cache_config=SimpleNamespace(block_size=16),
    )
    attention = UniformTypeKVCacheSpecs(
        block_size=640, kv_cache_specs={"mla.0": _mla(), "idx.0": _mla()}
    )
    config = _kv_cache_config(
        KVCacheGroupSpec(["mla.0", "idx.0"], attention),
        KVCacheGroupSpec(["kda.1"], _mamba(block_size=1280)),
        KVCacheGroupSpec(
            ["ring.2"],
            CircularBufferSpec(
                block_size=96, num_kv_heads=1, head_size=8, dtype=torch.uint8
            ),
        ),
    )
    assert engine_dcp_layout(vllm_config, config) == DcpLayout(16, 1, 640)
    # Without attention groups the configured block stands in.
    state_only = _kv_cache_config(KVCacheGroupSpec(["kda.1"], _mamba()))
    assert engine_dcp_layout(vllm_config, state_only) == DcpLayout(16, 1, 16)


def test_scheduler_and_worker_derive_one_dcp_layout(monkeypatch):
    """The layout P's scheduler advertises in kv_transfer_params (which D's
    scheduler sizes staging from) is the one P's workers advertise at the
    handshake (which D's workers plan the reads with)."""
    from . import test_nixl_desc_geometry as geometry

    def hma_vllm_config(**kwargs):
        vllm_config = create_vllm_config(**kwargs)
        vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = False
        return vllm_config

    monkeypatch.setattr(geometry, "create_vllm_config", hma_vllm_config)
    worker = geometry._make_mla_hybrid_worker(
        local_block_size=12, kernel_block_size=4, num_logical_blocks=8
    )
    scheduler = NixlPullConnectorScheduler(
        worker.vllm_config, "local-engine", worker.kv_cache_config
    )
    assert scheduler._dcp_layout == worker._dcp_layout
    assert worker._dcp_layout.block_size == (
        worker.block_size * worker._physical_blocks_per_logical_kv_block
    )


@pytest.mark.parametrize(
    ("groups", "kwargs", "match"),
    [
        ([], dict(transfer_mode="push"), "push transfers"),
        ([], dict(use_host_buffer=True), "host buffer"),
        ([], dict(pcp_size=2), "prefill context parallelism"),
        (
            [KVCacheGroupSpec(["mla.0"], _mla(), host_resident=True)],
            dict(),
            "host-resident",
        ),
        (
            [
                KVCacheGroupSpec(["mla.0"], _mla()),
                KVCacheGroupSpec(["draft.0"], _mla(dcp_sharded=False)),
            ],
            dict(),
            "DCP-replicated",
        ),
        (
            [
                KVCacheGroupSpec(["mla.0"], _mla()),
                KVCacheGroupSpec(["swa.0"], _mla(block_size=64)),
            ],
            dict(),
            "exactly one attention KV cache group, got 2",
        ),
        ([KVCacheGroupSpec(["kda.0"], _mamba())], dict(), "got 0"),
    ],
)
def test_relayout_config_problems(groups, kwargs, match):
    groups = groups or [
        KVCacheGroupSpec(["mla.0"], _mla()),
        KVCacheGroupSpec(["kda.1"], _mamba()),
    ]
    options = dict(transfer_mode="pull", use_host_buffer=False, pcp_size=1)
    options.update(kwargs)
    problem = relayout_config_problem(_kv_cache_config(*groups), **options)
    assert problem is not None and match in problem


def test_relayout_config_of_kimi_k31():
    """One packed attention group plus KDA state groups receive relayouts."""
    config = _kv_cache_config(
        KVCacheGroupSpec(["mla.0", "idx.0"], _mla()),
        KVCacheGroupSpec(["kda.1"], _mamba()),
        KVCacheGroupSpec(["kda.2"], _mamba()),
    )
    assert (
        relayout_config_problem(
            config, transfer_mode="pull", use_host_buffer=False, pcp_size=1
        )
        is None
    )


def _reason_worker(**attrs) -> NixlConnectorWorker:
    worker = object.__new__(NixlConnectorWorker)
    worker.kv_cache_config = _kv_cache_config(
        KVCacheGroupSpec(["mla.0"], _mla()), KVCacheGroupSpec(["kda.1"], _mamba())
    )
    worker.use_host_buffer = False
    worker.pcp_size = 1
    worker._mixed_mem_types = False
    worker._get_relayout_buffers = lambda: []
    for name, value in attrs.items():
        setattr(worker, name, value)
    return worker


def _raise(message: str):
    def raiser():
        raise ValueError(message)

    return raiser


@pytest.mark.parametrize(
    ("attrs", "match"),
    [
        (dict(_TRANSFER_MODE="push"), "push transfers"),
        (dict(use_host_buffer=True), "host buffer"),
        (dict(pcp_size=2), "prefill context parallelism"),
        (dict(_mixed_mem_types=True), "host-resident KV caches"),
        (dict(_get_relayout_buffers=_raise("no token-row layout")), "no token-row"),
    ],
)
def test_worker_relayout_unsupported_reasons(attrs, match):
    """Each reason rejects the peer at its handshake, keeping no peer state."""
    assert _reason_worker()._relayout_unsupported_reason() is None
    worker = _reason_worker(**attrs)
    reason = worker._relayout_unsupported_reason()
    assert reason is not None and match in reason

    worker.engine_id = "D"
    worker._dcp_layout = D_LAYOUT
    worker.dcp_size = 16
    worker.transfer_tp_size = 16
    worker._remote_dcp_layouts = {}
    worker._kv_row_layouts = {}
    worker._kv_cache_scales = {}
    worker._layer_specs = {}
    worker._mamba_state_geometry = None
    meta = SimpleNamespace(
        engine_id="P",
        region_members=[],
        kv_row_layouts={},
        kv_cache_scales={},
        mamba_state_geometry=None,
        cp_kv_cache_interleave_size=1,
        block_size=64,
        physical_blocks_per_logical_kv_block=20,
    )
    with pytest.raises(NotImplementedError, match=f"cannot receive.*{match}"):
        worker._check_peer_compatibility(meta, remote_tp_size=8, remote_dcp_size=8)
    assert worker._remote_dcp_layouts == {}


# ----------------------------------------------------------------------
# Scheduler: staging blocks
# ----------------------------------------------------------------------


@pytest.mark.parametrize("failed", [False, True])
def test_staging_blocks_follow_the_receive(failed):
    """Counted at admission, taken with the request's blocks, handed to the
    workers with the positions they were counted for, returned when the
    receive finishes (or fails)."""
    scheduler = _scheduler()
    request = _request()
    need = scheduler.get_num_staging_blocks(request, 0, 130_000)
    assert need == num_staging_blocks(P_LAYOUT, D_LAYOUT, 0, 130_000) > 0

    scheduler._take_staging_blocks("req")
    pool = scheduler._block_pool
    assert scheduler._staging_meta("req") == (list(range(100, 100 + need)), 0, 130_000)
    meta = NixlConnectorMetadata()
    meta.add_new_req_to_recv(
        "req", [[1]], _params(), staging=scheduler._staging_meta("req")
    )
    req_meta = meta.reqs_to_recv["req"]
    assert req_meta.staging_block_ids == list(range(100, 100 + need))
    assert req_meta.relayout_tokens == (0, 130_000)
    assert req_meta.remote.block_size == 1280

    output = (
        KVConnectorOutput(failed_recving={"req"})
        if failed
        else KVConnectorOutput(finished_recving={"req"})
    )
    scheduler.update_connector_output(output)
    assert pool.free_count == need
    assert scheduler._staging == {}
    # Reported again (e.g. failed and finished): freed once.
    scheduler.update_connector_output(KVConnectorOutput(finished_recving={"req"}))
    assert pool.free_count == need


@pytest.mark.parametrize(
    ("params", "num_external", "problem"),
    [
        # Same layout on both sides: whole pages pair.
        (dict(tp_size=16, dcp_size=16, block_size=640), 1000, None),
        # Same DCP size, other blocks: rows pair, no copy.
        (dict(tp_size=16, dcp_size=16, block_size=1280), 1000, None),
        # Nothing to load.
        (dict(), 0, None),
        # Not a version 18 NIXL peer.
        (dict(block_size=None), 1000, None),
        # A peer no mode supports: its handshake rejects it.
        (dict(tp_size=16), 1000, None),
        # Malformed layout parameters.
        (dict(cp_kv_cache_interleave_size=3), 1000, None),
        # An engine that cannot receive a relayout (its handshake rejects it).
        (dict(), 1000, "push transfers write whole pages."),
    ],
)
def test_no_staging_without_relayout(params, num_external, problem):
    scheduler = _scheduler(relayout_problem=problem)
    assert scheduler.get_num_staging_blocks(_request(**params), 0, num_external) == 0
    scheduler._take_staging_blocks("req")
    assert scheduler._staging_needs == {} and scheduler._staging == {}


def test_staging_for_the_reverse_direction():
    scheduler = _scheduler(P_LAYOUT)
    request = _request(tp_size=16, dcp_size=16, block_size=640)
    assert scheduler.get_num_staging_blocks(request, 10240, 50_000) == (
        num_staging_blocks(D_LAYOUT, P_LAYOUT, 10240, 60_240)
    )


def _lifecycle_scheduler() -> NixlPullConnectorScheduler:
    scheduler = _scheduler()
    scheduler._reqs_in_batch = set()
    scheduler._reqs_need_save = {}
    scheduler._reqs_need_recv = {}
    scheduler._reqs_need_send = {}
    scheduler._reqs_not_processed = set()
    scheduler.use_host_buffer = False
    scheduler.is_bidirectional_kv_xfer_enabled = False
    scheduler._is_hma_required = False
    scheduler.kv_cache_config = MagicMock(
        select_transfer_block_ids=lambda block_ids: block_ids
    )
    return scheduler


def _blocks(*block_ids):
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

    return KVCacheBlocks(blocks=([KVCacheBlock(block_id=b) for b in block_ids],))


def test_staging_is_taken_with_the_request_blocks():
    """`update_state_after_alloc` takes the counted staging, and the
    request's metadata carries it to the workers."""
    scheduler = _lifecycle_scheduler()
    request = create_request(do_remote_prefill=True)
    request.kv_transfer_params.update(_params())
    need = scheduler.get_num_staging_blocks(request, 0, 3000)
    assert need > 0

    scheduler.update_state_after_alloc(request, _blocks(1, 2), num_external_tokens=3000)
    assert scheduler._staging_needs == {}
    blocks, start, end = scheduler._staging_meta(request.request_id)
    assert len(blocks) == need and (start, end) == (0, 3000)
    meta = scheduler.build_connector_meta(MagicMock())
    req_meta = meta.reqs_to_recv[request.request_id]
    assert req_meta.relayout_tokens == (0, 3000)
    assert req_meta.staging_block_ids == blocks


def test_aborted_receive_keeps_staging_until_it_completes():
    """An abort while the READs or the row gather run frees the request's
    blocks only once the workers report the receive (`_aborted_kv_loads`);
    its staging, which they still write, is returned then too."""
    scheduler = _lifecycle_scheduler()
    request = create_request(do_remote_prefill=True)
    request.kv_transfer_params.update(_params())
    need = scheduler.get_num_staging_blocks(request, 0, 3000)
    scheduler.update_state_after_alloc(request, _blocks(1, 2), num_external_tokens=3000)
    scheduler.request_finished(request, [[1, 2]])
    assert request.request_id in scheduler._staging
    assert scheduler._block_pool.free_count == 0
    scheduler.update_connector_output(
        KVConnectorOutput(finished_recving={request.request_id})
    )
    assert scheduler._staging == {}
    assert scheduler._block_pool.free_count == need


def test_untaken_staging_count_is_dropped():
    """A count whose request never got its blocks (re-counted at the next
    admission attempt, or aborted before allocation) takes nothing."""
    scheduler = _lifecycle_scheduler()
    request = create_request(do_remote_prefill=True)
    request.kv_transfer_params.update(_params())
    assert scheduler.get_num_staging_blocks(request, 0, 3000) > 0
    # The next admission attempt counts again; nothing accumulates.
    assert scheduler.get_num_staging_blocks(request, 0, 0) == 0
    assert scheduler._staging_needs == {}

    assert scheduler.get_num_staging_blocks(request, 0, 3000) > 0
    scheduler.request_finished(request, [])
    assert scheduler._staging_needs == {} and scheduler._staging == {}
    assert scheduler._block_pool.next_id == 100  # nothing taken


def test_admission_counts_staging_blocks():
    """The scheduler admits an asynchronous load only if the request's blocks
    and the connector's staging blocks both fit."""

    def admitted(num_staging: int) -> bool:
        scheduler = create_scheduler(create_vllm_config(), num_blocks=12)
        request = create_request(num_tokens=4 * scheduler.block_size)
        scheduler.add_request(request)
        connector = create_mock_connector()
        connector.get_num_new_matched_tokens.return_value = (
            2 * scheduler.block_size,
            True,
        )
        connector.get_num_staging_blocks.return_value = num_staging
        connector.get_loaded_kv_cache_group_ids.return_value = (0,)
        connector.take_events.return_value = ()
        scheduler.connector = connector
        scheduler.schedule()
        connector.get_num_staging_blocks.assert_called_with(
            request, 0, 2 * scheduler.block_size
        )
        return request.request_id in scheduler.requests and (
            request.status.name == "WAITING_FOR_REMOTE_KVS"
        )

    free = 12 - 1  # the null block
    assert admitted(0)
    assert admitted(2)
    assert not admitted(free)


def test_multi_connector_routes_staging_to_the_loading_connector():
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiConnector,
    )

    multi = object.__new__(MultiConnector)
    multi._connectors = [Mock(), Mock()]
    multi._connectors[1].get_num_staging_blocks.return_value = 7
    multi._requests_to_connector = {"loaded": 1}
    loaded = SimpleNamespace(request_id="loaded")
    assert multi.get_num_staging_blocks(loaded, 3, 5) == 7
    multi._connectors[1].get_num_staging_blocks.assert_called_once_with(loaded, 3, 5)
    assert multi.get_num_staging_blocks(SimpleNamespace(request_id="other"), 0, 5) == 0
    multi._connectors[0].get_num_staging_blocks.assert_not_called()


# ----------------------------------------------------------------------
# Worker: reads of a RELAYOUT request
# ----------------------------------------------------------------------

ATTN, SSM = 0, 1
KERNEL = 64


def _mapping(local: DcpLayout, remote: DcpLayout, tp_rank: int, remote_tp: int):
    topology = object.__new__(TransferTopology)
    topology.tp_rank = tp_rank
    topology.tp_size = local.dcp_size
    topology.is_mla = True
    topology.total_num_kv_heads = 1
    topology.dcp_size = local.dcp_size
    topology.is_mamba = True
    return compute_tp_mapping(
        transfer_topology=topology,
        remote_tp_size=remote_tp,
        group_spec_types=(MLAAttentionSpec, MambaSpec),
        remote_dcp_size=remote.dcp_size,
        dcp_mode=DcpTransferMode.RELAYOUT,
    )


def _read_worker(local: DcpLayout, remote: DcpLayout, rank: int) -> NixlConnectorWorker:
    worker = object.__new__(NixlConnectorWorker)
    worker._remote_dcp_layouts = {"P": remote}
    worker._dcp_layout = local
    worker.dcp_size = local.dcp_size
    worker.dcp_rank = rank % local.dcp_size
    # Kernel blocks of 64 rows on both sides: 64-row units.
    worker.block_size = KERNEL
    worker._group_spec_types = (MLAAttentionSpec, MambaSpec)
    worker.transfer_topo = MagicMock()
    worker._peer_relayouts = {
        "P": SimpleNamespace(unit_rows=KERNEL, src_kernel_block_size=KERNEL)
    }
    worker._recv_failures = set()
    worker._handle_failed_transfer = MagicMock()
    return worker


def _recv_meta(
    local: DcpLayout,
    remote: DcpLayout,
    hit: int,
    start: int,
    end: int,
    staging: list[int] | None,
) -> ReqMeta:
    tokens_per_block = local.block_size * local.dcp_size
    local_blocks = list(range(50 + hit, 50 + -(-end // tokens_per_block)))
    remote_blocks = list(
        range(500, 500 + -(-end // (remote.block_size * remote.dcp_size)))
    )
    return ReqMeta(
        local_block_ids=[local_blocks, [77]],
        local_physical_block_ids=[local_blocks, [77]],
        tp_size=remote.dcp_size,
        dcp_size=remote.dcp_size,
        local_num_computed_blocks=(hit, 0),
        remote=RemoteMeta(
            block_ids=[remote_blocks, [9]],
            host="h",
            port=1,
            engine_id="P",
            request_id="p",
            num_tokens=end,
            block_size=remote.block_size,
            cp_kv_cache_interleave_size=remote.interleave,
        ),
        staging_block_ids=list(staging or ()),
        relayout_tokens=None if staging is None else (start, end),
    )


@pytest.mark.parametrize(
    ("remote", "local", "rank"),
    [(P_LAYOUT, D_LAYOUT, 3), (P_LAYOUT, D_LAYOUT, 11), (D_LAYOUT, P_LAYOUT, 5)],
    ids=["p8_to_d16_rank3", "p8_to_d16_rank11", "p16_to_d8_rank5"],
)
@pytest.mark.parametrize("hit", [0, 1, 2])
@pytest.mark.parametrize("num_tokens", [5, 3000])
def test_relayout_reads_match_the_plan(remote, local, rank, hit, num_tokens):
    """Over both directions, prefix hits and requests short enough to leave
    ranks without rows: staging sized by the scheduler fits the worker's plan
    at the finer kernel-block units, the attention reads are the planned ones
    from mapped attention sources, and state slots come 1:1 from the state
    sources."""
    start = hit * local.block_size * local.dcp_size
    end = start + num_tokens
    scheduler = _scheduler(local)
    request = _request(
        tp_size=remote.dcp_size, dcp_size=remote.dcp_size, block_size=remote.block_size
    )
    scheduler.get_num_staging_blocks(request, start, num_tokens)
    scheduler._take_staging_blocks("req")
    staging, s_start, s_end = scheduler._staging_meta("req")
    assert (s_start, s_end) == (start, end)

    worker = _read_worker(local, remote, rank)
    plan = _mapping(local, remote, rank, remote.dcp_size)
    meta = _recv_meta(local, remote, hit, start, end, staging)
    specs = worker._relayout_read_specs("req", meta, plan)
    assert specs is not None, worker._handle_failed_transfer.call_args
    expected = plan_staged_relayout(
        remote,
        local,
        rank % local.dcp_size,
        start,
        end,
        unit_rows=KERNEL,
        src_kernel_block_size=KERNEL,
        src_block_ids=meta.remote.block_ids[ATTN],
        dst_block_ids=meta.local_block_ids[ATTN],
        staging_block_ids=staging,
        dst_kernel_block_size=KERNEL,
    )
    assert np.array_equal(meta.relayout.dst_rows, expected.dst_rows)
    assert np.array_equal(meta.relayout.src_slots, expected.src_slots)
    assert np.array_equal(meta.relayout.staging_units, expected.staging_units)
    assert set(expected.reads) <= set(plan.source_ranks_per_group[ATTN])
    by_rank = {spec.remote_rank: spec for spec in specs}
    assert set(by_rank) == set(plan.all_source_ranks)
    for source, spec in by_rank.items():
        units = expected.reads.get(source, ([], []))
        assert (spec.remote_block_ids[ATTN], spec.local_block_ids[ATTN]) == units
        if source in plan.source_ranks_per_group[SSM]:
            assert spec.local_block_ids[SSM] == [77]
            assert spec.remote_block_ids[SSM] == [9]
        else:
            assert spec.local_block_ids[SSM] == spec.remote_block_ids[SSM] == []


def test_relayout_reads_from_a_replica_without_dcp():
    """A remote without DCP (TP8, DCP1) holds every row on every rank: the
    reads of its one DCP rank go to the replica the mapping chose."""
    remote = DcpLayout(1, 1, 1280)
    local = P_LAYOUT
    rank = 5
    scheduler = _scheduler(local)
    scheduler.transfer_tp_size = 8
    request = _request(tp_size=8, dcp_size=1)
    scheduler.get_num_staging_blocks(request, 0, 3000)
    scheduler._take_staging_blocks("req")
    staging, _, _ = scheduler._staging_meta("req")
    worker = _read_worker(local, remote, rank)
    plan = _mapping(local, remote, rank, 8)
    assert plan.source_ranks_per_group[ATTN] == (rank,)
    meta = _recv_meta(local, remote, 0, 0, 3000, staging)
    specs = worker._relayout_read_specs("req", meta, plan)
    assert specs is not None
    (spec,) = specs
    assert spec.remote_rank == rank and spec.local_block_ids[ATTN]


@pytest.mark.parametrize("problem", ["no_staging", "short_staging", "misaligned"])
def test_unplannable_relayout_fails_the_request(problem):
    """A receive whose relayout cannot be planned fails (and recomputes);
    it never raises on the worker."""
    worker = _read_worker(D_LAYOUT, P_LAYOUT, 3)
    plan = _mapping(D_LAYOUT, P_LAYOUT, 3, 8)
    staging = {"no_staging": None, "short_staging": [90], "misaligned": [90] * 9}
    meta = _recv_meta(D_LAYOUT, P_LAYOUT, 0, 0, 30_000, staging[problem])
    if problem == "misaligned":
        meta.relayout_tokens = (640 * 16, 30_000)
    assert worker._relayout_read_specs("req", meta, plan) is None
    worker._handle_failed_transfer.assert_called_once()
    assert meta.relayout is None


# ----------------------------------------------------------------------
# Worker: the caches the row gather moves, and the read units
# ----------------------------------------------------------------------

MLA_ROWS = KVRowLayout.token_major(16)
INDEXER_ROWS = KVRowLayout(4, ((0, 8), (32, 4)))
LAYER_ROWS = {"mla.0": MLA_ROWS, "idx.0": INDEXER_ROWS, "draft.0": MLA_ROWS}


def _gather_worker(rows: int = 8, padding: int = 24, **attrs) -> NixlConnectorWorker:
    """A packed padded row of MLA, indexer and drafter MLA pages plus a KDA
    state page per block, registered as one region per layer view."""
    worker = object.__new__(NixlConnectorWorker)
    worker.engine_id = "D"
    worker.tp_rank = 0
    worker.block_size = rows
    page_bytes = [rows * LAYER_ROWS[name].bytes_per_row for name in LAYER_ROWS]
    state_bytes = 40
    stride = sum(page_bytes) + state_bytes + padding
    num_blocks = 6
    storage = torch.zeros(num_blocks * stride, dtype=torch.uint8)
    worker._test_storage = storage
    names = [*LAYER_ROWS, "kda.1"]
    offsets = [0, page_bytes[0], sum(page_bytes[:2]), sum(page_bytes)]
    lens = [*page_bytes, state_bytes]
    views = [
        storage.as_strided((num_blocks, n), (stride, 1), offset)
        for offset, n in zip(offsets, lens)
    ]
    worker.region_members = [[name] for name in names]
    worker.region_names = names
    worker.device_kv_caches = dict(zip(names, views))
    worker.kv_caches_base_addr = {"D": {0: [v.data_ptr() for v in views]}}
    worker.block_len_per_layer = lens
    worker.block_stride_per_layer = [stride] * len(names)
    worker._ssm_region_indices = [3]
    worker._layer_specs = {
        **{name: _mla(block_size=rows) for name in LAYER_ROWS},
        "kda.1": _mamba(block_size=rows),
    }
    worker._kv_row_layouts = dict(LAYER_ROWS)
    worker._relayout_buffers = None
    worker._peer_relayouts = {}
    for name, value in attrs.items():
        setattr(worker, name, value)
    worker._test_layout = (offsets, stride, num_blocks)
    return worker


def test_row_gathers_address_the_packed_layer_pages():
    worker = _gather_worker()
    offsets, stride, _ = worker._test_layout
    assert worker._relayout_regions() == [0, 1, 2]
    (buffer,) = worker._get_relayout_buffers()
    assert buffer.region_ids == [0, 1, 2]
    assert buffer.regions == [
        RelayoutRegion(offsets[i], stride, 8, LAYER_ROWS[name])
        for i, name in enumerate(LAYER_ROWS)
    ]
    assert worker._get_relayout_buffers()[0] is buffer  # resolved once

    # Moving row 5 of block 1 (staged whole, one 8-row unit) to row 2 of
    # block 4 moves exactly each layer's row bytes, and nothing else (not the
    # state page or the padding).
    buf = worker._test_storage
    buf.copy_(torch.randint(0, 256, buf.shape, dtype=torch.uint8))
    before = buf.clone()
    one = StagedRelayout(
        {}, np.array([0]), np.array([5]), np.array([4]), np.array([2]), np.array([1])
    )
    relayout_rows_reference(buf, buffer.regions, list(LAYER_ROWS.values()), 8, one)
    changed = torch.zeros_like(buf, dtype=torch.bool)
    for i, rows in enumerate(LAYER_ROWS.values()):
        for (src_off, size), (dst_off, _) in zip(
            rows.row_byte_ranges(5), rows.row_byte_ranges(2)
        ):
            src = stride + offsets[i] + src_off
            dst = 4 * stride + offsets[i] + dst_off
            assert torch.equal(buf[dst : dst + size], before[src : src + size])
            changed[dst : dst + size] = True
    assert torch.equal(buf[~changed], before[~changed])


@pytest.mark.parametrize(
    ("attrs", "match"),
    [
        # A region whose layers store their rows differently.
        (
            dict(
                region_members=[["mla.0", "idx.0"], ["idx.0"], ["draft.0"], ["kda.1"]]
            ),
            "row layouts",
        ),
        # A layer without a declared row layout (e.g. on ROCm).
        (
            dict(_kv_row_layouts={"mla.0": MLA_ROWS, "draft.0": MLA_ROWS}),
            "token-row layout",
        ),
        # A page that is not block_size rows of its layout.
        (dict(block_len_per_layer=[100, 96, 128, 40]), "100 bytes"),
    ],
)
def test_row_gathers_reject_undeclared_pages(attrs, match):
    worker = _gather_worker(**attrs)
    with pytest.raises(ValueError, match=match):
        worker._get_relayout_buffers()


def _peer_meta_rows(**layouts) -> SimpleNamespace:
    return SimpleNamespace(engine_id="P", kv_row_layouts={**LAYER_ROWS, **layouts})


def test_peer_relayout_reads_the_peers_page_layout():
    """A peer whose indexer pages are twice as long keeps its scales twice
    as far: its gather reads them there (8-row kernel blocks staged as two
    4-row units) and writes them where this engine keeps them."""
    worker = _gather_worker(rows=4)
    peer_indexer = KVRowLayout(8, ((0, 8), (64, 4)))
    peer = worker._peer_relayout(
        _peer_meta_rows(**{"idx.0": peer_indexer}),
        unit_rows=4,
        src_kernel_block_size=8,
    )
    assert (peer.unit_rows, peer.src_kernel_block_size) == (4, 8)
    (gather,) = peer.gathers
    assert worker._get_relayout_buffers()[0].regions == gather.regions

    # Row 6 of a peer kernel block staged in blocks 1 and 3 (one 48-byte unit
    # each) to row 1 of block 5: its key from staged byte 6 * 8 = 48 (byte 0
    # of block 3), its scale from staged byte 64 + 6 * 4 = 88 (byte 40 of
    # block 3).
    buf = worker._test_storage
    buf.copy_(torch.randint(0, 256, buf.shape, dtype=torch.uint8))
    before = buf.clone()
    offsets, stride, _ = worker._test_layout
    job = StagedRelayout(
        {}, np.array([0]), np.array([6]), np.array([5]), np.array([1]), np.array([1, 3])
    )
    src_rows = [MLA_ROWS, peer_indexer, MLA_ROWS]
    relayout_rows_reference(buf, gather.regions, src_rows, 4, job)
    unit_bytes = 4 * INDEXER_ROWS.bytes_per_row
    key = 5 * stride + offsets[1] + 1 * 8
    scale = 5 * stride + offsets[1] + 4 * 8 + 1 * 4
    src_key = 3 * stride + offsets[1] + (6 * 8 - unit_bytes)
    src_scale = 3 * stride + offsets[1] + (64 + 6 * 4 - unit_bytes)
    assert torch.equal(buf[key : key + 8], before[src_key : src_key + 8])
    assert torch.equal(buf[scale : scale + 4], before[src_scale : src_scale + 4])


@pytest.mark.parametrize(
    ("layouts", "kernel", "match"),
    [
        # The peer does not declare a layer's rows.
        (dict(LAYER_ROWS, **{"idx.0": None}), 8, "lays out the rows"),
        # Kernel pages longer than the peer's kernel blocks.
        ({"idx.0": KVRowLayout(16, ((0, 8), (128, 4)))}, 8, "lays out the rows"),
        # Another KV format (row pieces).
        ({"idx.0": KVRowLayout(8, ((0, 4), (32, 8)))}, 8, "row pieces"),
    ],
)
def test_peer_relayout_rejects_unreadable_pages(layouts, kernel, match):
    worker = _gather_worker(rows=4)
    meta = _peer_meta_rows(**layouts)
    meta.kv_row_layouts = {k: v for k, v in meta.kv_row_layouts.items() if v}
    with pytest.raises(NotImplementedError, match=match):
        worker._peer_relayout(meta, unit_rows=4, src_kernel_block_size=kernel)


def test_relayout_peer_must_pair_every_gathered_region():
    worker = _gather_worker()
    pairs = SimpleNamespace(groups_match=True, local=[0, 1, 3])
    with pytest.raises(NotImplementedError, match=r"regions \[2\]"):
        worker._check_relayout_pairs("P", pairs)
    worker._check_relayout_pairs(
        "P", SimpleNamespace(groups_match=True, local=[0, 1, 2])
    )
    with pytest.raises(NotImplementedError, match="other KV cache groups"):
        worker._check_relayout_pairs(
            "P", SimpleNamespace(groups_match=False, local=[0, 1, 2])
        )


def _split_meta(block_size: int, block_lens: list[int], strides: list[int]):
    return SimpleNamespace(
        block_size=block_size,
        block_lens=list(block_lens),
        block_strides=list(strides),
        ple_block_len=None,
        physical_blocks_per_logical_kv_block=2,
        num_blocks=10,
        region_num_blocks=[10] * len(block_lens),
    )


def _split_worker(kernel_page_rows: int = 1) -> NixlConnectorWorker:
    worker = object.__new__(NixlConnectorWorker)
    worker.block_size = 64
    worker._kv_row_layouts = {"x": KVRowLayout(kernel_page_rows, ((0, 4),))}
    return worker


def test_split_padded_remote_blocks_keeps_their_stride():
    """Remote 96-row blocks inside a padded packed row split into 32-row
    units that local 64-row blocks hold whole; descriptor `block * split +
    sub` addresses the unit `sub` of block `block`."""
    meta = _split_meta(96, [96 * 4, 7], [1000, 1000])
    split = _split_worker()._split_remote_blocks(meta, required=True, state_regions={1})
    assert split == 3
    assert meta.block_size == 32 and meta.block_strides == [1000, 1000]
    assert meta.block_lens[0] == 128
    assert meta.physical_blocks_per_logical_kv_block == 6 and meta.num_blocks == 30
    descs = expand_strided_descs(
        NixlConnectorWorker._block_strided_descs(
            5000, meta.block_lens[0], meta.block_strides[0], 4, split, 0
        )
    )
    for block in range(4):
        for sub in range(split):
            addr = int(descs[block * split + sub, 0])
            assert addr == 5000 + block * 1000 + sub * meta.block_lens[0]


@pytest.mark.parametrize(
    ("kernel_page_rows", "block_lens", "state_regions", "match"),
    [
        # A token region that cannot be cut in three.
        (1, [100], set(), "cannot be split"),
        # The same odd length on a state-only region is fine (see above).
    ],
)
def test_split_remote_blocks_rejects_bad_units(
    kernel_page_rows, block_lens, state_regions, match
):
    worker = _split_worker(kernel_page_rows)
    meta = _split_meta(96, block_lens, block_lens)
    with pytest.raises(NotImplementedError, match=match):
        worker._split_remote_blocks(meta, required=True, state_regions=state_regions)
    # Without `required` (DIRECT hybrids) the blocks stay whole, unpaired.
    meta = _split_meta(96, block_lens, block_lens)
    assert worker._split_remote_blocks(meta) == 1
    assert meta.block_size == 96 and meta.block_lens == block_lens


def test_split_remote_blocks_keeps_kernel_pages_whole_only_for_direct_pairs():
    """32-row units split 64-row indexer kernel pages: pairing them directly
    would scramble their scales, but a relayout reads them in the peer's page
    layout (`RowGather`)."""
    worker = _split_worker(64)
    meta = _split_meta(96, [96 * 4], [96 * 4])
    assert worker._split_remote_blocks(meta) == 1
    assert worker._split_remote_blocks(meta, required=True) == 3


# ----------------------------------------------------------------------
# Worker: two-phase receive completion
# ----------------------------------------------------------------------


class _FakeEvent:
    def __init__(self):
        self.done = False

    def query(self):
        return self.done


def _job(n: int) -> StagedRelayout:
    rows = np.arange(n, dtype=np.int64)
    return StagedRelayout({}, rows, rows, rows + 100, rows, rows + 50)


def _completion_worker(event: _FakeEvent) -> NixlConnectorWorker:
    worker = object.__new__(NixlConnectorWorker)
    worker.transfer_topo = MagicMock()
    worker._relayouting = {}
    worker._recv_failures = set()
    worker._replicated_pcp_done_sending = set()
    worker._failed_recv_reqs = queue.Queue()
    worker._recving_metadata = {}
    worker._recving_transfers = {}
    worker._pending_recv_notifs = {}
    worker._is_hma_required = True
    worker.use_host_buffer = False
    worker.enable_heterogeneous_attn_post_process = False
    worker.tp_rank = 0
    worker.done = set()
    worker._get_new_notifs = lambda: set()
    worker._pop_done_transfers = lambda transfers: (set(worker.done), set())
    worker._send_pending_recv_notifs = MagicMock()
    worker._sync_device_after_direct_recv = MagicMock()
    worker._reap_expired_send_leases = MagicMock()
    worker._cleanup_remote_engines = MagicMock()
    worker._launch_relayouts = MagicMock(return_value=(event, "rows"))
    worker._peer_relayouts = {"P": "peer P"}
    return worker


def _relayout_meta(n: int) -> ReqMeta:
    meta = _recv_meta(D_LAYOUT, P_LAYOUT, 0, 0, 3000, [90, 91, 92])
    meta.relayout = _job(n)
    return meta


def test_receive_reported_only_after_the_relayout():
    event = _FakeEvent()
    worker = _completion_worker(event)
    worker._recving_metadata = {"a": _relayout_meta(3), "b": _relayout_meta(2)}
    worker.done = {"a", "b"}

    results = worker.get_transfer_results()
    # READs done: P is notified, the gathers run, the requests wait for them.
    assert results.finished_recving == set()
    assert worker._send_pending_recv_notifs.call_count == 2
    worker._launch_relayouts.assert_called_once()
    peer, job = worker._launch_relayouts.call_args.args
    assert peer == "peer P"
    assert len(job.src_slots) == len(job.dst_blocks) == 5
    # Each job's slots follow the staging units of the jobs merged before it.
    assert job.src_slots.tolist() == [0, 1, 2, 3, 4]
    assert sorted(job.staging_units.tolist()) == [50, 50, 51, 51, 52]
    # The row table stays referenced until the gather completes.
    assert worker._relayouting == {"a": (event, "rows"), "b": (event, "rows")}

    worker.done = set()
    assert worker.get_transfer_results().finished_recving == set()
    event.done = True
    assert worker.get_transfer_results().finished_recving == {"a", "b"}
    assert worker._relayouting == {}
    worker._launch_relayouts.assert_called_once()


def test_failed_receive_skips_the_relayout():
    event = _FakeEvent()
    worker = _completion_worker(event)
    worker._recving_metadata = {"a": _relayout_meta(3)}
    worker._recv_failures = {"a"}

    results = worker.get_transfer_results()
    assert results.finished_recving == {"a"}
    assert results.failed_recving == {"a"}
    worker._launch_relayouts.assert_not_called()
    assert worker._relayouting == {}


def test_relayout_mode_is_selected_per_peer():
    """The worker's handshake and the scheduler's staging use one selection."""
    assert (
        select_dcp_transfer_mode(D_LAYOUT, P_LAYOUT, tp_sizes=(16, 8))
        is DcpTransferMode.RELAYOUT
    )
