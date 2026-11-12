# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL wiring of the DCP row relayout (`DcpTransferMode.RELAYOUT`): staging
blocks reserved by the decode scheduler, the reads planned by the worker, and
receives reported only once the row gather completed."""

import queue
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from vllm.distributed.kv_transfer.kv_connector.dcp_layout import DcpLayout
from vllm.distributed.kv_transfer.kv_connector.dcp_relayout import (
    StagedRelayout,
    num_staging_blocks,
    plan_staged_relayout,
)
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
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
    NixlConnectorWorker,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
from vllm.v1.outputs import KVConnectorOutput

pytestmark = pytest.mark.cpu_test

# Kimi K3.1 shapes: P TP8=DCP8 with 1280-token blocks, D TP16=DCP16 with 640.
P_LAYOUT = DcpLayout(8, 1, 1280)
D_LAYOUT = DcpLayout(16, 1, 640)


def _params(**overrides) -> dict:
    params = dict(
        do_remote_prefill=True,
        remote_block_ids=[[1, 2]],
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


def _scheduler(layout: DcpLayout = D_LAYOUT) -> NixlPullConnectorScheduler:
    scheduler = object.__new__(NixlPullConnectorScheduler)
    scheduler._dcp_layout = layout
    scheduler.transfer_tp_size = layout.dcp_size
    scheduler._has_mamba = True
    scheduler._staging_needs = {}
    scheduler._staging_blocks = {}
    scheduler._heartbeat_by_engine = {}
    scheduler._heartbeat_req_engine = {}
    scheduler.bind_gpu_block_pool(_FakeBlockPool())
    return scheduler


def _request(**params) -> SimpleNamespace:
    return SimpleNamespace(request_id="req", kv_transfer_params=_params(**params))


@pytest.mark.parametrize("failed", [False, True])
def test_staging_blocks_follow_the_receive(failed):
    """Counted at admission, taken with the request's blocks, returned when
    the receive finishes (or fails)."""
    scheduler = _scheduler()
    request = _request()
    need = scheduler.get_num_staging_blocks(request, 0, 130_000)
    assert need == num_staging_blocks(P_LAYOUT, D_LAYOUT, 0, 130_000) > 0

    scheduler._take_staging_blocks("req")
    pool = scheduler._block_pool
    assert [b.block_id for b in scheduler._staging_blocks["req"]] == list(
        range(100, 100 + need)
    )
    meta = NixlConnectorMetadata()
    meta.add_new_req_to_recv(
        "req",
        [[1]],
        _params(
            remote_engine_id="P", remote_request_id="p", remote_host="h", remote_port=1
        ),
        staging_block_ids=[b.block_id for b in scheduler._staging_blocks["req"]],
    )
    assert meta.reqs_to_recv["req"].staging_block_ids == list(range(100, 100 + need))
    assert meta.reqs_to_recv["req"].remote.block_size == 1280

    output = (
        KVConnectorOutput(failed_recving={"req"})
        if failed
        else KVConnectorOutput(finished_recving={"req"})
    )
    scheduler.update_connector_output(output)
    assert pool.free_count == need
    assert scheduler._staging_blocks == {}
    # Reported again (e.g. failed and finished): freed once.
    scheduler.update_connector_output(KVConnectorOutput(finished_recving={"req"}))
    assert pool.free_count == need


@pytest.mark.parametrize(
    ("params", "num_external"),
    [
        # Same layout on both sides: whole pages pair.
        (dict(tp_size=16, dcp_size=16, block_size=640), 1000),
        # Nothing to load.
        (dict(), 0),
        # A peer older than version 18 is rejected at its handshake.
        (dict(block_size=None), 1000),
        # A peer no mode supports is rejected at its handshake.
        (dict(tp_size=16), 1000),
    ],
)
def test_no_staging_without_relayout(params, num_external):
    scheduler = _scheduler()
    assert scheduler.get_num_staging_blocks(_request(**params), 0, num_external) == 0
    scheduler._take_staging_blocks("req")
    assert scheduler._staging_blocks == {}


def test_staging_for_the_reverse_direction():
    scheduler = _scheduler(P_LAYOUT)
    request = _request(tp_size=16, dcp_size=16, block_size=640)
    assert scheduler.get_num_staging_blocks(request, 10240, 50_000) == (
        num_staging_blocks(D_LAYOUT, P_LAYOUT, 10240, 60_240)
    )


# ----------------------------------------------------------------------
# Worker: reads of a RELAYOUT request
# ----------------------------------------------------------------------

ATTN, SSM = 0, 1


def _read_worker(dcp_rank: int) -> NixlConnectorWorker:
    worker = object.__new__(NixlConnectorWorker)
    worker._remote_dcp_layouts = {"P": P_LAYOUT}
    worker._dcp_layout = D_LAYOUT
    worker.dcp_size = 16
    worker.dcp_rank = dcp_rank
    # Kernel blocks of 64 rows on both sides: 64-row units.
    worker.block_size = 64
    worker._group_spec_types = (FullAttentionSpec, MambaSpec)
    worker.transfer_topo = MagicMock()
    worker.transfer_topo.get_engine_info.return_value = SimpleNamespace(
        remote_block_size=64
    )
    worker._recv_failures = set()
    return worker


def _recv_meta(num_tokens: int | None = 3000, **remote) -> ReqMeta:
    return ReqMeta(
        local_block_ids=[[50, 51, 52], [77]],
        local_physical_block_ids=[[50, 51, 52], [77]],
        tp_size=8,
        dcp_size=8,
        local_num_computed_blocks=(0, 0),
        remote=RemoteMeta(
            block_ids=[[5, 6, 7], [9]],
            host="h",
            port=1,
            engine_id="P",
            request_id="p",
            num_tokens=num_tokens,
            block_size=remote.get("block_size", 1280),
            cp_kv_cache_interleave_size=remote.get("interleave", 1),
        ),
        staging_block_ids=[90, 91, 92],
    )


@pytest.mark.parametrize("dcp_rank", [3, 11])
def test_relayout_reads_stage_attention_and_read_state_directly(dcp_rank):
    """P8 -> D16 hybrid: D rank r stages P rank r % 8's units for attention,
    and reads its state slot 1:1 from its state source rank (P rank r // 2)."""
    worker = _read_worker(dcp_rank)
    plan = SimpleNamespace(
        all_source_ranks=(dcp_rank % 8, dcp_rank // 2),
        source_ranks_per_group=((dcp_rank % 8,), (dcp_rank // 2,)),
    )
    meta = _recv_meta()
    specs = worker._relayout_read_specs("req", meta, plan)
    expected = plan_staged_relayout(
        P_LAYOUT,
        D_LAYOUT,
        dcp_rank,
        0,
        3000,
        unit_rows=64,
        src_block_ids=[5, 6, 7],
        dst_block_ids=[50, 51, 52],
        staging_block_ids=[90, 91, 92],
        dst_kernel_block_size=64,
    )
    assert meta.relayout is not None
    assert np.array_equal(meta.relayout.dst_rows, expected.dst_rows)
    by_rank = {spec.remote_rank: spec for spec in specs}
    attn = by_rank[dcp_rank % 8]
    assert (attn.remote_block_ids[ATTN], attn.local_block_ids[ATTN]) == (
        expected.reads[dcp_rank % 8]
    )
    assert attn.local_block_ids[ATTN]
    state = by_rank[dcp_rank // 2]
    assert state.local_block_ids[SSM] == [77] and state.remote_block_ids[SSM] == [9]
    if dcp_rank // 2 != dcp_rank % 8:
        assert state.local_block_ids[ATTN] == [] and attn.local_block_ids[SSM] == []


def test_relayout_reads_check_the_request_layout():
    worker = _read_worker(0)
    plan = SimpleNamespace(all_source_ranks=(0,), source_ranks_per_group=((0,), (0,)))
    with pytest.raises(RuntimeError, match="handshake advertised"):
        worker._relayout_read_specs("req", _recv_meta(block_size=640), plan)


def test_relayout_reads_need_the_remote_token_count():
    worker = _read_worker(0)
    worker._handle_failed_transfer = MagicMock()
    plan = SimpleNamespace(all_source_ranks=(0,), source_ranks_per_group=((0,), (0,)))
    meta = _recv_meta(num_tokens=None)
    assert worker._relayout_read_specs("req", meta, plan) is None
    worker._handle_failed_transfer.assert_called_once()
    assert meta.relayout is None


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
    return StagedRelayout({}, rows, rows, rows + 100, rows)


def _completion_worker(event: _FakeEvent) -> NixlConnectorWorker:
    worker = object.__new__(NixlConnectorWorker)
    worker._peer_incompatible_error = None
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
    worker._launch_relayouts = MagicMock(return_value=event)
    return worker


def _relayout_meta(n: int) -> ReqMeta:
    meta = _recv_meta()
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
    src_blocks, _, dst_blocks, _ = worker._launch_relayouts.call_args.args
    assert len(src_blocks) == len(dst_blocks) == 5
    assert worker._relayouting == {"a": event, "b": event}

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
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
        select_dcp_transfer_mode,
    )

    assert (
        select_dcp_transfer_mode(D_LAYOUT, P_LAYOUT, has_mamba=True, tp_sizes=(16, 8))
        is DcpTransferMode.RELAYOUT
    )
