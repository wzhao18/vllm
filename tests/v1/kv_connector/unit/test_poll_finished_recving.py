# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Async KV loads reported on the earliest step output.

A worker drains loads that finished cleanly right before it sends a step's
output (`KVConnectorBase_V1.poll_finished_recving`), so the scheduler can see
them on the output of a step older than the one that issued them.
"""

import itertools
from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_transfer import kv_transfer_state
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.v1.executor.multiproc_executor import WorkerProc
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    KVConnectorOutput,
    ModelRunnerOutput,
)
from vllm.v1.request import RequestStatus

from .utils import (
    assert_scheduler_empty,
    create_mock_connector,
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def agent(monkeypatch) -> MagicMock:
    connector = MagicMock(spec=KVConnectorBase_V1)
    connector.poll_finished_recving.return_value = set()
    monkeypatch.setattr(kv_transfer_state, "_KV_CONNECTOR_AGENT", connector)
    return connector


def _kv_output(output: ModelRunnerOutput) -> KVConnectorOutput:
    assert output.kv_connector_output is not None
    return output.kv_connector_output


def test_base_connector_reports_nothing_early():
    assert KVConnectorBase_V1.poll_finished_recving(MagicMock()) == set()


def test_poll_into_without_connector_is_a_noop(monkeypatch):
    monkeypatch.setattr(kv_transfer_state, "_KV_CONNECTOR_AGENT", None)
    output = create_model_runner_output([], finished_recving={"a"})

    assert kv_transfer_state.poll_finished_recving_into(output) is output
    assert _kv_output(output).finished_recving == {"a"}


def test_poll_into_merges_into_a_new_set(agent):
    in_band = {"a"}
    output = create_model_runner_output([], finished_recving=in_band)
    agent.poll_finished_recving.return_value = {"b"}

    assert kv_transfer_state.poll_finished_recving_into(output) is output
    assert _kv_output(output).finished_recving == {"a", "b"}
    assert in_band == {"a"}


def test_poll_into_attaches_kv_output_to_a_copy_when_missing(agent):
    output = create_model_runner_output([])
    assert output.kv_connector_output is None
    agent.poll_finished_recving.return_value = {"b"}

    polled = kv_transfer_state.poll_finished_recving_into(output)

    assert polled is not output
    assert polled.req_ids is output.req_ids
    assert _kv_output(polled).finished_recving == {"b"}
    assert output.kv_connector_output is None


def test_poll_into_never_writes_the_shared_empty_output(agent):
    agent.poll_finished_recving.return_value = {"b"}

    output = kv_transfer_state.poll_finished_recving_into(EMPTY_MODEL_RUNNER_OUTPUT)

    assert output is not EMPTY_MODEL_RUNNER_OUTPUT
    assert _kv_output(output).finished_recving == {"b"}
    assert EMPTY_MODEL_RUNNER_OUTPUT.kv_connector_output is None


def test_poll_into_leaves_output_untouched_when_nothing_finished(agent):
    assert (
        kv_transfer_state.poll_finished_recving_into(EMPTY_MODEL_RUNNER_OUTPUT)
        is EMPTY_MODEL_RUNNER_OUTPUT
    )
    agent.poll_finished_recving.assert_called_once_with()


def test_poll_into_survives_a_failing_connector(agent):
    output = create_model_runner_output([], finished_recving={"a"})
    agent.poll_finished_recving.side_effect = RuntimeError("boom")

    assert kv_transfer_state.poll_finished_recving_into(output) is output
    assert _kv_output(output).finished_recving == {"a"}


def _make_worker_proc() -> tuple[WorkerProc, list]:
    worker_proc = object.__new__(WorkerProc)
    sent: list = []
    worker_proc.worker_response_mq = MagicMock()
    worker_proc.worker_response_mq.enqueue.side_effect = sent.append
    return worker_proc, sent


class _FailingAsyncOutput(AsyncModelRunnerOutput):
    def __init__(self, kv_connector_output: KVConnectorOutput | None):
        self._kv_connector_output = kv_connector_output

    def get_output(self) -> ModelRunnerOutput:
        raise RuntimeError("boom")

    def get_kv_connector_output(self) -> KVConnectorOutput | None:
        return self._kv_connector_output


class _AsyncOutput(_FailingAsyncOutput):
    def __init__(self, output: ModelRunnerOutput):
        super().__init__(output.kv_connector_output)
        self._output = output

    def get_output(self) -> ModelRunnerOutput:
        return self._output


@pytest.mark.parametrize("is_async", [False, True])
def test_enqueue_output_reports_polled_loads(agent, is_async):
    worker_proc, sent = _make_worker_proc()
    output = create_model_runner_output([], finished_recving={"a"})
    agent.poll_finished_recving.return_value = {"b"}

    worker_proc.enqueue_output(_AsyncOutput(output) if is_async else output)

    ((status, sent_output),) = sent
    assert status == WorkerProc.ResponseStatus.SUCCESS
    assert _kv_output(sent_output).finished_recving == {"a", "b"}


def test_enqueue_output_does_not_write_the_shared_empty_output(agent):
    worker_proc, sent = _make_worker_proc()
    agent.poll_finished_recving.return_value = {"b"}

    worker_proc.enqueue_output(EMPTY_MODEL_RUNNER_OUTPUT)

    ((status, sent_output),) = sent
    assert status == WorkerProc.ResponseStatus.SUCCESS
    assert _kv_output(sent_output).finished_recving == {"b"}
    assert EMPTY_MODEL_RUNNER_OUTPUT.kv_connector_output is None


def test_enqueue_output_sends_the_output_when_the_poll_raises(agent):
    worker_proc, sent = _make_worker_proc()
    output = create_model_runner_output([], finished_recving={"a"})
    agent.poll_finished_recving.side_effect = RuntimeError("boom")

    worker_proc.enqueue_output(_AsyncOutput(output))

    ((status, sent_output),) = sent
    assert status == WorkerProc.ResponseStatus.SUCCESS
    assert _kv_output(sent_output).finished_recving == {"a"}


@pytest.mark.parametrize(
    "kv_connector_output", [None, KVConnectorOutput(finished_recving={"a"})]
)
def test_enqueue_output_does_not_poll_on_failure(agent, kv_connector_output):
    worker_proc, sent = _make_worker_proc()

    worker_proc.enqueue_output(_FailingAsyncOutput(kv_connector_output))

    ((status, _),) = sent
    assert status != WorkerProc.ResponseStatus.SUCCESS
    agent.poll_finished_recving.assert_not_called()


@pytest.mark.parametrize("result", [None, {"key": "value"}, RuntimeError("boom")])
def test_enqueue_output_does_not_poll_for_other_results(agent, result):
    worker_proc, sent = _make_worker_proc()

    worker_proc.enqueue_output(result)

    assert len(sent) == 1
    agent.poll_finished_recving.assert_not_called()


def _rank_output(finished_recving=None, failed_recving=None) -> ModelRunnerOutput:
    return create_model_runner_output(
        [], finished_recving=finished_recving, failed_recving=failed_recving
    )


def test_aggregator_emits_once_when_ranks_report_on_different_outputs():
    world_size = 8
    aggregator = KVOutputAggregator(expected_finished_count=world_size)

    # Ranks 0-3 report on output k, ranks 4-7 on output k + 1.
    output_k = aggregator.aggregate(
        [_rank_output({"r"}) for _ in range(4)] + [_rank_output() for _ in range(4)]
    )
    output_k1 = aggregator.aggregate(
        [_rank_output() for _ in range(4)] + [_rank_output({"r"}) for _ in range(4)]
    )
    output_k2 = aggregator.aggregate([_rank_output() for _ in range(world_size)])

    assert _kv_output(output_k).finished_recving is None
    assert _kv_output(output_k1).finished_recving == {"r"}
    assert _kv_output(output_k2).finished_recving is None


@pytest.mark.parametrize("failing_output", [0, 1, 2])
def test_aggregator_reports_failure_whatever_the_arrival_order(failing_output):
    world_size = 8
    aggregator = KVOutputAggregator(expected_finished_count=world_size)
    # Rank 0 fails in-band; ranks 1-7 reported early, before or after it.
    rounds: list[list[ModelRunnerOutput]] = [
        [_rank_output() for _ in range(world_size)] for _ in range(3)
    ]
    rounds[failing_output][0] = _rank_output({"r"}, failed_recving={"r"})
    early_round = (failing_output + 1) % 3
    for rank in range(1, world_size):
        rounds[early_round][rank] = _rank_output({"r"})

    results = [_kv_output(aggregator.aggregate(outputs)) for outputs in rounds]

    emitted = [i for i, r in enumerate(results) if r.finished_recving]
    assert emitted == [max(failing_output, early_round)]
    assert results[emitted[0]].finished_recving == {"r"}
    assert results[emitted[0]].failed_recving == {"r"}


def _make_load_scheduler(load_req_id: str, num_external_tokens: int):
    scheduler = create_scheduler(create_vllm_config())
    scheduler.connector = create_mock_connector()
    scheduler.connector.get_loaded_kv_cache_group_ids.return_value = (0,)
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        lambda request, _: (num_external_tokens, True)
        if request.request_id == load_req_id
        else (0, False)
    )
    scheduler.connector.request_finished.return_value = (False, None)
    scheduler.connector.take_events.return_value = ()
    return scheduler


_req_ids = itertools.count()


@pytest.mark.parametrize("abort_while_loading", [False, True])
def test_scheduler_accepts_load_reported_on_an_older_output(abort_while_loading):
    """A load issued by step S1 can finish before S0's output is sent and ride
    it; the scheduler applies S0's output after scheduling S1."""
    block_size = 16
    running = create_request(request_id=next(_req_ids), num_tokens=block_size)
    loading = create_request(request_id=next(_req_ids), num_tokens=4 * block_size)
    scheduler = _make_load_scheduler(loading.request_id, 2 * block_size)
    block_pool = scheduler.kv_cache_manager.block_pool

    scheduler.add_request(running)
    output_s0 = scheduler.schedule()
    num_free_blocks = block_pool.get_num_free_blocks()

    scheduler.add_request(loading)
    pending = [scheduler.schedule()]
    assert loading.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert loading.request_id not in pending[0].num_scheduled_tokens

    if abort_while_loading:
        scheduler.finish_requests(loading.request_id, RequestStatus.FINISHED_ABORTED)
        # The load may still be writing the blocks, so they stay allocated.
        assert block_pool.get_num_free_blocks() < num_free_blocks

    scheduler.update_from_output(
        output_s0,
        create_model_runner_output([running], finished_recving={loading.request_id}),
    )

    if abort_while_loading:
        # Released now (returned to the pool once in-flight steps finish).
        assert loading.request_id not in scheduler.requests
    else:
        pending.append(scheduler.schedule())
        assert loading.status == RequestStatus.RUNNING
        assert pending[-1].num_scheduled_tokens[loading.request_id] == 2 * block_size

    scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)
    pending.append(scheduler.schedule())
    for scheduler_output in pending:
        scheduler.update_from_output(scheduler_output, create_model_runner_output([]))
    assert_scheduler_empty(scheduler)
