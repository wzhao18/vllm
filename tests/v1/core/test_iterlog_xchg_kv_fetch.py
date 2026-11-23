# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-iteration log: DP pull-up exchange (X) and KV connector load (RK) records."""

import json
from types import SimpleNamespace

import pytest

from vllm.v1.metrics import iterlog
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler, mock_kv

pytestmark = pytest.mark.cpu_test


def _records(log) -> list:
    log._w.close()
    with open(log._w.path) as f:
        return [json.loads(line) for line in f]


@pytest.fixture
def engine_log(tmp_path, monkeypatch):
    monkeypatch.setattr(iterlog, "ITERLOG_DIR", str(tmp_path))
    return iterlog.EngineIterLog({})


def _req(rid: str, finished: bool = False, computed: int = 0):
    return SimpleNamespace(
        request_id=rid,
        session_id=None,
        arrival_time=0.0,
        num_prompt_tokens=64,
        max_tokens=2,
        num_computed_tokens=computed,
        is_finished=lambda: finished,
    )


def test_exchange_record_names_its_fields_and_step(engine_log):
    engine_log.dp_exchange(1.0, 1.25, 512, 9000, 2048, True)
    engine_log.dp_exchange(2.0, 2.5, 0, 0, 1024, False)
    recs = _records(engine_log)
    fields = recs[0][1]["fields"]["X"].split()
    xs = [dict(zip(fields, r[1:])) for r in recs if r[0] == "X"]
    assert xs == [
        dict(k=0, t_xchg0=1.0, t_xchg1=1.25, num_tokens=512, demand=9000,
             max_tokens=2048, in_sched=1),
        dict(k=0, t_xchg0=2.0, t_xchg1=2.5, num_tokens=0, demand=0,
             max_tokens=1024, in_sched=0),
    ]


def test_kv_load_record_lifecycle(engine_log):
    loaded, dropped, plain = _req("a"), _req("b"), _req("c")
    for r in (loaded, dropped, plain):
        engine_log.add_request(r)
    loaded.num_computed_tokens = 48
    engine_log.kv_fetch(loaded, "in_progress")
    engine_log.kv_fetch(loaded, "completed_waiting")
    engine_log.kv_fetch(loaded, None)  # admitted to compute
    dropped.num_computed_tokens = 32
    engine_log.kv_fetch(dropped, "in_progress")
    dropped.is_finished = lambda: True
    engine_log.kv_fetch(dropped, None)  # finished before admission
    engine_log.kv_fetch(plain, None)  # no load: no record
    recs = _records(engine_log)
    fields = recs[0][1]["fields"]["RK"].split()
    rks = [dict(zip(fields, r[1:])) for r in recs if r[0] == "RK"]
    assert [r["idx"] for r in rks] == [0, 1]
    a, b = rks
    assert a["t_queued"] is None
    assert a["t_issue"] <= a["t_done"] <= a["t_admit"]
    assert a["n_computed"] == 48
    assert b["t_done"] is None and b["t_admit"] is None and b["n_computed"] == 32


def test_scheduler_reports_async_load_stages():
    block_size = 16
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        use_kv_connector=mock_kv(matched_tokens=2 * block_size, is_async=True),
        block_size=block_size,
    )
    seen = []
    scheduler.set_kv_fetch_log(lambda req, stage: seen.append((req.request_id, stage)))
    (req,) = create_requests(
        num_requests=1, num_tokens=3 * block_size, max_tokens=2,
        block_size=block_size, req_ids=["r"],
    )
    scheduler.add_request(req)
    out = scheduler.schedule()
    assert req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.update_from_output(
        out,
        ModelRunnerOutput(
            req_ids=[], req_id_to_index={}, sampled_token_ids=[],
            kv_connector_output=KVConnectorOutput(finished_recving={"r"}),
        ),
    )
    out = scheduler.schedule()
    assert "r" in out.num_scheduled_tokens
    stages = [stage for rid, stage in seen if rid == "r"]
    # Queued for a remote prefill on arrival (the mock connector's params), then
    # issued, completed, and admitted to compute.
    assert stages == ["waiting_to_start", "in_progress", "completed_waiting", None]
