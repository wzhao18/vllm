# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""WorkerIterLog rank selection: global rank 0 only by default; every rank (own file per rank) with
VLLM_ITERLOG_ALL_RANKS=1. CPU only: GPU timing level 0 creates no CUDA events."""

import os

import pytest
import torch.distributed as dist

from vllm.v1.metrics import iterlog


def _fake_dist(monkeypatch, rank: int) -> None:
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda *a, **k: rank)


@pytest.fixture
def iterlog_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(iterlog, "ITERLOG_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_ITERLOG_GPU", "0")
    return tmp_path


def _worker_files(d):
    return sorted(f for f in os.listdir(d) if f.startswith("worker-"))


def test_nonzero_rank_off_by_default(iterlog_dir, monkeypatch):
    monkeypatch.delenv("VLLM_ITERLOG_ALL_RANKS", raising=False)
    _fake_dist(monkeypatch, 3)
    assert iterlog.WorkerIterLog.maybe_create() is None
    assert _worker_files(iterlog_dir) == []


def test_all_ranks_writes_one_file_per_rank(iterlog_dir, monkeypatch):
    monkeypatch.setenv("VLLM_ITERLOG_ALL_RANKS", "1")
    _fake_dist(monkeypatch, 3)
    w = iterlog.WorkerIterLog.maybe_create()
    assert w is not None
    files = _worker_files(iterlog_dir)
    assert len(files) == 1 and files[0].startswith("worker-r3-")


def test_rank0_unchanged(iterlog_dir, monkeypatch):
    monkeypatch.delenv("VLLM_ITERLOG_ALL_RANKS", raising=False)
    _fake_dist(monkeypatch, 0)
    w = iterlog.WorkerIterLog.maybe_create()
    assert w is not None
    files = _worker_files(iterlog_dir)
    assert len(files) == 1 and files[0].startswith("worker-r0-")


def test_off_without_iterlog_dir(monkeypatch):
    monkeypatch.setattr(iterlog, "ITERLOG_DIR", None)
    monkeypatch.setenv("VLLM_ITERLOG_ALL_RANKS", "1")
    _fake_dist(monkeypatch, 3)
    assert iterlog.WorkerIterLog.maybe_create() is None


def test_engine_iterlog_records_reported_cached_tokens(iterlog_dir):
    """RC: the prefix-cache tokens the engine reports (usage cached_tokens),
    split local / external, once per prefill stats output."""
    import json
    from types import SimpleNamespace

    from vllm.v1.metrics.stats import PrefillStats

    log = iterlog.EngineIterLog({})
    req = SimpleNamespace(
        request_id="r0",
        session_id=None,
        arrival_time=0.0,
        num_prompt_tokens=100,
        max_tokens=1,
    )
    log.add_request(req)
    so = object()
    log._pending.append([so, 0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, None])
    ps = PrefillStats(
        num_prompt_tokens=100,
        num_computed_tokens=36,
        num_cached_tokens=64,
        num_local_cached_tokens=16,
        num_external_cached_tokens=48,
    )
    out = SimpleNamespace(
        request_id="r0", new_token_ids=[7], finish_reason=None, prefill_stats=ps
    )
    log.on_update(so, 1.0, 2.0, {0: SimpleNamespace(outputs=[out])})
    log._w.close()

    recs = [json.loads(line) for line in open(log._w.path)]
    assert ["RC", 0, 2.0, 64, 16, 48] in recs
    assert ["RT", 0, 2.0, 0] in recs
