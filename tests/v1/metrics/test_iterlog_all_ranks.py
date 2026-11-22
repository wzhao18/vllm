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
