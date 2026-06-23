# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest

from vllm.v1.simple_kv_offload import copy_backend


class _FakeEvent:
    def __init__(self) -> None:
        self.recorded_stream = None

    def record(self, stream) -> None:
        self.recorded_stream = stream


class _FakeStream:
    def __init__(self) -> None:
        self.waited_stream = None

    def wait_stream(self, stream) -> None:
        self.waited_stream = stream


def test_launch_copy_splits_blocks_by_group(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = copy_backend.DmaCopyBackend()
    backend._load_params_by_group = [  # type: ignore[list-item]
        "params-g0",
        "params-g1",
    ]
    backend._load_stream = _FakeStream()  # type: ignore[assignment]
    events: list[copy_backend.CopyEvent] = []
    calls = []

    def fake_copy_blocks(src_blocks, dst_blocks, params):
        calls.append((list(src_blocks), list(dst_blocks), params))
        return f"workspace-{params}"

    monkeypatch.setattr(copy_backend, "copy_blocks", fake_copy_blocks)
    monkeypatch.setattr(copy_backend.torch, "Event", _FakeEvent)

    backend.launch_copy(
        src_blocks=[10, 11, 12],
        dst_blocks=[20, 21, 22],
        group_ids=[1, 0, 1],
        is_store=False,
        event_idx=7,
        events_list=events,
    )

    assert calls == [
        ([10, 12], [20, 22], "params-g1"),
        ([11], [21], "params-g0"),
    ]
    assert len(events) == 1
    assert events[0][0] == 7
    assert events[0][2] == ["workspace-params-g1", "workspace-params-g0"]
    assert events[0][1].recorded_stream is backend._load_stream


def test_launch_store_waits_for_current_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = copy_backend.DmaCopyBackend()
    backend._store_params_by_group = ["params-g0"]  # type: ignore[list-item]
    backend._store_stream = _FakeStream()  # type: ignore[assignment]
    current_stream = _FakeStream()
    events: list[copy_backend.CopyEvent] = []

    monkeypatch.setattr(copy_backend, "copy_blocks", lambda *_args: None)
    monkeypatch.setattr(copy_backend.torch, "Event", _FakeEvent)
    monkeypatch.setattr(
        copy_backend.torch.cuda, "current_stream", lambda: current_stream
    )

    backend.launch_copy(
        src_blocks=[10],
        dst_blocks=[20],
        group_ids=[0],
        is_store=True,
        event_idx=8,
        events_list=events,
    )

    assert backend._store_stream.waited_stream is current_stream
    assert events[0][1].recorded_stream is backend._store_stream
