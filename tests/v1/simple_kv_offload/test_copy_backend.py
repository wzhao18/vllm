# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
from types import SimpleNamespace

import torch

from vllm.v1.simple_kv_offload import cuda_mem_ops
from vllm.v1.simple_kv_offload.copy_backend import DmaCopyBackend


def test_build_params_sets_source_access_order(monkeypatch) -> None:
    monkeypatch.setattr(cuda_mem_ops, "_batch_memcpy_fn", object())

    src = {"layer": torch.empty((2, 8), dtype=torch.int8)}
    dst = {"layer": torch.empty((2, 8), dtype=torch.int8)}
    stream = SimpleNamespace(cuda_stream=123)

    stream_ordered = cuda_mem_ops.build_params(
        src,
        dst,
        stream,  # type: ignore[arg-type]
        is_src_access_order_any=False,
    )
    relaxed = cuda_mem_ops.build_params(
        src,
        dst,
        stream,  # type: ignore[arg-type]
        is_src_access_order_any=True,
    )

    assert stream_ordered.attrs.srcAccessOrder == 1
    assert relaxed.attrs.srcAccessOrder == 3


def test_launch_copy_preserves_ready_event() -> None:
    backend = DmaCopyBackend()
    backend._store_params = object()  # type: ignore[assignment]
    backend._queue = queue.SimpleQueue()
    ready_event = object()
    events_list = []

    backend.launch_copy(
        [1],
        [2],
        is_store=True,
        event_idx=7,
        events_list=events_list,
        ready_event=ready_event,  # type: ignore[arg-type]
    )

    item = backend._queue.get_nowait()
    expected = ([1], [2], backend._store_params, True, 7, events_list, ready_event)
    assert item == expected
