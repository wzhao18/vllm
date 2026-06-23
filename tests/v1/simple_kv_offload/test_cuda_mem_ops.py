# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes

import numpy as np
import pytest

from vllm.v1.simple_kv_offload import cuda_mem_ops


class _FakePlatform:
    @staticmethod
    def is_rocm() -> bool:
        return False


def _make_params() -> cuda_mem_ops.BatchMemcpyParams:
    return cuda_mem_ops.BatchMemcpyParams(
        src_bases=np.array([1000, 2000], dtype=np.uint64),
        dst_bases=np.array([5000, 7000], dtype=np.uint64),
        bpb=np.array([10, 20], dtype=np.uint64),
        src_num_blocks=np.array([8, 8], dtype=np.int64),
        dst_num_blocks=np.array([8, 8], dtype=np.int64),
        num_layers=2,
        attrs=cuda_mem_ops._CUmemcpyAttributes(srcAccessOrder=1),
        stream_handle=99,
    )


def test_copy_blocks_keeps_batch_args_alive(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict[str, int]] = []

    def fake_batch_memcpy(
        dst_ptr,
        src_ptr,
        sizes_ptr,
        total,
        attrs_ptr,
        attrs_idx_ptr,
        num_attrs,
        fail_idx_ptr,
        stream,
    ):
        calls.append(
            {
                "dst_ptr": dst_ptr,
                "src_ptr": src_ptr,
                "sizes_ptr": sizes_ptr,
                "total": total,
                "num_attrs": num_attrs,
                "stream": stream,
            }
        )
        return 0

    monkeypatch.setattr(cuda_mem_ops, "current_platform", _FakePlatform())
    monkeypatch.setattr(cuda_mem_ops, "_batch_memcpy_fn", fake_batch_memcpy)

    workspace = cuda_mem_ops.copy_blocks([1, 3], [2, 4], _make_params())

    assert workspace is not None
    assert workspace.src_all.tolist() == [1010, 1030, 2020, 2060]
    assert workspace.dst_all.tolist() == [5020, 5040, 7040, 7080]
    assert workspace.sz_all.tolist() == [10, 10, 20, 20]

    assert calls == [
        {
            "dst_ptr": workspace.dst_all.ctypes.data,
            "src_ptr": workspace.src_all.ctypes.data,
            "sizes_ptr": workspace.sz_all.ctypes.data,
            "total": 4,
            "num_attrs": 1,
            "stream": 99,
        }
    ]


def test_copy_blocks_uses_per_call_fail_idx(monkeypatch: pytest.MonkeyPatch):
    def fake_batch_memcpy(
        dst_ptr,
        src_ptr,
        sizes_ptr,
        total,
        attrs_ptr,
        attrs_idx_ptr,
        num_attrs,
        fail_idx_ptr,
        stream,
    ):
        fail_idx = ctypes.cast(
            fail_idx_ptr, ctypes.POINTER(ctypes.c_size_t)
        ).contents
        fail_idx.value = 3
        return 123

    monkeypatch.setattr(cuda_mem_ops, "current_platform", _FakePlatform())
    monkeypatch.setattr(cuda_mem_ops, "_batch_memcpy_fn", fake_batch_memcpy)

    with pytest.raises(RuntimeError, match=r"err=123 failIdx=3"):
        cuda_mem_ops.copy_blocks([1, 3], [2, 4], _make_params())


def test_copy_blocks_rejects_out_of_range_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cuda_mem_ops, "current_platform", _FakePlatform())
    monkeypatch.setattr(
        cuda_mem_ops,
        "_batch_memcpy_fn",
        lambda *_args: pytest.fail("copy should not be launched"),
    )

    with pytest.raises(RuntimeError, match=r"src block id out of range"):
        cuda_mem_ops.copy_blocks([8], [0], _make_params())

    with pytest.raises(RuntimeError, match=r"dst block id out of range"):
        cuda_mem_ops.copy_blocks([0], [8], _make_params())
