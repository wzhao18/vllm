# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest
import torch

from vllm.v1.kv_offload.base import CanonicalKVCacheRef
from vllm.v1.kv_offload.cpu.gpu_worker import (
    _check_block_ids_in_range,
    _check_copy_ref,
    _validate_copy_descriptors,
)


def test_check_block_ids_rejects_out_of_range() -> None:
    tensor = torch.empty((2, 8), dtype=torch.int8)

    with pytest.raises(RuntimeError, match="out-of-range source block IDs"):
        _check_block_ids_in_range(
            np.array([0, 2], dtype=np.int64), tensor, "source", job_id=3
        )


def test_check_copy_ref_rejects_oversized_sub_block() -> None:
    src_tensor = torch.empty((2, 8), dtype=torch.int8)
    dst_tensor = torch.empty((2, 24), dtype=torch.int8)
    data_ref = CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=9)

    with pytest.raises(RuntimeError, match="exceeds source sub-block size"):
        _check_copy_ref(
            data_ref,
            src_tensor,
            dst_tensor,
            src_block_size_factor=1,
            dst_block_size_factor=3,
            job_id=4,
        )


def test_check_copy_ref_allows_sub_block_sized_copy() -> None:
    src_tensor = torch.empty((2, 8), dtype=torch.int8)
    dst_tensor = torch.empty((2, 24), dtype=torch.int8)
    data_ref = CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=8)

    _check_copy_ref(
        data_ref,
        src_tensor,
        dst_tensor,
        src_block_size_factor=1,
        dst_block_size_factor=3,
        job_id=5,
    )


def test_validate_copy_descriptors_rejects_out_of_range_pointer() -> None:
    src_tensor = torch.empty((2, 8), dtype=torch.int8)
    dst_tensor = torch.empty((2, 8), dtype=torch.int8)

    with pytest.raises(RuntimeError, match="outside registered source tensors"):
        _validate_copy_descriptors(
            np.array([src_tensor.data_ptr() + src_tensor.numel()], dtype=np.int64),
            np.array([dst_tensor.data_ptr()], dtype=np.int64),
            np.array([8], dtype=np.int64),
            [src_tensor],
            [dst_tensor],
            job_id=6,
        )


def test_validate_copy_descriptors_rejects_overlapping_destinations() -> None:
    src_tensor = torch.empty((2, 16), dtype=torch.int8)
    dst_tensor = torch.empty((2, 16), dtype=torch.int8)

    with pytest.raises(RuntimeError, match="overlapping destination descriptors"):
        _validate_copy_descriptors(
            np.array([src_tensor.data_ptr(), src_tensor.data_ptr() + 8]),
            np.array([dst_tensor.data_ptr(), dst_tensor.data_ptr() + 4]),
            np.array([8, 8]),
            [src_tensor],
            [dst_tensor],
            job_id=7,
        )
