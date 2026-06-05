# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest
import torch

from vllm.v1.kv_offload.base import CanonicalKVCacheRef
from vllm.v1.kv_offload.cpu.gpu_worker import (
    _check_block_ids_in_range,
    _check_copy_ref,
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
