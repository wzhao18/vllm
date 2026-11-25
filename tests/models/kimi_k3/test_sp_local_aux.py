# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shard-local drafter aux states (``ops/sp_local_aux.py``) under CUDA graphs:
the graph wrapper's weak references keep the ``SpLocalAux`` type and shard, and
the local path is taken while capturing (it was refused, so graph steps
gathered the packed 6-wide aux rows)."""

from unittest import mock

import pytest
import torch

from vllm.models.kimi_k3.nvidia.ops import sp_local_aux
from vllm.utils.torch_utils import weak_ref_tensors


@pytest.mark.skipif(not torch.cuda.is_available(), reason="weak_ref_tensor is a CUDA op")
def test_weak_refs_keep_shard_local_type():
    shards = [torch.zeros(4, 8, device="cuda"), torch.ones(4, 8, device="cuda")]
    aux = sp_local_aux.SpLocalAux(shards, num_rows=32)
    out = weak_ref_tensors((torch.zeros(32, 8, device="cuda"), aux))
    assert isinstance(out, tuple)
    ref = out[1]
    assert isinstance(ref, sp_local_aux.SpLocalAux)
    assert ref.shard == 4 and ref.num_rows == 32
    assert all(torch.equal(a, b) for a, b in zip(ref, shards))


def test_local_aux_taken_while_capturing():
    with mock.patch("torch.cuda.is_current_stream_capturing", return_value=True):
        assert sp_local_aux.use_local_aux(sp_local_aux.MIN_TOKENS)
    assert not sp_local_aux.use_local_aux(sp_local_aux.MIN_TOKENS - 1)
