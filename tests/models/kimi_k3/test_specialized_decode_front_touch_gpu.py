# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 SP decode MoE-front L2 touch (``cutedsl.common.touch``) on Rubin: the
touch kernel walks every range of its table (several ranges, sizes that are not a
multiple of the 16 KiB chunk), completes, and leaves the touched memory unchanged;
a later GEMM over the touched weights is bitwise unchanged. One GPU, seconds."""

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("needs a Rubin GPU (compute capability 10.7)", allow_module_level=True)

import cuda.bindings.driver as cuda  # noqa: E402

from vllm.models.common.ops.moe_tail_fusion.primitives import to_cute  # noqa: E402
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.touch import (  # noqa: E402,E501
    compile_touch,
    sm_count,
    touch_table,
)

DEVICE = torch.device("cuda", 0)


@pytest.mark.parametrize("sizes", [(4480 * 7168,), (896 * 7168, 3584 * 7168, 40008)])
def test_touch_leaves_memory_unchanged(sizes):
    torch.manual_seed(0)
    tensors = [torch.randn(s, device=DEVICE).to(torch.bfloat16) for s in sizes]
    copies = [t.clone() for t in tensors]
    table = touch_table(tensors, DEVICE)
    touch = compile_touch(len(tensors), sm_count(DEVICE), DEVICE.index)
    stream = torch.cuda.Stream(device=DEVICE)
    for _ in range(3):
        touch(to_cute(table, 16), cuda.CUstream(stream.cuda_stream))
    stream.synchronize()
    for t, c in zip(tensors, copies):
        assert torch.equal(t, c)


def test_gemm_after_touch_bitwise():
    torch.manual_seed(1)
    w = (torch.randn(4480, 7168, device=DEVICE) * 0.02).to(torch.bfloat16)
    x = torch.randn(64, 7168, device=DEVICE).to(torch.bfloat16)
    ref = x @ w.t()
    table = touch_table([w], DEVICE)
    touch = compile_touch(1, sm_count(DEVICE), DEVICE.index)
    stream = torch.cuda.Stream(device=DEVICE)
    stream.wait_stream(torch.cuda.current_stream(DEVICE))
    touch(to_cute(table, 16), cuda.CUstream(stream.cuda_stream))
    out = x @ w.t()
    torch.cuda.current_stream(DEVICE).wait_stream(stream)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)


def test_touch_table_refuses_misaligned():
    t = torch.empty(1025, dtype=torch.bfloat16, device=DEVICE)[1:]
    with pytest.raises(ValueError):
        touch_table([t], DEVICE)
