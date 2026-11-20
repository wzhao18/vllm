# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 prefill: fused W_UK + RoPE concat + static FP8 query (sm_107)
is bitwise equal to the stock bmm / copy / QuantFP8 passes."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda()
    or torch.cuda.get_device_capability() != (10, 7),
    reason="sm_107 kernel",
)


def _stock(q, w_uk_t, q_scale):
    """bmm into the NoPE columns, RoPE copy, then QuantFP8's static per-tensor
    quant as the CUDA op (forward_cuda) runs it."""
    from vllm import _custom_ops as ops

    t, h, d = q.shape
    p, lat = w_uk_t.shape[1], w_uk_t.shape[2]
    cat = q.new_empty((t, h, lat + d - p))
    torch.bmm(q[..., :p].transpose(0, 1), w_uk_t, out=cat[..., :lat].transpose(0, 1))
    cat[..., lat:].copy_(q[..., p:])
    out, _ = ops.scaled_fp8_quant(cat.reshape(t, -1), q_scale)
    return out.view(cat.shape)


@pytest.mark.parametrize("t", [1, 300, 4097])
@pytest.mark.parametrize("scale", [1.0, 0.5])
@pytest.mark.parametrize("amp", [1.0, 40.0])
def test_fused_query_bitwise(t, scale, amp):
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm import uk_query

    torch.manual_seed(t)
    h, p, r, lat = 96, 128, 64, 512
    q = (torch.randn(t, h, p + r, device="cuda") * 2.0 * amp).to(torch.bfloat16)
    w = (torch.randn(h, p, lat, device="cuda") * 0.5).to(torch.bfloat16)
    q_scale = torch.tensor([scale], device="cuda", dtype=torch.float32)
    out = uk_query.fused_query(q, w, q_scale)
    assert out is not None and out.dtype == torch.float8_e4m3fn
    ref = _stock(q, w, q_scale)
    assert out.shape == ref.shape
    # forward_native (x * (1 / scale), clamp, cast) gives the same bytes
    native = uk_query.reference(q, w, q_scale)
    assert torch.equal(native.view(torch.uint8), ref.view(torch.uint8))
    mism = (out.view(torch.uint8) != ref.view(torch.uint8)).sum().item()
    assert mism == 0, f"{mism} of {out.numel()} bytes differ"


def test_fused_query_declines_unsupported():
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm import uk_query

    q = torch.randn(8, 96, 192, device="cuda").to(torch.bfloat16)
    w = torch.randn(96, 128, 512, device="cuda").to(torch.bfloat16)
    s = torch.ones(1, device="cuda")
    assert uk_query.fused_query(q.float(), w, s) is None
    assert uk_query.fused_query(q[:, :, :190], w, s) is None
