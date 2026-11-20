# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 prefill: the W_UV up-projection with the DSA output gate in its
epilogue (sm_107) is bitwise equal to the stock bmm + ``attn * g.sigmoid()``."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda()
    or torch.cuda.get_device_capability() != (10, 7),
    reason="sm_107 kernel",
)

H, P, L, V = 96, 128, 512, 128


def _w_uv(layout: str) -> torch.Tensor:
    """W_UV (H, L, V) as the MLA layer holds it: a transposed view of the
    (L, H, P + V) kv_b_proj split ("model"), or contiguous (the ShardQ copy)."""
    w = torch.randn((L, H, P + V), device="cuda", dtype=torch.bfloat16) * 0.05
    w_uv = w[..., P:].transpose(0, 1)
    return w_uv if layout == "model" else w_uv.contiguous()


def _stock(latent, w_uv, g):
    t = latent.shape[0]
    attn = latent.new_empty((t, H * V))
    torch.bmm(latent.transpose(0, 1), w_uv, out=attn.view(t, H, V).transpose(0, 1))
    return attn * g.sigmoid()


@pytest.mark.parametrize("t", [512, 1000, 4097])
@pytest.mark.parametrize("amp", [1.0, 30.0])
@pytest.mark.parametrize("layout", ["model", "contiguous"])
def test_v_up_gate_bitwise(t, amp, layout):
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm import uv_gate

    torch.manual_seed(t)
    latent = torch.randn((t, H, L), device="cuda", dtype=torch.bfloat16) * amp
    w_uv = _w_uv(layout)
    g = torch.randn((t, H * V), device="cuda", dtype=torch.bfloat16) * 4
    out = torch.full((t, H * V), float("nan"), device="cuda", dtype=torch.bfloat16)
    assert uv_gate.v_up_gate(latent, w_uv, g, out)
    ref = _stock(latent, w_uv, g)
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))


def test_v_up_gate_row_stride():
    """The (T, H * L) latent as a view of a wider buffer and a strided gate."""
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm import uv_gate

    t = 777
    buf = torch.randn((t, H * L + 64), device="cuda", dtype=torch.bfloat16)
    latent = buf[:, : H * L].view(t, H, L)
    w_uv = _w_uv("model")
    gbuf = torch.randn((t, H * V + 128), device="cuda", dtype=torch.bfloat16)
    g = gbuf[:, : H * V]
    out = torch.empty((t, H * V), device="cuda", dtype=torch.bfloat16)
    assert uv_gate.v_up_gate(latent, w_uv, g, out)
    assert torch.equal(out.view(torch.int16), _stock(latent, w_uv, g).view(torch.int16))


def test_v_up_gate_declines_unsupported():
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm import uv_gate

    w_uv = _w_uv("model")
    small = torch.randn((64, H, L), device="cuda", dtype=torch.bfloat16)
    g = torch.randn((64, H * V), device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(g)
    assert not uv_gate.v_up_gate(small, w_uv, g, out)  # decode-sized: stock
    latent = torch.randn((600, H, L), device="cuda", dtype=torch.float16)
    g = torch.randn((600, H * V), device="cuda", dtype=torch.bfloat16)
    assert not uv_gate.v_up_gate(latent, w_uv, g, torch.empty_like(g))
