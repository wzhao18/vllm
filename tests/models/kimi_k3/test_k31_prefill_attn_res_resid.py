# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill AttnRes dataflow C producers (specialized_prefill.attn_res_resid):
the sm_107 MXFP8 GEMM whose TMA store reduce-adds its BF16 output into the
AttnRes prefix must equal the stock prefix rounding bf16(P + bf16(x @ W.T))
bit for bit, and its GEMM must match torch._scaled_mm."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability((10, 7)), reason="sm_107 only"
)

N = 7168


class _QA:
    def __init__(self, data, scale):
        self.data, self.scale = data, scale


def _mx(t: torch.Tensor):
    from flashinfer import mxfp8_quantize

    q, s = mxfp8_quantize(t, is_sf_swizzled_layout=True)
    return q, s.view(torch.uint8).reshape(-1)


@pytest.mark.parametrize("m,k", [(4096, 6144), (5000, 12288), (16384, 6144)])
def test_add_into_is_stock_prefix_rounding(m: int, k: int):
    from vllm.models.kimi_k3.nvidia.specialized_prefill import attn_res_resid as rr

    torch.manual_seed(0)
    w, ws = _mx(torch.randn(N, k, device="cuda", dtype=torch.bfloat16) * 0.02)
    x, xs = _mx(torch.randn(m, k, device="cuda", dtype=torch.bfloat16))
    gemm = rr.ResidGemm.from_rows(w, ws)
    qa = _QA(x, xs)
    y = torch.empty(m, N, device="cuda", dtype=torch.bfloat16)
    gemm.store_into(qa, y)
    # Real prefix magnitudes, a zero row and signed zeros / tiny values.
    p0 = (torch.randn(m, N, device="cuda") * 4).bfloat16()
    p0[0] = 0
    p0[1] = -y[1]
    p0[2] = (torch.randn(N, device="cuda") * 1e-30).bfloat16()
    p = p0.clone()
    gemm.add_into(qa, p)
    ref = (p0.float() + y.float()).bfloat16()
    assert torch.equal(p, ref), int((p != ref).sum())
    # Twice: the prefix accumulates in place.
    gemm.add_into(qa, p)
    assert torch.equal(p, (ref.float() + y.float()).bfloat16())
    # The GEMM is the stock MXFP8 product (accumulation order may differ).
    ref_mm = torch._scaled_mm(
        x, w.t(), scale_a=xs.view(torch.float8_e8m0fnu),
        scale_b=ws.view(torch.float8_e8m0fnu), out_dtype=torch.bfloat16,
    )  # fmt: skip
    torch.testing.assert_close(y.float(), ref_mm.float(), rtol=2e-2, atol=2e-2)


def test_shape_checks():
    from vllm.models.kimi_k3.nvidia.specialized_prefill import attn_res_resid as rr

    w, ws = _mx(torch.randn(N, 256, device="cuda", dtype=torch.bfloat16))
    x, xs = _mx(torch.randn(256, 256, device="cuda", dtype=torch.bfloat16))
    gemm = rr.ResidGemm.from_rows(w, ws)
    with pytest.raises(ValueError):
        gemm.add_into(_QA(x, xs), torch.empty(256, N, device="cuda", dtype=torch.float32))
    with pytest.raises(ValueError):
        gemm.add_into(_QA(x, xs), torch.empty(128, N, device="cuda", dtype=torch.bfloat16))
