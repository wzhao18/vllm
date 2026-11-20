# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The pinned prefill cuBLASLt tactics compute the stock GEMM (F.linear for BF16,
``torch._scaled_mm`` for MXFP8) at prefill row counts, and the linear dispatch
routes only prefill-sized calls to them."""

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.is_device_capability((10, 7))),
    reason="cuBLASLt tactics are pinned for SM107",
)


def _pins():
    from vllm.models.kimi_k3.nvidia.specialized_prefill import gemm_tactics

    return sorted(gemm_tactics.PINS)


@pytest.mark.parametrize("key", _pins())
@pytest.mark.parametrize("rows", [4096, 4100])
def test_pinned_tactic_matches_stock(key, rows):
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )
    from vllm.models.kimi_k3.nvidia.specialized_prefill import gemm_tactics

    dtype, n, k = key
    plan = gemm_tactics.plan(dtype, n, k)
    assert plan is not None
    gen = torch.Generator(device="cuda").manual_seed(n + k + rows)
    x = torch.randn(rows, k, device="cuda", generator=gen).bfloat16()
    w = (0.02 * torch.randn(n, k, device="cuda", generator=gen)).bfloat16()
    if dtype == "bf16":
        ref = F.linear(x, w)
        out = gemm_tactics.mm(plan, x, w)
    else:
        xq, xs = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
        wq, ws = mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=True)
        ref = torch._scaled_mm(
            xq,
            wq.t(),
            scale_a=xs.view(torch.float8_e8m0fnu),
            scale_b=ws.view(torch.float8_e8m0fnu),
            out_dtype=torch.bfloat16,
        )
        out = gemm_tactics.mm(plan, xq, wq, xs.view(torch.uint8), ws.view(torch.uint8))
    torch.accelerator.synchronize()
    # Same math (FP32 accumulation over K, one BF16 rounding); only the order of
    # the FP32 sums inside a tile may differ: at most one BF16 ulp.
    diff = (out.float() - ref.float()).abs()
    ulp = ref.float().abs().clamp_min(1e-30) * 2.0**-7
    assert bool((diff <= ulp).all()), (
        f"{key} rows={rows}: {(diff > 0).float().mean().item():.2e} of outputs "
        f"differ, max {(diff / ulp).max().item():.2f} ulp"
    )


def test_bf16_dispatch_routes_prefill_rows_only():
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.models.kimi_k3.nvidia.specialized_prefill import gemm_tactics

    bf16 = [key for key in gemm_tactics.PINS if key[0] == "bf16"]
    if not bf16:
        pytest.skip("no BF16 pins")
    _, n, k = bf16[0]

    class Lin(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                (0.02 * torch.randn(n, k, device="cuda")).bfloat16(),
                requires_grad=False,
            )
            self.quant_method = UnquantizedLinearMethod()

    lin = Lin()
    assert gemm_tactics.enable(lin) == 1
    method = lin.quant_method
    assert isinstance(method, UnquantizedLinearMethod)
    calls = []
    real_mm = gemm_tactics.mm
    gemm_tactics.mm = lambda *a, **kw: calls.append(a[1].shape[0]) or real_mm(*a, **kw)
    try:
        for rows in (16, gemm_tactics.MIN_ROWS - 1, gemm_tactics.MIN_ROWS):
            x = torch.randn(rows, k, device="cuda").bfloat16()
            out = method.apply(lin, x)
            assert out.shape == (rows, n)
    finally:
        gemm_tactics.mm = real_mm
    assert calls == [gemm_tactics.MIN_ROWS]


def test_low_latency_linear_takes_the_pin_at_prefill_rows():
    """The latent-MoE down_proj carries the Kimi low-latency method (small-row
    CuTe plans); its prefill rows must reach the pin, not the stock fallback."""
    from vllm.models.kimi_k3.nvidia.low_latency_gemm import (
        KimiK3LowLatencyLinearMethod,
    )
    from vllm.models.kimi_k3.nvidia.specialized_prefill import gemm_tactics

    key = ("bf16", 3584, 7168)
    if key not in gemm_tactics.PINS:
        pytest.skip("no latent down_proj pin")
    _, n, k = key

    class Lin(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                (0.02 * torch.randn(n, k, device="cuda")).bfloat16(),
                requires_grad=False,
            )
            self.quant_method = KimiK3LowLatencyLinearMethod({}, {})

    lin = Lin()
    assert gemm_tactics.enable(lin) == 1
    method = lin.quant_method
    assert isinstance(method, KimiK3LowLatencyLinearMethod)
    calls = []
    real_mm = gemm_tactics.mm
    gemm_tactics.mm = lambda *a, **kw: calls.append(a[1].shape[0]) or real_mm(*a, **kw)
    try:
        for rows in (8, gemm_tactics.MIN_ROWS - 1, gemm_tactics.MIN_ROWS):
            x = torch.randn(rows, k, device="cuda").bfloat16()
            out = method.apply(lin, x)
            assert out.shape == (rows, n)
            if rows == gemm_tactics.MIN_ROWS:
                ref = torch.nn.functional.linear(x, lin.weight)
                torch.testing.assert_close(out, ref, rtol=1.6e-2, atol=1e-3)
    finally:
        gemm_tactics.mm = real_mm
    assert calls == [gemm_tactics.MIN_ROWS]
