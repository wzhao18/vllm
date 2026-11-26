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


def _cases():
    from vllm.models.kimi_k3.nvidia.specialized_prefill import gemm_tactics

    return [
        (key, rows)
        for key in _pins()
        for rows in sorted(
            {gemm_tactics.min_rows(*key), gemm_tactics.min_rows(*key) + 4, 4096, 4100}
        )
    ]


@pytest.mark.parametrize("key,rows", _cases())
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
    from vllm.models.kimi_k3.nvidia.specialized_prefill import gemm_tactics

    bf16 = [key for key in gemm_tactics.PINS if key[0] == "bf16"]
    if not bf16:
        pytest.skip("no BF16 pins")
    # One pin on the default floor and every prefill-worker pin on its own floor.
    for key in {
        bf16[0],
        *(key for key in bf16 if key in gemm_tactics.PREFILL_WORKER_PINS),
    }:
        _check_dispatch(*key[1:])


def test_prefill_worker_pins_stay_off_elsewhere():
    """Decode / aggregated instances keep the stock dispatch for the prefill-worker
    pins at every row count (decode graph captures reach 2048 rows)."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.models.kimi_k3.nvidia.specialized_prefill import gemm_tactics

    for _, n, k in gemm_tactics.PREFILL_WORKER_PINS:
        lin = _lin(n, k)
        assert gemm_tactics.enable(lin, prefill_worker=False) == 0
        assert type(lin.quant_method) is UnquantizedLinearMethod


def _lin(n, k):
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    class Lin(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                (0.02 * torch.randn(n, k, device="cuda")).bfloat16(),
                requires_grad=False,
            )
            self.quant_method = UnquantizedLinearMethod()

    return Lin()


def _check_dispatch(n, k):
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.models.kimi_k3.nvidia.specialized_prefill import gemm_tactics

    floor = gemm_tactics.min_rows("bf16", n, k)
    lin = _lin(n, k)
    assert gemm_tactics.enable(lin, prefill_worker=True) == 1
    method = lin.quant_method
    assert isinstance(method, UnquantizedLinearMethod)
    calls = []
    real_mm = gemm_tactics.mm
    gemm_tactics.mm = lambda *a, **kw: calls.append(a[1].shape[0]) or real_mm(*a, **kw)
    try:
        for rows in (16, floor - 1, floor):
            x = torch.randn(rows, k, device="cuda").bfloat16()
            out = method.apply(lin, x)
            assert out.shape == (rows, n)
    finally:
        gemm_tactics.mm = real_mm
    assert calls == [floor]
