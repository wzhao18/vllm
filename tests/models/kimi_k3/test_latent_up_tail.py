# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MegaMoE latent up-projection with the in-place beta-add (specialized decode,
``gemm.latent_up``): stock's ``residual.addmm_`` up to the FP32 summation order."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda()
    or torch.cuda.get_device_capability() != (10, 7),
    reason="sm_107 (Rubin) only",
)


@pytest.mark.parametrize("t", [32, 64, 96])
def test_latent_up_matches_addmm(t):
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.latent_up import (
        PreparedLatentUp,
    )

    torch.manual_seed(t)
    n, k = 7168, 3584
    x = torch.randn(t, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    residual = torch.randn(t, n, device="cuda", dtype=torch.bfloat16)
    ref = residual.clone().addmm_(x, w.t())  # stock: FP32 sum + residual, one rounding
    exact = (x.float() @ w.float().t() + residual.float()).to(torch.bfloat16)

    up = PreparedLatentUp(w, (t,)).prepare()
    out = residual.clone()
    got = up.run(x, out)
    assert got.data_ptr() == out.data_ptr()  # in place
    torch.testing.assert_close(out.float(), exact.float(), atol=2e-2, rtol=1.6e-2)
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=1.6e-2)
    # Every element is one rounding of the same FP32 sum up to its order: at most
    # one BF16 ulp from the FP64 reference.
    exact64 = (x.double() @ w.double().t() + residual.double())
    ulp = exact64.abs().clamp_min(2**-126).log2().floor().exp2() * 2**-7
    assert ((out.double() - exact64).abs() <= ulp * 1.01).float().mean() > 0.999
    # Deterministic (rank-ordered split-K sum).
    again = residual.clone()
    up.run(x, again)
    assert torch.equal(out, again)


def test_latent_up_refuses_unprepared_rows():
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.latent_up import (
        PreparedLatentUp,
    )

    w = torch.zeros(7168, 3584, device="cuda", dtype=torch.bfloat16)
    assert not PreparedLatentUp.supports(128)
    with pytest.raises(ValueError):
        PreparedLatentUp(w, (128,))
