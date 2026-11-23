# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The drafter's token-wide 2-CTA split-K GEMM (``cutedsl.gemm.wide``) against the
stock BF16 linear's precision: one FP32 sum (order may differ) rounded once, so at
most one BF16 ulp from an FP32 reference; with ``silu`` the stock silu_and_mul on
the BF16 gate / up (gate or up may sit one ulp off, so two ulps). Repeated launches
check that the split counters re-arm (graph replays reuse them)."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(100)),
    reason="tcgen05 2-CTA GEMM targets SM100+",
)


def _ulps(got, ref):
    """Max error in BF16 ulps of the reference, with an FP32-reorder allowance of
    1e-4 of the tensor's magnitude for outputs near cancellation."""
    ref = ref.float()
    ulp = (ref.abs().clamp_min(1e-30).log2().floor() - 7).exp2()
    slack = (got.float() - ref).abs() - 1e-4 * ref.abs().amax()
    return (slack.clamp_min(0) / ulp).max().item()


@pytest.mark.parametrize(
    "rows,n,k,split,silu",
    [
        (288, 512, 1024, 3, False),
        (192, 768, 1536, 4, False),
        (96, 512, 2048, 1, False),
        (288, 1024, 1024, 3, True),
        (96, 512, 768, 2, True),
    ],
)
def test_wide_gemm(rows, n, k, split, silu):
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.wide import (
        PreparedWide,
    )

    torch.manual_seed(rows + n + k + split)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    gemm = PreparedWide(w, {rows: split}, silu=silu).prepare()
    out = torch.empty(rows, n // 2 if silu else n, device="cuda", dtype=torch.bfloat16)
    full = x.float() @ w.float().t()
    if silu:
        g, u = full[:, : n // 2].bfloat16(), full[:, n // 2 :].bfloat16()
        h =torch.nn.functional.silu(g.float()).bfloat16().float()
        ref = (h * u.float()).bfloat16()
        bound = 2
    else:
        ref = full.bfloat16()
        bound = 1
    for _ in range(3):
        out.fill_(float("nan"))
        gemm.run(x, out)
        torch.cuda.synchronize()
        assert not out.isnan().any()
        assert _ulps(out, ref) <= bound


@pytest.mark.parametrize(
    "rows,n,k,split,partials",
    [
        (96, 3584, 7168, 7, False),  # latent_down at r24: uneven token regions
        (96, 4480, 7168, 5, True),  # router | latent_down: n % 256 != 0, partials
        # latent up partials (sp_mlp.LATENT_UP_PARTS): uneven 3-way token regions
        # at 128 / 160, even at 192
        (128, 7168, 3584, 3, True),
        (160, 7168, 3584, 3, True),
        (192, 7168, 3584, 3, True),
    ],
)
def test_wide_gemm_decode_front(rows, n, k, split, partials):
    """The SP decode MoE front shapes with staggered K order (``rotate``); with
    ``partials`` the split-K FP32 partials summed in split order (route_stage's
    sum) must match the FP32 reference to FP32-reorder accuracy."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.wide import (
        PreparedWide,
    )

    torch.manual_seed(rows + n + split)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    policy = {rows: (split, (1,), None, (0, 3), False, True)}
    gemm = PreparedWide(w, policy, partials=partials).prepare()
    full = x.float() @ w.float().t()
    if partials:
        out = torch.empty(split, rows, n, device="cuda", dtype=torch.float32)
    else:
        out = torch.empty(rows, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        out.fill_(float("nan"))
        gemm.run(x, out)
        torch.cuda.synchronize()
        assert not out.isnan().any()
        if partials:
            got = out[0].clone()
            for s in range(1, split):
                got += out[s]
            assert (got - full).abs().max() <= 1e-5 * full.abs().max()
        else:
            assert _ulps(out, full.bfloat16()) <= 1
