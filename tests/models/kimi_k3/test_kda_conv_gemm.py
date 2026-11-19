# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KDA in_proj Q|K|V GEMM with the conv epilogue: bit for bit equal to the
in_proj GEMM + ``kda_conv.qkv_conv`` (outputs and FP32 conv state)."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and current_platform.is_device_capability_family(100)
        and torch.cuda.get_device_capability()[1] >= 7
    ),
    reason="sm_107 block-scaled GEMM",
)


def _operands(m, n, k):
    from flashinfer import mxfp8_quantize

    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05
    xq, xs = mxfp8_quantize(x, is_sf_swizzled_layout=True)
    wq, ws = mxfp8_quantize(w, is_sf_swizzled_layout=True)
    return xq, xs, wq, ws


@pytest.mark.parametrize("drain", [True, False])
@pytest.mark.parametrize(
    "seqlens",
    [[1000], [1, 2, 3, 130, 125, 256, 4], [300, 1, 127, 2, 260]],
)
def test_conv_gemm_bitwise(seqlens, drain):
    from vllm.models.kimi_k3.nvidia.ops import kda_conv
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm.kda_conv_gemm import (  # noqa: E501
        KdaConvGemm,
        reference_inv_norms,
    )

    torch.manual_seed(1)
    m, p, k = 1024, 256, 512
    n = 3 * p
    xq, xs, wq, ws = _operands(m, n, k)
    cw = torch.randn(n, 4, device="cuda") * 0.5
    qsl = torch.tensor(
        [0] + torch.tensor(seqlens).cumsum(0).tolist(), dtype=torch.int32, device="cuda"
    )
    nseq = len(seqlens)
    lines = torch.arange(2, nseq + 2, dtype=torch.int32, device="cuda")
    has_init = torch.arange(nseq, device="cuda") % 2 == 1
    state0 = torch.randn(nseq + 3, n, 4, device="cuda").bfloat16().float()[:, :, :3]

    proj = torch._scaled_mm(
        xq, wq.t(), scale_a=xs.view(torch.float8_e8m0fnu),
        scale_b=ws.view(torch.float8_e8m0fnu), out_dtype=torch.bfloat16,
    )  # fmt: skip
    st_ref = state0.clone()
    ref = kda_conv.qkv_conv(proj, cw, st_ref, qsl, lines, has_init, p)
    st = state0.clone()
    gemm = KdaConvGemm(wq, ws, cw, p, tactic=((512, 256), (2, 1), 256, 8, "n"), drain=drain)
    y, norms = gemm(xq, xs, st, qsl, lines, has_init, with_norms=True)
    tot = int(qsl[-1])
    torch.testing.assert_close(y[:, :tot], ref[:, :tot], rtol=0, atol=0)
    torch.testing.assert_close(st, st_ref, rtol=0, atol=0)
    torch.testing.assert_close(
        norms[:, :tot], reference_inv_norms(ref[:, :tot], p), rtol=1e-5, atol=0
    )
