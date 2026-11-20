# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerics of the shared-expert SiTU + MXFP8 quant in the routed experts'
MUFU-light form (``mxfp8_act`` mode 4, the MegaMoE FC1 SiTU) against the
precise-tanhf form (mode 1, stock ``situ_and_mul`` + MXFP8 quant).

The fast form is not bitwise equal; the gate is BF16 rounding noise: the
BF16-rounded activations differ by at most one BF16 ulp, and after the MXFP8
quant the codes differ by at most one E4M3 step on a small fraction of
elements, with the same E8M0 scales up to one binade."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(100)),
    reason="MXFP8 producers target SM100+",
)

# Kimi K3 shared-expert SiTU betas, plus generic ones (beta/2 not an integer).
BETAS = [(4.0, 25.0), (4.0, None), (3.0, 10.0)]


def _dequant(qa, rows, d):
    from vllm.models.kimi_k3.nvidia.mxfp8 import unswizzle_mxfp8_scale

    sf = unswizzle_mxfp8_scale(qa.scale, rows, d).to(torch.int32)
    scale = torch.exp2((sf - 127).float()).repeat_interleave(32, dim=1)
    return qa.data.float() * scale, sf


def _situ_ref(x, beta, linear_beta):
    d = x.shape[-1] // 2
    g, u = x[:, :d].double(), x[:, d:].double()
    gate = beta * torch.tanh(g / beta) * torch.sigmoid(g)
    up = linear_beta * torch.tanh(u / linear_beta) if linear_beta else u
    return gate * up


@pytest.mark.parametrize("beta,linear_beta", BETAS)
@pytest.mark.parametrize("std", [0.5, 3.0, 12.0])
def test_situ_fast_within_bf16_noise(beta, linear_beta, std):
    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act

    torch.manual_seed(int(std * 10))
    rows, d = 4096, 1536
    x = (torch.randn(rows, 2 * d, device="cuda") * std).to(torch.bfloat16)
    # Saturation / special values: large |g|, |u|, exact zeros.
    x[0, :64] = torch.linspace(-80, 80, 64, device="cuda").to(torch.bfloat16)
    x[0, d : d + 64] = torch.linspace(-300, 300, 64, device="cuda").to(torch.bfloat16)
    x[1] = 0
    precise = mxfp8_act.situ_and_mul_quantize(x, beta, linear_beta)
    fast = mxfp8_act.situ_and_mul_quantize_fast(x, beta, linear_beta)
    p, sp = _dequant(precise, rows, d)
    f, sf = _dequant(fast, rows, d)
    assert torch.isfinite(f).all()
    assert (sf - sp).abs().max().item() <= 1
    # Error vs the FP64 reference, per MXFP8 block, in units of the block's
    # E4M3 step at its scale: the fast form adds at most BF16 noise.
    ref = _situ_ref(x, beta, linear_beta)
    # Largest E4M3 step in a block ([256, 448] codes): 32 x the block scale.
    step = torch.exp2((sp - 127).double()).repeat_interleave(32, dim=1) * 32
    err_p = ((p.double() - ref).abs() / step).max().item()
    err_f = ((f.double() - ref).abs() / step).max().item()
    mismatch = (fast.data.view(torch.uint8) != precise.data.view(torch.uint8))
    frac = mismatch.float().mean().item()
    print(
        f"beta={beta} lbeta={linear_beta} std={std}: code mismatch {frac:.2e}, "
        f"max err/step precise {err_p:.3f} fast {err_f:.3f}"
    )
    assert frac < 2e-2, frac
    assert err_f <= err_p + 1.0 + 1e-6


def test_situ_auto_dispatch_by_rows():
    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act

    torch.manual_seed(0)
    for rows, fast in ((64, False), (mxfp8_act.SITU_FAST_MIN_ROWS, True)):
        x = (torch.randn(rows, 512, device="cuda") * 4).to(torch.bfloat16)
        got = mxfp8_act.situ_and_mul_quantize_auto(x, 4.0, 25.0)
        fn = (
            mxfp8_act.situ_and_mul_quantize_fast
            if fast
            else mxfp8_act.situ_and_mul_quantize
        )
        want = fn(x, 4.0, 25.0)
        assert torch.equal(got.data.view(torch.uint8), want.data.view(torch.uint8))
        assert torch.equal(got.scale, want.scale)


def _ulp_dist(a, b):
    """BF16 ulp distance (+0 and -0 coincide)."""

    def ordered(t):
        i = t.view(torch.int16).int()
        return torch.where(i < 0, -(i & 0x7FFF), i)

    return (ordered(a) - ordered(b)).abs()


@pytest.mark.parametrize("beta,linear_beta", BETAS)
@pytest.mark.parametrize("rows,d", [(4096, 1536), (257, 18432)])
def test_situ_fast_bf16_is_mode4_values(beta, linear_beta, rows, d):
    """The BF16 fast form (prefill dense SiTU MLP) returns exactly the BF16
    values mode 4 quantizes, within one BF16 ulp of the precise form."""
    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act
    from vllm.models.kimi_k3.nvidia.ops.situ_and_mul import situ_and_mul

    torch.manual_seed(rows + d)
    x = (torch.randn(rows, 2 * d, device="cuda") * 4.0).to(torch.bfloat16)
    x[0, :64] = torch.linspace(-80, 80, 64, device="cuda").to(torch.bfloat16)
    x[0, d : d + 64] = torch.linspace(-300, 300, 64, device="cuda").to(torch.bfloat16)
    x[1] = 0
    fast = mxfp8_act.situ_and_mul_fast_bf16(x, beta, linear_beta)
    precise = situ_and_mul(x, beta, linear_beta)
    assert torch.isfinite(fast.float()).all()
    assert _ulp_dist(fast, precise).max().item() <= 1
    want = mxfp8_act.situ_and_mul_quantize_fast(x, beta, linear_beta)
    got = mxfp8_act.quantize(fast)
    assert torch.equal(got.data.view(torch.uint8), want.data.view(torch.uint8))
    assert torch.equal(got.scale, want.scale)

