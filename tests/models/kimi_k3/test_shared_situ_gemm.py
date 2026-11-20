# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill shared gate_up GEMM with the SiTU + MXFP8 epilogue (sm_107) is
bitwise equal to cuBLASLt gate_up + situ_and_mul_quantize_auto.

Exact-input operands (small-integer FP8, power-of-two scales) make every
summation order give the same FP32 accumulator, so the comparison checks the
epilogue bytes (values and scales), padding rows included.
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability((10, 7)), reason="sm_107 only"
)


def _operands(m, n, k, dev):
    g = torch.Generator(device=dev).manual_seed(m)
    x = torch.randint(-2, 3, (m, k), device=dev, generator=g).float()
    w = torch.randint(-2, 3, (n, k), device=dev, generator=g).float()
    sx = torch.randint(122, 126, (((m + 127) // 128) * 128 * k // 32,), device=dev,
                       generator=g).to(torch.uint8)  # fmt: skip
    sw = torch.randint(117, 121, (n * k // 32,), device=dev, generator=g).to(torch.uint8)
    return x.to(torch.float8_e4m3fn), sx, w.to(torch.float8_e4m3fn), sw


@pytest.mark.parametrize("m", [16384, 4100, 2048])
@pytest.mark.parametrize("betas", [(4.0, 25.0), (4.0, None)])
def test_shared_situ_gemm_bitwise(m, betas):
    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm import (
        shared_situ_gemm as ssg,
    )

    beta, lbeta = betas
    dev = torch.device("cuda")
    inter, k = 1536, 7168
    x, sx, w, sw = _operands(m, 2 * inter, k, dev)
    y = torch._scaled_mm(
        x, w.t(),
        scale_a=sx.view(torch.float8_e8m0fnu), scale_b=sw.view(torch.float8_e8m0fnu),
        out_dtype=torch.bfloat16,
    )  # fmt: skip
    ref = mxfp8_act.Mxfp8Buffer(m, inter, dev)
    ref.data.view(torch.uint8).zero_()
    ref.scale.zero_()
    mxfp8_act.situ_and_mul_quantize_auto(y, beta, lbeta, out=ref)

    wi, swi = w.clone(), sw.clone()
    ssg.interleave_gate_up_rows(wi, swi)
    out = mxfp8_act.Mxfp8Buffer(m, inter, dev)
    out.data.view(torch.uint8).fill_(0x7F)
    out.scale.zero_()
    ssg.SharedSituGemm(wi, swi, beta, lbeta)(x, sx, out)
    torch.cuda.synchronize()
    assert torch.equal(out.data.view(torch.uint8), ref.data.view(torch.uint8))
    assert torch.equal(out.scale, ref.scale)


@pytest.mark.parametrize("m", [4100])
def test_shared_situ_gemm_die_split_bitwise(m):
    """The die-split scheduler (weight halves homed per die, per-die tile
    claims) writes the same bytes at the deployed width (inter 6144)."""
    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm import (
        die_sched,
    )
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm import (
        shared_situ_gemm as ssg,
    )

    dev = torch.device("cuda")
    inter, k = 6144, 7168
    x, sx, w, sw = _operands(m, 2 * inter, k, dev)
    wi, swi = w.clone(), sw.clone()
    ssg.interleave_gate_up_rows(wi, swi)
    tactic = ssg.die_tactic(wi)
    if tactic is None:
        pytest.skip("single-die device")
    ref = mxfp8_act.Mxfp8Buffer(m, inter, dev)
    ssg.SharedSituGemm(wi, swi, 4.0, 25.0)(x, sx, ref)
    gemm = ssg.SharedSituGemm(die_sched.localize_rows(wi), swi, 4.0, 25.0, tactic=tactic)
    out = mxfp8_act.Mxfp8Buffer(m, inter, dev)
    for _ in range(2):  # the second launch runs on the counters the first reset
        out.data.view(torch.uint8).fill_(0x7F)
        out.scale.zero_()
        gemm(x, sx, out)
        torch.cuda.synchronize()
        assert torch.equal(out.data.view(torch.uint8), ref.data.view(torch.uint8))
        assert torch.equal(out.scale, ref.scale)
