# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KDA prefill in_proj Q|K|V GEMM with the conv + SiLU epilogue (sm_107,
specialized_prefill.sm107_gemm.kda_conv_gemm) is bitwise equal to the
cuBLASLt in_proj followed by ops/kda_qkv_conv: Q|K|V planes and the FP32 conv
state, for sequence starts inside and at 128-row blocks, sequences shorter
than the conv window, padding rows past the last sequence and accumulators
deep enough below zero to take the exact ``div.full`` SiLU path.

Exact-input operands (small-integer FP8, power-of-two scales) make every
summation order give the same FP32 accumulator, so the comparison checks the
epilogue.
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability((10, 7)), reason="sm_107 only"
)

PART, K = 512, 1024


def _operands(m, n, k, dev, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randint(-2, 3, (m, k), device=dev, generator=g).float()
    w = torch.randint(-2, 3, (n, k), device=dev, generator=g).float()
    sx = torch.randint(118, 122, (((m + 127) // 128) * 128 * k // 32,), device=dev,
                       generator=g).to(torch.uint8)  # fmt: skip
    sw = torch.randint(117, 121, (n * k // 32,), device=dev, generator=g).to(torch.uint8)
    return x.to(torch.float8_e4m3fn), sx, w.to(torch.float8_e4m3fn), sw


def _run(seqlens, m, deep=False, tactic=None, seed=0):
    from vllm.models.kimi_k3.nvidia.ops import kda_conv
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm.kda_conv_gemm import (  # noqa: E501
        KdaConvGemm,
    )

    dev = torch.device("cuda")
    n = 3 * PART
    xq, xs, wq, ws = _operands(m, n, K, dev, seed)
    proj = torch._scaled_mm(
        xq, wq.t(),
        scale_a=xs.view(torch.float8_e8m0fnu), scale_b=ws.view(torch.float8_e8m0fnu),
        out_dtype=torch.bfloat16,
    )  # fmt: skip
    g = torch.Generator(device=dev).manual_seed(seed + 1)
    cw = torch.randn(n, 4, device=dev, generator=g) * 0.5
    if deep:
        # Columns whose conv reaches acc < -87 (d = 1 + exp(-acc) > 2^126).
        cw[::7] *= 60.0 / float(proj.float().abs().mean())
    cw = cw.contiguous()
    qsl = torch.tensor([0, *torch.tensor(seqlens).cumsum(0).tolist()], dtype=torch.int32,
                       device=dev)  # fmt: skip
    nseq = len(seqlens)
    lines = torch.randperm(2 * nseq + 2, device=dev, generator=g)[:nseq].to(torch.int32)
    has_init = torch.arange(nseq, device=dev) % 2 == 1
    state0 = torch.randn(2 * nseq + 2, n, 4, device=dev, generator=g)
    state0 = state0.to(torch.bfloat16).float()[:, :, :3]

    st_ref = state0.clone()
    ref = kda_conv.qkv_conv(proj, cw, st_ref, qsl, lines, has_init, PART)
    gemm = KdaConvGemm(wq, ws, cw, PART, tactic=tactic)
    st = state0.clone()
    out, _ = gemm(xq, xs, st, qsl, lines, has_init)
    torch.cuda.synchronize()
    tot = int(qsl[-1])
    assert torch.equal(out[:, :tot], ref[:, :tot])
    assert torch.equal(st, st_ref)
    if deep:
        # Tiny nonzero outputs only come from the exact path's band.
        y = ref[:, :tot].float()
        assert bool(((y != 0) & (y.abs() < 1e-30)).any())


@pytest.mark.parametrize(
    "seqlens,m",
    [
        ([1365] * 3, 4096),
        ([1, 2, 3, 125, 128, 3, 256, 600, 5], 1300),
        ([127, 1, 128, 129, 2], 512),
        ([700, 300], 1100),
    ],
)
def test_kda_conv_gemm_bitwise(seqlens, m):
    _run(seqlens, m)


def test_kda_conv_gemm_exact_silu_path():
    _run([900, 513, 7], 1420, deep=True, seed=3)


def test_kda_conv_gemm_old_column_pass():
    # The previous per-row column pass (kept for the inverse-norm variant).
    _run([1, 2, 3, 125, 128, 3, 256, 600, 5], 1300,
         tactic=((512, 256), (4, 1), 256, 8, "n", 0, 0, "ct1"))  # fmt: skip
