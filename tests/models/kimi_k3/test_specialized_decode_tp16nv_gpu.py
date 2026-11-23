# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 TP16 (DA) NVFP4 checkpoint, specialized decode table entries, one GPU:

- ``test_tp16_kda_in_pre``: the TP16 KDA in-projection shard [3216, 7168] (online
  MXFP8, ``gemm.mxfp8.MX_TUNED``) runs the prewait CuTe DSL kernel ("pre") at 17-128
  rows. At those rows it is bitwise FlashInfer's fi_cute kernel on the same operands
  and within the FP32 accumulation bound of the FP64 product, as stock's kind
  (scaled_mm) is; both kinds read the same MXFP8 activation (FlashInfer's
  ``mxfp8_quantize`` semantics) and weight bytes.

Needs a Rubin (SM107) GPU (the prewait kernel's architecture)."""

import pytest
import torch

N, K = 3216, 7168
ROWS = (17, 24, 64, 128)


def _rubin():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)


pytestmark = pytest.mark.skipif(not _rubin(), reason="needs a Rubin (SM107) GPU")


def _weight(gen, dev):
    """FP8 rows and a constant E8M0 scale (2^0): the F8_128x4 swizzle of a
    constant is the constant, so the FP64 reference reads the weight directly."""
    w = (torch.randn((N, K), generator=gen, device=dev) * 2).to(torch.float8_e4m3fn)
    scale = torch.full(
        (-(-N // 128) * 128 * (K // 32),), 127, device=dev, dtype=torch.uint8
    )
    return w, scale


def _activation(x):
    """FP64 values of x's MXFP8 quantization (FlashInfer's ``mxfp8_quantize``, scales
    unswizzled), and its FP8 bytes."""
    import flashinfer

    q, sf = flashinfer.mxfp8_quantize(x, is_sf_swizzled_layout=False)
    sf = sf.view(torch.uint8).reshape(x.shape[0], -1)[:, : K // 32]
    exps = sf.to(torch.float64) - 127
    return q.to(torch.float64) * torch.exp2(exps).repeat_interleave(32, dim=1), q


def test_tp16_kda_in_pre():
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.mxfp8 import (
        Mxfp8Projection,
        mxfp8_tactics,
        stock_kind,
    )

    dev = torch.device("cuda")
    tactics = mxfp8_tactics(N, K, (16, *ROWS, 129))
    assert tactics[16] == stock_kind(N, K, 16) and tactics[129] == stock_kind(N, K, 129)
    assert all(tactics[r] == "pre" for r in ROWS), tactics
    gen = torch.Generator(device=dev).manual_seed(5)
    weight = _weight(gen, dev)
    for r in ROWS:
        x = torch.randn((r, K), generator=gen, device=dev, dtype=torch.bfloat16)
        outs = {}
        for kind in ("pre", "fi_cute", "scaled_mm"):
            proj = Mxfp8Projection(weight, [r], family=f"t{kind}", tactics={r: kind})
            q = proj.input.quantize(x)
            out = torch.empty((r, N), device=dev, dtype=torch.bfloat16)
            proj.run(q, out)
            outs[kind] = out
        torch.cuda.synchronize()
        assert torch.isfinite(outs["pre"]).all()
        assert torch.equal(outs["pre"], outs["fi_cute"]), f"R{r}: pre != fi_cute"
        # FP64 reference on the same MXFP8 operands
        xd, xq = _activation(x)
        assert torch.equal(xq.view(torch.uint8), q.data.view(torch.uint8))
        wd = weight[0].to(torch.float64)
        exact = xd @ wd.t()
        acc = K * 2.0**-24 * (xd.abs() @ wd.abs().t())
        top = exact.abs() + acc
        ulp = torch.exp2(torch.floor(torch.log2(top.clamp_min(1e-30))))
        limit = (acc + ulp * 2.0**-8) * 1.001
        for kind in ("pre", "scaled_mm"):
            err = (outs[kind].double() - exact).abs()
            assert (err <= limit).all(), f"R{r} {kind}: beyond the FP32 bound"
