# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 SP (TP4) decode MXFP8 projections (NVFP4 checkpoint: KDA in / o
projections and the replicated shared expert are online MXFP8) on Rubin:
``gemm.mxfp8.Mxfp8Projection`` at the D3-32 per-GPU shapes vs stock's online-MXFP8
linear (``KimiK3Mxfp8Kernel``: FlashInfer quantize, then the CuTe DSL GEMM at <= 16
rows or cuBLASLt ``_scaled_mm`` above):

* the input quantization into the shared capacity scratch is bitwise FlashInfer's
  ``mxfp8_quantize`` (values and swizzled scales of the R rows);
* the kind stock picks at R is bitwise stock; every other kind is within one BF16
  rounding of an FP32 dequantized reference (the FP32 summation order differs);
* a pre-quantized ``Mxfp8Buffer`` input (the fused producers' path) equals the BF16
  input path bitwise;
* a captured graph replays every R; unknown kinds / unprepared rows raise.

One GPU, well under a minute."""

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("needs a Rubin GPU (compute capability 10.7)", allow_module_level=True)

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (  # noqa: E402
    mxfp8_e4m3_quantize,
    swizzle_mxfp8_scale,
)
from vllm.models.kimi_k3.nvidia.mxfp8 import use_cute_dsl  # noqa: E402
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.mxfp8 import (  # noqa: E402
    MX_KINDS,
    Mxfp8Projection,
    DENSE_MAX_ROWS,
    DENSE_TOKEN_SPLIT,
    mxfp8_tactics,
    stock_kind,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.shapes import (  # noqa: E402
    Unsupported,
)

DEVICE = torch.device("cuda", 0)
# family: [N, K] per GPU at TP4 (KDA) / replicated (shared expert)
SHAPES = {
    "kda_in": (12448, 7168),
    "o_proj": (7168, 3072),
    "shared_gu": (12288, 7168),
    "shared_down": (7168, 6144),
}
# D3-32: R = 16 r gathered rows (KDA), n = 4 r shard rows (shared expert)
ROWS = {
    "kda_in": (4, 16, 20, 128, 384, 512),
    "o_proj": (4, 16, 128, 256, 384),
    "shared_gu": (1, 4, 16, 32, 96, 128),
    "shared_down": (1, 4, 32, 96, 128),
}


def mx_weight(n, k, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    w = (torch.randn(n, k, device=DEVICE, generator=g) * 0.02).bfloat16()
    wq, ws = mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=False)
    ws = ws.view(torch.uint8).reshape(n, k // 32)
    return wq.contiguous(), swizzle_mxfp8_scale(ws, M=n, K=k).contiguous(), ws


def stock(x, wq, sw, n, k):
    """``KimiK3Mxfp8Kernel.apply_weights`` on a BF16 input."""
    import flashinfer

    xq, xs = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
    if use_cute_dsl(n, k, x.shape[0]):
        return flashinfer.mm_mxfp8(
            xq, wq.t(), xs, sw, out_dtype=torch.bfloat16, backend="cute-dsl"
        )
    return torch._scaled_mm(
        xq, wq.t(), scale_a=xs.view(torch.float8_e8m0fnu),
        scale_b=sw.view(torch.float8_e8m0fnu), out_dtype=torch.bfloat16,
    )  # fmt: skip


def reference(x, wq, ws):
    """FP32 dequantized reference (FlashInfer's activation quantization)."""
    xq, xs = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=False)
    r, k = x.shape
    xs = xs.view(torch.uint8).reshape(r, k // 32).float()
    xd = (xq.float().view(r, k // 32, 32) * torch.exp2(xs - 127).unsqueeze(-1))
    n = wq.shape[0]
    wd = wq.float().view(n, k // 32, 32) * torch.exp2(ws.float() - 127).unsqueeze(-1)
    return xd.view(r, k) @ wd.view(n, k).t()


@pytest.mark.parametrize("family", list(SHAPES))
def test_kinds_vs_stock(family):
    torch.cuda.set_device(DEVICE)
    n, k = SHAPES[family]
    wq, sw, ws = mx_weight(n, k, 1)
    rows = ROWS[family]
    for kind in MX_KINDS:
        rows = [
            r for r in ROWS[family]
            if kind != "dense_mx8"
            or (r <= DENSE_MAX_ROWS and ((n, k) not in DENSE_TOKEN_SPLIT or r % 128 == 0))
        ]  # fmt: skip
        proj = Mxfp8Projection(
            (wq, sw), rows, family=f"t_{family}", tactics=dict.fromkeys(rows, kind)
        )
        for r in rows:
            x = torch.randn(r, k, device=DEVICE).bfloat16()
            out = torch.empty(r, n, device=DEVICE, dtype=torch.bfloat16)
            proj.run(x, out)
            # the quantized input in the capacity scratch is FlashInfer's
            view = proj.input.view(r)
            xq, xs = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
            assert torch.equal(view.data.view(torch.uint8), xq.view(torch.uint8))
            assert torch.equal(view.scale[: xs.numel()], xs.view(torch.uint8))
            if kind == stock_kind(n, k, r):
                assert torch.equal(out, stock(x, wq, sw, n, k)), (kind, r)
            ref = reference(x, wq, ws)
            tol = ref.abs().amax(dim=1, keepdim=True) * 2.0**-7 + 1e-3
            assert ((out.float() - ref).abs() <= tol).all(), (kind, r)
            # pre-quantized input path == BF16 input path
            again = torch.empty_like(out)
            proj.run(view, again)
            assert torch.equal(again, out), (kind, r)


def test_policy_and_graph():
    torch.cuda.set_device(DEVICE)
    n, k = SHAPES["shared_down"]
    wq, sw, _ = mx_weight(n, k, 2)
    rows = (4, 32, 96)
    tactics = mxfp8_tactics(n, k, rows)
    assert set(tactics) == set(rows) and set(tactics.values()) <= set(MX_KINDS)
    proj = Mxfp8Projection((wq, sw), rows, family="t_graph")
    xs = {r: torch.randn(r, k, device=DEVICE).bfloat16() for r in rows}
    outs = {r: torch.empty(r, n, device=DEVICE, dtype=torch.bfloat16) for r in rows}
    for r in rows:  # warm (JIT) outside the capture
        proj.run(xs[r], outs[r])
    graphs = {}
    for r in rows:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            proj.run(xs[r], outs[r])
        graphs[r] = g
    for r in rows:
        xs[r].copy_(torch.randn_like(xs[r], dtype=torch.float32).bfloat16())
        graphs[r].replay()
        torch.cuda.synchronize()
        want = torch.empty_like(outs[r])
        proj.run(xs[r], want)
        assert torch.equal(outs[r], want)
    with pytest.raises(Unsupported):
        proj.run(torch.randn(8, k, device=DEVICE).bfloat16(), outs[4])
    with pytest.raises(Unsupported):
        Mxfp8Projection((wq, sw), rows, family="t_bad", tactics=dict.fromkeys(rows, "x"))
