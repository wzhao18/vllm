# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 SP decode ``ltpin`` GEMMs (cuBLASLt on algorithms pinned per shape and
rows, ``specialized_decode.cutedsl.gemm.ltpin``) on Rubin, at the D3-32 per-GPU
shapes:

* every pinned shape builds (the loaded cuBLASLt is the pinned version and
  ``cublasLtMatmulAlgoCheck`` accepts every descriptor) and is within one output
  rounding of an FP32 reference (BF16 out) or FP32 summation-order noise (FP32 out);
* the beta-add shape (latent_up) adds into the residual in place, as stock's
  ``residual.addmm_``;
* MXFP8 ltpin equals the dequantized FP32 reference within one BF16 rounding;
* ``DecodeProjection`` dispatches the "ltpin" kind (plain and residual), a captured
  graph replays it, and unknown shapes / unprepared rows raise.

One GPU, well under a minute."""

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("needs a Rubin GPU (compute capability 10.7)", allow_module_level=True)

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (  # noqa: E402
    mxfp8_e4m3_quantize,
    swizzle_mxfp8_scale,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm import (  # noqa: E402
    ltpin,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.decode import (  # noqa: E402
    DecodeProjection,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.mxfp8 import (  # noqa: E402
    Mxfp8Projection,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.shapes import (  # noqa: E402
    Unsupported,
)

DEVICE = torch.device("cuda", 0)
BF16 = {  # name: (N, K, out32, beta)
    "dsa_in": (5344, 7168, False, False),
    "dsa_o": (7168, 3072, False, False),
    "q_b": (4608, 1536, False, False),
    "wq_b": (4096, 1536, False, False),
    "latent_down": (3584, 7168, False, False),
    "latent_up": (7168, 3584, False, True),
}
MXFP8 = {"kda_in": (12448, 7168), "shared_gu": (12288, 7168), "shared_down": (7168, 6144)}


def _rand(shape, scale=1.0, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return (torch.randn(*shape, device=DEVICE, generator=g) * scale).bfloat16()


def test_version_and_table():
    assert ltpin.version() // 100 == ltpin.CUBLASLT_VERSION // 100
    for (name, rows), (hexdesc, ws) in ltpin.PINS.items():
        assert name in ltpin.SHAPES.values() and len(hexdesc) == 128 and ws >= 0, name


@pytest.mark.parametrize("name", list(BF16))
def test_bf16_pins(name):
    torch.cuda.set_device(DEVICE)
    n, k, out32, beta = BF16[name]
    rows = ltpin.pinned_rows(name) + (ltpin.pinned_rows(name)[0] - 8,)
    gemm = ltpin.LtPin(f"t_{name}", n, k, rows, out32=out32, beta=beta, device=DEVICE)
    w = _rand((n, k), 0.02, 1)
    for r in rows:
        x = _rand((r, k), 1.0, r)
        ref = x.float() @ w.float().t()
        if beta:
            residual = _rand((r, n), 1.0, 7 + r)
            ref = ref + residual.float()
            out = residual.clone()
        else:
            out = torch.empty(r, n, device=DEVICE,
                              dtype=torch.float32 if out32 else torch.bfloat16)  # fmt: skip
        gemm.run(w, x, out)
        scale = ref.abs().amax(dim=1, keepdim=True)
        tol = scale * (1e-5 if out32 else 2.0**-7) + 1e-4
        assert ((out.float() - ref).abs() <= tol).all(), (name, r)
        if out32:  # stock's BF16 -> FP32 cuBLAS GEMM: summation order only
            stock = torch.mm(x, w.t(), out_dtype=torch.float32)
            assert ((out - stock).abs() <= scale * 1e-5 + 1e-5).all(), (name, r)


def mx_weight(n, k, seed):
    w = _rand((n, k), 0.02, seed)
    wq, ws = mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=False)
    ws = ws.view(torch.uint8).reshape(n, k // 32)
    return wq.contiguous(), swizzle_mxfp8_scale(ws, M=n, K=k).contiguous(), ws


@pytest.mark.parametrize("name", list(MXFP8))
def test_mxfp8_pins(name):
    torch.cuda.set_device(DEVICE)
    n, k = MXFP8[name]
    wq, sw, ws = mx_weight(n, k, 3)
    rows = ltpin.pinned_rows(name)
    proj = Mxfp8Projection((wq, sw), rows, family=f"t_{name}",
                           tactics=dict.fromkeys(rows, "ltpin"))  # fmt: skip
    wd = wq.float().view(n, k // 32, 32) * torch.exp2(ws.float() - 127).unsqueeze(-1)
    for r in rows:
        x = _rand((r, k), 1.0, r)
        out = torch.empty(r, n, device=DEVICE, dtype=torch.bfloat16)
        proj.run(x, out)
        xq, xs = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=False)
        xs = xs.view(torch.uint8).reshape(r, k // 32).float()
        xd = xq.float().view(r, k // 32, 32) * torch.exp2(xs - 127).unsqueeze(-1)
        ref = xd.view(r, k) @ wd.view(n, k).t()
        tol = ref.abs().amax(dim=1, keepdim=True) * 2.0**-7 + 1e-3
        assert ((out.float() - ref).abs() <= tol).all(), (name, r)
        stock = torch._scaled_mm(
            proj.input.view(r).data, wq.t(),
            scale_a=proj.input.view(r).scale.view(torch.float8_e8m0fnu),
            scale_b=sw.view(torch.float8_e8m0fnu), out_dtype=torch.bfloat16,
        )  # fmt: skip
        assert ((out.float() - stock.float()).abs() <= tol).all(), (name, r)


def test_projection_kind_and_graph():
    torch.cuda.set_device(DEVICE)
    n, k = 7168, 3584  # latent_up: beta-add
    w = _rand((n, k), 0.02, 6)
    rows = (32, 64, 96)
    proj = DecodeProjection(w, rows, dict.fromkeys(rows, ("ltpin", None)),
                            family="latent_up")  # fmt: skip
    xs = {r: _rand((r, k), 1.0, r) for r in rows}
    res = {r: _rand((r, n), 1.0, 100 + r) for r in rows}
    want = {r: torch.addmm(res[r].float(), xs[r].float(), w.float().t()) for r in rows}
    bufs = {r: res[r].clone() for r in rows}
    for r in rows:  # warm outside the capture
        proj.run(xs[r], residual=bufs[r])
    graphs = {}
    for r in rows:
        bufs[r].copy_(res[r])
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            proj.run(xs[r], residual=bufs[r])
        graphs[r] = g
    for r in rows:
        bufs[r].copy_(res[r])
        graphs[r].replay()
        torch.cuda.synchronize()
        tol = want[r].abs().amax(dim=1, keepdim=True) * 2.0**-7 + 1e-4
        assert ((bufs[r].float() - want[r]).abs() <= tol).all(), r
    with pytest.raises(Unsupported):  # beta-add plan without a residual
        proj.run(xs[32], torch.empty(32, n, device=DEVICE, dtype=torch.bfloat16))
    with pytest.raises(Unsupported):
        proj.run(_rand((8, k)), residual=_rand((8, n)))
    with pytest.raises(Unsupported):  # no pinned shape
        ltpin.LtPin("t_bad", 1000, 1024, (32,), device=DEVICE)
