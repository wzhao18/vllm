# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 SP decode MXFP8 kind "pre" (``gemm.sm107.prewait``: the vendored FlashInfer
sm_107 CuTe DSL block-scaled GEMM, PDL launch, weight k-blocks issued before
griddepcontrol.wait) on Rubin:

* kda_in at the D3-32 rows: bitwise equal to cuBLASLt ``_scaled_mm`` (stock's kind above
  16 rows: both sum K in order in FP32 with one BF16 rounding) and to the same kernel
  without PDL / prologue;
* a captured graph of back-to-back "pre" projections (PDL edges) replays bitwise;
* tactics the host cannot serve raise.

One GPU, well under a minute."""

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 7):
    pytest.skip("needs an sm_107 (Rubin) GPU", allow_module_level=True)

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (  # noqa: E402
    mxfp8_e4m3_quantize,
    swizzle_mxfp8_scale,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.mxfp8 import (  # noqa: E402
    DIE_ROWS,
    Mxfp8Projection,
    pre_die_static,
    pre_tactic,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.sm107.prewait import (  # noqa: E402
    Mxfp8PreGemm,
    die_weight_rows,
)

DEVICE = torch.device("cuda", 0)
N, K = 12448, 7168  # kda_in per GPU (TP4)
ROWS = (128, 256, 384)


def mx_weight(n, k, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    w = (torch.randn(n, k, device=DEVICE, generator=g) * 0.02).bfloat16()
    wq, ws = mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=False)
    ws = ws.view(torch.uint8).reshape(n, k // 32)
    return wq.contiguous(), swizzle_mxfp8_scale(ws, M=n, K=k).contiguous()


def scaled_mm(xq, xs, wq, sw):
    return torch._scaled_mm(
        xq, wq.t(), scale_a=xs.view(torch.float8_e8m0fnu),
        scale_b=sw.view(torch.float8_e8m0fnu), out_dtype=torch.bfloat16,
    )  # fmt: skip


def test_pre_bitwise_kda_in():
    torch.cuda.set_device(DEVICE)
    wq, sw = mx_weight(N, K, 1)
    proj = Mxfp8Projection((wq, sw), ROWS, family="t_pre", tactics=dict.fromkeys(ROWS, "pre"))
    for r in ROWS:
        x = torch.randn(r, K, device=DEVICE).bfloat16()
        xq, xs = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
        out = torch.empty(r, N, device=DEVICE, dtype=torch.bfloat16)
        proj.run(x, out)
        assert torch.equal(out, scaled_mm(xq, xs, wq, sw)), r
        tactic, _, _ = pre_tactic(N, K, r)
        plain = Mxfp8PreGemm(N, K, r, tactic, use_pdl=False, prewait=0, device=DEVICE)
        again = torch.empty_like(out)
        plain(xq, xs, wq, sw, again)
        assert torch.equal(again, out), r


def test_pre_graph_chain():
    torch.cuda.set_device(DEVICE)
    weights = [mx_weight(N, K, s) for s in (2, 3)]
    rows = (128, 384)
    projs = [
        Mxfp8Projection(w, rows, family=f"t_pre_g{i}", tactics=dict.fromkeys(rows, "pre"))
        for i, w in enumerate(weights)
    ]
    for r in rows:
        x = torch.randn(r, K, device=DEVICE).bfloat16()
        q = projs[0].input.quantize(x)
        outs = [torch.empty(r, N, device=DEVICE, dtype=torch.bfloat16) for _ in projs]
        for p, o in zip(projs, outs):  # warm (compile) outside the capture
            p.run(q, o)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for p, o in zip(projs, outs):
                p.run(q, o)
        x.copy_(torch.randn_like(x, dtype=torch.float32).bfloat16())
        q = projs[0].input.quantize(x)
        g.replay()
        torch.cuda.synchronize()
        for (wq, sw), o in zip(weights, outs):
            assert torch.equal(o, scaled_mm(q.data, q.scale, wq, sw)), r


def test_pre_die_static():
    """Die-static placement (one cluster per TPC / SM, die-local weight rows): bitwise to
    _scaled_mm over back-to-back PDL calls on both counter sets; the landing gate resets."""
    torch.cuda.set_device(DEVICE)
    wq, sw = mx_weight(N, K, 4)
    rows_d0 = DIE_ROWS[(N, K)]
    wd = die_weight_rows(wq, rows_d0)
    for r in ROWS:
        tactic, pre, cap = pre_tactic(N, K, r)
        if pre_die_static(N, K, r)[0] is not None:
            cap = pre_die_static(N, K, r)[0]
        g = Mxfp8PreGemm(N, K, r, tactic, prewait=pre, max_stages=cap, die_static=True,
                         die_rows=rows_d0, device=DEVICE)
        x = torch.randn(r, K, device=DEVICE).bfloat16()
        xq, xs = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
        outs = [torch.empty(r, N, device=DEVICE, dtype=torch.bfloat16) for _ in range(3)]
        for o in outs:
            g(xq, xs, wd, sw, o)
        ref = scaled_mm(xq, xs, wq, sw)
        for o in outs:
            assert torch.equal(o, ref), r
        torch.cuda.synchronize()
        assert not g.aux.view(torch.int32)[:, [4, 6]].any(), r


def test_pre_rejects():
    with pytest.raises(ValueError):
        Mxfp8PreGemm(N, K, 128, ((256, 96), (2, 1), True, 256), device=DEVICE)
    with pytest.raises(ValueError):
        Mxfp8PreGemm(N, 256, 128, ((128, 64), (1, 1), False, 128), prewait=8, device=DEVICE)
