# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 SP decode ``moe_front`` (router | latent down GEMM + router top-k +
MegaMoE staging in one launch, ``specialized_decode.cutedsl.moe.moe_front``):

* its FP32 GEMM rows match ``h @ W^T`` (FP32 accumulation, summation order aside);
* given those rows, every staged buffer (routes, weights, quantized latent, SF,
  dead rows, route-error flag) is bitwise what ``route_stage`` (itself bitwise the
  stock top-k + stager chain, test_specialized_decode_route_stage_gpu.py) stages;
* the production form (router logits only) stages the same bytes, and repeated
  calls (in-kernel counter reset) too.
One GPU, a few seconds."""

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
    pytest.skip("needs an sm_10x GPU", allow_module_level=True)

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import (  # noqa: E402
    moe_front,
    route_stage,
)

E, TOPK, LATENT, K, CAP, SCALE = 896, 16, 3584, 7168, 128, 2.827


def _buffers(nvfp4, dev):
    vec = 16 if nvfp4 else 32
    g = torch.Generator(device=dev).manual_seed(7)
    x = torch.randint(0, 256, (CAP, LATENT // (2 if nvfp4 else 1)), generator=g,
                      device=dev, dtype=torch.uint8)  # fmt: skip
    sf = torch.randint(0, 256, (CAP, LATENT // vec + 4), generator=g, device=dev,
                       dtype=torch.uint8)  # fmt: skip
    idx = torch.randint(0, E, (CAP, TOPK), generator=g, device=dev, dtype=torch.int32)
    sw = torch.rand((CAP, TOPK), generator=g, device=dev)
    err = torch.zeros(2, dtype=torch.int64, device=dev)
    return [x, sf, idx, sw, err]


def _same(got, ref):
    for name, a, b in zip(("x", "x_sf", "topk_idx", "topk_weights", "err"), got, ref):
        if a.dtype == torch.float32:
            a, b = a.view(torch.int32), b.view(torch.int32)
        bad = (a != b).nonzero()
        assert bad.numel() == 0, f"{name}: {bad.shape[0]} mismatches, first {bad[:4].tolist()}"


@pytest.mark.parametrize("nvfp4", [True, False])
@pytest.mark.parametrize("n", [32, 64, 96])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("tiles", ["config", "tr256"])
def test_moe_front_matches_route_stage(nvfp4, n, padded, tiles):
    dev = torch.device("cuda")
    torch.manual_seed(n + 2 * nvfp4 + 4 * padded)
    h = torch.randn((n, K), device=dev).to(torch.bfloat16)
    w = (torch.randn((E + LATENT, K), device=dev) * 0.02).to(torch.bfloat16)
    w[E + 3 * 128 : E + 3 * 128 + 16] *= 1e3  # some large latent blocks
    w[E : E + 64] = 0  # zero latent blocks
    bias = (torch.randn(E, device=dev) * 0.1).float()
    pad = (torch.rand(n, device=dev) < 0.3) if padded else None
    # tr256: 256-row weight tiles (a tile holds router and latent rows), 8-CTA clusters
    cfg = moe_front.config_for(n) if tiles == "config" else (8, 2, 2, 2, 2, 2, 256)
    stamps = torch.empty(0, dtype=torch.int64, device=dev)
    cnt = torch.zeros(4, dtype=torch.int32, device=dev)
    mod = moe_front._module()

    # Test form: y holds the FP32 rows of the whole GEMM.
    y = torch.full((n, E + LATENT), float("nan"), device=dev)
    got = _buffers(nvfp4, dev)
    mod.moe_front(w, h, y, cnt, bias, *got[:5], pad, E, E, True, SCALE, nvfp4, 1.0,
                  *cfg, stamps)  # fmt: skip
    torch.accelerator.synchronize()
    ref_y = (h.double() @ w.double().t()).float()
    assert torch.isfinite(y).all()
    # Error relative to the sum of |products| (the accumulation's own scale: the test
    # weights include 1e3-scaled rows whose dot products cancel).
    mag = (h.double().abs() @ w.double().abs().t()).float() + 1e-3

    def scaled(a):
        return ((a - ref_y).abs() / mag).max().item()

    # Tensor-core FP32 accumulation is not IEEE-exact: hold it to cuBLAS's error on
    # the same BF16 -> FP32 GEMM (the stock router path).
    cublas = scaled(torch.mm(h, w.t(), out_dtype=torch.float32))
    err = scaled(y)
    where = ""
    if err > max(4.0 * cublas, 1e-5):  # where (debug aid in the assertion message)
        e = (y - ref_y).abs() / mag
        tiles = e.amax(dim=0).view(-1, 128).amax(dim=1)
        bad = (tiles > 2.0 * cublas).nonzero().flatten().tolist()
        r, c = divmod(int(e.argmax()), e.shape[1])
        where = f"; bad 128-col tiles {bad}; worst row {r} col {c} y {y[r, c].item()} ref {ref_y[r, c].item()}"
    assert err <= max(4.0 * cublas, 1e-5), f"GEMM rows: scaled error {err} vs cuBLAS {cublas}{where}"
    assert cnt.eq(0).all(), f"counters not reset: {cnt.tolist()}"

    ref = _buffers(nvfp4, dev)
    route_stage._module().route_stage(
        y, E, bias, *ref[:5], pad, E, True, SCALE, nvfp4, 1.0
    )  # fmt: skip
    torch.accelerator.synchronize()
    _same(got, ref)

    # Production form (router logits only), twice: the same bytes.
    for _ in range(2):
        y_r = torch.empty((n, E), device=dev)
        again = _buffers(nvfp4, dev)
        mod.moe_front(w, h, y_r, cnt, bias, *again[:5], pad, E, E, True, SCALE,
                      nvfp4, 1.0, *cfg, stamps)  # fmt: skip
        torch.accelerator.synchronize()
        assert torch.equal(y_r.view(torch.int32), y[:, :E].view(torch.int32))
        _same(again, ref)


def test_moe_front_rejects_bad_shapes():
    dev = torch.device("cuda")
    h = torch.zeros((32, K + 64), device=dev, dtype=torch.bfloat16)
    w = torch.zeros((E + LATENT, K), device=dev, dtype=torch.bfloat16)
    b = _buffers(True, dev)
    with pytest.raises(RuntimeError):
        moe_front._module().moe_front(
            w, h, torch.zeros((32, E), device=dev), torch.zeros(4, dtype=torch.int32,
            device=dev), torch.zeros(E, device=dev), *b, None, E, E, True, 1.0, True,
            1.0, 5, 4, 4, 3, 1, 2, 128, torch.empty(0, dtype=torch.int64, device=dev),
        )  # fmt: skip
