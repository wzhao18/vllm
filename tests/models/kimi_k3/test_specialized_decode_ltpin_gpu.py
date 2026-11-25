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
* the router | latent down concatenation (FP32 out, ``runtime.sp_mlp``): the router
  logits match stock's BF16 -> FP32 router GEMM to FP32 summation-order noise.

One GPU, well under a minute."""

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("needs a Rubin GPU (compute capability 10.7)", allow_module_level=True)

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm import (  # noqa: E402
    ltpin,
)

DEVICE = torch.device("cuda", 0)
BF16 = {  # name: (N, K, out32, beta)
    "dsa_in": (5344, 7168, False, False),
    "dsa_o": (7168, 3072, False, False),
    "q_b": (4608, 1536, False, False),
    "wq_b": (4096, 1536, False, False),
    "kda_in_tp16": (3216, 7168, False, False),
    "dsa_in_tp16": (3040, 7168, False, False),
    "latent_down": (3584, 7168, False, False),
    "latent_up": (7168, 3584, False, True),
    "router_latent": (4480, 7168, True, False),
}


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


def test_tp16_dispatch_rows():
    """The TP16 decode prepares a pinned GEMM only at rows measured for it, above its
    split-K rows: wq_b (``dsa.query_stage.index_pin``, rows above 128) and the KDA /
    DSA in-projections (``runtime.providers.inproj_pin``); other rows keep cuBLAS."""
    from types import SimpleNamespace

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.query_stage import (
        index_pin,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.providers import (
        inproj_pin,
    )

    torch.cuda.set_device(DEVICE)
    wq = _rand((4096, 1536), 0.02, 1)
    rows = [64, 128, 256, 320, 384, 512]
    assert index_pin(wq, rows, rubin=False) is None
    assert sorted(index_pin(wq, rows, rubin=True).plans) == [512]
    split_k = SimpleNamespace(kernels={64: None, 128: None})
    for kind, n in (("kda", 3216), ("dsa", 3040)):
        w = _rand((n, 7168), 0.02, 2)
        pin = inproj_pin(w, kind, split_k, rows, 212, 16)
        assert sorted(pin.plans) == [512], kind
        assert inproj_pin(w, kind, split_k, rows, 212, 8) is None
        assert inproj_pin(w, kind, split_k, rows[:3], 212, 16) is None
    assert inproj_pin(_rand((2880, 7168)), "mla", split_k, rows, 212, 16) is None
