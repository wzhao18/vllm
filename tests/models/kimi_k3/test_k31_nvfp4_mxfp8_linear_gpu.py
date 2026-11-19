# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 NVFP4 export: its MXFP8 shared-expert linears, loaded as stored, on Rubin.

The export stores the shared experts (and the KDA projections) as ModelOpt MXFP8:
E4M3 weight rows + per-32 UE8M0 scale bytes, ``[N, K]`` / ``[N, K / 32]`` row-major,
gate and up as separate checkpoint tensors. This test builds the real stock layers
(``MergedColumnParallelLinear`` gate_up / ``RowParallelLinear`` down, the export's
quantization config, single rank), loads checkpoint-format tensors through the
layers' weight loaders, runs ``process_weights_after_loading`` and checks:

* the specialized decode reads the layer's storage as exactly the checkpoint bytes
  (``mxfp8_weight``: weight rows bitwise, unswizzled scales bitwise);
* stock ``apply`` and the specialized decode ``Mxfp8Projection`` (its per-R kind) are
  within one BF16 rounding of an FP32 product of the dequantized checkpoint weight and
  FlashInfer's MXFP8 activation quantization.

One GPU, D3-32 per-GPU shapes (replicated shared expert: 7168 -> 2 x 6144 -> 7168).
"""

import os
import tempfile

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("needs a Rubin GPU (compute capability 10.7)", allow_module_level=True)

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (  # noqa: E402
    mxfp8_e4m3_quantize,
)

DEV = torch.device("cuda", 0)
HIDDEN, INTER = 7168, 6144
L = "language_model.model.layers.2.block_sparse_moe.shared_experts"
ROWS = (1, 4, 16, 32, 96, 128)


def _export_config() -> dict:
    layers = {}
    for base in ("language_model.model.layers", "model.layers", "layers"):
        for proj in ("gate_proj", "up_proj", "down_proj", "gate_up_proj"):
            layers[f"{base}.2.block_sparse_moe.shared_experts.{proj}"] = {
                "quant_algo": "MXFP8"
            }
    return {
        "quant_method": "modelopt_mixed",
        "quant_algo": "MIXED_PRECISION",
        "group_size": 16,
        "ignore": [],
        "quantized_layers": layers,
    }


def _ckpt(n, k, seed):
    """Checkpoint tensors of one MXFP8 linear: E4M3 [n, k], UE8M0 bytes [n, k/32]."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    w = (torch.randn(n, k, device=DEV, generator=g) * 0.02).bfloat16()
    wq, ws = mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=False)
    return wq.contiguous(), ws.view(torch.uint8).reshape(n, k // 32).contiguous()


def _reference(x, wq, ws):
    xq, xs = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=False)
    r, k = x.shape
    xs = xs.view(torch.uint8).reshape(r, k // 32).float()
    xd = xq.float().view(r, k // 32, 32) * torch.exp2(xs - 127).unsqueeze(-1)
    n = wq.shape[0]
    wd = wq.float().view(n, k // 32, 32) * torch.exp2(ws.float() - 127).unsqueeze(-1)
    return xd.view(r, k) @ wd.view(n, k).t()


def _close(out, ref):
    tol = ref.abs().amax(dim=1, keepdim=True) * 2.0**-7 + 1e-3
    return bool(((out.float() - ref).abs() <= tol).all())


@pytest.fixture(scope="module")
def layers():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear,
        RowParallelLinear,
    )
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptMixedPrecisionConfig,
    )
    from vllm.models.kimi_k3.nvidia import model as kimi_model

    torch.cuda.set_device(DEV)
    tmp = tempfile.TemporaryDirectory()
    store = os.path.join(tmp.name, "store")
    with set_current_vllm_config(VllmConfig()), torch.device(DEV):
        init_distributed_environment(
            world_size=1, rank=0, local_rank=0,
            distributed_init_method=f"file://{store}", backend="nccl",
        )  # fmt: skip
        initialize_model_parallel(1, 1)
        qcfg = ModelOptMixedPrecisionConfig.from_config(_export_config())
        qcfg.packed_modules_mapping = dict(
            kimi_model.KimiLinearModel.packed_modules_mapping
        )
        gu = MergedColumnParallelLinear(
            HIDDEN, [INTER, INTER], bias=False, quant_config=qcfg,
            params_dtype=torch.bfloat16, prefix=f"{L}.gate_up_proj",
        )  # fmt: skip
        down = RowParallelLinear(
            INTER, HIDDEN, bias=False, quant_config=qcfg,
            params_dtype=torch.bfloat16, prefix=f"{L}.down_proj",
        )  # fmt: skip
        gate, up, dn = (
            _ckpt(INTER, HIDDEN, 1),
            _ckpt(INTER, HIDDEN, 2),
            _ckpt(HIDDEN, INTER, 3),
        )
        for shard, (wq, ws) in enumerate((gate, up)):
            gu.weight.weight_loader(gu.weight, wq, shard)
            gu.weight_scale.weight_loader(gu.weight_scale, ws, shard)
        down.weight.weight_loader(down.weight, dn[0])
        down.weight_scale.weight_loader(down.weight_scale, dn[1])
        for layer in (gu, down):
            layer.quant_method.process_weights_after_loading(layer)
        ckpt = {
            "shared_gu": (torch.cat((gate[0], up[0])), torch.cat((gate[1], up[1]))),
            "shared_down": dn,
        }
        yield {"shared_gu": gu, "shared_down": down}, ckpt
    cleanup_dist_env_and_memory()
    tmp.cleanup()


@pytest.mark.parametrize("family", ["shared_gu", "shared_down"])
def test_modelopt_mxfp8_linear_as_stored(layers, family):
    from vllm.models.kimi_k3.nvidia.mxfp8 import (
        is_mxfp8_linear,
        unswizzle_mxfp8_scale,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.mxfp8 import (
        Mxfp8Projection,
        mxfp8_tactics,
        mxfp8_weight,
    )

    mods, ckpt = layers
    layer = mods[family]
    wq, ws = ckpt[family]
    n, k = wq.shape
    assert is_mxfp8_linear(layer)
    w, s = mxfp8_weight(layer, family)
    assert torch.equal(w.view(torch.uint8), wq.view(torch.uint8))
    assert torch.equal(unswizzle_mxfp8_scale(s, n, k), ws)

    proj = Mxfp8Projection(
        (w, s), ROWS, family=f"t_{family}", tactics=mxfp8_tactics(n, k, ROWS)
    )
    g = torch.Generator(device=DEV).manual_seed(n + k)
    for r in ROWS:
        x = torch.randn(r, k, device=DEV, generator=g).bfloat16()
        ref = _reference(x, wq, ws)
        assert _close(layer.quant_method.apply(layer, x), ref), ("stock", r)
        out = torch.empty(r, n, device=DEV, dtype=torch.bfloat16)
        proj.run(x, out)
        assert _close(out, ref), ("sdec", r)
