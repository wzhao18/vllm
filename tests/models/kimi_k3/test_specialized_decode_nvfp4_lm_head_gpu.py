# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 NVFP4 checkpoint (``modelopt_mixed``): its LM head on the fused verify
kernels (``runtime.verify.plain_lm_head``). The checkpoint's quant config resolves
the LM head (in ``ignore``) to ``UnquantizedLinearMethod``, which ``plain_lm_head``
accepts; on the checkpoint's own ``language_model.lm_head.weight`` (a TP8 and a
TP16 vocabulary shard), the fused BF16-logits GEMV (1-8 rows) and the evict-first
GEMM (17-32 rows) stay within one BF16 ulp of that method's ``apply`` (F.linear,
the stock ``compute_logits`` GEMM), deterministic across calls.

The checkpoint directory is ``KIMI_K31_NVFP4_CHECKPOINT`` (a test input; skipped
without it)."""

import json
import os

import pytest
import torch

from vllm.platforms import current_platform

CHECKPOINT = os.environ.get("KIMI_K31_NVFP4_CHECKPOINT", "")
NAME = "language_model.lm_head.weight"

pytestmark = [
    pytest.mark.skipif(
        not current_platform.is_device_capability_family(100),
        reason="the specialized decode kernels need SM100/SM103/SM107",
    ),
    pytest.mark.skipif(
        not os.path.isfile(os.path.join(CHECKPOINT, "hf_quant_config.json")),
        reason="needs the NVFP4 checkpoint (KIMI_K31_NVFP4_CHECKPOINT)",
    ),
]


def _method():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptMixedPrecisionConfig,
    )
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    with open(os.path.join(CHECKPOINT, "hf_quant_config.json")) as f:
        config = ModelOptMixedPrecisionConfig.from_config(json.load(f))
    layer = ParallelLMHead.__new__(ParallelLMHead)  # (the type is what it reads)
    torch.nn.Module.__init__(layer)
    with set_current_vllm_config(VllmConfig()):
        return config.get_quant_method(layer, "language_model.lm_head")


def _weight(vocab_rank, dev):
    from safetensors import safe_open

    with open(os.path.join(CHECKPOINT, "model.safetensors.index.json")) as f:
        shard = json.load(f)["weight_map"][NAME]
    with safe_open(os.path.join(CHECKPOINT, shard), "pt", device="cpu") as f:
        rows = f.get_slice(NAME)[:vocab_rank]
    return rows.to(dev, torch.bfloat16).contiguous()


@pytest.mark.parametrize("tp", [8, 16])
def test_nvfp4_lm_head_fused_verify(tp):
    pytest.importorskip("cutlass")
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.verify import lmhead
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.verify import (
        plain_lm_head,
    )

    method = _method()
    assert isinstance(method, UnquantizedLinearMethod) and plain_lm_head(method)
    dev = torch.device("cuda", torch.accelerator.current_device_index())
    weight = _weight(163840 // tp, dev)
    assert lmhead.supports(weight)
    stub = torch.nn.Module()
    stub.weight = torch.nn.Parameter(weight, requires_grad=False)
    gemv = lmhead.PreparedLmHeadLogits(weight, dev)
    wide = lmhead.PreparedWideLmHead(weight, dev)
    gen = torch.Generator(device=dev).manual_seed(tp)
    for rows in (1, 4, 8, 17, 24, 32):
        x = (torch.randn(rows, weight.shape[1], generator=gen, device=dev) * 2).bfloat16()
        head = gemv if rows <= 8 else wide
        out = head.run(x).clone()
        again = head.run(x).clone()
        ref = method.apply(stub, x)  # the stock compute_logits GEMM (F.linear)
        torch.cuda.synchronize()
        assert torch.equal(out, again), rows
        exact = x.float() @ weight.float().t()
        # one BF16 ulp of the value, or of the row's largest logit near zero
        floor = 2**-8 * exact.abs().amax(1, keepdim=True)
        tol = 2**-7 * torch.maximum(exact.abs(), floor)
        assert bool(((out.float() - ref.float()).abs() <= 2 * tol).all()), rows
        assert bool(((out.float() - exact).abs() <= tol).all()), rows
