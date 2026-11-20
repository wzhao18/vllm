# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 specialized prefill MLA FP8 KV insert: bitwise vs the stock
concat_and_cache_mla (per-tensor FP8 cache, unit scale)."""

import pytest
import torch

from vllm import _custom_ops as ops

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


def _inputs(num_tokens: int, blocks: int, block_size: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    kv = torch.randn(num_tokens, 576, device="cuda", generator=g) * 40
    kv = kv.to(torch.bfloat16)
    # saturation, NaN, infinities and signed zeros
    kv[0, :8] = torch.tensor(
        [1e4, -1e4, float("inf"), float("-inf"), float("nan"), -0.0, 0.0, 449.0]
    ).to(torch.bfloat16)
    kv_c = kv[:, :512]  # strided views, as the model's kv_a split
    k_pe = kv[:, 512:].unsqueeze(1)
    perm = torch.randperm(blocks * block_size, device="cuda", generator=g)
    slots = perm[:num_tokens].to(torch.int64)
    slots[1::7] = -1
    cache = torch.zeros(blocks, block_size, 576, dtype=torch.uint8, device="cuda")
    return kv_c, k_pe, slots, cache


@pytest.mark.parametrize("num_tokens", [1, 15, 16, 1000, 16384])
def test_kv_insert_bitwise(num_tokens):
    from vllm.models.kimi_k3.nvidia.specialized_prefill import kv_insert

    kv_c, k_pe, slots, cache = _inputs(num_tokens, 2048, 64, num_tokens)
    ref = cache.clone()
    scale = torch.ones(1, dtype=torch.float32, device="cuda")
    ops.concat_and_cache_mla(
        kv_c, k_pe.squeeze(1), ref, slots, kv_cache_dtype="fp8", scale=scale,
        skip_scale=True,
    )
    k_pe2 = k_pe.squeeze(1)
    assert kv_insert.applicable(kv_c, k_pe2, cache, slots, "fp8", True)
    kv_insert.insert(kv_c, k_pe2, cache, slots)
    torch.cuda.synchronize()
    assert torch.equal(cache, ref)


def test_kv_insert_scope():
    from vllm.models.kimi_k3.nvidia.specialized_prefill import kv_insert

    kv_c, k_pe, slots, cache = _inputs(16, 4, 64, 0)
    k_pe2 = k_pe.squeeze(1)
    assert not kv_insert.applicable(kv_c, k_pe2, cache, slots, "fp8", False)
    assert not kv_insert.applicable(kv_c, k_pe2, cache, slots, "auto", True)
    assert not kv_insert.applicable(
        kv_c, k_pe2, cache, slots.to(torch.int32), "fp8", True
    )
