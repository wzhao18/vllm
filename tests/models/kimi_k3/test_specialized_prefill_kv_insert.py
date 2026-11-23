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


@pytest.mark.parametrize("block_size", [64, 2048])
@pytest.mark.parametrize("dcp_rank", [0, 3, 7])
def test_kv_insert_dcp_local_slots_bitwise(dcp_rank, block_size):
    """TP8+DCP8 prefill worker (PFLHF P4): the DCP-local slot map (interleave 1)
    holds -1 for every token another DCP rank owns; the insert writes exactly
    the rows stock concat_and_cache_mla writes (skip_scale per-tensor FP8),
    including the 2048-token pool blocks of the TP8 slot-packed KV groups."""
    from vllm.models.kimi_k3.nvidia.specialized_prefill import kv_insert

    dcp, num_tokens = 8, 4096
    kv_c, k_pe, _, cache = _inputs(num_tokens, 8192 // block_size, block_size, dcp_rank)
    t = torch.arange(num_tokens, device="cuda")
    local = t // dcp  # this rank's position of its owned tokens
    slots = torch.where(t % dcp == dcp_rank, 3 * 64 + local, -1)
    slots = slots.to(torch.int64).contiguous()
    ref = cache.clone()
    scale = torch.ones(1, dtype=torch.float32, device="cuda")
    k_pe2 = k_pe.squeeze(1)
    ops.concat_and_cache_mla(
        kv_c, k_pe2, ref, slots, kv_cache_dtype="fp8", scale=scale, skip_scale=True
    )
    assert kv_insert.applicable(kv_c, k_pe2, cache, slots, "fp8", True)
    kv_insert.insert(kv_c, k_pe2, cache, slots)
    torch.cuda.synchronize()
    assert torch.equal(cache, ref)


def test_backend_routes_dcp_insert(monkeypatch):
    """FlashInferMLASparseImpl.do_kv_cache_update takes the specialized insert
    when ``_prefill_kv_insert`` is set (no specialized sparse MLA under DCP),
    bitwise equal to stock."""
    from types import SimpleNamespace

    from vllm.models.kimi_k3.nvidia.specialized_prefill import kv_insert
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
        FlashInferMLASparseImpl,
    )

    kv_c, k_pe, slots, cache = _inputs(1000, 64, 64, 11)
    ref = cache.clone()
    scale = torch.ones(1, dtype=torch.float32, device="cuda")
    ops.concat_and_cache_mla(
        kv_c, k_pe.squeeze(1), ref, slots, kv_cache_dtype="fp8", scale=scale,
        skip_scale=True,
    )  # fmt: skip
    calls = []
    real = kv_insert.insert
    monkeypatch.setattr(
        kv_insert, "insert", lambda *a: (calls.append(1), real(*a))[1]
    )
    impl = SimpleNamespace(_prefill_kv_insert=True, _prefill_sparse_mla=None)
    FlashInferMLASparseImpl.do_kv_cache_update(
        impl, kv_c, k_pe, cache, slots, "fp8", scale, True
    )
    torch.cuda.synchronize()
    assert calls == [1]
    assert torch.equal(cache, ref)


def test_prefill_dcp_supported_policy(monkeypatch):
    """Producer + per-tensor FP8 + D 512/64 + sm_107 only (CPU)."""
    from types import SimpleNamespace

    from vllm.models.kimi_k3.nvidia.specialized_prefill import kv_insert
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform, "is_device_capability", lambda c: c in (107, (10, 7))
    )

    def cfg(producer):
        kt = None if producer is None else SimpleNamespace(is_kv_producer=producer)
        return SimpleNamespace(kv_transfer_config=kt)

    assert kv_insert.prefill_dcp_supported(cfg(True), "fp8", 512, 64)
    assert kv_insert.prefill_dcp_supported(cfg(True), "fp8_e4m3", 512, 64)
    assert not kv_insert.prefill_dcp_supported(cfg(False), "fp8", 512, 64)
    assert not kv_insert.prefill_dcp_supported(cfg(None), "fp8", 512, 64)
    assert not kv_insert.prefill_dcp_supported(cfg(True), "auto", 512, 64)
    assert not kv_insert.prefill_dcp_supported(cfg(True), "fp8_ds_mla", 512, 64)
    assert not kv_insert.prefill_dcp_supported(cfg(True), "fp8", 256, 64)
    monkeypatch.setattr(current_platform, "is_device_capability", lambda c: False)
    assert not kv_insert.prefill_dcp_supported(cfg(True), "fp8", 512, 64)
