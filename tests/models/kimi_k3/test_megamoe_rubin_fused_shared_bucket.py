# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rubin MegaMoE fused MXFP8 shared expert on EP16 SP decode (PFLHF D3, CPU).

A decode server whose only bucket is GenPhase above RUBIN_WIDE_EP_DECODE_TOKENS
tokens per rank (EP16 SP decode shards of 192 / 242) gets the 128-token fused
shared bucket from ``enable_fused_shared``; servers that already have a bucket
at or below it, MXFP4 GenPhase servers and P/D prefill workers are unchanged."""

import dataclasses
from types import SimpleNamespace

import pytest

from vllm.models.kimi_k3.nvidia.megamoe_rubin import config as C
from vllm.models.kimi_k3.nvidia.megamoe_rubin import experts as X

HIDDEN, INTER = 7168, 6144  # shared expert (hidden, 2 x 3072)


def _layer(tokens: int, world: int, kind: str = "nvfp4", prefill: bool = False):
    X.select_rubin_bucket.cache_clear()
    problem = C.RubinMoeProblem(quant_kind=kind)
    buckets, _ = X.select_rubin_buckets(problem, world, tokens, prefill_worker=prefill)
    return SimpleNamespace(
        prefill_worker=prefill,
        buckets=buckets,
        bucket=buckets[-1],
        problem=problem,
        ep_size=world,
    )


def _enable(layer) -> bool:
    return X.KimiK3MegaMoEExperts.enable_fused_shared(layer, HIDDEN, INTER)


@pytest.mark.parametrize("tokens", [192, 242, 1024])
def test_ep16_genphase_server_adds_fused_bucket(tokens):
    layer = _layer(tokens, 16)
    (main,) = layer.buckets
    assert main.variant == "genphase" and main.max_tokens == tokens
    assert _enable(layer)
    small, big = layer.buckets
    assert small.max_tokens == X.RUBIN_WIDE_EP_DECODE_TOKENS
    assert small.variant == "genphase"
    assert small.shared == (HIDDEN, INTER)
    assert small.knobs == X.RUBIN_GENPHASE_FUSED_SHARED_KNOBS
    assert small.max_sm_count is None
    # the server's own bucket is untouched (rows above 128 keep the side stream)
    assert big == main and big.shared is None
    assert layer.bucket == big
    # idempotent (a rebuilt runtime calls it again)
    assert _enable(layer)
    assert [b.max_tokens for b in layer.buckets] == [128, tokens]


def test_ep16_select_bucket_capacities():
    layer = _layer(192, 16)
    _enable(layer)
    sel = lambda n: next(b for b in layer.buckets if b.max_tokens >= n)  # noqa: E731
    assert sel(1).shared is not None
    assert sel(128).shared is not None
    assert sel(129).shared is None and sel(192).max_tokens == 192


def test_small_genphase_server_unchanged():
    layer = _layer(128, 16)
    assert _enable(layer)
    (only,) = layer.buckets
    assert only.max_tokens == 128 and only.shared == (HIDDEN, INTER)


def test_ep32_wide_ep_unchanged():
    layer = _layer(2048, 32)
    before = [b.max_tokens for b in layer.buckets]
    assert _enable(layer)
    assert [b.max_tokens for b in layer.buckets] == before == [128, 2048]
    assert layer.buckets[0].shared is not None and layer.buckets[1].shared is None


def test_inference_decode_bucket_server_unchanged():
    """C512-style decode (inference 256 decode bucket) on the stock path: no
    GenPhase bucket in its buckets (GenPhase without the fused shared was slower
    than the 256 inference bucket at the EP16 decode shape)."""
    layer = _layer(2048, 16)
    assert [(b.variant, b.max_tokens) for b in layer.buckets] == [
        ("inference", 256), ("inference", 2048)]


def test_ep16_inference_server_fuses_shared_bucket():
    """The same server on the specialized decode (enable_fused_shared) adds the
    EP32 128-token GenPhase bucket with the shared expert fused in (measured
    faster: RUBIN_FUSED_DECODE_EP_SIZES); larger steps keep their buckets."""
    layer = _layer(2048, 16)
    layer.fused_decode_eligible = X.RUBIN_FUSED_DECODE_NOTE in _notes(2048, 16)
    before = layer.buckets
    assert _enable(layer)
    small, *rest = layer.buckets
    assert (small.variant, small.max_tokens) == ("genphase", X.RUBIN_WIDE_EP_DECODE_TOKENS)
    assert small.knobs == X.RUBIN_GENPHASE_FUSED_SHARED_KNOBS
    assert small.shared == (HIDDEN, INTER) and small.max_sm_count is None
    assert tuple(rest) == before
    assert _enable(layer)  # idempotent
    assert [b.max_tokens for b in layer.buckets] == [128, 256, 2048]


@pytest.mark.parametrize("prefill,kind", [(True, "nvfp4"), (False, "mxfp4_mxfp8")])
def test_prefill_and_mxfp4_unchanged(prefill, kind):
    layer = _layer(192, 16, kind=kind, prefill=prefill)
    before = layer.buckets
    assert not _enable(layer)
    assert layer.buckets == before


def test_inserted_bucket_keeps_placement():
    """The added bucket inherits the server bucket's die placement."""
    layer = _layer(242, 16)
    (main,) = layer.buckets
    placed = dataclasses.replace(main, die_sm_mask32=(0x0F0F0F0F,) * 8)
    layer.buckets, layer.bucket = (placed,), placed
    _enable(layer)
    assert layer.buckets[0].die_sm_mask32 == placed.die_sm_mask32


@pytest.mark.parametrize("world,tokens,eligible", [
    (16, 2048, True), (16, 4096, True), (16, 192, False), (32, 2048, False),
    (64, 2048, False), (8, 2048, False)])
def test_fused_decode_eligibility(world, tokens, eligible):
    assert (X.RUBIN_FUSED_DECODE_NOTE in _notes(tokens, world)) == eligible


def test_ep16_fused_decode_opt_out(monkeypatch):
    monkeypatch.setenv("VLLM_K31_EP_FUSED_DECODE", "0")
    assert X.RUBIN_FUSED_DECODE_NOTE not in _notes(2048, 16)


def test_ineligible_inference_server_unchanged():
    layer = _layer(2048, 64)
    layer.fused_decode_eligible = False
    before = layer.buckets
    assert not _enable(layer)
    assert layer.buckets == before


def _notes(tokens, world):
    X.select_rubin_bucket.cache_clear()
    problem = C.RubinMoeProblem(quant_kind="nvfp4")
    return X.select_rubin_buckets(problem, world, tokens, prefill_worker=False)[1]
