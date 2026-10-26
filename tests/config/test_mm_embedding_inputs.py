# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VllmConfig._resolve_mm_embedding_inputs: EC/KV consumers accept embedding inputs
unless VLLM_MM_EMBEDS_ON_CONSUMERS=0."""

from types import SimpleNamespace

import pytest

from vllm.config.vllm import VllmConfig


def _config(*, kv_consumer=False, ec_consumer=False, enable_mm_embeds=False):
    mm_config = SimpleNamespace(
        allow_missing_mm_embeddings=False, enable_mm_embeds=enable_mm_embeds
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(multimodal_config=mm_config),
        kv_transfer_config=SimpleNamespace(is_kv_consumer=kv_consumer),
        ec_transfer_config=SimpleNamespace(is_ec_consumer=ec_consumer),
    )


@pytest.mark.parametrize(
    "kv_consumer, ec_consumer", [(True, False), (False, True), (True, True)]
)
def test_consumer_enables_mm_embeds_by_default(monkeypatch, kv_consumer, ec_consumer):
    monkeypatch.delenv("VLLM_MM_EMBEDS_ON_CONSUMERS", raising=False)
    config = _config(kv_consumer=kv_consumer, ec_consumer=ec_consumer)
    VllmConfig._resolve_mm_embedding_inputs(config)
    mm_config = config.model_config.multimodal_config
    assert mm_config.allow_missing_mm_embeddings
    assert mm_config.enable_mm_embeds


@pytest.mark.parametrize("kv_consumer, ec_consumer", [(True, False), (False, True)])
def test_consumer_opt_out_keeps_mm_embeds_off(monkeypatch, kv_consumer, ec_consumer):
    monkeypatch.setenv("VLLM_MM_EMBEDS_ON_CONSUMERS", "0")
    config = _config(kv_consumer=kv_consumer, ec_consumer=ec_consumer)
    VllmConfig._resolve_mm_embedding_inputs(config)
    mm_config = config.model_config.multimodal_config
    assert mm_config.allow_missing_mm_embeddings
    assert not mm_config.enable_mm_embeds


def test_opt_out_keeps_explicit_mm_embeds(monkeypatch):
    monkeypatch.setenv("VLLM_MM_EMBEDS_ON_CONSUMERS", "0")
    config = _config(kv_consumer=True, enable_mm_embeds=True)
    VllmConfig._resolve_mm_embedding_inputs(config)
    assert config.model_config.multimodal_config.enable_mm_embeds


@pytest.mark.parametrize("opt_out", ["0", "1"])
def test_non_consumer_untouched(monkeypatch, opt_out):
    monkeypatch.setenv("VLLM_MM_EMBEDS_ON_CONSUMERS", opt_out)
    config = _config()
    VllmConfig._resolve_mm_embedding_inputs(config)
    mm_config = config.model_config.multimodal_config
    assert not mm_config.allow_missing_mm_embeddings
    assert not mm_config.enable_mm_embeds


def test_text_only_model_untouched(monkeypatch):
    monkeypatch.delenv("VLLM_MM_EMBEDS_ON_CONSUMERS", raising=False)
    config = _config(kv_consumer=True)
    config.model_config.multimodal_config = None
    VllmConfig._resolve_mm_embedding_inputs(config)
