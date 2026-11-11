# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Kimi K3 fused GEMM-RS/AR policy (CPU only): GEMM-RS under sequence
parallelism and GEMM-AR otherwise are on by policy, independent of any environment
switch, except GEMM-RS on the specialized decode's SP topology."""

from types import SimpleNamespace

import pytest

from vllm import envs
from vllm.model_executor.kernels.linear.cute_dsl import gemm_rs_ar
from vllm.models.kimi_k3.nvidia import model


@pytest.fixture
def init_calls(monkeypatch):
    calls = []

    def fake_init(vllm_config, *, N, all_reduce):
        calls.append((N, all_reduce))
        return True

    monkeypatch.setattr(gemm_rs_ar, "maybe_init_gemm_rs_ar", fake_init)
    return calls


def _config(hidden_size=7168):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(hidden_size=hidden_size)
        )
    )


@pytest.mark.parametrize("env", [False, True])
@pytest.mark.parametrize("sp", [False, True])
def test_on_by_policy(monkeypatch, init_calls, sp, env):
    monkeypatch.setattr(envs, "VLLM_ENABLE_GEMM_RS", env, raising=False)
    monkeypatch.setattr(model, "specialized_decode_config", lambda c: None)
    assert model.maybe_init_gemm_rs_ar(_config(), use_sequence_parallel=sp)
    assert init_calls == [(7168, not sp)]


def test_specialized_decode_sp_keeps_unfused_rs(monkeypatch, init_calls):
    decode = SimpleNamespace(sequence_parallel=True)
    monkeypatch.setattr(model, "specialized_decode_config", lambda c: decode)
    assert not model.maybe_init_gemm_rs_ar(_config(), use_sequence_parallel=True)
    assert init_calls == []
    # Without SP (TP8/16 specialized decode) the GEMM-AR policy is unchanged.
    assert model.maybe_init_gemm_rs_ar(_config(), use_sequence_parallel=False)
    assert init_calls == [(7168, True)]


def test_capability_fallback_reported(monkeypatch):
    monkeypatch.setattr(model, "specialized_decode_config", lambda c: None)
    monkeypatch.setattr(
        gemm_rs_ar, "maybe_init_gemm_rs_ar", lambda c, *, N, all_reduce: False
    )
    assert not model.maybe_init_gemm_rs_ar(_config(), use_sequence_parallel=True)


@pytest.mark.parametrize("bound", [False, True])
def test_dsa_binding_skips_specialized_decode(monkeypatch, bound):
    """The DSA o_proj takes the fused GEMM-RS/AR only without the specialized
    decode: the C256 decode role (TP16 non-SP, latency_specialized) keeps its
    stock GEMM + all-reduce DSA o_proj."""
    monkeypatch.setattr(model, "specialized_decode_config", lambda c: None)
    assert model.dsa_runs_gemm_rs_ar(_config(), bound) is bound
    decode = SimpleNamespace(sequence_parallel=False)
    monkeypatch.setattr(model, "specialized_decode_config", lambda c: decode)
    assert model.dsa_runs_gemm_rs_ar(_config(), bound) is False
