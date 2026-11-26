# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA-graph capture default of a Kimi K3 P/D prefill worker.

The prefill worker of a P/D deployment (``pd_prefill_worker``) captures up to
min(max_num_batched_tokens, 8192) tokens by default; decode instances and non-P/D
serving keep vLLM's default, and explicit capture settings are never overridden.
The final sizes are checked through ``VllmConfig._set_cudagraph_sizes``.
"""

from types import SimpleNamespace

import pytest

from vllm.config import CompilationConfig, CUDAGraphMode, VllmConfig
from vllm.model_executor.models.config import KimiK3ForConditionalGenerationConfig
from vllm.models.kimi_k3.nvidia.model import (
    PD_PREFILL_CAPTURE_MAX_TOKENS,
    pd_prefill_capture_sizes,
)
from vllm.platforms import current_platform

# The PR 1906 production prefill render pins these (max 2048).
RENDER_SIZES = [7, 8, 14, 16, 28, 32, 56, 64, 112, 128, 168, 192, 224, 256, 384, 512,
                640, 768, 896, 1024, 1280, 1536, 1792, 2048]  # fmt: skip


@pytest.fixture(autouse=True)
def _cuda(monkeypatch):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)


def _kv(role):
    if role is None:
        return None
    return SimpleNamespace(
        kv_role=role,
        is_kv_producer=role in ("kv_producer", "kv_both"),
        is_kv_consumer=role in ("kv_consumer", "kv_both"),
    )


def _cfg(role, extra=None, mnbt=16384, sizes=None, max_size=None, eager=False):
    """A VllmConfig stand-in with the fields the hook and _set_cudagraph_sizes read
    (TP8+DCP8 prefill / TP16+DCP16 decode recipe: max-num-seqs 32, DSpark k7)."""
    # unset knobs are omitted (the config rejects an explicit None)
    knobs = {}
    if sizes is not None:
        knobs["cudagraph_capture_sizes"] = sizes
    if max_size is not None:
        knobs["max_cudagraph_capture_size"] = max_size
    compilation = CompilationConfig(
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE, **knobs
    )
    return SimpleNamespace(
        kv_transfer_config=_kv(role),
        additional_config=extra or {},
        model_config=SimpleNamespace(enforce_eager=eager),
        compilation_config=compilation,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=mnbt, max_num_seqs=32),
        speculative_config=None,
        uniform_decode_query_len=8,
        num_speculative_tokens=7,
        performance_mode="balanced",
    )


def _final(cfg, hook=True):
    if hook:
        KimiK3ForConditionalGenerationConfig._default_pd_prefill_capture_sizes(cfg)
    VllmConfig._set_cudagraph_sizes(cfg)
    c = cfg.compilation_config
    return list(c.cudagraph_capture_sizes), c.max_cudagraph_capture_size


def test_capture_sizes_shape():
    sizes = pd_prefill_capture_sizes(16384)
    assert PD_PREFILL_CAPTURE_MAX_TOKENS == 8192
    assert sizes[:4] == [1, 2, 4, 8] and sizes[-3:] == [2048, 4096, 8192]
    assert sizes == sorted(set(sizes))
    fine = [s for s in sizes if s <= 2048]
    assert fine == [1, 2, 4, *range(8, 256, 8), *range(256, 2049, 16)]
    # capped at the step budget, which is itself captured
    assert pd_prefill_capture_sizes(8192)[-2:] == [4096, 8192]
    assert pd_prefill_capture_sizes(6000)[-3:] == [2048, 4096, 6000]
    assert pd_prefill_capture_sizes(3000)[-2:] == [2048, 3000]
    assert pd_prefill_capture_sizes(1000)[-1] == 1000
    assert max(pd_prefill_capture_sizes(1000)) == 1000


@pytest.mark.parametrize("role", ["kv_producer", "kv_both"])
def test_prefill_role_gets_extended_capture(role):
    sizes, max_size = _final(_cfg(role))
    assert max_size == 8192
    assert sizes == pd_prefill_capture_sizes(16384)
    assert sizes[-3:] == [2048, 4096, 8192]


def test_prefill_role_capped_by_step_budget():
    sizes, max_size = _final(_cfg("kv_producer", mnbt=4096))
    assert max_size == 4096 and 8192 not in sizes


@pytest.mark.parametrize(
    ("role", "extra"),
    [
        (None, {}),  # unified serving
        ("kv_consumer", {}),  # P/D decode
        ("kv_consumer", {"decode_backend": "latency_specialized"}),
        # a decode server launched kv_both on the specialized decode
        ("kv_both", {"decode_backend": "latency_specialized"}),
    ],
)
def test_decode_and_non_pd_unchanged(role, extra):
    with_hook = _final(_cfg(role, extra))
    stock = _final(_cfg(role, extra), hook=False)
    assert with_hook == stock
    assert with_hook[1] < 2048


@pytest.mark.parametrize(
    ("sizes", "max_size"),
    [(RENDER_SIZES, 2048), (RENDER_SIZES, None), (None, 2048), (None, 512)],
)
def test_explicit_capture_is_kept(sizes, max_size):
    """An explicit recipe (the PR 1906 render pins sizes and the max) is never
    overridden: the prefill sizes must come from the recipe."""
    cfg = _cfg("kv_both", sizes=list(sizes) if sizes else None, max_size=max_size)
    with_hook = _final(cfg)
    cfg2 = _cfg("kv_both", sizes=list(sizes) if sizes else None, max_size=max_size)
    assert with_hook == _final(cfg2, hook=False)
    assert with_hook[1] == (max_size or 2048)


def test_render_with_prefill_sizes():
    """The render diff (sizes + 4096, 8192, --max-cudagraph-capture-size 8192)
    yields the approved capture on the prefill worker."""
    cfg = _cfg("kv_both", sizes=RENDER_SIZES + [4096, 8192], max_size=8192)
    sizes, max_size = _final(cfg)
    assert max_size == 8192 and sizes[-3:] == [2048, 4096, 8192]


def test_eager_and_non_cuda_untouched(monkeypatch):
    cfg = _cfg("kv_producer", eager=True)
    KimiK3ForConditionalGenerationConfig._default_pd_prefill_capture_sizes(cfg)
    assert cfg.compilation_config.cudagraph_capture_sizes is None
    monkeypatch.setattr(current_platform, "is_cuda", lambda: False)
    cfg = _cfg("kv_producer")
    KimiK3ForConditionalGenerationConfig._default_pd_prefill_capture_sizes(cfg)
    assert cfg.compilation_config.cudagraph_capture_sizes is None
    assert cfg.compilation_config.max_cudagraph_capture_size is None
