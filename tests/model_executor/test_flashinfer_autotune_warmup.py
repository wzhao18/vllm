# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.model_executor.layers.fused_moe.utils import (
    fi_moe_tune_max_num_tokens,
)
from vllm.model_executor.warmup.kernel_warmup import (
    _FLASHINFER_BF16_AUTOTUNE_MAX_TOKENS,
    _flashinfer_autotune_token_counts,
    _run_flashinfer_autotune_dummy_runs,
)


def _moe_config(
    *,
    deferred: bool,
    deferred_limit: int,
    max_num_tokens: int = 2048,
    dp_size: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        use_deferred_moe_finalize=deferred,
        defer_moe_finalize_max_num_tokens=deferred_limit,
        max_num_tokens=max_num_tokens,
        dp_size=dp_size,
    )


def _runner(
    *moe_configs: SimpleNamespace,
    max_tokens: int = 8192,
    linear_backend: str | None = None,
) -> SimpleNamespace:
    modules = [SimpleNamespace(moe_config=config) for config in moe_configs]
    modules.append(SimpleNamespace())
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_tokens),
        vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(linear_backend=linear_backend)
        ),
        get_model=lambda: SimpleNamespace(modules=lambda: iter(modules)),
    )


@pytest.mark.parametrize(
    ("deferred", "limit", "expected"),
    [
        (False, 128, 8192),
        (True, 128, 128),
        (True, -1, 8192),
        (True, 16384, 8192),
    ],
)
def test_fi_moe_tune_max_num_tokens(
    deferred: bool, limit: int, expected: int
) -> None:
    config = _moe_config(deferred=deferred, deferred_limit=limit)

    assert (
        fi_moe_tune_max_num_tokens(config, deferred_finalize=deferred) == expected
    )


def test_flashinfer_autotune_token_counts_cover_deferred_limits() -> None:
    runner = _runner(
        _moe_config(deferred=True, deferred_limit=128),
        _moe_config(deferred=True, deferred_limit=64),
        _moe_config(deferred=True, deferred_limit=128),
        _moe_config(deferred=False, deferred_limit=16),
    )

    assert _flashinfer_autotune_token_counts(runner) == (8192, 128, 64)


def test_flashinfer_autotune_token_counts_preserve_bf16_warmup() -> None:
    runner = _runner(
        _moe_config(
            deferred=True,
            deferred_limit=_FLASHINFER_BF16_AUTOTUNE_MAX_TOKENS,
        ),
        _moe_config(deferred=True, deferred_limit=128),
        linear_backend="flashinfer_cutedsl",
    )

    assert _flashinfer_autotune_token_counts(runner) == (
        8192,
        128,
        _FLASHINFER_BF16_AUTOTUNE_MAX_TOKENS,
    )


def test_flashinfer_autotune_token_counts_ignore_out_of_range_limits() -> None:
    runner = _runner(
        _moe_config(deferred=True, deferred_limit=-1),
        _moe_config(deferred=True, deferred_limit=0),
        _moe_config(deferred=True, deferred_limit=8192),
        _moe_config(deferred=True, deferred_limit=16384),
    )

    assert _flashinfer_autotune_token_counts(runner) == (8192,)


def test_flashinfer_autotune_secondary_runs_skip_attention() -> None:
    runner = _runner(
        _moe_config(deferred=True, deferred_limit=128),
        max_tokens=8192,
    )
    calls: list[dict[str, object]] = []
    runner._dummy_run = lambda **kwargs: calls.append(kwargs)

    _run_flashinfer_autotune_dummy_runs(runner)

    assert calls == [
        {
            "num_tokens": 8192,
            "skip_eplb": True,
            "is_profile": True,
            "randomize_inputs": True,
            "skip_attn": False,
        },
        {
            "num_tokens": 128,
            "skip_eplb": True,
            "is_profile": True,
            "randomize_inputs": True,
            "skip_attn": True,
        },
    ]
