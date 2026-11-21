# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup warmup of the logprobs kernel (`_topk_log_softmax_kernel`).

A request whose logprobs / prompt_logprobs / logprob_token_ids count differs
from the startup warmup's must not compile the kernel in the serving loop.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.sampling_params import MAX_LOGPROB_TOKEN_IDS
from vllm.utils import jit_monitor
from vllm.v1.worker.gpu import warmup
from vllm.v1.worker.gpu.sample import logprob
from vllm.v1.worker.gpu.sample.logprob import (
    compute_token_logprobs,
    compute_topk_scores,
    token_logprobs_warmup_counts,
    warmup_token_logprobs,
)

MAX_LOGPROBS = 20  # ModelConfig.max_logprobs default
MAX_COUNT = 1 + max(MAX_LOGPROBS, MAX_LOGPROB_TOKEN_IDS)


def test_warm_list():
    # One count per (TOPK_BLOCK_SIZE, count == 1, count % 16 == 0) variant.
    assert token_logprobs_warmup_counts(MAX_COUNT) == [
        1, 2, 3, 5, 9, 16, 17, 32, 33, 48, 65, 80, 129
    ]  # fmt: skip
    assert token_logprobs_warmup_counts(MAX_LOGPROBS, odd_only=True) == [
        1, 3, 5, 9, 17
    ]  # fmt: skip
    # max_logprobs=-1 (vocab-size requests): TOPK_BLOCK_SIZE is capped at 1024.
    # Blocks 1, 2, 4, 8 once, 16..1024 with and without divisibility by 16.
    vocab_counts = token_logprobs_warmup_counts(1 + 163840)
    assert len(vocab_counts) == 4 + 7 * 2
    assert max(vocab_counts) < 1024


def _topk_events(events: list[str]) -> list[str]:
    return [e for e in events if "_topk_log_softmax_kernel" in e]


def _serve_every_count(logits: torch.Tensor) -> None:
    """Every token-id count and layout the server's logprob paths can pass."""
    num_rows = logits.shape[0]
    sampled = torch.zeros(num_rows, dtype=torch.int64, device=logits.device)
    for num_logprobs in range(MAX_COUNT):
        # Sample / rejection-sampler / prompt logprobs: 1 + num_logprobs columns.
        compute_topk_scores(logits, num_logprobs, sampled)
        # logprob_token_ids (up to MAX_LOGPROB_TOKEN_IDS) widen the rows.
        ids = torch.zeros(num_rows, num_logprobs + 1, dtype=torch.int64)
        compute_token_logprobs(logits, ids.to(logits.device))
    for num_ids in range(1, MAX_LOGPROBS + 1):
        # prompt_logprob_token_ids chunks slice a [rows, num_ids] array at any row.
        ids = torch.zeros(2 * num_rows + 1, num_ids, dtype=torch.int64)
        ids = ids.to(logits.device)
        for start in (0, 1):
            compute_token_logprobs(logits, ids[start : start + num_rows])
    torch.accelerator.synchronize()


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="needs a GPU")
@pytest.mark.parametrize(
    "dtype, vocab_size, row_offset",
    [
        (torch.bfloat16, 163840, 0),
        (torch.float32, 163840, 0),
        # Rows not a multiple of 16 bytes, scored from an odd row (row chunks).
        (torch.bfloat16, 32001, 1),
    ],
)
def test_warmup_compiles_every_count(dtype, vocab_size, row_offset):
    rows = torch.randn(row_offset + 3, vocab_size, dtype=dtype, device="cuda")
    logits = rows[row_offset:]
    kernel = logprob._topk_log_softmax_kernel

    kernel.device_caches.clear()
    with jit_monitor.record_jit_events() as cold:
        _serve_every_count(logits)
    # Without the warmup the counts compile while serving.
    assert _topk_events(cold)

    kernel.device_caches.clear()
    warmup_token_logprobs(logits, MAX_COUNT, max_unaligned_count=MAX_LOGPROBS)
    with jit_monitor.record_jit_events() as warm:
        _serve_every_count(logits)
    assert not _topk_events(warm)


@pytest.mark.parametrize(
    "logprobs_mode, expected",
    [
        ("raw_logprobs", [("raw", MAX_LOGPROBS)]),
        ("processed_logprobs", [("raw", MAX_LOGPROBS), (torch.float32, 0)]),
        ("raw_logits", []),
        ("processed_logits", []),
    ],
)
def test_warmup_logprobs_kernels_layouts(monkeypatch, logprobs_mode, expected):
    """The runner warms the compute_logits rows (raw), plus FP32 rows in the
    processed modes; the logits modes never run the kernel."""
    raw = torch.zeros(1, 64, dtype=torch.bfloat16)
    calls = []
    monkeypatch.setattr(
        warmup, "current_platform", SimpleNamespace(is_cpu=lambda: False)
    )
    monkeypatch.setattr(
        warmup,
        "warmup_token_logprobs",
        lambda layout, max_count, max_unaligned_count: calls.append(
            (layout, max_count, max_unaligned_count)
        ),
    )
    runner = SimpleNamespace(
        is_pooling_model=False,
        is_last_pp_rank=True,
        sampler=SimpleNamespace(logprobs_mode=logprobs_mode),
        model_config=SimpleNamespace(
            max_logprobs=MAX_LOGPROBS,
            dtype=torch.bfloat16,
            get_hidden_size=lambda: 8,
        ),
        device="cpu",
        vocab_size=64,
        batch_sharder=None,
        model=SimpleNamespace(compute_logits=lambda hidden_states: raw),
    )
    warmup._warmup_logprobs_kernels(runner)
    assert len(calls) == len(expected)
    for (layout, max_count, unaligned), (kind, want_unaligned) in zip(calls, expected):
        assert max_count == MAX_COUNT
        assert unaligned == want_unaligned
        if kind == "raw":
            assert layout is raw
        else:
            assert layout.dtype == kind and layout.shape == raw.shape
