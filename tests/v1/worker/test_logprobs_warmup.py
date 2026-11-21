# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup warmup of the logprobs kernels (`_topk_log_softmax_kernel`,
`_fill_logprob_token_ids_kernel`).

A request whose logprobs / prompt_logprobs / logprob_token_ids count differs
from the startup warmup's must not compile a kernel in the serving loop.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.sampling_params import MAX_LOGPROB_TOKEN_IDS, SamplingParams
from vllm.utils import jit_monitor
from vllm.v1.worker.gpu import warmup
from vllm.v1.worker.gpu.sample import logprob
from vllm.v1.worker.gpu.sample.logprob import (
    LogprobTokenIdsState,
    compute_token_logprobs,
    compute_topk_scores,
    token_logprobs_warmup_counts,
    warmup_logprob_token_ids,
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


def _events(events: list[str], kernel: str) -> list[str]:
    return [e for e in events if kernel in e]


def _topk_events(events: list[str]) -> list[str]:
    return _events(events, "_topk_log_softmax_kernel")


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


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="needs a GPU")
def test_warmup_compiles_every_logprob_token_ids_width():
    """Requests with `logprob_token_ids` fill [rows, 1 + max(num_logprobs,
    longest list)] token ids; every width must be compiled at startup."""
    num_rows = 2
    logits = torch.randn(num_rows, 4096, device="cuda")
    state = LogprobTokenIdsState(num_rows, torch.device("cuda"))
    for req_idx in range(num_rows):
        state.add_request(req_idx, SamplingParams(logprobs=1, logprob_token_ids=[1]))
    state.apply_staged_writes()
    sampled = torch.zeros(num_rows, dtype=torch.int64, device="cuda")
    mapping = torch.arange(num_rows, dtype=torch.int64, device="cuda")

    def serve():
        for num_logprobs in range(MAX_LOGPROBS + 1):
            for longest in range(1, MAX_LOGPROB_TOKEN_IDS + 1):
                compute_topk_scores(
                    logits,
                    num_logprobs,
                    sampled,
                    logprob_token_ids_state=state,
                    expanded_idx_mapping=mapping,
                    max_per_req_token_ids=longest,
                )
        torch.accelerator.synchronize()

    kernel = logprob._fill_logprob_token_ids_kernel
    warmup_token_logprobs(logits, MAX_COUNT)
    kernel.device_caches.clear()
    with jit_monitor.record_jit_events() as cold:
        serve()
    assert _events(cold, "_fill_logprob_token_ids_kernel")

    kernel.device_caches.clear()
    warmup_logprob_token_ids(state, sampled, mapping, MAX_COUNT - 1)
    with jit_monitor.record_jit_events() as warm:
        serve()
    assert not _events(warm, "_fill_logprob_token_ids_kernel")
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
    """The runner warms the logprob_token_ids fill in every mode, and the
    log-softmax on the compute_logits rows (raw), plus FP32 rows in the
    processed modes; the logits modes gather instead."""
    raw = torch.zeros(1, 64, dtype=torch.bfloat16)
    calls = []
    fills = []
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
    monkeypatch.setattr(
        warmup,
        "warmup_logprob_token_ids",
        lambda state, sampled, mapping, max_num_cols: fills.append(
            (state, sampled.dtype, mapping.dtype, max_num_cols)
        ),
    )
    state = object()
    runner = SimpleNamespace(
        is_pooling_model=False,
        is_last_pp_rank=True,
        sampler=SimpleNamespace(
            logprobs_mode=logprobs_mode, logprob_token_ids_state=state
        ),
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
    assert fills == [(state, torch.int64, torch.int64, MAX_COUNT - 1)]
    assert len(calls) == len(expected)
    for (layout, max_count, unaligned), (kind, want_unaligned) in zip(calls, expected):
        assert max_count == MAX_COUNT
        assert unaligned == want_unaligned
        if kind == "raw":
            assert layout is raw
        else:
            assert layout.dtype == kind and layout.shape == raw.shape
