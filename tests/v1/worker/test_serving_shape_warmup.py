# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serving-shape warmup (full-budget mixed steps) and its JIT verification."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from vllm.utils import jit_monitor
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu import warmup

pytestmark = pytest.mark.cpu_test

BUDGET = 8192


def _runner(dp_rank=0, num_spec=2, max_num_seqs=256, max_model_len=131072):
    return SimpleNamespace(
        is_pooling_model=False,
        is_encoder_decoder=False,
        max_num_reqs=max_num_seqs,
        decode_query_len=num_spec + 1,
        num_speculative_steps=num_spec,
        max_model_len=max_model_len,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=BUDGET),
        model_config=SimpleNamespace(get_vocab_size=lambda: 163840),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=64))
            ],
            num_blocks=100000,
        ),
        vllm_config=SimpleNamespace(
            num_lookahead_tokens=num_spec,
            is_mm_encoder_only=False,
            parallel_config=SimpleNamespace(data_parallel_rank=dp_rank),
        ),
        model_state=SimpleNamespace(max_encoder_len=0),
        kv_connector=SimpleNamespace(set_disabled=lambda disabled: None),
        kv_block_zeroer=None,
    )


def _run(runner, variant, monkeypatch):
    monkeypatch.setattr(warmup.torch.accelerator, "synchronize", lambda: None)
    steps: list[SchedulerOutput] = []
    assert warmup.run_serving_shape_warmup(
        runner, steps.append, lambda grammar_output: None, variant
    )
    return steps


@pytest.mark.parametrize("variant", [0, 1, 2])
def test_steps_fill_the_budget_with_mixed_rows(variant, monkeypatch):
    runner = _runner()
    step1, step2, step3, cleanup = _run(runner, variant, monkeypatch)
    for step in (step1, step2, step3):
        assert step.total_num_scheduled_tokens <= BUDGET
        assert step.total_num_scheduled_tokens == sum(
            step.num_scheduled_tokens.values()
        )
    num_decodes = len(step1.scheduled_new_reqs)
    assert num_decodes > 1
    # Step 2: spec-decode rows + a fresh prompt chunk longer than DSA top-k.
    (long_req,) = step2.scheduled_new_reqs
    assert long_req.num_computed_tokens == 0
    first = step2.num_scheduled_tokens[long_req.req_id]
    assert first > 2048
    assert len(step2.scheduled_spec_decode_tokens) == num_decodes
    # Step 3: the prompt continues over its first chunk, plus a cache hit.
    cached = step3.scheduled_cached_reqs
    assert long_req.req_id in cached.req_ids
    assert cached.num_computed_tokens[cached.req_ids.index(long_req.req_id)] == first
    (hit,) = step3.scheduled_new_reqs
    assert hit.num_computed_tokens > 2048
    assert cleanup.finished_req_ids == {
        *step3.num_scheduled_tokens,
    }
    if variant == 0:
        # The cache hit's new token is held back for step 3.
        assert step2.total_num_scheduled_tokens == BUDGET - 1
        assert step3.total_num_scheduled_tokens == BUDGET


def test_verification_shapes_differ_per_dp_rank_and_from_warmup(monkeypatch):
    shapes = {
        (variant, rank): warmup._serving_step_shapes(_runner(dp_rank=rank), variant)
        for variant in range(warmup.NUM_SERVING_WARMUP_VARIANTS + 1)
        for rank in range(4)
    }
    verify = warmup.NUM_SERVING_WARMUP_VARIANTS
    assert len({shapes[(verify, rank)] for rank in range(4)}) == 4
    for rank in range(4):
        for variant in range(verify):
            assert shapes[(verify, rank)] != shapes[(variant, rank)]


def test_skips_when_model_len_is_too_short(monkeypatch):
    runner = _runner(max_model_len=4)
    assert not warmup.run_serving_shape_warmup(
        runner, lambda out: None, lambda grammar_output: None, 0
    )


def test_compile_in_verification_fails_startup(monkeypatch):
    variants: list[int] = []

    def fake_run(runner, execute, sample, variant):
        variants.append(variant)
        if variant == warmup.NUM_SERVING_WARMUP_VARIANTS:
            jit_monitor._handle_jit_event(
                backend="Triton",
                event="kernel JIT compilation",
                fn_name="_dcp_a2a_pack_send_kernel",
            )
        return True

    @contextmanager
    def fake_record():
        events: list[str] = []
        jit_monitor._recorded_events = events
        try:
            yield events
        finally:
            jit_monitor._recorded_events = None

    monkeypatch.setattr(warmup, "run_serving_shape_warmup", fake_run)
    monkeypatch.setattr(jit_monitor, "record_jit_events", fake_record)
    with pytest.raises(warmup.JitWarmupGapError, match="_dcp_a2a_pack_send_kernel"):
        warmup.serving_shape_warmup(None, None, None)
    assert variants == list(range(warmup.NUM_SERVING_WARMUP_VARIANTS + 1))


def test_clean_verification_passes(monkeypatch):
    @contextmanager
    def fake_record():
        yield []

    monkeypatch.setattr(warmup, "run_serving_shape_warmup", lambda *a: True)
    monkeypatch.setattr(jit_monitor, "record_jit_events", fake_record)
    warmup.serving_shape_warmup(None, None, None)
