# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serving-shape warmup (full-budget mixed steps) and its JIT verification."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from vllm.utils import jit_monitor
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu import warmup

pytestmark = pytest.mark.cpu_test

BUDGET = 8192


def _runner(
    dp_rank=0,
    num_spec=2,
    max_num_seqs=256,
    max_model_len=131072,
    num_blocks=100000,
    kv_cache_specs=None,
    dcp_size=1,
    budget=BUDGET,
):
    if kv_cache_specs is None:
        kv_cache_specs = [SimpleNamespace(block_size=64, dcp_sharded=False)]
    return SimpleNamespace(
        is_pooling_model=False,
        is_encoder_decoder=False,
        max_num_reqs=max_num_seqs,
        decode_query_len=num_spec + 1,
        num_speculative_steps=num_spec,
        max_model_len=max_model_len,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=budget),
        model_config=SimpleNamespace(get_vocab_size=lambda: 163840),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=spec) for spec in kv_cache_specs
            ],
            num_blocks=num_blocks,
        ),
        dcp_size=dcp_size,
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


def test_ladder_steps_cover_every_token_count_octave(monkeypatch):
    """Small serving steps (65 tokens up to the budget, single-sequence prefills
    among them) run at startup in every power-of-two octave, so kernels that
    pick a tile or a specialization by token or sequence count do not compile
    in the first small mixed serving step. Same shapes on every DP rank."""
    runner = _runner()
    ladder = warmup._serving_ladder_shapes(runner)
    assert ladder == warmup._serving_ladder_shapes(_runner(dp_rank=3))
    monkeypatch.setattr(warmup.torch.accelerator, "synchronize", lambda: None)
    totals, single = [], []
    for i, shapes in enumerate(ladder):
        steps: list[SchedulerOutput] = []
        assert warmup.run_serving_shape_warmup(
            runner, steps.append, lambda grammar_output: None, 100 + i, ladder=shapes
        )
        step1, step2, step3, cleanup = steps
        assert len(step1.scheduled_new_reqs) == 1
        single.append(step1.total_num_scheduled_tokens)
        for step in (step1, step2, step3):
            assert step.total_num_scheduled_tokens <= BUDGET
            totals.append(step.total_num_scheduled_tokens)
        assert cleanup.finished_req_ids == {*step3.num_scheduled_tokens}
    for k in range(6, BUDGET.bit_length() - 1):
        assert any(2**k < t <= 2 ** (k + 1) for t in totals), k
    assert min(single) <= 128 and max(single) > BUDGET // 4


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


def _full_blocks(runner) -> int:
    return warmup._serving_step_blocks(runner, warmup._serving_step_shapes(runner, 0))[
        0
    ]


@pytest.mark.parametrize("dp_rank", [0, 3])
def test_small_kv_pool_shrinks_the_decode_batch_instead_of_skipping(
    dp_rank, monkeypatch
):
    """A decode server whose KV pool holds fewer warmup requests than
    max_num_seqs (2026-11-12 C256: 4326 blocks for 4457) still runs every
    variant, including the verification step, with a smaller decode batch."""
    full = _full_blocks(_runner(dp_rank=dp_rank))
    runner = _runner(dp_rank=dp_rank, num_blocks=full - 50)
    full_decodes = warmup._serving_step_shapes(runner, 0)[0]
    cap = warmup._serving_decode_cap(runner)
    assert cap is not None and 1 <= cap < full_decodes
    shapes = []
    for variant in range(warmup.NUM_SERVING_WARMUP_VARIANTS + 1):
        steps = _run(runner, variant, monkeypatch)
        num_decodes = len(steps[0].scheduled_new_reqs)
        assert 1 <= num_decodes <= cap
        shapes.append(warmup._serving_step_shapes(runner, variant, cap))
        required = warmup._serving_step_blocks(runner, shapes[-1])[0]
        assert required < runner.kv_cache_config.num_blocks
    # The verification step still differs from the warmed shapes.
    assert all(shapes[-1] != shape for shape in shapes[:-1])


def test_kv_pool_too_small_for_one_request_fails_startup(monkeypatch):
    runner = _runner(num_blocks=8)
    with pytest.raises(RuntimeError, match="cannot fit one decode request"):
        warmup.run_serving_shape_warmup(
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


def _held_blocks_per_request(steps: list[SchedulerOutput]) -> dict[str, list[int]]:
    """Most blocks each request holds per KV group over the steps, as the
    worker's block table accumulates them (``BlockTables.append_block_ids``)."""
    held: dict[str, list[int]] = {}
    peak: dict[str, list[int]] = {}
    for step in steps:
        for new_req in step.scheduled_new_reqs:
            held[new_req.req_id] = [len(ids) for ids in new_req.block_ids]
        cached = step.scheduled_cached_reqs
        for req_id, new_block_ids in zip(cached.req_ids, cached.new_block_ids):
            if new_block_ids is not None:
                held[req_id] = [
                    n + len(ids) for n, ids in zip(held[req_id], new_block_ids)
                ]
        for req_id, counts in held.items():
            peak[req_id] = [max(a, b) for a, b in zip(peak.get(req_id, counts), counts)]
    return peak


@pytest.mark.parametrize(
    ("block_size", "max_model_len", "budget"),
    [
        # BUG-DPARITY-1: K3.1 DCP4 decode, attention groups at 4096-token
        # blocks (raised to the Mamba page), --max-model-len 16384: one block
        # per row; the 2045 + 3067-token ladder set took 2 blocks.
        (4096, 16384, 4096),
        (64, 4096, BUDGET),
    ],
)
def test_dcp_sharded_groups_fit_the_block_table_rows(
    block_size, max_model_len, budget, monkeypatch
):
    """Every warmup step set must fit the worker's block-table rows. A
    DCP-sharded group's block holds ``block_size * dcp`` tokens of a request
    (the scheduler allocates so, and the rows are
    ``cdiv(max_model_len, block_size * dcp)`` wide), so the warmup must not
    count ``block_size``-token blocks for it."""
    dcp = 4
    specs = [
        SimpleNamespace(block_size=block_size, dcp_sharded=True),
        SimpleNamespace(block_size=block_size, dcp_sharded=False),
    ]
    runner = _runner(
        num_spec=3,
        max_num_seqs=96,
        max_model_len=max_model_len,
        kv_cache_specs=specs,
        dcp_size=dcp,
        budget=budget,
    )
    row_capacity = [
        cdiv(max_model_len, block_size * dcp),
        cdiv(max_model_len, block_size),
    ]
    monkeypatch.setattr(warmup.torch.accelerator, "synchronize", lambda: None)
    step_sets = [
        dict(ladder=shapes) for shapes in warmup._serving_ladder_shapes(runner)
    ] + [dict() for _ in range(warmup.NUM_SERVING_WARMUP_VARIANTS + 1)]
    assert len(step_sets) > warmup.NUM_SERVING_WARMUP_VARIANTS + 1
    longest = 0
    for i, kwargs in enumerate(step_sets):
        steps: list[SchedulerOutput] = []
        if not warmup.run_serving_shape_warmup(
            runner, steps.append, lambda grammar_output: None, 100 + i, **kwargs
        ):
            continue
        for req_id, held in _held_blocks_per_request(steps).items():
            longest = max(longest, held[1])
            for group, (n, cap) in enumerate(zip(held, row_capacity)):
                assert n <= cap, (
                    f"step set {i}: request {req_id} holds {n} blocks of group "
                    f"{group}, the block-table row holds {cap}"
                )
    # The ladder reaches prompts longer than one unsharded block.
    assert longest > 1
