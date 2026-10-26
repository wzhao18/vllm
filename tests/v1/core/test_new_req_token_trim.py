# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""New requests ship to v2 workers only the token ids the workers read.

The engine pickles a new request's data and every worker unpickles and stages
it in the step that admits the request, so a long prompt admitted after a
prefix-cache hit or a KV transfer used to stall the GPU in proportion to its
full length. ``trim_new_request_data`` keeps the ids from the first uncomputed
position on, unless a worker-side reader of earlier positions is active.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched import scheduler as scheduler_module
from vllm.v1.core.sched.output import (
    NewRequestData,
    needs_full_token_ids,
    trim_new_request_data,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.worker.gpu.states import RequestState

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test

PROMPT_LEN = 100


def _request(sampling_params: SamplingParams | None = None, **kwargs) -> Request:
    params = sampling_params or SamplingParams(max_tokens=8)
    params.update_from_generation_config({}, 2)
    return Request(
        request_id="r",
        prompt_token_ids=list(range(1000, 1000 + PROMPT_LEN)),
        sampling_params=params,
        pooling_params=None,
        **kwargs,
    )


def _new_req_data(request: Request, num_computed_tokens: int) -> NewRequestData:
    request.num_computed_tokens = num_computed_tokens
    return NewRequestData.from_request(request, ([1, 2, 3],), request._all_token_ids)


@pytest.mark.parametrize(
    "params, full",
    [
        (SamplingParams(), False),
        (SamplingParams(temperature=1.0, top_p=0.95, ignore_eos=True), False),
        (SamplingParams(repetition_penalty=1.1), True),
        (SamplingParams(frequency_penalty=0.5), True),
        (SamplingParams(presence_penalty=0.5), True),
        (SamplingParams(prompt_logprobs=0), True),
        (SamplingParams(thinking_token_budget=16), True),
    ],
)
def test_needs_full_token_ids(params: SamplingParams, full: bool) -> None:
    assert needs_full_token_ids(params) is full


def test_needs_full_token_ids_bad_words() -> None:
    params = SamplingParams()
    params._bad_words_token_ids = [[5, 6]]
    assert needs_full_token_ids(params)


def test_trim_keeps_the_uncomputed_tail() -> None:
    request = _request()
    all_ids = request._all_token_ids
    new_req = _new_req_data(request, 92)

    assert trim_new_request_data(new_req, request)
    assert new_req.prefill_token_start == 92
    assert new_req.prefill_token_ids == list(range(1092, 1100))
    assert new_req.prefill_len == PROMPT_LEN
    # The prompt list is not shipped; its length is.
    assert new_req.prompt_token_ids is None
    assert new_req.prompt_len == PROMPT_LEN
    # The engine's own lists are replaced in the data, never mutated.
    assert request._all_token_ids is all_ids
    assert len(request.prompt_token_ids) == len(all_ids) == PROMPT_LEN

    # The worker-bound payload no longer scales with the prompt: 8 ids
    # instead of the prompt list plus the prefill list (2 x 100).
    untrimmed = _new_req_data(_request(), 92)
    assert len(untrimmed.prompt_token_ids) + len(untrimmed.prefill_token_ids) == 200
    assert len(new_req.prefill_token_ids) == 8


def test_no_tail_trim_without_a_hit() -> None:
    request = _request()
    new_req = _new_req_data(request, 0)
    assert not trim_new_request_data(new_req, request)
    assert new_req.prefill_token_start == 0
    assert new_req.prefill_token_ids == request._all_token_ids
    # The duplicate prompt list is still dropped.
    assert new_req.prompt_token_ids is None and new_req.prompt_len == PROMPT_LEN


@pytest.mark.parametrize(
    "params",
    [
        SamplingParams(repetition_penalty=1.1, max_tokens=8),
        SamplingParams(prompt_logprobs=1, max_tokens=8),
        SamplingParams(thinking_token_budget=4, max_tokens=8),
    ],
)
def test_full_ids_for_readers_of_earlier_positions(params: SamplingParams) -> None:
    request = _request(params)
    new_req = _new_req_data(request, 92)
    assert not trim_new_request_data(new_req, request)
    assert new_req.prefill_token_start == 0
    assert new_req.prefill_len == PROMPT_LEN
    assert new_req.prefill_token_ids == request._all_token_ids


def test_untouched_for_resumable_requests() -> None:
    request = _request(resumable=True)
    new_req = _new_req_data(request, 92)
    assert not trim_new_request_data(new_req, request)
    assert new_req.prompt_token_ids == request.prompt_token_ids
    assert new_req.prefill_token_start == 0


class _StagedWrites:
    """Records ``StagedWriteTensor`` writes and applies them to a numpy row."""

    def __init__(self, shape: tuple[int, ...], fill: int = 0):
        self.np = np.full(shape, fill, dtype=np.int64)
        self.writes: list[tuple] = []

    def stage_write(self, index: int, start: int, x) -> None:
        self.writes.append((index, start, list(x)))
        self.np[index, start : start + len(x)] = x

    def stage_write_elem(self, index: int, x: int) -> None:
        self.writes.append((index, x))
        self.np[index] = x


def _request_state(max_len: int = 128) -> RequestState:
    # Only the host-side bookkeeping of RequestState.add_request; the device
    # buffers are replaced by recorders (no GPU on CPU test runners).
    state = object.__new__(RequestState)
    state.req_id_to_index, state.index_to_req_id = {}, {}
    state.free_indices = [0]
    state.num_speculative_steps = 0
    state.max_seq_len = np.zeros(1, dtype=np.int32)
    state.prompt_len = SimpleNamespace(np=np.zeros(1, dtype=np.int32))
    state.prefill_len = SimpleNamespace(np=np.zeros(1, dtype=np.int32))
    state.total_len = _StagedWrites((1,))
    state.num_computed_tokens = _StagedWrites((1,))
    state.num_computed_prefill_tokens = np.zeros(1, dtype=np.int32)
    state.num_computed_tokens_np = np.zeros(1, dtype=np.int32)
    state.all_token_ids = _StagedWrites((1, max_len), fill=-1)
    return state


def _prefill_reads(state: RequestState, query_len: int, lookahead: int) -> list:
    # What _prepare_prefill_inputs_kernel loads for the request: the step's
    # prefill inputs and the next prefill tokens below prefill_len.
    row = state.all_token_ids.np[0]
    start = int(state.num_computed_tokens_np[0])
    end = int(state.prefill_len.np[0])
    inputs = row[start : start + query_len].tolist()
    ahead = [
        int(row[p]) if p < end else 0
        for p in range(start + query_len, start + query_len + lookahead)
    ]
    return inputs + ahead


def test_worker_reads_are_unchanged_by_the_trim() -> None:
    request = _request()
    full = _new_req_data(request, 92)
    trimmed = _new_req_data(_request(), 92)
    assert trim_new_request_data(trimmed, _request())

    states = []
    for data in (full, trimmed):
        state = _request_state()
        state.add_request(
            req_id="r",
            prompt_len=data.prompt_len,
            all_token_ids=data.prefill_token_ids,
            num_computed_tokens=data.num_computed_tokens,
            max_tokens=8,
            start=data.prefill_token_start,
        )
        states.append(state)
    full_state, trimmed_state = states

    assert trimmed_state.all_token_ids.writes[0][:2] == (0, 92)
    for state in states:
        assert state.prefill_len.np[0] == PROMPT_LEN
        assert state.prompt_len.np[0] == PROMPT_LEN
        assert state.total_len.np[0] == PROMPT_LEN
    # Every position a worker reads from the trimmed row matches the full row.
    for query_len, lookahead in ((8, 1), (4, 4), (1, 8)):
        assert _prefill_reads(trimmed_state, query_len, lookahead) == (
            _prefill_reads(full_state, query_len, lookahead)
        )
    np.testing.assert_array_equal(
        trimmed_state.all_token_ids.np[0, 92:PROMPT_LEN],
        full_state.all_token_ids.np[0, 92:PROMPT_LEN],
    )


def test_worker_rejects_a_start_past_the_computed_tokens() -> None:
    with pytest.raises(AssertionError):
        _request_state().add_request(
            req_id="r",
            prompt_len=PROMPT_LEN,
            all_token_ids=list(range(4)),
            num_computed_tokens=90,
            max_tokens=8,
            start=96,
        )


def _engine_config(**overrides) -> SimpleNamespace:
    model_config = SimpleNamespace(uses_mrope=False, logits_processors=None)
    config = SimpleNamespace(
        model_config=model_config, speculative_config=None, watermark_config=None
    )
    for key, value in overrides.items():
        if key in ("uses_mrope", "logits_processors"):
            setattr(model_config, key, value)
        else:
            setattr(config, key, value)
    return config


@pytest.mark.parametrize(
    "overrides, supported",
    [
        ({}, True),
        ({"uses_mrope": True}, False),
        ({"logits_processors": ["pkg.mod:Proc"]}, False),
        ({"watermark_config": object()}, False),
        ({"speculative_config": SimpleNamespace(method="ngram")}, False),
        ({"speculative_config": SimpleNamespace(method="ngram_gpu")}, False),
        ({"speculative_config": SimpleNamespace(method="dspark")}, True),
    ],
)
def test_engine_wide_readers(monkeypatch, overrides: dict, supported: bool) -> None:
    monkeypatch.setattr(scheduler_module, "entry_points", lambda group: [])
    config = _engine_config(**overrides)
    assert scheduler_module.new_req_token_trim_supported(config) is supported


def test_engine_wide_readers_plugins_and_opt_out(monkeypatch) -> None:
    monkeypatch.setattr(scheduler_module, "entry_points", lambda group: ["plugin"])
    assert not scheduler_module.new_req_token_trim_supported(_engine_config())
    monkeypatch.setattr(scheduler_module, "entry_points", lambda group: [])
    monkeypatch.setenv("VLLM_TRIM_NEW_REQUEST_TOKEN_IDS", "0")
    assert not scheduler_module.new_req_token_trim_supported(_engine_config())


def _model_runner_output(req_ids: list[str], sampled: list[list[int]]):
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=sampled,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


@pytest.mark.parametrize("use_v2_model_runner", [True, False])
def test_scheduler_ships_the_tail_after_a_prefix_hit(use_v2_model_runner) -> None:
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        use_v2_model_runner=use_v2_model_runner,
        block_size=16,
    )
    assert scheduler.trim_new_req_token_ids
    assert scheduler.use_v2_model_runner is use_v2_model_runner
    req_a, req_b = create_requests(
        num_requests=2,
        num_tokens=64,
        same_prompt=True,
        ignore_eos=True,
        max_tokens=4,
        block_size=16,
    )
    scheduler.add_request(req_a)
    out = scheduler.schedule()
    (new_a,) = out.scheduled_new_reqs
    assert new_a.num_computed_tokens == 0 and new_a.prefill_token_start == 0
    scheduler.update_from_output(out, _model_runner_output([req_a.request_id], [[7]]))

    scheduler.add_request(req_b)
    out = scheduler.schedule()
    (new_b,) = out.scheduled_new_reqs
    hit = new_b.num_computed_tokens
    assert hit > 0
    if use_v2_model_runner:
        assert new_b.prefill_token_start == hit
        assert new_b.prefill_token_ids == req_b._all_token_ids[hit:]
        assert new_b.prefill_len == 64
        assert new_b.prompt_token_ids is None and new_b.prompt_len == 64
        assert scheduler.num_new_reqs_trimmed == 1
    else:
        # The V1 model runner reads the prompt list: nothing is trimmed.
        assert new_b.prefill_token_start == 0
        assert new_b.prompt_token_ids == req_b.prompt_token_ids
        assert scheduler.num_new_reqs_trimmed == 0
    # The engine keeps the request's full ids (prefix hashing, detokenization).
    assert len(req_b._all_token_ids) == 64 and len(req_b.prompt_token_ids) == 64


def test_scheduler_opt_out(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_TRIM_NEW_REQUEST_TOKEN_IDS", "0")
    scheduler = create_scheduler(enable_prefix_caching=True, use_v2_model_runner=True)
    assert not scheduler.trim_new_req_token_ids
    (req,) = create_requests(num_requests=1, num_tokens=32)
    scheduler.add_request(req)
    (new_req,) = scheduler.schedule().scheduled_new_reqs
    assert new_req.prompt_token_ids == req.prompt_token_ids


def test_trim_counter_log_points() -> None:
    log_bucket = scheduler_module._log_bucket
    points = [n for n in range(1, 4100) if log_bucket(n - 1) != log_bucket(n)]
    assert points[:11] == [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    assert points[11:] == [2048, 3072, 4096]
