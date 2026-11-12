# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A request whose remote KV finished loading is admitted even while a long
running prompt (e.g. a recompute) would take the whole token budget."""

import pytest

from vllm.v1.request import RequestStatus

from .utils import (
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)

pytestmark = pytest.mark.cpu_test

BUDGET = 64


def _output(scheduler, finished_recving: set[str] | None = None):
    # A token for every running request (a partial prompt's is discarded).
    return create_model_runner_output(
        list(scheduler.running), finished_recving=finished_recving
    )


def test_loaded_request_not_starved_by_running_prompt_chunks():
    vllm_config = create_vllm_config(max_num_batched_tokens=BUDGET)
    scheduler = create_scheduler(vllm_config)
    block_size = vllm_config.cache_config.block_size

    loaded = create_request(
        request_id=1,
        block_size=block_size,
        num_tokens=3 * block_size,
        do_remote_prefill=True,
    )
    # A long local prompt: its chunks alone fill the budget for many steps.
    long_prompt = create_request(
        request_id=2, block_size=block_size, num_tokens=20 * BUDGET
    )
    scheduler.add_request(loaded)
    scheduler.add_request(long_prompt)

    # Step 1: the load starts, the long prompt takes the whole budget.
    out = scheduler.schedule()
    assert loaded.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert out.num_scheduled_tokens[long_prompt.request_id] == BUDGET
    scheduler.update_from_output(out, _output(scheduler))

    # Step 2: the load finishes.
    out = scheduler.schedule()
    scheduler.update_from_output(out, _output(scheduler, {loaded.request_id}))
    assert loaded.request_id in scheduler.finished_recving_kv_req_ids

    # Step 3: the loaded request is admitted in the same step; the prompt
    # chunk leaves it the tokens it needs.
    out = scheduler.schedule()
    assert loaded.status == RequestStatus.RUNNING
    need = out.num_scheduled_tokens[loaded.request_id]
    assert need >= 1
    assert out.num_scheduled_tokens[long_prompt.request_id] == BUDGET - need
    assert out.total_num_scheduled_tokens == BUDGET


def test_no_reserve_without_ready_loaded_requests():
    vllm_config = create_vllm_config(max_num_batched_tokens=BUDGET)
    scheduler = create_scheduler(vllm_config)
    block_size = vllm_config.cache_config.block_size
    loaded = create_request(
        request_id=1,
        block_size=block_size,
        num_tokens=3 * block_size,
        do_remote_prefill=True,
    )
    long_prompt = create_request(
        request_id=2, block_size=block_size, num_tokens=20 * BUDGET
    )
    scheduler.add_request(loaded)
    scheduler.add_request(long_prompt)
    out = scheduler.schedule()
    scheduler.update_from_output(out, _output(scheduler))
    # The load is still in flight: nothing is held back from the prompt.
    assert scheduler._loaded_kv_admission_reserve(BUDGET) == 0
    out = scheduler.schedule()
    assert out.num_scheduled_tokens[long_prompt.request_id] == BUDGET
