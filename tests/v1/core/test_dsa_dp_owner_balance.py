# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA-DP per-step owner balance: every owner card's modeled DSA cost of a step
stays close to the others', with long-context requests on one owner."""

import numpy as np
import pytest

from vllm.models.kimi_k3.nvidia.ops import dsa_dp
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.outputs import ModelRunnerOutput

from .test_prefix_caching import make_request
from .utils import create_scheduler

pytestmark = pytest.mark.cpu_test

TP = 4
CHUNK = 64
STEP = 16 * CHUNK


@pytest.fixture(autouse=True)
def _dsa_dp(monkeypatch):
    init_none_hash(sha256)
    monkeypatch.setattr(dsa_dp, "ENABLED", True)
    monkeypatch.setattr(dsa_dp, "CHUNK", CHUNK)
    # A long prompt's late chunks cost ~3x a fresh one, as a 4096-token chunk at
    # 400k context does at K3.1 (C0 = 180k there).
    monkeypatch.setattr(dsa_dp, "COST_C0", 1024.0)


def _scheduler():
    sched = create_scheduler(
        max_num_seqs=64,
        max_num_batched_tokens=STEP,
        long_prefill_token_threshold=CHUNK,
        num_blocks=4096,
        block_size=16,
    )
    sched.vllm_config.parallel_config.tensor_parallel_size = TP
    # Owners are set by the test.
    sched._assign_dsa_owner = lambda request: None
    return sched


def _add(sched, rid: str, owner: int, num_tokens: int):
    req = make_request(
        rid,
        [hash(rid) % 1000 + i for i in range(num_tokens)],
        16,
        sha256,
        sampling_params=SamplingParams(max_tokens=1),
    )
    req.dsa_owner = owner
    sched.add_request(req)
    return req


def _step(sched, dp: bool):
    if dp:
        sched.set_dp_prefill_exchange(lambda own, demand: own)
    ctx = {r.request_id: r.num_computed_tokens for r in sched.running}
    out = sched.schedule()
    cost = np.zeros(TP)
    for rid, n in out.num_scheduled_tokens.items():
        req = sched.requests[rid]
        c = ctx.get(rid, req.num_computed_tokens - n)
        cost[req.dsa_owner] += dsa_dp.chunk_cost(n, c)
    ids = list(out.num_scheduled_tokens)
    done = [
        sched.requests[r].num_computed_tokens >= sched.requests[r].num_prompt_tokens
        for r in ids
    ]
    sched.update_from_output(
        out,
        ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={r: i for i, r in enumerate(ids)},
            sampled_token_ids=[[1] if d else [] for d in done],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    return out, cost


@pytest.mark.parametrize("dp", [False, True])
@pytest.mark.parametrize("pull_headroom", [False, True])
def test_owner_cost_balanced_per_step(dp, pull_headroom, monkeypatch):
    """Owner 0 serves long prompts (its chunks get expensive as context grows),
    owners 1-3 short ones. Running chunks must count against their owner's
    share whether or not the DP pull-up books them (it is off when KV headroom
    is short), and no running prompt may wait two steps in a row."""
    sched = _scheduler()
    monkeypatch.setattr(sched, "_dp_pull_kv_headroom", lambda: pull_headroom)
    for i in range(6):
        _add(sched, f"long{i}", 0, 40 * CHUNK)
    for o in range(1, TP):
        for i in range(60):
            _add(sched, f"short{o}_{i}", o, 3 * CHUNK)

    ratios = []
    waited: dict[str, int] = {}
    max_wait = 0
    for step in range(24):
        out, cost = _step(sched, dp)
        if step >= 4 and cost.min() > 0:
            ratios.append(cost.max() / cost.mean())
        for r in sched.running:
            rid = r.request_id
            if r.num_computed_tokens < r.num_prompt_tokens:
                waited[rid] = (
                    0 if rid in out.num_scheduled_tokens else (waited.get(rid, 0) + 1)
                )
                max_wait = max(max_wait, waited[rid])
    assert len(ratios) >= 10
    assert float(np.mean(ratios)) <= 1.15 and max_wait <= 1, (
        float(np.mean(ratios)),
        max_wait,
    )
