# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA-DP admission at K3.1 token scale: block size 4096, Mamba block-aligned
splits, DSpark spec decode, per-card DSA regions; P/D decode engines and local
prefill. Each test fails on an admission deadlock: requests waiting, nothing
running or loading, every request skipped each step."""

import random
from dataclasses import replace

import pytest
import torch

from tests.v1.core.test_dsa_dp_regions import _turn, prefix_owner  # noqa: F401
from tests.v1.core.utils import create_scheduler
from tests.v1.kv_connector.unit.utils import MockKVConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import SupportsHMA
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import RequestStatus

TP = 4
BS = 4096
SHARED = 2561
REGION = 3741
POOL = list(range(1000, 61000))


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    init_none_hash(sha256)


def _sched(num_spec: int = 7, whole_blocks: bool = False) -> Scheduler:
    cfg = KVCacheConfig(
        num_blocks=SHARED,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["dsa.attn"],
                MLAAttentionSpec(
                    block_size=BS, num_kv_heads=1, head_size=576, dtype=torch.float32
                ),
            ),
            KVCacheGroupSpec(
                ["swa.attn"],
                SlidingWindowSpec(
                    block_size=BS,
                    num_kv_heads=1,
                    head_size=64,
                    dtype=torch.float32,
                    sliding_window=BS,
                ),
            ),
        ],
    )
    cfg = replace(
        cfg,
        dsa_dp_group_ids=(0,),
        dsa_dp_shared_blocks=SHARED,
        dsa_dp_region_blocks=REGION,
        dsa_dp_num_regions=TP,
    )
    base = create_scheduler(
        enable_prefix_caching=True,
        block_size=BS,
        max_num_seqs=128,
        max_num_batched_tokens=8192,
        max_model_len=1 << 20,
        use_kv_connector=MockKVConfig(matched_tokens=0, is_async=True),
        num_speculative_tokens=num_spec,
    )
    base.vllm_config.parallel_config.tensor_parallel_size = TP
    base.vllm_config.cache_config.num_gpu_blocks = SHARED
    sched = Scheduler(
        vllm_config=base.vllm_config,
        kv_cache_config=cfg,
        block_size=BS,
        log_stats=True,
        structured_output_manager=base.structured_output_manager,
    )
    # K3.1 is hybrid (KDA): prompt chunks end on Mamba blocks.
    sched.need_mamba_block_aligned_split = True

    class _PDMock(type(sched.connector), SupportsHMA):
        # The prompt but its last token, as NIXL transfers it, or only its
        # whole blocks, as a block-granular store hit.
        def get_num_new_matched_tokens(self, request, num_computed_tokens):
            if request.num_computed_tokens > 0:
                return 0, False
            n = request.num_tokens - 1 - num_computed_tokens
            if whole_blocks:
                n = n // BS * BS
            return max(n, 0), n > 0

        def request_finished_all_groups(self, request, block_ids):
            return False, None

    sched.connector.__class__ = _PDMock
    return sched


def _req(rid: str, length: int, owner: int, off: int, max_tokens: int = 50):
    r = _turn(rid, [POOL[(off + i) % len(POOL)] for i in range(length)])
    r.sampling_params.max_tokens = max_tokens
    r.max_tokens = max_tokens
    r.kv_transfer_params = {
        "do_remote_prefill": True,
        "remote_dsa_owner": owner,
        "tp_size": TP,
    }
    return r


def _feed(sched: Scheduler, out, finished_recving: set[str]) -> None:
    ids = list(out.num_scheduled_tokens)
    mro = ModelRunnerOutput(
        req_ids=ids,
        req_id_to_index={r: i for i, r in enumerate(ids)},
        sampled_token_ids=[
            [] if sched.requests[r].is_prefill_chunk else [7] for r in ids
        ],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    mro.kv_connector_output = KVConnectorOutput(finished_recving=finished_recving)
    sched.update_from_output(out, mro)


def _step(sched: Scheduler) -> None:
    # Loads started by this schedule() finish at once; a reported load is
    # promoted by the next schedule().
    out = sched.schedule()
    _feed(sched, out, _loading(sched))


def _loading(sched: Scheduler) -> set[str]:
    return {
        r.request_id
        for r in sched.requests.values()
        if r.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    }


@pytest.mark.usefixtures("prefix_owner")
@pytest.mark.parametrize("k", [1, 3, 5, 7])
def test_last_prompt_token_near_block_end(k):
    """The loaded KV ends k tokens before a block end; the owner's share of the
    remaining token must not be cut back to that block end (below the k + 1
    spec decode rows), which stopped admission for every request."""
    sched = _sched(num_spec=k)
    a = _req("a", 161 * BS - k + 1, 0, 0)
    sched.add_request(a)
    _step(sched)
    others = [
        _req(f"w{i}", 100_000 + 37 * i, i % TP, 1000 + 977 * i) for i in range(60)
    ]
    for r in others:
        sched.add_request(r)
    for _ in range(30):
        _step(sched)
    assert a.num_computed_tokens >= a.num_prompt_tokens, (
        a.status,
        a.num_computed_tokens,
    )
    started = sum(1 for r in others if r.num_computed_tokens > 0)
    assert started == len(others), f"only {started}/{len(others)} loads started"


@pytest.mark.usefixtures("prefix_owner")
def test_long_context_loads_skewed_owners():
    """600-740K-token block-granular loads (a remaining tail of up to 4095
    tokens each) with owners skewed 44/19/15/22, closed-loop concurrency 128:
    loads must not over-commit an owner's region, and loads still queued must
    not hold the owners' cost cap below every loaded request's tail."""
    rng = random.Random(0)
    sched = _sched(whole_blocks=True)
    probs = [0.44, 0.19, 0.15, 0.22]
    loading: dict[str, int] = {}
    reported: set[str] = set()
    n = idle = finished = 0
    for step in range(200):
        while sum(1 for r in sched.requests.values() if not r.is_finished()) < 128:
            owner = rng.choices(range(TP), probs)[0]
            length = rng.randint(600_000, 740_000)
            sched.add_request(
                _req(f"r{n}", length, owner, 7919 * n, rng.randint(20, 100))
            )
            n += 1
        out = sched.schedule()
        waiting = _loading(sched)
        reported &= waiting
        for rid in waiting - loading.keys() - reported:
            loading[rid] = rng.randint(0, 6)
        done = {rid for rid, left in loading.items() if left <= 0}
        reported |= done
        loading = {rid: left - 1 for rid, left in loading.items() if left > 0}
        _feed(sched, out, done)
        finished += len(out.finished_req_ids)
        stuck = not out.num_scheduled_tokens and not sched.running and not loading
        idle = idle + 1 if stuck else 0
        assert idle < 5, f"admission deadlock at step {step} ({finished} finished)"
    assert finished > 0


@pytest.mark.usefixtures("prefix_owner")
def test_equal_head_chunks_over_the_token_budget():
    """Local prefill with 4096-token Mamba-aligned chunks and an 8192-token step:
    one fresh request per owner makes every owner's head chunk the same,
    indivisible cost and all four together exceed the step. The owner shares
    must not leave the step empty."""
    sched = _sched()
    sched.connector = None
    for i in range(4):
        r = _turn(f"r{i}", [POOL[(7919 * i + j) % len(POOL)] for j in range(5 * BS)])
        r.sampling_params.max_tokens = 4
        r.max_tokens = 4
        sched.add_request(r)
    scheduled = 0
    for _ in range(10):
        out = sched.schedule()
        scheduled += out.total_num_scheduled_tokens
        _feed(sched, out, set())
    assert scheduled > 0
