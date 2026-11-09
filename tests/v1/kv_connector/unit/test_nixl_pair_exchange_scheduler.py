# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCP pair mode, scheduler side (design A4).

A decoder builds each pull's exchange plan when it allocates the pull's blocks,
and emits it exactly once, with the connector metadata of the first step that
can promote the request, so every rank exchanges the blocks before a forward
reads them. These tests run the real core scheduler and NIXL connector of a
TP16/DCP16 decoder pulling from TP8/DCP8 prefillers.
"""

import math
import random
from collections import deque
from typing import Literal

import pytest
import torch

from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.dcp_pair_exchange import (
    NixlPairModeError,
    PairUnit,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import (
    NixlPullConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    NixlPullConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_scheduler import (
    NixlPushConnectorScheduler,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from .utils import create_model_runner_output, create_request, create_vllm_config

pytestmark = pytest.mark.cpu_test

B = 16  # Tokens per block on each rank.
P_DCP, D_DCP = 8, 16
P_BLOCK, D_BLOCK = B * P_DCP, B * D_DCP  # Tokens per P and per D block.


@pytest.fixture(autouse=True)
def _one_torch_thread():
    num_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(num_threads)


def _make_scheduler(
    *,
    opt_in: object = True,
    kda_group: int | None = None,
    async_scheduling: bool = False,
    max_num_seqs: int = 16,
    kv_load_failure_policy: Literal["recompute", "fail"] = "fail",
    num_blocks: int = 128,
) -> Scheduler:
    """The scheduler of a TP16/DCP16 decoder, sized as the engine sizes it.

    Args:
        opt_in: The ``dcp_pair_exchange`` extra config value; None omits it.
        kda_group: Where to insert a KDA-like state group among the groups;
            None for an attention-only model.
        async_scheduling: Use the async scheduler (steps overlap).
        max_num_seqs: The running request limit.
        kv_load_failure_policy: The decoder's KV load failure policy.
        num_blocks: The KV cache pool size.

    """
    vllm_config = create_vllm_config(
        block_size=B,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=4 * D_BLOCK,
        kv_connector_extra_config=(
            None if opt_in is None else {"dcp_pair_exchange": opt_in}
        ),
        kv_load_failure_policy=kv_load_failure_policy,
    )
    vllm_config.parallel_config.decode_context_parallel_size = D_DCP
    vllm_config.scheduler_config.async_scheduling = async_scheduling
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    groups = [
        KVCacheGroupSpec(
            ["attn"],
            FullAttentionSpec(
                block_size=B, num_kv_heads=1, head_size=16, dtype=torch.float16
            ),
        )
    ]
    if kda_group is not None:
        groups.insert(
            kda_group,
            KVCacheGroupSpec(
                ["kda"],
                MambaSpec(
                    block_size=D_BLOCK,
                    shapes=((16,),),
                    dtypes=(torch.float16,),
                    mamba_cache_mode="align",
                ),
            ),
        )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups
    )
    block_size, hash_block_size = resolve_kv_cache_block_sizes(
        kv_cache_config, vllm_config
    )
    scheduler_cls = AsyncScheduler if async_scheduling else Scheduler
    return scheduler_cls(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=block_size,
        hash_block_size=hash_block_size,
    )


def _pull_request(
    request_id: int,
    num_tokens: int,
    *,
    kda_group: int | None = None,
    remote_dcp_size: int = P_DCP,
    remote_num_tokens: int | None = None,
    num_remote_blocks: int | None = None,
    **kwargs,
) -> Request:
    """A decoder request whose prompt a prefiller computed, minus the backoff
    token the decoder recomputes when the model has state groups."""
    request = create_request(
        request_id=request_id,
        num_tokens=num_tokens,
        do_remote_prefill=True,
        block_size=D_BLOCK,
        **kwargs,
    )
    num_loaded = num_tokens if kda_group is None else num_tokens - 1
    if num_remote_blocks is None:
        num_remote_blocks = cdiv(num_loaded, B * remote_dcp_size)
    first = 1000 * request_id
    remote_block_ids = [list(range(first, first + num_remote_blocks))]
    if kda_group is not None:
        remote_block_ids.insert(kda_group, [first - 1])
    assert request.kv_transfer_params is not None
    request.kv_transfer_params.update(
        remote_block_ids=remote_block_ids,
        remote_num_tokens=num_loaded
        if remote_num_tokens is None
        else remote_num_tokens,
        tp_size=remote_dcp_size,
        dcp_size=remote_dcp_size,
    )
    return request


def _pulled_units(
    output: SchedulerOutput, req_id: str, group: int = 0
) -> list[PairUnit]:
    """The units the workers' own block matching implies for the pull that
    ``output`` posts: the D blocks the parity-0 ranks pull, partnered when the
    parity-1 ranks pull them too (design A2)."""
    recv = output.kv_connector_metadata.reqs_to_recv[req_id]
    assert recv.remote is not None
    worker = object.__new__(NixlPullConnectorWorker)
    pulled = [
        worker._apply_dcp_prefix_caching(
            list(recv.local_block_ids[group]),
            list(recv.remote.block_ids[group]),
            remote_rank=rank % P_DCP,
            local_dcp_size=D_DCP,
            local_dcp_rank=rank,
            remote_dcp_size=P_DCP,
            local_num_computed_blocks=recv.local_num_computed_blocks[group],
        )[0]
        for rank in range(D_DCP)
    ]
    even, odd = pulled[0], pulled[P_DCP]
    assert pulled == [even] * P_DCP + [odd] * P_DCP
    # The partner pulls the same blocks, or all but the last.
    assert odd == even[: len(odd)] and len(odd) >= len(even) - 1
    return [PairUnit(group, b, b in odd) for b in even]


class _Engine:
    """Steps a scheduler as EngineCore does. Records the plans the connector
    metadata of each schedule() call emits, and the call promoting each request
    out of WAITING_FOR_REMOTE_KVS."""

    def __init__(self, scheduler: Scheduler):
        assert scheduler.connector is not None
        self.scheduler = scheduler
        self.nixl = scheduler.connector.connector_scheduler
        assert isinstance(self.nixl, NixlPullConnectorScheduler)
        self.requests: dict[str, Request] = {}
        self.emitted: list[dict[str, list[PairUnit]]] = []
        self.promoted: dict[str, int] = {}
        promote = scheduler._update_waiting_for_remote_kv

        def record_promotion(request: Request) -> None:
            self.promoted.setdefault(request.request_id, len(self.emitted))
            promote(request)

        scheduler._update_waiting_for_remote_kv = (  # type: ignore[method-assign]
            record_promotion
        )

    def add(self, *requests: Request) -> None:
        for request in requests:
            self.requests[request.request_id] = request
            self.scheduler.add_request(request)

    def abort(self, req_id: str) -> None:
        self.scheduler.finish_requests(req_id, RequestStatus.FINISHED_ABORTED)

    def schedule(self) -> SchedulerOutput:
        output = self.scheduler.schedule()
        meta = output.kv_connector_metadata
        assert isinstance(meta, NixlConnectorMetadata)
        self.emitted.append(meta.reqs_to_pair_exchange)
        return output

    def update(
        self,
        output: SchedulerOutput,
        finished: set[str] | None = None,
        failed: set[str] | None = None,
        eos: bool = False,
    ) -> None:
        """Process ``output``, reporting the pulls that ended on every rank.
        Failures come with the completion, as the output aggregator sends
        them."""
        runner_output = create_model_runner_output(
            [self.requests[r] for r in output.num_scheduled_tokens],
            finished_recving=(finished or set()) | (failed or set()) or None,
            failed_recving=failed,
            use_eos=eos,
        )
        self.scheduler.update_from_output(output, runner_output)

    def step(self, **kwargs) -> SchedulerOutput:
        output = self.schedule()
        self.update(output, **kwargs)
        return output

    def emissions(self, req_id: str) -> list[int]:
        return [i for i, plans in enumerate(self.emitted) if req_id in plans]

    def assert_idle(self) -> None:
        assert not self.nixl._pair_plans
        assert not self.nixl._pair_ready


@pytest.mark.parametrize("num_p_blocks", [5, 6], ids=["odd_P_blocks", "even_P_blocks"])
@pytest.mark.parametrize(
    "kda_group",
    [None, 1, 0],
    ids=["attention", "attention+kda", "kda+attention"],
)
def test_plan_is_emitted_once_with_the_promoting_step(
    kda_group: int | None, num_p_blocks: int
):
    """The plan of the blocks the workers pull is built at allocation, held
    while the pull runs, and emitted by the step that promotes the request."""
    engine = _Engine(_make_scheduler(kda_group=kda_group))
    attn = 1 if kda_group == 0 else 0
    num_tokens = (num_p_blocks - 1) * P_BLOCK + 5 + (kda_group is not None)
    request = _pull_request(1, num_tokens, kda_group=kda_group)
    req_id = request.request_id
    engine.add(request)

    output = engine.step()
    units = _pulled_units(output, req_id, group=attn)
    recv = output.kv_connector_metadata.reqs_to_recv[req_id]
    # One unit per D block; an odd P block count leaves the last one alone.
    assert [u.block_id for u in units] == recv.local_block_ids[attn]
    assert [u.has_partner for u in units] == [True, True, num_p_blocks % 2 == 0]
    assert engine.nixl._pair_plans == {req_id: units}

    engine.step()
    assert engine.nixl._pair_plans == {req_id: units}
    engine.step(finished={req_id})
    assert engine.nixl._pair_ready == {req_id: units}
    assert not engine.nixl._pair_plans

    output = engine.step()
    assert output.kv_connector_metadata.reqs_to_pair_exchange == {req_id: units}
    assert req_id in output.num_scheduled_tokens
    engine.step(eos=True)
    engine.step()
    assert engine.emissions(req_id) == [3]
    assert engine.promoted[req_id] == 3
    engine.assert_idle()


@pytest.mark.parametrize("failure", [False, True], ids=["ok", "one_rank_failed"])
def test_plan_waits_for_the_pull_to_end_on_every_rank(failure: bool):
    """Ranks end a pull at different steps; one may retry its reads after a
    remote disconnect, reporting nothing until the retry ends. The workers'
    output aggregator reports the pull, with any rank's failure, once every
    rank ended it: the plan is kept until then, then emitted or dropped."""
    engine = _Engine(_make_scheduler())
    request = _pull_request(1, 3 * P_BLOCK)
    req_id = request.request_id
    engine.add(request)
    aggregator = KVOutputAggregator(expected_finished_count=D_DCP)
    retrying_rank, failing_rank = 3, 5
    ends = {rank: 1 for rank in range(D_DCP)} | {retrying_rank: 3}

    def rank_output(output: SchedulerOutput, rank: int, step: int) -> ModelRunnerOutput:
        ended = ends[rank] == step
        return create_model_runner_output(
            [engine.requests[r] for r in output.num_scheduled_tokens],
            finished_recving={req_id} if ended else None,
            failed_recving={req_id}
            if ended and failure and rank == failing_rank
            else None,
        )

    for step in range(5):
        output = engine.schedule()
        if step == 0:
            units = engine.nixl._pair_plans[req_id]
        outputs = [rank_output(output, rank, step) for rank in range(D_DCP)]
        engine.scheduler.update_from_output(output, aggregator.aggregate(outputs))
        if step < 3:
            assert engine.nixl._pair_plans == {req_id: units}
    assert engine.emissions(req_id) == ([] if failure else [4])
    engine.assert_idle()


def test_async_scheduling_emits_with_the_first_step_that_can_promote():
    """Steps run one ahead of their outputs. The output ending the pull is
    processed after the next step was scheduled without the plan; the step
    after emits it and promotes the request."""
    scheduler = _make_scheduler(async_scheduling=True)
    assert scheduler.defer_block_free  # Overlapping batches, as on GPUs.
    engine = _Engine(scheduler)
    request = _pull_request(1, 3 * P_BLOCK + 7)
    req_id = request.request_id
    engine.add(request)

    in_flight = deque([engine.schedule(), engine.schedule()])
    engine.update(in_flight.popleft())
    in_flight.append(engine.schedule())
    engine.update(in_flight.popleft(), finished={req_id})
    for _ in range(3):
        in_flight.append(engine.schedule())
        engine.update(in_flight.popleft())
    assert engine.emissions(req_id) == [3]
    assert engine.promoted[req_id] == 3


def test_zero_token_step_emits_the_plan_once():
    """A step that schedules nothing still emits the plan. The request is
    promoted by a later step, which must not emit it again."""
    engine = _Engine(_make_scheduler())
    request = _pull_request(1, 2 * P_BLOCK + 1)
    req_id = request.request_id
    engine.add(request)
    engine.step()
    engine.step(finished={req_id})

    engine.scheduler.set_pause_state(PauseState.PAUSED_ALL)
    output = engine.step()
    assert output.total_num_scheduled_tokens == 0
    assert set(output.kv_connector_metadata.reqs_to_pair_exchange) == {req_id}
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS

    engine.scheduler.set_pause_state(PauseState.UNPAUSED)
    output = engine.step()
    assert req_id in output.num_scheduled_tokens
    assert engine.emissions(req_id) == [2]
    assert engine.promoted[req_id] == 3


def test_max_num_seqs_defers_the_promotion_not_the_emission():
    """With every running slot taken, the plan is still emitted right after
    the pull; the promotion that follows later emits nothing."""
    engine = _Engine(_make_scheduler(max_num_seqs=1))
    pulled = _pull_request(1, 3 * P_BLOCK)
    engine.add(pulled)
    engine.step()
    engine.add(
        create_request(request_id=2, num_tokens=8, max_tokens=3, block_size=D_BLOCK)
    )
    engine.step(finished={pulled.request_id})

    engine.step()
    assert engine.emissions(pulled.request_id) == [2]
    assert pulled.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    while pulled.request_id not in engine.promoted:
        assert len(engine.emitted) < 10
        engine.step()
    assert engine.promoted[pulled.request_id] > 2
    engine.step()
    assert engine.emissions(pulled.request_id) == [2]


@pytest.mark.parametrize("policy", ["recompute", "fail"])
def test_failed_pull_emits_no_plan(policy: Literal["recompute", "fail"]):
    """A failed pull drops its plan: its request recomputes from token 0 or
    fails, and no exchange may touch its blocks."""
    engine = _Engine(_make_scheduler(kv_load_failure_policy=policy))
    request = _pull_request(1, 3 * P_BLOCK)
    engine.add(request)
    engine.step()
    engine.step(failed={request.request_id})
    engine.assert_idle()
    engine.step()
    engine.step()
    assert not any(engine.emitted)
    if policy == "fail":
        assert request.status == RequestStatus.FINISHED_ERROR
    else:
        assert request.status == RequestStatus.RUNNING
        assert request.num_computed_tokens >= request.num_prompt_tokens
    engine.assert_idle()


@pytest.mark.parametrize("when", ["pulling", "before_the_pull_ends", "pulled"])
def test_abort_before_emission_emits_no_plan(when: str):
    """An aborted request's blocks are freed when its pull ends, or at once
    once it has ended: its plan must never be emitted."""
    engine = _Engine(_make_scheduler())
    request = _pull_request(1, 3 * P_BLOCK)
    req_id = request.request_id
    engine.add(request)
    engine.step()
    if when == "pulling":
        engine.abort(req_id)
        engine.assert_idle()
        engine.step(finished={req_id})
    elif when == "before_the_pull_ends":
        # EngineCore processes aborts before the step's output.
        output = engine.schedule()
        engine.abort(req_id)
        engine.update(output, finished={req_id})
    else:
        engine.step(finished={req_id})
        engine.abort(req_id)
    engine.assert_idle()
    engine.step()
    assert not any(engine.emitted)
    assert req_id not in engine.scheduler.requests


@pytest.mark.parametrize("promoted", [True, False], ids=["promoted", "zero_token"])
def test_abort_after_emission_keeps_the_emitted_units(promoted: bool):
    """The exchange of an emitted plan is queued on the workers even if the
    request is aborted now and its blocks freed (the worker's reuse guard
    covers that); the emitted units stay as sent and are not emitted again."""
    engine = _Engine(_make_scheduler())
    request = _pull_request(1, 3 * P_BLOCK)
    req_id = request.request_id
    engine.add(request)
    engine.step()
    engine.step(finished={req_id})
    units = engine.nixl._pair_ready[req_id]

    if not promoted:
        engine.scheduler.set_pause_state(PauseState.PAUSED_ALL)
    output = engine.schedule()
    assert (req_id in output.num_scheduled_tokens) is promoted
    engine.abort(req_id)
    engine.update(output)
    engine.scheduler.set_pause_state(PauseState.UNPAUSED)
    engine.step()

    emitted = output.kv_connector_metadata.reqs_to_pair_exchange
    assert emitted == {req_id: units}
    assert emitted[req_id] is units
    assert engine.emissions(req_id) == [2]
    assert req_id not in engine.scheduler.requests
    engine.assert_idle()


@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("seed", range(4))
def test_random_lifecycles(seed: int, async_scheduling: bool):
    """A4 invariants over random arrivals, pull times, failures, aborts and
    paused (0-token) steps, with one running slot to defer promotions. A plan
    is emitted iff its pull succeeded and its request was alive at the next
    schedule(); it is emitted by that call only, with the units of the blocks
    the workers pulled; and no request is promoted before its plan is
    emitted."""
    rng = random.Random(seed)
    engine = _Engine(
        _make_scheduler(
            async_scheduling=async_scheduling,
            max_num_seqs=1,
            kv_load_failure_policy="recompute",
        )
    )
    arrivals = [
        _pull_request(i, rng.randint(1, 5 * P_BLOCK), max_tokens=rng.randint(1, 4))
        for i in range(1, 11)
    ]
    plans: dict[str, list[PairUnit]] = {}
    ends: dict[str, int] = {}  # The output whose processing reports the end.
    failed: set[str] = set()
    ready_at: dict[str, int] = {}  # The schedule() call that must emit.
    aborted_at: dict[str, int] = {}
    in_flight: deque[tuple[int, SchedulerOutput]] = deque()
    depth = 2 if async_scheduling else 1  # Steps scheduled per output.
    step = 0
    paused = False
    while arrivals or engine.scheduler.requests or in_flight:
        assert step < 400, "the requests did not drain"
        if arrivals and rng.random() < 0.3:
            engine.add(arrivals.pop())
        live = [
            r.request_id
            for r in engine.scheduler.requests.values()
            if not r.is_finished()
        ]
        if live and rng.random() < 0.08:
            victim = rng.choice(live)
            aborted_at[victim] = len(engine.emitted)
            engine.abort(victim)
        if paused != (pause := rng.random() < 0.1):
            paused = pause
            engine.scheduler.set_pause_state(
                PauseState.PAUSED_ALL if pause else PauseState.UNPAUSED
            )
        output = engine.schedule()
        for req_id, recv in output.kv_connector_metadata.reqs_to_recv.items():
            if recv.awaiting_kvs:
                plans[req_id] = _pulled_units(output, req_id)
                assert engine.nixl._pair_plans[req_id] == plans[req_id]
                ends[req_id] = step + rng.randint(0, 3)
                if rng.random() < 0.15:
                    failed.add(req_id)
        in_flight.append((step, output))
        drain = not arrivals and not engine.scheduler.requests
        while in_flight and (len(in_flight) >= depth or drain):
            index, done = in_flight.popleft()
            ended = {r for r, end in ends.items() if end == index}
            engine.update(done, finished=ended - failed, failed=ended & failed)
            ready_at.update((r, len(engine.emitted)) for r in ended - failed)
        step += 1

    assert any(engine.emitted) and failed and aborted_at
    assert set().union(*engine.emitted) <= set(plans)
    for req_id, units in plans.items():
        if req_id in ready_at and aborted_at.get(req_id, math.inf) > ready_at[req_id]:
            assert engine.emissions(req_id) == [ready_at[req_id]], req_id
            assert engine.emitted[ready_at[req_id]][req_id] == units
        else:
            assert engine.emissions(req_id) == [], req_id
    for req_id, promoted_at in engine.promoted.items():
        if req_id not in failed:
            assert engine.emissions(req_id)[0] <= promoted_at, req_id
    engine.assert_idle()


def test_plan_covers_only_the_blocks_past_the_local_prefix_hit():
    """Two turns: the second hits the m = 2 D blocks the first one cached, so
    its plan lists only its uncached blocks, the first of them fed by P blocks
    2m and 2m + 1."""
    engine = _Engine(_make_scheduler())
    m = 2
    turn1 = _pull_request(1, m * D_BLOCK + 5, common_prefix_len=m * D_BLOCK)
    engine.add(turn1)
    engine.step()
    engine.step(finished={turn1.request_id})
    engine.step()  # Promotes turn 1, caching its blocks.

    turn2 = _pull_request(2, 4 * D_BLOCK + 3, common_prefix_len=m * D_BLOCK)
    engine.add(turn2)
    output = engine.step()
    recv = output.kv_connector_metadata.reqs_to_recv[turn2.request_id]
    assert recv.local_num_computed_blocks == (m,)
    blocks = engine.scheduler.kv_cache_manager.get_block_ids(turn2.request_id)[0]
    turn1_blocks = engine.scheduler.kv_cache_manager.get_block_ids(turn1.request_id)
    assert blocks[:m] == turn1_blocks[0][:m]
    units = engine.nixl._pair_plans[turn2.request_id]
    assert units == _pulled_units(output, turn2.request_id)
    assert [u.block_id for u in units] == blocks[m:]
    # P blocks 4 to 8 (nine in all): pairs (4, 5) and (6, 7), then 8 alone.
    assert [u.has_partner for u in units] == [True, True, False]


def test_pull_that_loads_nothing_plans_nothing():
    """A request that does not read external KV only releases the prefiller's
    blocks: nothing is pulled, so nothing is planned or exchanged."""
    engine = _Engine(_make_scheduler())
    request = _pull_request(1, 3 * P_BLOCK)
    assert request.sampling_params is not None
    request.sampling_params.skip_reading_prefix_cache = True
    request.prefix_cache_hit_limit = request.get_prefix_cache_hit_limit()
    engine.add(request)

    output = engine.step()
    recv = output.kv_connector_metadata.reqs_to_recv[request.request_id]
    assert not recv.awaiting_kvs and not recv.local_block_ids
    assert request.status == RequestStatus.RUNNING
    engine.step(eos=True)
    engine.step()
    assert not any(engine.emitted)
    engine.assert_idle()


@pytest.mark.parametrize(
    "skew", ["fewer_tokens", "fewer_blocks", "unreported", "dcp_unreported"]
)
def test_prefiller_skew_stops_the_decoder(skew: str):
    """P computing less than D loads, or not saying how much or at which DCP
    size, is configuration skew: in pair mode it stops the engine when the
    plan is built, instead of zero-filling the tail as it does today or
    leaving the pulled rows unexchanged."""

    def request() -> Request:
        num_tokens = 3 * P_BLOCK + 7
        request = _pull_request(
            1,
            num_tokens,
            remote_num_tokens=num_tokens - 1 if skew == "fewer_tokens" else None,
            num_remote_blocks=3 if skew == "fewer_blocks" else None,
        )
        assert request.kv_transfer_params is not None
        if skew == "unreported":
            del request.kv_transfer_params["remote_num_tokens"]
        elif skew == "dcp_unreported":
            del request.kv_transfer_params["dcp_size"]
        return request

    scheduler = _make_scheduler()
    scheduler.add_request(request())
    with pytest.raises(NixlPairModeError):
        scheduler.schedule()

    scheduler = _make_scheduler(opt_in=None)
    scheduler.add_request(request())
    assert scheduler.schedule().kv_connector_metadata.reqs_to_recv


@pytest.mark.parametrize(
    ("opt_in", "remote_dcp_size", "pair_mode"),
    [
        (None, P_DCP, False),
        (False, P_DCP, False),
        # Not a JSON boolean: the workers' startup checks reject it.
        ("true", P_DCP, False),
        # Same DCP size: the pull places tokens as the decoder reads them.
        (True, D_DCP, True),
        # A quarter of the DCP size: the handshake rejects the peer.
        (True, P_DCP // 2, True),
    ],
)
def test_no_plan_outside_pair_mode(
    opt_in: object, remote_dcp_size: int, pair_mode: bool
):
    """Without the opt-in, or for a peer that is not half this DCP size, the
    connector state and metadata stay as they are without pair mode."""
    engine = _Engine(_make_scheduler(opt_in=opt_in))
    assert engine.nixl._pair_mode is pair_mode
    request = _pull_request(1, 3 * P_BLOCK + 1, remote_dcp_size=remote_dcp_size)
    req_id = request.request_id
    engine.add(request)
    engine.step()
    engine.assert_idle()
    engine.step(finished={req_id})
    engine.assert_idle()
    engine.step(eos=True)
    engine.step()
    assert engine.emitted == [{}] * 4
    assert engine.promoted == {req_id: 2}
    engine.assert_idle()


def test_pair_groups_are_the_dcp_sharded_attention_groups():
    """KDA state groups and attention groups that DCP does not shard are not
    exchanged, and a push decoder, which has no pull to fix up, exchanges
    nothing."""
    mla = MLAAttentionSpec(
        block_size=B,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.uint8,
        cache_dtype_str="fp8_ds_mla",
    )
    kda = MambaSpec(block_size=D_BLOCK, shapes=((16,),), dtypes=(torch.float16,))
    replicated = FullAttentionSpec(
        block_size=B,
        num_kv_heads=1,
        head_size=16,
        dtype=torch.float16,
        dcp_sharded=False,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["kda.0"], kda),
            KVCacheGroupSpec(
                ["mla.0", "mla.1"],
                UniformTypeKVCacheSpecs(
                    block_size=B, kv_cache_specs={"mla.0": mla, "mla.1": mla}
                ),
            ),
            KVCacheGroupSpec(["kda.1"], kda),
            KVCacheGroupSpec(["draft"], replicated),
        ],
    )
    for opt_in, groups in ((True, (1,)), (None, ())):
        vllm_config = create_vllm_config(
            kv_connector_extra_config=(
                None if opt_in is None else {"dcp_pair_exchange": opt_in}
            )
        )
        scheduler = NixlPullConnectorScheduler(vllm_config, "D", kv_cache_config)
        assert scheduler._pair_groups == groups
        push = NixlPushConnectorScheduler(vllm_config, "D", kv_cache_config)
        assert not push._pair_mode and push._pair_groups == ()


def test_plans_reach_every_worker_through_the_scheduler_output_queue():
    """The units cross the executor's scheduler-output queue intact and in
    order, which every rank walks in lockstep: by shared memory to the ranks on
    the scheduler's node and over its socket to the ranks on the other node of
    a TP16 decoder."""
    engine = _Engine(_make_scheduler())
    requests = [_pull_request(1, 3 * P_BLOCK + 9), _pull_request(2, 5 * P_BLOCK)]
    engine.add(*requests)
    engine.step()
    engine.step(finished={r.request_id for r in requests})
    output = engine.schedule()
    sent = list(output.kv_connector_metadata.reqs_to_pair_exchange.items())
    assert len(sent) == 2

    writer = MessageQueue(n_reader=2, n_local_reader=1, connect_ip="127.0.0.1")
    handle = writer.export_handle()
    readers = [MessageQueue.create_from_handle(handle, rank) for rank in (0, 1)]
    try:
        writer.wait_until_ready()
        for reader in readers:
            reader.wait_until_ready()
        # As MultiprocExecutor.collective_rpc sends execute_model.
        writer.enqueue(("execute_model", (output,), {}, None))
        for reader in readers:
            method, (received,), _, _ = reader.dequeue(timeout=10)
            assert method == "execute_model"
            meta = received.kv_connector_metadata
            assert isinstance(meta, NixlConnectorMetadata)
            assert list(meta.reqs_to_pair_exchange.items()) == sent
            assert all(
                type(unit) is PairUnit
                for units in meta.reqs_to_pair_exchange.values()
                for unit in units
            )
    finally:
        writer.shutdown()
        for reader in readers:
            reader.shutdown()
