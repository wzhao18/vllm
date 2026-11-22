# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DP prefill pull-up (engine side): every running iteration of a DP engine core
takes part in exactly one exchange of the step's scheduled token count over the
engine DP group (gloo, CPU only), whether or not it schedules anything."""

from types import SimpleNamespace

import pytest
import torch.multiprocessing as mp

from vllm.distributed.utils import stateless_init_torch_distributed_process_group
from vllm.utils.network_utils import get_open_port
from vllm.v1.engine.core import DPEngineCoreProc

pytestmark = pytest.mark.cpu_test


class _Sched:
    """Calls the armed exchange from schedule() when it has requests, like
    Scheduler.schedule(); records what came back."""

    def __init__(self, counts):
        self.counts = list(counts)
        self.exchange = None
        self.got = []

        self.max_num_scheduled_tokens = 16384
        self.caps = []

    def set_dp_prefill_exchange(self, exchange):
        self.exchange = exchange

    def set_dp_step_token_cap(self, cap):
        self.caps.append(cap)

    def dp_prefill_demand(self, num_scheduled_tokens):
        return 0


def _fake_core(pg, sched, engines_running=True, pull_up=True):
    core = SimpleNamespace(
        dp_group=pg,
        scheduler=sched,
        engines_running=engines_running,
        dp_prefill_pull_up=pull_up,
        _dp_prefill_exchanged=False,
        _dp_step_token_floor=2048,
        _dp_moves=[],
        _dp_abort_counts=[],
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(cpu_distributed_timeout=None)
        ),
    )
    core._exchange_dp_prefill_tokens = (
        lambda n, d=0: DPEngineCoreProc._exchange_dp_prefill_tokens(core, n, d)
    )

    def step():
        # A step either schedules (count >= 0: schedule() calls the exchange
        # once) or, with no local requests (count None), does not.
        count = sched.counts.pop(0)
        if count is not None and sched.exchange is not None:
            ex, sched.exchange = sched.exchange, None
            sched.got.append(ex(count))
        return bool(count)

    core._process_engine_step = step
    return core


def _worker(rank, world_size, port, steps, out):
    pg = stateless_init_torch_distributed_process_group(
        "127.0.0.1", port, rank, world_size, backend="gloo"
    )
    sched = _Sched(s[rank] for s in steps)
    core = _fake_core(pg, sched)
    for _ in steps:
        DPEngineCoreProc._process_engine_step_dp(core)
        assert sched.exchange is None
    # An idle rank's exchange (no schedule) returns the max too; collect via a
    # direct call so every rank can check it.
    out[rank] = (sched.got, DPEngineCoreProc._exchange_dp_prefill_tokens(core, rank))


@pytest.mark.parametrize("world_size", [2, 3, 4])
def test_one_exchange_per_step_on_every_rank(world_size):
    # Per step, per rank: tokens scheduled, or None = no schedule() (idle rank).
    steps = [
        [24576, 8192, None, 16384, 0][:world_size],
        [None] * world_size,
        [32768] + [None] * (world_size - 1),
        [5000 + 100 * r for r in range(world_size)],
    ]
    port = get_open_port()
    with mp.Manager() as m:
        out = m.dict()
        procs = [
            mp.get_context("spawn").Process(
                target=_worker, args=(r, world_size, port, steps, out)
            )
            for r in range(world_size)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=120)
            assert p.exitcode == 0
        for r in range(world_size):
            got, last = out[r]
            expected = [max(c or 0 for c in s) for s in steps if s[r] is not None]
            assert got == expected
            assert last == world_size - 1


def test_not_armed_when_engines_idle_or_disabled():
    for running, pull_up in ((False, True), (True, False)):
        sched = _Sched([7])
        core = _fake_core(None, sched, engines_running=running, pull_up=pull_up)
        assert DPEngineCoreProc._process_engine_step_dp(core) is True
        assert sched.got == [] and sched.exchange is None


def test_dp_step_token_cap_policy():
    from vllm.v1.engine.core import dp_step_token_cap

    # Idle or light group: the floor (largest CUDA-graph batch).
    assert dp_step_token_cap(0, 16, 16384, 2048) == 2048
    assert dp_step_token_cap(16 * 1000, 16, 16384, 2048) == 2048
    # Mean outstanding prompt tokens per rank, rounded up to 128.
    assert dp_step_token_cap(16 * 6000, 16, 16384, 2048) == 6016
    assert dp_step_token_cap(100_000, 16, 16384, 2048) == 6272
    # Saturated group: no cap (the configured budget).
    assert dp_step_token_cap(16 * 16384, 16, 16384, 2048) is None
    assert dp_step_token_cap(10**7, 16, 16384, 2048) is None
    # Floor never above the budget.
    assert dp_step_token_cap(0, 4, 1024, 2048) == 1024


def _cap_worker(rank, world_size, port, out):
    pg = stateless_init_torch_distributed_process_group(
        "127.0.0.1", port, rank, world_size, backend="gloo"
    )
    sched = _Sched([])
    core = _fake_core(pg, sched)
    # Every rank sets the same cap from the group's total demand.
    got = DPEngineCoreProc._exchange_dp_prefill_tokens(core, 100 * rank, 4096 * rank)
    # A long cold prompt on one rank counts as one full step (the budget):
    # 16384 (clipped from 10**6) + 0 + 0 + 0 -> 4096 per rank.
    DPEngineCoreProc._exchange_dp_prefill_tokens(core, 0, 10**6 if rank == 0 else 0)
    out[rank] = (got, sched.caps)


def test_exchange_sets_the_same_cap_on_every_rank():
    world_size = 4
    port = get_open_port()
    with mp.Manager() as m:
        out = m.dict()
        ps = [
            mp.get_context("spawn").Process(
                target=_cap_worker, args=(r, world_size, port, out)
            )
            for r in range(world_size)
        ]
        for p in ps:
            p.start()
        for p in ps:
            p.join(timeout=120)
            assert p.exitcode == 0
        # demand 0 + 4096 + 8192 + 12288 = 24576 -> 6144 per rank
        for rank in range(world_size):
            assert out[rank] == (300, [6144, 4096])


def _idle_flag_worker(rank, world_size, port, sched_tokens, demand, out):
    pg = stateless_init_torch_distributed_process_group(
        "127.0.0.1", port, rank, world_size, backend="gloo"
    )
    sched = _Sched([sched_tokens[rank]])
    core = _fake_core(pg, sched)
    DPEngineCoreProc._exchange_dp_prefill_tokens(core, sched_tokens[rank], demand[rank])
    out.put((rank, sched.dp_group_has_idle_rank))


@pytest.mark.parametrize(
    "sched_tokens,demand,expected",
    [
        ([4096, 0], [8000, 0], True),  # rank 1 has nothing, rank 0 has work
        ([4096, 2048], [8000, 0], False),  # rank 1 schedules something
        ([0, 0], [0, 0], False),  # no prompt work anywhere
    ],
)
def test_exchange_sets_idle_rank_flag(sched_tokens, demand, expected):
    ctx = mp.get_context("spawn")
    port = get_open_port()
    out = ctx.Queue()
    procs = [
        ctx.Process(
            target=_idle_flag_worker,
            args=(r, 2, port, sched_tokens, demand, out),
        )
        for r in range(2)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    got = dict(out.get(timeout=5) for _ in range(2))
    assert got == {0: expected, 1: expected}
