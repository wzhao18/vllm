# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DP late binding (engine side): an uncommitted prompt queued on a busy DP rank
moves to an idle rank of the same engine over the DP group (gloo, CPU only),
arrives there as the same request, and a client abort sent to the original rank
is forwarded to the rank that holds it."""

import time

import pytest
import torch.multiprocessing as mp

from vllm.distributed.utils import stateless_init_torch_distributed_process_group
from vllm.sampling_params import SamplingParams
from vllm.utils.network_utils import get_open_port
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import DPEngineCoreProc
from vllm.v1.request import Request, RequestStatus

pytestmark = pytest.mark.cpu_test


class _Sched:
    """The scheduler surface the DP move path uses: a waiting list of
    uncommitted requests (all movable) plus committed prompt work."""

    def __init__(self, demand, requests=()):
        self.max_num_scheduled_tokens = 16384
        self.demand = demand
        self.requests = {r.request_id: r for r in requests}
        self.exchange = None
        self.dp_bound_here = None
        self.finished = []
        self.dp_listed_local_hit = {}

    def set_dp_prefill_exchange(self, exchange):
        self.exchange = exchange

    def set_dp_step_token_cap(self, cap):
        pass

    def dp_prefill_demand(self, num_scheduled_tokens):
        return self.demand

    def dp_move_demand(self):
        return self.demand

    def dp_movable_requests(self, limit):
        return [(rid, r.num_prompt_tokens) for rid, r in self.requests.items()][:limit]

    def dp_export_request(self, request_id):
        return self.requests.pop(request_id, None)

    def add_request(self, request):
        self.requests[request.request_id] = request

    def finish_requests(self, request_ids, status):
        for rid in request_ids:
            if self.requests.pop(rid, None) is not None:
                self.finished.append((rid, status))

    def get_kv_connector(self):
        return None


def _request(rid, n, max_tokens=1):
    ecr = EngineCoreRequest(
        request_id=rid,
        prompt_token_ids=list(range(n)),
        mm_features=None,
        sampling_params=SamplingParams(
            max_tokens=max_tokens,
            extra_args={"kv_transfer_params": {"do_remote_decode": True}},
        ),
        pooling_params=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=0,
        client_index=3,
    )
    return Request.from_engine_core_request(ecr, None)


def _core(pg, sched, local_size):
    core = object.__new__(DPEngineCoreProc)
    core.dp_group = pg
    core.scheduler = sched
    core.engines_running = True
    core.dp_prefill_pull_up = True
    core._dp_prefill_exchanged = False
    core._dp_step_token_floor = 2048
    core.vllm_config = type(
        "C", (), {"parallel_config": type("P", (), {"cpu_distributed_timeout": None})}
    )
    core._dp_move_local_size = local_size
    core._dp_move_slots = 4
    core._dp_movable_ids = []
    core._dp_moves = []
    core._dp_abort_counts = []
    core._dp_moved_to = {}
    core._dp_abort_forward = {}
    core._dp_move_stats = [0, 0, 0, 0]
    core._dp_foreign = {}
    core._dp_out_forward = {}
    core._dp_out_counts = []
    core.iterlog = None
    core.mm_receiver_cache = None
    core.request_block_hasher = None
    core.has_coordinator = False
    core.current_wave = 0

    def step():
        # Every step schedules (one exchange), with this rank's committed work.
        ex, sched.exchange = sched.exchange, None
        ex(0, sched.demand)
        return True

    core._process_engine_step = step
    return core


def _worker(rank, world_size, port, out):
    pg = stateless_init_torch_distributed_process_group(
        "127.0.0.1", port, rank, world_size, backend="gloo"
    )
    if rank == 0:
        # Busy rank: 16000 committed prompt tokens and a queued 12000-token
        # prompt (P-side truncated: max_tokens capped at 1 after arrival).
        req = _request("a", 12000, max_tokens=7)
        req.max_tokens = 1
        req.kv_transfer_params["_p_side_truncated"] = True
        sched = _Sched(16000, [req])
    else:
        sched = _Sched(0)
    core = _core(pg, sched, world_size)
    # Step 1: the exchange plans the move; step 2 runs it before scheduling.
    DPEngineCoreProc._process_engine_step_dp(core)
    planned = [(m.src, m.dst, m.num_tokens) for m in core._dp_moves]
    DPEngineCoreProc._process_engine_step_dp(core)
    held = sorted(sched.requests)
    moved = None
    if "a" in sched.requests:
        r = sched.requests["a"]
        moved = (
            r.num_prompt_tokens,
            list(r.prompt_token_ids[:3]),
            r.max_tokens,
            r.client_index,
            r.status == RequestStatus.WAITING,
            dict(r.kv_transfer_params),
        )
    # The client aborts "a" at the rank it sent it to (rank 0): forwarded at
    # the next exchange, applied at the step after.
    if rank == 0:
        DPEngineCoreProc.abort_requests(core, ["a"])
    DPEngineCoreProc._process_engine_step_dp(core)
    DPEngineCoreProc._process_engine_step_dp(core)
    out.put((rank, planned, held, moved, list(sched.finished), sorted(sched.requests)))


def test_queued_prompt_moves_to_idle_rank_and_abort_follows():
    world_size = 2
    ctx = mp.get_context("spawn")
    port = get_open_port()
    out = ctx.Queue()
    procs = [
        ctx.Process(target=_worker, args=(r, world_size, port, out))
        for r in range(world_size)
    ]
    for p in procs:
        p.start()
    got = dict((x[0], x[1:]) for x in (out.get(timeout=120) for _ in procs))
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    for rank in range(world_size):
        planned = got[rank][0]
        assert planned == [(0, 1, 12000)]
    _, held0, moved0, fin0, left0 = got[0]
    _, held1, moved1, fin1, left1 = got[1]
    assert held0 == [] and moved0 is None
    assert held1 == ["a"]
    n, head, max_tokens, client_index, waiting, kvp = moved1
    # Same prompt, cap and client; not truncated a second time.
    assert (n, head, max_tokens, client_index, waiting) == (12000, [0, 1, 2], 1, 3, True)
    assert kvp == {"do_remote_decode": True, "_p_side_truncated": True}
    assert fin0 == [] and left0 == []
    assert fin1 == [("a", RequestStatus.FINISHED_ABORTED)] and left1 == []


def _nomove_worker(rank, world_size, port, out):
    pg = stateless_init_torch_distributed_process_group(
        "127.0.0.1", port, rank, world_size, backend="gloo"
    )
    sched = _Sched(16000, [_request(f"r{rank}", 1000)])
    core = _core(pg, sched, world_size)
    DPEngineCoreProc._process_engine_step_dp(core)
    planned = list(core._dp_moves)
    DPEngineCoreProc._process_engine_step_dp(core)
    out.put((rank, planned, sorted(sched.requests)))


def test_balanced_group_moves_nothing():
    ctx = mp.get_context("spawn")
    port = get_open_port()
    out = ctx.Queue()
    procs = [ctx.Process(target=_nomove_worker, args=(r, 2, port, out)) for r in range(2)]
    for p in procs:
        p.start()
    got = dict((x[0], x[1:]) for x in (out.get(timeout=120) for _ in procs))
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    assert got == {0: ([], ["r0"]), 1: ([], ["r1"])}


def _cross_worker(rank, world_size, port, out):
    import queue

    from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason

    pg = stateless_init_torch_distributed_process_group(
        "127.0.0.1", port, rank, world_size, backend="gloo"
    )
    # Two "nodes" of two ranks. Node 0 is busy on both ranks: the queued
    # 24000-token prompt on rank 0 cannot go to rank 1, only to node 1 (gain
    # 20000 >= DP_CROSS_NODE_MIN_GAIN).
    if rank == 0:
        sched = _Sched(20000, [_request("x", 24000)])
    else:
        sched = _Sched(30000 if rank == 1 else 0)
    core = _core(pg, sched, 2)
    core.output_queue = queue.Queue()
    core.engine_index = rank
    DPEngineCoreProc._process_engine_step_dp(core)
    planned = [(m.src, m.dst) for m in core._dp_moves]
    DPEngineCoreProc._process_engine_step_dp(core)
    foreign = dict(core._dp_foreign)
    if "x" in sched.requests:
        # This rank computes it: its finished output is diverted to the home rank.
        sched.requests.pop("x")
        outputs = {
            0: EngineCoreOutputs(
                engine_index=rank,
                outputs=[
                    EngineCoreOutput(
                        request_id="x",
                        new_token_ids=[7],
                        finish_reason=FinishReason.LENGTH,
                        kv_transfer_params={"remote_engine_id": f"e{rank}"},
                    )
                ],
            )
        }
        DPEngineCoreProc._dp_divert_foreign_outputs(core, outputs)
        assert outputs[0].outputs == []
    DPEngineCoreProc._process_engine_step_dp(core)
    DPEngineCoreProc._process_engine_step_dp(core)
    emitted = []
    while not core.output_queue.empty():
        client_index, eco = core.output_queue.get()
        emitted.extend(
            (client_index, eco.engine_index, o.request_id, o.new_token_ids,
             o.kv_transfer_params)
            for o in eco.outputs
        )
    out.put((rank, planned, foreign, emitted, dict(core._dp_moved_to),
             dict(core._dp_foreign)))


def test_cross_node_move_forwards_output_to_home_rank():
    ctx = mp.get_context("spawn")
    port = get_open_port()
    out = ctx.Queue()
    procs = [ctx.Process(target=_cross_worker, args=(r, 4, port, out)) for r in range(4)]
    for p in procs:
        p.start()
    got = dict((x[0], x[1:]) for x in (out.get(timeout=120) for _ in procs))
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    for rank in range(4):
        assert got[rank][0] == [(0, 2)]
    # Rank 2 (node 1) took it in as a foreign request.
    assert got[2][1] == {"x": 0}
    # The home rank emitted the output to its client as its own engine; nothing
    # is left in either side's bookkeeping.
    assert got[0][2] == [(0, 0, "x", [7], {"remote_engine_id": "e2"})]
    assert got[2][2] == []
    assert got[0][3] == {} and got[2][4] == {}


def _bind_worker(rank, world_size, port, out):
    pg = stateless_init_torch_distributed_process_group(
        "127.0.0.1", port, rank, world_size, backend="gloo"
    )
    if rank == 0:
        sched = _Sched(16000, [_request("a", 12000), _request("b", 1000)])
    else:
        sched = _Sched(0)
    core = _core(pg, sched, world_size)
    DPEngineCoreProc._process_engine_step_dp(core)
    bound_after_plan = sorted(sched.dp_bound_here)
    DPEngineCoreProc._process_engine_step_dp(core)
    out.put((rank, bound_after_plan, sorted(sched.dp_bound_here), sorted(sched.requests)))


def test_plan_binds_requests_where_they_will_run():
    """Bind at load-issue time: the planner moves "a" to the idle rank (bound
    there on arrival) and binds "b" at home (it starts within the horizon)."""
    ctx = mp.get_context("spawn")
    port = get_open_port()
    out = ctx.Queue()
    procs = [ctx.Process(target=_bind_worker, args=(r, 2, port, out)) for r in range(2)]
    for p in procs:
        p.start()
    got = dict((x[0], x[1:]) for x in (out.get(timeout=120) for _ in procs))
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    bound0, _, held0 = got[0]
    _, bound1_after_move, held1 = got[1]
    assert bound0 == ["b"] and held0 == ["b"]
    assert "a" in bound1_after_move and held1 == ["a"]


def test_pending_handover_keeps_the_group_running():
    """A rank holding a forwarded output (or a planned move / abort forward)
    reports unfinished work, so the DP group keeps stepping until it is handed
    over (Brcff warmup, 2026-11-22: outputs parked while every engine went
    idle never reached the client)."""
    core = object.__new__(DPEngineCoreProc)
    core._dp_moves = []
    core._dp_abort_forward = {}
    core._dp_out_forward = {}
    core._dp_abort_counts = []
    core._dp_out_counts = []
    assert not DPEngineCoreProc._dp_has_pending_handover(core)
    core._dp_out_forward = {0: [("c", "out", False)]}
    assert DPEngineCoreProc._dp_has_pending_handover(core)
    core._dp_out_forward = {}
    core._dp_out_counts = [[0, 1], [0, 0]]
    assert DPEngineCoreProc._dp_has_pending_handover(core)
