# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 SP decode, the serving path of the deferred RecoverSSM (C512: 24 KDA heads
per rank, FP32 state, q4), end to end through the objects the runner uses:

* runner state: ``recoverssm.bind_model_state`` (via ``SpecializedDecode.
  bind_model_state``) installs ``DeferredRecoverSSMState`` on the model state, wraps
  ``add_request`` (a new or resumed request's pending count is reset) and
  ``prepare_attn`` (binds the KDA metadata builders: pending counts, the step's
  request slots; None for captures);
* metadata: ``state._build_deferred`` stages each speculative row's record slot from
  the step's request slots (-1 past the live rows);
* after sampling: ``KDADeferredCommitContext.commit(pending=, record_rows=, eager=)``
  (``_defer_commit_kernel``: the accepted count becomes the record slot's pending
  replay; ``eager`` rows are committed now), as ``_DeferredStepCommit`` calls it;
* next verify: ``SpKdaCore`` "deferred" (KdaWide folds the pending replay in its own
  pass; KdaSplit / KdaWS flush first through ``KDARecoverSSMDeferred.flush``).

Against the "triton" record core + the stock Triton commit after sampling, over
``STEPS`` steps with partial acceptance, a request without a verified window, eager
rows, and a request replaced mid-sequence (new state in its slot, ``add_request``).

KdaWide's fold runs the stock commit's math bit for bit, so while every commit is a
fold the chains stay equal bit for bit. A flush (KdaSplit / KdaWS) or an eager row
commits through the FP32 commit (``cutedsl.kda.commit``), the stock association in
another FP32 order: from then on the states, corrections and outputs are compared
within FP32 order (the bounds of test_specialized_decode_sp_recoverssm_gpu.py, no
drift over the steps); conv windows, keys and gates stay bit for bit. One Rubin GPU."""

from types import SimpleNamespace

import pytest
import torch

from tests.models.kimi_k3 import test_specialized_decode_sp_kda_gpu as kda
from tests.models.kimi_k3.test_specialized_decode_sp_recoverssm_gpu import (
    accepted_counts,
    commit,
    msg,
    relative,
)
from vllm.models.kimi_k3.nvidia.kda_metadata import (
    KimiK3KDAMetadata,
    KimiK3KDAMetadataBuilder,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda.sp import (
    FOLDS,
    SpKdaCore,
    record_kind,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.recoverssm import state as S
from vllm.models.kimi_k3.nvidia.specialized_decode.recoverssm.context import (
    FP32_DEFERRED,
    KDADeferredCommitContext,
    KDARecoverSSMDeferred,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.decode import (
    SpecializedDecode,
)

Q, HEADS, DIM, MAX_REQS = kda.Q, kda.HEADS, kda.DIM, kda.MAX_REQS
STEPS = 8
NEW_REQUEST_STEP = 4  # after this step's sampling, request 3 is replaced
REQUESTS = [
    pytest.param(1, id="split-n1"),
    pytest.param(9, id="wide-n9"),
    pytest.param(52, id="wide-n52"),
    pytest.param(64, id="wide-n64"),
    pytest.param(71, id="ws-n71"),
]
Module = SimpleNamespace


class _State:
    """The fields ``install_model_state`` reads on a ``KimiK3ModelState``; records the
    base ``add_request`` / ``prepare_attn`` calls."""

    def __init__(self):
        self.recoverssm = object()  # (RecoverSSM on: --use-replayssm)
        self.max_num_reqs = MAX_REQS
        self.device = kda.DEVICE
        self.calls = []

    def add_request(self, req_index, new_req_data):
        self.calls.append(("add_request", req_index))

    def prepare_attn(self, input_batch, *args, for_capture=False, **kwargs):
        self.calls.append(("prepare_attn", for_capture))
        return {}


def _builder():
    """A ``KimiK3KDAMetadataBuilder`` with only the fields ``install_builder`` and
    ``_build_deferred`` read (no KV spec / runner)."""
    builder = KimiK3KDAMetadataBuilder.__new__(KimiK3KDAMetadataBuilder)
    builder.use_recoverssm = True
    builder.recoverssm_context = None
    builder.device = kda.DEVICE
    builder.num_spec = Q - 1
    builder.vllm_config = Module(
        scheduler_config=Module(max_num_seqs=MAX_REQS),
        cache_config=Module(mamba_cache_mode="none"),
        num_speculative_tokens=Q - 1,
    )
    return builder


def _groups(builder):
    group = Module(get_metadata_builder=lambda ubatch: builder)
    other = Module(get_metadata_builder=lambda ubatch: object())  # not a KDA builder
    return [[group, other]]


def test_bind_model_state_installs_runner_side():
    """``SpecializedDecode.bind_model_state`` binds only the SP "deferred" step;
    the wrapped ``add_request`` resets pending after the base, ``prepare_attn`` binds
    the builders (request slots None for a capture) before the base."""
    state = _State()
    for sp in (None, Module(records="triton"), Module(records="none")):
        SpecializedDecode.bind_model_state(Module(sp=sp), state)
        assert not isinstance(state.recoverssm, S.DeferredRecoverSSMState), sp
    SpecializedDecode.bind_model_state(Module(sp=Module(records="deferred")), state)
    deferred = state.recoverssm
    assert isinstance(deferred, S.DeferredRecoverSSMState)
    assert deferred.pending.shape == (MAX_REQS,) and int(deferred.pending.sum()) == 0

    deferred.pending.fill_(3)
    state.add_request(7, None)
    assert state.calls[-1] == ("add_request", 7)
    assert int(deferred.pending[7]) == 0 and int(deferred.pending.sum()) == 3 * (
        MAX_REQS - 1
    )

    builder = _builder()
    idx = torch.tensor([5, 2, 9], dtype=torch.int32, device=kda.DEVICE)
    state.prepare_attn(Module(idx_mapping=idx), None, None, None, _groups(builder), None)
    assert state.calls[-1] == ("prepare_attn", False)
    assert builder.recoverssm_pending is deferred.pending
    assert builder.recoverssm_request_slots is idx
    assert builder.recoverssm_record_rows.shape == (MAX_REQS,)
    rows = builder.recoverssm_record_rows
    state.prepare_attn(
        Module(idx_mapping=idx), None, None, None, _groups(builder), None,
        for_capture=True,
    )  # fmt: skip
    assert state.calls[-1] == ("prepare_attn", True)
    assert builder.recoverssm_request_slots is None
    assert builder.recoverssm_record_rows is rows  # rebound, not reinstalled


def test_build_deferred_stages_record_rows():
    """A step's record rows = its request slots of the speculative rows, -1 past
    them (and for a capture, every row)."""
    builder = _builder()
    pending = torch.zeros(MAX_REQS, dtype=torch.int32, device=kda.DEVICE)
    S.install_builder(builder, pending)
    builder.recoverssm_record_rows.fill_(77)  # stale values of an earlier step
    slots = torch.tensor([11, 4, 30, 8, 2], dtype=torch.int32, device=kda.DEVICE)
    builder.recoverssm_request_slots = slots
    m = Module(num_reqs=5)
    context = Module()
    state_indices = torch.arange(1, 6, dtype=torch.int32, device=kda.DEVICE)
    spec_rows = torch.tensor([1, 3, 4], dtype=torch.int64, device=kda.DEVICE)
    got = S._build_deferred(builder, m, context, 3, spec_rows, None, state_indices)
    assert got.pending is pending and got.eager is None
    assert got.record_rows[:3].tolist() == [4, 8, 2]
    assert got.record_rows[3:5].tolist() == [-1, -1]
    assert got.state_indices.tolist() == [1, 2, 3]
    builder.recoverssm_request_slots = None  # a capture
    got = S._build_deferred(builder, m, context, 3, spec_rows, None, state_indices)
    assert got.record_rows[:5].tolist() == [-1] * 5


def test_build_hands_the_step_commit_its_deferred_operands():
    """The installed ``build`` wraps the stock build: a RecoverSSM step with the
    ``fp32_deferred`` context gets ``recoverssm_deferred`` and a step commit that the
    stock ``commit_recoverssm_state`` (after sampling) calls with the step's record
    rows, pending counts and eager mask."""
    n = 6
    builder = _builder()
    calls = []
    context = KDADeferredCommitContext.__new__(KDADeferredCommitContext)
    context.backend = FP32_DEFERRED
    context.commit = lambda *args, **kwargs: calls.append((args, kwargs))
    state_indices = torch.arange(1, n + 1, dtype=torch.int32, device=kda.DEVICE)
    starts = torch.arange(0, Q * n + 1, Q, dtype=torch.int32, device=kda.DEVICE)

    def stock_build(common_prefix_len, m, num_accepted_tokens=None,
                    num_decode_draft_tokens_cpu=None, fast_build=False):  # fmt: skip
        commit_md = Module(state_indices=state_indices.view(n, 1),
                           query_start_loc=starts, request_indices=None, align=None)  # fmt: skip
        return Module(recoverssm_commit=commit_md, recoverssm_context=context,
                      num_spec_decodes=n)  # fmt: skip

    builder.build = stock_build
    pending = torch.zeros(MAX_REQS, dtype=torch.int32, device=kda.DEVICE)
    S.install_builder(builder, pending)
    slots = torch.tensor([9, 3, 40, 7, 1, 22], dtype=torch.int32, device=kda.DEVICE)
    builder.recoverssm_request_slots = slots
    md = builder.build(0, Module(num_reqs=n))
    deferred = md.recoverssm_deferred
    assert isinstance(deferred, KDARecoverSSMDeferred) and deferred.pending is pending
    assert deferred.record_rows[:n].equal(slots)
    assert isinstance(md.recoverssm_context, S._DeferredStepCommit)
    accepted = torch.tensor([4, 1, 2, 3, 0, 4], dtype=torch.int32, device=kda.DEVICE)
    assert KimiK3KDAMetadata.commit_recoverssm_state(md, accepted) is None
    (args, kwargs), = calls
    assert args[0] is accepted and args[1].equal(state_indices)
    assert args[2].equal(starts)
    assert kwargs["record_rows"] is builder.recoverssm_record_rows
    assert kwargs["pending"] is pending and kwargs["eager"] is None


def _new_request(kv, ref_kv, slot, seed):
    """A new request's prefilled state and conv window in ``slot`` of both caches."""
    g = torch.Generator(device=kda.DEVICE).manual_seed(seed)
    state = torch.randn(HEADS, 128, 128, generator=g, device=kda.DEVICE) * 0.1
    conv = torch.randn(3 * DIM, Q + 2, generator=g, device=kda.DEVICE)
    for caches in (kv, ref_kv):
        caches[1][slot].copy_(state)
        caches[0][slot].copy_(conv.to(caches[0].dtype))


@pytest.mark.parametrize("eager_rows", [False, True], ids=["deferred", "eager"])
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("n", REQUESTS)
def test_deferred_serving_path_matches_stock(n, graph, eager_rows):
    kind = record_kind(HEADS, n)
    assert kind is not None, n
    rows, slots = Q * n, 2 * n + 2
    stock, ours = kda.layer(), kda.layer()  # the same weights, their own records
    S.install_layers([Module(self_attn=ours)])  # records per record slot
    assert ours.deferred_recoverssm
    kv = kda.caches(slots)
    ref_kv = tuple(t.clone() for t in kv)
    _, metadata = kda.step(n, slots)
    checkpoints = metadata.spec_state_indices_tensor[:, 0].contiguous()
    starts = torch.arange(0, rows + 1, Q, dtype=torch.int32, device=kda.DEVICE)

    # Runner side: the model state's deferred RecoverSSM and the step's record rows
    # (request slots = idx_mapping of the batch rows; one row per request here).
    runner = _State()
    SpecializedDecode.bind_model_state(Module(sp=Module(records="deferred")), runner)
    pending = runner.recoverssm.pending
    builder = _builder()
    idx_mapping = torch.randperm(MAX_REQS, generator=torch.Generator().manual_seed(4))
    idx_mapping = idx_mapping[:n].to(torch.int32).to(kda.DEVICE)
    runner.prepare_attn(Module(idx_mapping=idx_mapping), None, None, None,
                        _groups(builder), None)  # fmt: skip
    ours.kv_cache = list(kv)
    context = KDADeferredCommitContext.create(
        [ours], spec_query_len=Q, max_num_reqs=MAX_REQS
    )
    assert context.backend == FP32_DEFERRED

    def build_step():
        """The metadata build's deferred operands (``state._build_deferred``)."""
        deferred = S._build_deferred(
            builder, Module(num_reqs=n), context, n, None, None, checkpoints
        )
        return Module(**vars(metadata), recoverssm_deferred=deferred)

    ref_core = SpKdaCore(stock, (rows,), records="triton", ctx=kda.context())
    ref_core.prepare_caches(list(ref_kv))
    core = SpKdaCore(ours, (rows,), records="deferred", ctx=kda.context())
    core.prepare_caches(list(kv))
    projected = torch.empty(rows, kda.PITCH, dtype=torch.bfloat16, device=kda.DEVICE)
    out, ref_out = (
        torch.empty(rows, DIM, dtype=torch.bfloat16, device=kda.DEVICE)
        for _ in range(2)
    )
    caches, ref_caches = list(kv), list(ref_kv)
    step_md = build_step()
    assert step_md.recoverssm_deferred.record_rows[:n].equal(idx_mapping)

    replay = None
    if graph:
        torch.cuda.synchronize()
        replay = torch.cuda.CUDAGraph()
        with torch.cuda.graph(replay):  # (captured, not run): the persistent buffers
            core.run(projected, out, caches=caches, metadata=step_md)

    folded = 0
    replaced = 3 if n > 3 else None
    exact = kind in FOLDS  # (a flush or an eager commit: FP32 order from then on)
    errors = []
    for step in range(STEPS):
        projected.copy_(kda.step(n, slots, seed=100 + step)[0])
        ref_core.run(projected, ref_out, caches=ref_caches, metadata=metadata)
        folded += int(pending.sum())
        if replay is not None:
            replay.replay()
        else:
            step_md = build_step()  # a fresh deferred per step (flush once)
            core.run(projected, out, caches=caches, metadata=step_md)
        torch.cuda.synchronize()
        exact &= kind in FOLDS
        where = f"{kind} step {step}" + ("" if exact else " (FP32 order)")
        got, want = ours.recoverssm_records, stock.recoverssm_records
        index = idx_mapping.long()
        kda.assert_bitwise(f"{where} conv window", kv[0], ref_kv[0])
        kda.assert_bitwise(f"{where} key / gate", got.key_gate[index],
                           want.key_gate[:n])  # fmt: skip
        if exact:
            kda.assert_bitwise(f"{where} checkpoints", kv[1], ref_kv[1])
            kda.assert_bitwise(f"{where} output", out, ref_out)
            kda.assert_bitwise(f"{where} corrections", got.correction[index],
                               want.correction[:n])  # fmt: skip
        else:
            live = checkpoints.long()
            errors.append(relative(kv[1][live], ref_kv[1][live]))
            tol = dict(rtol=1e-4 * (step + 1), atol=1e-5 * (step + 1))
            torch.testing.assert_close(kv[1], ref_kv[1], **tol,
                                       msg=msg(f"{where} checkpoints"))  # fmt: skip
            torch.testing.assert_close(got.correction[index], want.correction[:n],
                                       **tol, msg=msg(f"{where} corrections"))  # fmt: skip
            torch.testing.assert_close(out.float(), ref_out.float(), rtol=2e-2,
                                       atol=2e-2, msg=msg(f"{where} output"))  # fmt: skip

        # After sampling: the stock chain commits; ours goes through the deferred
        # context's commit as _DeferredStepCommit calls it.
        accepted = accepted_counts(n, step)
        if step % 3 == 2 and n > 2:
            accepted[2] = 0  # (a request without a verified window: no commit)
        eager = None
        if eager_rows and step % 2 == 1:
            mask = torch.zeros(n, dtype=torch.int32)
            mask[0] = 1
            mask[n - 1] = 1
            eager = mask.to(kda.DEVICE)
        commit(stock, ref_kv, accepted, checkpoints, starts)
        ours.kv_cache = list(kv)
        context.commit(
            accepted, checkpoints, starts,
            record_rows=builder.recoverssm_record_rows, pending=pending, eager=eager,
        )  # fmt: skip
        torch.cuda.synchronize()
        want_pending = accepted.clone()
        if eager is not None:
            want_pending[eager.bool()] = 0
            # eager rows are committed now (FP32 commit: the stock state in FP32 order)
            live = checkpoints.long()[eager.bool()]
            torch.testing.assert_close(
                kv[1][live], ref_kv[1][live], rtol=1e-4 * (step + 1),
                atol=1e-5 * (step + 1), msg=msg(f"{where} eager committed"),
            )  # fmt: skip
            exact = False
        assert pending[idx_mapping.long()].tolist() == want_pending.tolist(), where
        others = torch.ones(MAX_REQS, dtype=torch.bool, device=kda.DEVICE)
        others[idx_mapping.long()] = False
        assert int(pending[others].abs().sum()) == 0, where

        if replaced is not None and step == NEW_REQUEST_STEP:
            # Request 3 finished; a new request takes its row, request slot and
            # state slot with a freshly prefilled state (both chains).
            _new_request(kv, ref_kv, int(checkpoints[replaced]), seed=900 + step)
            runner.add_request(int(idx_mapping[replaced]), None)
            assert int(pending[idx_mapping[replaced].long()]) == 0
    assert folded > 0
    print(f"{kind} n={n} graph={graph} eager={eager_rows}: state rel error {errors}")
    assert max(errors, default=0.0) <= 1e-4, errors
