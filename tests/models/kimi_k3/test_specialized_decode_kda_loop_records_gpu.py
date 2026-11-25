# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 TP16 KDA verify rows under ``--use-replayssm`` (6 heads per rank, q8, FP32
state, FP32 conv): the record core ``KdaVerifyLoopRecords`` (``sp.SpKdaCore`` kind
"loop") against

* the plain ``KdaVerifyLoopStock`` verify it extends (nothing pending): the outputs and
  the conv window bit for bit (same operations), the checkpoint left as it is;
* the ``KdaWide`` record core (pending commits of 1..q tokens): the folded checkpoint
  bit for bit (both run the stock Triton commit's instructions), the recorded keys and
  raw gates bit for bit, the corrections and outputs within FP32 reduction order;
* the stock chain (Triton RecoverSSM verify + Triton commit after sampling) over several
  steps with partial acceptance, ours through the deferred commit context: committed
  states within FP32 order (no drift), conv windows bit for bit.

One Rubin GPU."""

from types import SimpleNamespace

import pytest
import torch

# (skips the module off Rubin / without cutlass, as its own tests)
from tests.models.kimi_k3 import test_specialized_decode_sp_kda_gpu as kda
from tests.models.kimi_k3.test_specialized_decode_sp_recoverssm_gpu import (
    msg,
    relative,
)
from vllm.models.kimi_k3.nvidia.ops.recoverssm import KDARecoverSSMCommitContext
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda import prepared, sp
from vllm.models.kimi_k3.nvidia.specialized_decode.recoverssm import state as S
from vllm.models.kimi_k3.nvidia.specialized_decode.recoverssm.context import (
    KDADeferredCommitContext,
)

Module = SimpleNamespace
HEADS, DIM, Q, PITCH, MAX_REQS = 6, 768, 8, 3216, 72
STEPS = 4


@pytest.fixture(autouse=True)
def tp16(monkeypatch):
    """The SP test helpers at the TP16 KDA shape (6 heads, q8, FP32 conv)."""
    shape = dict(
        HEADS=HEADS, DIM=DIM, Q=Q, PITCH=PITCH, MAX_REQS=MAX_REQS,
        CONV_DTYPE=torch.float32,
    )  # fmt: skip
    for name, value in shape.items():
        monkeypatch.setattr(kda, name, value)
    monkeypatch.setitem(sp.RECORD_KINDS, HEADS, {"*": "1-1024:loop"})
    sp.executable.cache_clear()
    prepared.executable.cache_clear()
    yield
    sp.executable.cache_clear()
    prepared.executable.cache_clear()


def deferred_metadata(metadata, record_rows, pending):
    return Module(
        **vars(metadata),
        recoverssm_deferred=Module(
            record_rows=record_rows, pending=pending, flush=lambda: None
        ),
    )


def core_of(attention, rows, kind):
    core = sp.SpKdaCore(attention, (rows,), records="deferred", ctx=kda.context())
    core.kinds[rows] = kind
    return core


def random_records(attention, seed):
    g = torch.Generator(device=kda.DEVICE).manual_seed(seed)
    rec = attention.recoverssm_records
    rec.correction.copy_(
        torch.randn(rec.correction.shape, generator=g, device=kda.DEVICE) * 0.05
    )
    keys = torch.randn(rec.key_gate.shape, generator=g, device=kda.DEVICE) * 0.5
    rec.key_gate.copy_(keys.to(rec.key_gate.dtype))


@pytest.mark.parametrize("n", [1, 4, 16, 37])
def test_loop_records_nothing_pending_equals_loopstock(n, monkeypatch):
    rows, slots = Q * n, Q * n + 2
    ours, plain = kda.layer(), kda.layer()
    ours.deferred_recoverssm = True
    kv = kda.caches(slots)
    ref_kv = tuple(t.clone() for t in kv)
    projected, metadata = kda.step(n, slots)
    checkpoints = metadata.spec_state_indices_tensor  # [n, 1]
    record_rows = torch.randperm(MAX_REQS, generator=torch.Generator().manual_seed(4))
    record_rows = record_rows[:n].to(torch.int32).to(kda.DEVICE)
    pending = torch.zeros(MAX_REQS, dtype=torch.int32, device=kda.DEVICE)
    out = torch.empty(rows, DIM, dtype=torch.bfloat16, device=kda.DEVICE)
    core = core_of(ours, rows, "loop")
    core.prepare_caches(list(kv))
    core.run(projected, out, caches=list(kv),
             metadata=deferred_metadata(metadata, record_rows, pending))  # fmt: skip

    # The plain LoopStock verify from the same checkpoint: slot 0 the checkpoint,
    # the other q - 1 token states into slots no request owns.
    used = set(checkpoints[:, 0].tolist())
    free = [s for s in range(1, slots) if s not in used]
    table = torch.empty(n, Q, dtype=torch.int32)
    table[:, 0] = checkpoints[:, 0].cpu()
    table[:, 1:] = torch.tensor(free[: n * (Q - 1)]).view(n, Q - 1)
    table = table.to(kda.DEVICE)
    monkeypatch.setattr(prepared, "wide_kind", lambda *a, **k: prepared.FP32_WIDE)
    loopstock = prepared.PreparedKda(plain, {(n, Q): (PITCH, 0)}, True)
    loopstock.prepare_caches(list(ref_kv))
    want = torch.empty_like(out)
    accepted = torch.ones(n, dtype=torch.int32, device=kda.DEVICE)
    loopstock.run(projected, want, caches=list(ref_kv), slots=table,
                  accepted=accepted)  # fmt: skip
    torch.cuda.synchronize()
    kda.assert_bitwise("output", out, want)
    kda.assert_bitwise("conv window", kv[0], ref_kv[0])
    live = checkpoints[:, 0].long()
    original = kda.caches(slots)[1]
    kda.assert_bitwise("checkpoint", kv[1][live], original[live])


@pytest.mark.parametrize("n", [1, 7, 16, 40])
def test_loop_records_fold_matches_wide(n):
    rows, slots = Q * n, 2 * n + 2
    loop, wide = kda.layer(), kda.layer()
    for attention in (loop, wide):
        attention.deferred_recoverssm = True
        random_records(attention, seed=9)
    kv = kda.caches(slots)
    ref_kv = tuple(t.clone() for t in kv)
    projected, metadata = kda.step(n, slots)
    record_rows = torch.randperm(MAX_REQS, generator=torch.Generator().manual_seed(5))
    record_rows = record_rows[:n].to(torch.int32).to(kda.DEVICE)
    g = torch.Generator().manual_seed(6)
    pending = torch.randint(0, Q + 1, (MAX_REQS,), generator=g).to(torch.int32)
    pending[record_rows[0].long().cpu()] = Q  # (at least one full replay)
    pending = pending.to(kda.DEVICE)
    md = deferred_metadata(metadata, record_rows, pending)
    out, ref_out = (
        torch.empty(rows, DIM, dtype=torch.bfloat16, device=kda.DEVICE)
        for _ in range(2)
    )
    ours = core_of(loop, rows, "loop")
    ours.prepare_caches(list(kv))
    ours.run(projected, out, caches=list(kv), metadata=md)
    ref = core_of(wide, rows, "wide")
    ref.prepare_caches(list(ref_kv))
    ref.run(projected, ref_out, caches=list(ref_kv), metadata=md)
    torch.cuda.synchronize()
    kda.assert_bitwise("folded checkpoints", kv[1], ref_kv[1])
    kda.assert_bitwise("conv window", kv[0], ref_kv[0])
    index = record_rows.long()
    got, want = loop.recoverssm_records, wide.recoverssm_records
    kda.assert_bitwise("keys / raw gates", got.key_gate[index], want.key_gate[index])
    torch.testing.assert_close(
        got.correction[index], want.correction[index], rtol=1e-4, atol=1e-5
    )
    torch.testing.assert_close(out.float(), ref_out.float(), rtol=2e-2, atol=2e-2)


def accepted_counts(n, step):
    g = torch.Generator().manual_seed(1000 + step)
    counts = torch.randint(1, Q + 1, (n,), generator=g)
    counts[0] = Q
    if n > 1:
        counts[1] = 1
    return counts.to(torch.int32).to(kda.DEVICE)


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("n", [2, 16, 33])
def test_loop_records_chain_matches_stock(n, graph):
    rows, slots = Q * n, 2 * n + 2
    ours, stock = kda.layer(), kda.layer()  # the same weights, their own records
    S.install_layers([Module(self_attn=ours)])  # records per record slot
    kv = kda.caches(slots)
    ref_kv = tuple(t.clone() for t in kv)
    _, metadata = kda.step(n, slots)
    checkpoints = metadata.spec_state_indices_tensor[:, 0].contiguous()
    starts = torch.arange(0, rows + 1, Q, dtype=torch.int32, device=kda.DEVICE)
    record_rows = torch.randperm(MAX_REQS, generator=torch.Generator().manual_seed(4))
    record_rows = record_rows[:n].to(torch.int32).to(kda.DEVICE)
    pending = torch.zeros(MAX_REQS, dtype=torch.int32, device=kda.DEVICE)
    full_rows = torch.full((MAX_REQS,), -1, dtype=torch.int32, device=kda.DEVICE)
    full_rows[:n] = record_rows
    md = deferred_metadata(metadata, full_rows, pending)
    ours.kv_cache = list(kv)
    context = KDADeferredCommitContext.create(
        [ours], spec_query_len=Q, max_num_reqs=MAX_REQS
    )
    core = core_of(ours, rows, "loop")
    core.prepare_caches(list(kv))
    projected = torch.empty(rows, PITCH, dtype=torch.bfloat16, device=kda.DEVICE)
    out = torch.empty(rows, DIM, dtype=torch.bfloat16, device=kda.DEVICE)
    caches = list(kv)
    replay = None
    if graph:
        torch.cuda.synchronize()
        replay = torch.cuda.CUDAGraph()
        with torch.cuda.graph(replay):  # (captured, not run)
            core.run(projected, out, caches=caches, metadata=md)
    errors = []
    for step in range(STEPS):
        projected.copy_(kda.step(n, slots, seed=100 + step)[0])
        want = kda.stock(stock, ref_kv, projected, metadata)
        if replay is not None:
            replay.replay()
        else:
            core.run(projected, out, caches=caches, metadata=md)
        torch.cuda.synchronize()
        where = f"step {step}"
        kda.assert_bitwise(f"{where} conv window", kv[0], ref_kv[0])
        torch.testing.assert_close(out.float(), want.float(), rtol=2e-2, atol=2e-2,
                                   msg=msg(f"{where} output"))  # fmt: skip
        got, ref = ours.recoverssm_records, stock.recoverssm_records
        index = record_rows.long()
        kda.assert_bitwise(f"{where} keys", got.key_gate[index][..., :128],
                           ref.key_gate[:n][..., :128])  # fmt: skip
        accepted = accepted_counts(n, step)
        # stock: the Triton commit now; ours: the accepted counts become pending
        stock.kv_cache = ref_kv
        KDARecoverSSMCommitContext.create(
            [stock], spec_query_len=Q, max_num_reqs=MAX_REQS, backend="triton"
        ).commit(accepted, checkpoints, starts)
        ours.kv_cache = list(kv)
        context.commit(accepted, checkpoints, starts, record_rows=full_rows,
                       pending=pending, eager=None)  # fmt: skip
        torch.cuda.synchronize()
        assert pending[index].tolist() == accepted.tolist(), where
    # The last pending commits folded by one more verify: the states then compare.
    projected.copy_(kda.step(n, slots, seed=999)[0])
    if replay is not None:
        replay.replay()
    else:
        core.run(projected, out, caches=caches, metadata=md)
    kda.stock(stock, ref_kv, projected, metadata)
    torch.cuda.synchronize()
    live = checkpoints.long()
    error = relative(kv[1][live], ref_kv[1][live])
    errors.append(error)
    torch.testing.assert_close(kv[1][live], ref_kv[1][live], rtol=1e-3, atol=1e-4,
                               msg=msg("folded state"))  # fmt: skip
    print(f"loop n={n} graph={graph}: folded state relative error {errors}")
    assert max(errors) <= 1e-4, errors
