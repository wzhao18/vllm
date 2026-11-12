# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 SP decode, deferred RecoverSSM folded into KdaWide (C512: 24 KDA heads
per rank, FP32 state, q4): over ``STEPS`` consecutive verify steps of the same
requests with partial acceptance, the "deferred" record core (each verify first folds
the previous step's accepted tokens into the checkpoint, ``wide.KdaWide.fold``)
against the "triton" record core + the stock Triton commit after sampling
(``_commit_kda_state_kernel`` + ``_compact_conv_state_kernel``), each on its own copy
of the caches. The fold runs the commit's math bit for bit, so the checkpoints, the
conv windows, the outputs and the records are equal bit for bit at every step. Eager
and with the deferred verify captured once in a CUDA graph. One Rubin GPU."""

import pytest
import torch

from tests.models.kimi_k3 import test_specialized_decode_sp_kda_gpu as kda
from tests.models.kimi_k3.test_specialized_decode_sp_recoverssm_gpu import (
    accepted_counts,
    commit,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda.sp import (
    FOLDS,
    SpKdaCore,
    record_kind,
)

Q, HEADS, DIM, MAX_REQS = kda.Q, kda.HEADS, kda.DIM, kda.MAX_REQS
STEPS = 8
REQUESTS = [
    pytest.param(9, id="wide-n9"),
    pytest.param(33, id="wide-n33"),
    pytest.param(52, id="wide-n52"),
    pytest.param(70, id="wide-n70"),
]


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("n", REQUESTS)
def test_fold_matches_stock_commit_then_verify(n, graph):
    assert record_kind(HEADS, n) in FOLDS, n
    rows, slots = Q * n, 2 * n + 2
    stock, ours = kda.layer(), kda.layer()  # the same weights, their own records
    ours.deferred_recoverssm = True
    kv = kda.caches(slots)
    ref_kv = tuple(t.clone() for t in kv)
    _, metadata = kda.step(n, slots)
    checkpoints = metadata.spec_state_indices_tensor[:, 0].contiguous()
    starts = torch.arange(0, rows + 1, Q, dtype=torch.int32, device=kda.DEVICE)
    record_rows = torch.randperm(MAX_REQS, generator=torch.Generator().manual_seed(4))
    record_rows = record_rows[:n].to(torch.int32).to(kda.DEVICE)
    pending = torch.zeros(MAX_REQS, dtype=torch.int32, device=kda.DEVICE)

    def no_flush():
        raise AssertionError("a folding core flushed")

    deferred = kda.Module(
        **vars(metadata),
        recoverssm_deferred=kda.Module(
            record_rows=record_rows, pending=pending, flush=no_flush
        ),
    )
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

    def verify():
        core.run(projected, out, caches=caches, metadata=deferred)

    replay = None
    if graph:
        torch.cuda.synchronize()
        replay = torch.cuda.CUDAGraph()
        with torch.cuda.graph(replay):  # (captured, not run)
            verify()

    folded = 0
    for step in range(STEPS):
        projected.copy_(kda.step(n, slots, seed=100 + step)[0])
        ref_core.run(projected, ref_out, caches=ref_caches, metadata=metadata)
        if replay is not None:
            replay.replay()
        else:
            verify()
        torch.cuda.synchronize()
        where = f"step {step}"
        # The fold committed the previous step's tokens as the stock commit did.
        kda.assert_bitwise(f"{where} checkpoints", kv[1], ref_kv[1])
        kda.assert_bitwise(f"{where} conv window", kv[0], ref_kv[0])
        kda.assert_bitwise(f"{where} output", out, ref_out)
        got, want = ours.recoverssm_records, stock.recoverssm_records
        index = record_rows.long()
        kda.assert_bitwise(f"{where} corrections", got.correction[index],
                           want.correction[:n])  # fmt: skip
        kda.assert_bitwise(f"{where} key / gate", got.key_gate[index],
                           want.key_gate[:n])  # fmt: skip
        folded += int(pending.sum())
        # After sampling: the stock chain commits, the deferred one records.
        accepted = accepted_counts(n, step)
        if step % 3 == 2 and n > 2:
            accepted[2] = 0  # (a request without a verified window: no commit)
        commit(stock, ref_kv, accepted, checkpoints, starts)
        pending.zero_()
        pending[index] = accepted
    assert folded > 0
