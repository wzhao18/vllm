# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 SP decode, KdaFlow (``cutedsl.kda.flow``: persistent record core with the
deferred RecoverSSM fold), C512 shape (24 KDA heads per rank, FP32 state, q4).

Over ``STEPS`` consecutive verify steps of the same requests with partial acceptance:
KdaFlow in "deferred" mode (each verify first folds the previous step's accepted
tokens into the checkpoint) against KdaWS in "triton" mode + the stock Triton commit
after sampling (``_commit_kda_state_kernel`` + ``_compact_conv_state_kernel``), each
on its own copy of the caches; once more with the verify captured in a CUDA graph.
KdaFlow's fold is the commit's math and its verify is the stock math with another
FP32 reduction order of the recurrence's dot products: conv windows, recorded keys
and raw gates equal bit for bit, checkpoints and corrections within FP32
reduction-order tolerance, outputs within BF16 tolerance. One Rubin GPU."""

import pytest
import torch

from tests.models.kimi_k3 import test_specialized_decode_sp_kda_gpu as kda
from tests.models.kimi_k3.test_specialized_decode_sp_recoverssm_gpu import (
    accepted_counts,
    commit,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda import flow, sp

Q, HEADS, DIM, MAX_REQS = kda.Q, kda.HEADS, kda.DIM, kda.MAX_REQS
STEPS = 6


def core_of(kind, attention, rows, records, caches):
    saved = dict(sp.RECORD_KINDS)
    sp.RECORD_KINDS[HEADS] = f"1-:{kind}"
    try:
        core = sp.SpKdaCore(attention, (rows,), records=records, ctx=kda.context())
    finally:
        sp.RECORD_KINDS.clear()
        sp.RECORD_KINDS.update(saved)
    assert core.kinds[rows] == kind
    core.prepare_caches(list(caches))
    return core


@pytest.mark.parametrize("stages", [1])
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("n", [3, 33, 96, 128])
def test_flow_fold_matches_ws_and_stock_commit(n, graph, stages, monkeypatch):
    monkeypatch.setattr(flow, "max_stages", lambda *args: stages)
    sp.executable.cache_clear()
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
    ref_core = core_of("ws", stock, rows, "triton", ref_kv)
    core = core_of("flow", ours, rows, "deferred", kv)
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
        # One eager launch first, on scratch caches (as the runner warms up before
        # capturing): loads the kernel module outside the capture.
        scratch = [t.clone() for t in kv]
        core.run(projected, out, caches=scratch, metadata=deferred)
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
        torch.testing.assert_close(kv[1], ref_kv[1], rtol=1e-4, atol=1e-5,
                                   msg=f"{where} checkpoints")  # fmt: skip
        kda.assert_bitwise(f"{where} conv window", kv[0], ref_kv[0])
        torch.testing.assert_close(out.float(), ref_out.float(), rtol=2e-2,
                                   atol=2e-2, msg=f"{where} output")  # fmt: skip
        got, want = ours.recoverssm_records, stock.recoverssm_records
        index = record_rows.long()
        torch.testing.assert_close(got.correction[index], want.correction[:n],
                                   rtol=1e-4, atol=1e-5,
                                   msg=f"{where} corrections")  # fmt: skip
        kda.assert_bitwise(f"{where} key / gate", got.key_gate[index],
                           want.key_gate[:n])  # fmt: skip
        folded += int(pending.sum())
        accepted = accepted_counts(n, step)
        if step % 3 == 2 and n > 2:
            accepted[2] = 0  # (a request without a verified window: no commit)
        commit(stock, ref_kv, accepted, checkpoints, starts)
        pending.zero_()
        pending[index] = accepted
    assert folded > 0
    sp.executable.cache_clear()
