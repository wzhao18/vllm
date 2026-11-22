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


def core_of(kind, attention, rows, records, caches, mx_out=False):
    saved = dict(sp.RECORD_KINDS)
    sp.RECORD_KINDS[HEADS] = f"1-:{kind}"
    try:
        core = sp.SpKdaCore(
            attention, (rows,), records=records, ctx=kda.context(), mx_out=mx_out
        )
    finally:
        sp.RECORD_KINDS.clear()
        sp.RECORD_KINDS.update(saved)
    assert core.kinds[rows] == kind
    core.prepare_caches(list(caches))
    return core


@pytest.mark.parametrize("stages", [1])
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("n", [3, 33, 96, 128, 141, 144, 160, 176])
def test_flow_fold_matches_ws_and_stock_commit(n, graph, stages, monkeypatch):
    monkeypatch.setattr(flow, "max_stages", lambda *args, **kwargs: stages)
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


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("n", [33, 96])
def test_flow_mxfp8_output_is_the_quantized_bf16_output(n, graph):
    """KdaFlow writing the online-MXFP8 o_proj's input itself (``mx_out``) equals its
    BF16 output quantized by ``ops.mxfp8_act.quantize`` (the standalone kernel it
    replaces), E4M3 bytes and F8_128x4 scales bit for bit, over folding steps."""
    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.mxfp8 import (
        Mxfp8Input,
    )

    sp.executable.cache_clear()
    rows, slots = Q * n, 2 * n + 2
    plain, mx = kda.layer(), kda.layer()  # the same weights, their own records
    plain.deferred_recoverssm = mx.deferred_recoverssm = True
    kv = kda.caches(slots)
    mx_kv = tuple(t.clone() for t in kv)
    _, metadata = kda.step(n, slots)
    record_rows = torch.randperm(MAX_REQS, generator=torch.Generator().manual_seed(5))
    record_rows = record_rows[:n].to(torch.int32).to(kda.DEVICE)
    pending = torch.zeros(MAX_REQS, dtype=torch.int32, device=kda.DEVICE)
    deferred = kda.Module(
        **vars(metadata),
        recoverssm_deferred=kda.Module(
            record_rows=record_rows, pending=pending, flush=None
        ),
    )
    core = core_of("flow", plain, rows, "deferred", kv)
    mx_core = core_of("flow", mx, rows, "deferred", mx_kv, mx_out=True)
    assert mx_core.writes_mxfp8(rows) and not core.writes_mxfp8(rows)
    projected = torch.empty(rows, kda.PITCH, dtype=torch.bfloat16, device=kda.DEVICE)
    out = torch.empty(rows, DIM, dtype=torch.bfloat16, device=kda.DEVICE)
    got = Mxfp8Input(rows, DIM, kda.DEVICE).view(rows)
    want = Mxfp8Input(rows, DIM, kda.DEVICE).view(rows)
    # (the core leaves the scales of the last tile's rows past R as they are)
    got.scale.zero_()
    caches, mx_caches = list(kv), list(mx_kv)

    def verify():
        core.run(projected, out, caches=caches, metadata=deferred)
        mx_core.run(projected, got, caches=mx_caches, metadata=deferred)

    replay = None
    if graph:
        scratch, mx_scratch = [t.clone() for t in kv], [t.clone() for t in mx_kv]
        core.run(projected, out, caches=scratch, metadata=deferred)
        mx_core.run(projected, got, caches=mx_scratch, metadata=deferred)
        torch.cuda.synchronize()
        replay = torch.cuda.CUDAGraph()
        with torch.cuda.graph(replay):
            verify()

    for step in range(3):
        projected.copy_(kda.step(n, slots, seed=200 + step)[0])
        if replay is not None:
            replay.replay()
        else:
            verify()
        mxfp8_act.quantize(out, out=want)
        torch.cuda.synchronize()
        where = f"step {step}"
        assert out.abs().amax() > 0, where
        for name, a, b in (("E4M3", got.data, want.data),
                           ("scales", got.scale, want.scale)):  # fmt: skip
            diff = a.view(torch.uint8) != b.view(torch.uint8)
            assert not diff.any(), f"{where} {name}: {int(diff.sum())} differ"
        pending.zero_()
        pending[record_rows.long()] = accepted_counts(n, step)
    sp.executable.cache_clear()


def test_flow_item_build_per_shape():
    """KdaFlow's item-capacity build per shape at 24 heads on Rubin's 212 SMs: the
    16-item build up to 141 requests (unchanged), the 20-item build from 142 to
    176, none above (the SP KDA core raises Unsupported)."""
    for n in (1, 21, 96, 128, 141):
        assert flow.item_build(24 * n, n, 212)[2] == 16, n
    for n in (142, 144, 160, 176):
        assert flow.item_build(24 * n, n, 212)[2] == 20, n
    assert flow.item_build(24 * 177, 177, 212)[2] is None
