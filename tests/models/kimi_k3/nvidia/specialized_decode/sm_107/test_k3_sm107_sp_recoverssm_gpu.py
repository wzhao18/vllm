# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from the source branch's tests/models/kimi_k3/test_specialized_decode_sp_recoverssm_gpu.py for the SM107
# specialized decode (vllm/models/kimi_k3/nvidia/specialized_decode/sm_107).
"""Kimi K3 SP decode, RecoverSSM over many steps (C512: 24 KDA heads per rank, FP32
state, q4, "triton" records): ``cutedsl.kda.sp.SpKdaCore`` + the stock Triton commit
against the stock RecoverSSM verify (``KimiK3DeltaAttention._recoverssm_verify``,
then ``o_norm``) + the same commit, each on its own copy of the caches, for
``STEPS`` consecutive decode steps of the same requests with partial acceptance
(random accepted counts 1..4 per request and step, all-accepted and one-accepted
requests in every step).

The two chains get the same projected rows every step, so the conv window (raw
inputs) and the recorded keys must stay equal bit for bit at every step; the FP32
checkpoint state, corrections and outputs differ only by FP32 reduction order, which
the commit carries forward: the state stays within the bound below at every step
(no drift past it after ``STEPS`` commits). Run eagerly and with the specialized
verify captured once in a CUDA graph and replayed every step (as the FULL decode
graphs run it), at one request count per record core (KdaSplit, KdaWS, KdaWide,
KdaStream, KdaWS at 128 requests).

The single-step comparison (records, deferred mode) is
test_k3_sm107_sp_kda_gpu.py; the block-level chain (AG -> in-proj -> core
-> o_proj + RS) against the stock layer on 4 GPUs is
tests/distributed/test_kimi_k3_sm107_sdec_sp_layers.py. One Rubin GPU."""

import pytest
import torch

# (skips the module off Rubin / without cutlass, as its own tests)
from tests.models.kimi_k3.nvidia.specialized_decode.sm_107 import test_k3_sm107_sp_kda_gpu as kda
from vllm.models.kimi_k3.nvidia.ops.recoverssm import KDARecoverSSMCommitContext
from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.kda.sp import (
    SpKdaCore,
    record_kind,
)
from vllm.platforms import current_platform  # noqa: E402

# The SM107 specialized decode (specialized_decode/sm_107) serves VR200 only.
_SM107_ONLY = pytest.mark.skipif(
    not current_platform.is_device_capability((10, 7)),
    reason="the SM107 specialized decode runs on compute capability 10.7 only",
)
pytestmark = _SM107_ONLY

Q, HEADS, DIM, MAX_REQS = kda.Q, kda.HEADS, kda.DIM, kda.MAX_REQS
STEPS = 8
REQUESTS = [
    pytest.param(1, id="split-n1"),
    pytest.param(9, id="ws-n9"),
    pytest.param(16, id="wide-n16"),
    pytest.param(33, id="stream-n33"),
    pytest.param(128, id="ws-n128"),
]


def accepted_counts(n, step):
    """Accepted tokens per request (1..q, as the sampler commits them): request 0
    accepts every draft, request 1 none (n > 1), the rest at random."""
    g = torch.Generator().manual_seed(1000 + step)
    counts = torch.randint(1, Q + 1, (n,), generator=g)
    counts[0] = Q
    if n > 1:
        counts[1] = 1
    return counts.to(torch.int32).to(kda.DEVICE)


def commit(owner, caches, accepted, checkpoints, starts):
    """The stock Triton commit of ``owner``'s records into ``caches``."""
    owner.kv_cache = caches
    context = KDARecoverSSMCommitContext.create(
        [owner], spec_query_len=Q, max_num_reqs=MAX_REQS, backend="triton"
    )
    context.commit(accepted, checkpoints, starts)


def msg(prefix):
    """An ``assert_close`` message naming the step."""
    return lambda message: f"{prefix}: {message}"


def relative(got, want):
    return float((got.double() - want.double()).norm() / want.double().norm())


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("n", REQUESTS)
def test_recoverssm_chain_matches_stock(n, graph):
    assert kda.is_conv_state_dim_first()
    assert record_kind(HEADS, n) is not None, n
    rows, slots = Q * n, 2 * n + 2
    ours, stock = kda.layer(), kda.layer()  # the same weights, their own records
    kv = kda.caches(slots)
    ref_kv = tuple(t.clone() for t in kv)
    _, metadata = kda.step(n, slots)  # the requests' checkpoint slots, accepted 1
    checkpoints = metadata.spec_state_indices_tensor[:, 0].contiguous()
    starts = torch.arange(0, rows + 1, Q, dtype=torch.int32, device=kda.DEVICE)

    core = SpKdaCore(ours, (rows,), records="triton", ctx=kda.context())
    core.prepare_caches(list(kv))
    projected = torch.empty(rows, kda.PITCH, dtype=torch.bfloat16, device=kda.DEVICE)
    out = torch.empty(rows, DIM, dtype=torch.bfloat16, device=kda.DEVICE)
    caches = list(kv)

    def verify():
        core.run(projected, out, caches=caches, metadata=metadata)

    replay = None
    if graph:
        torch.cuda.synchronize()
        replay = torch.cuda.CUDAGraph()
        with torch.cuda.graph(replay):  # (captured, not run)
            verify()

    errors = []
    for step in range(STEPS):
        projected.copy_(kda.step(n, slots, seed=100 + step)[0])
        # The stock chain's verify, then the specialized one, from their own caches.
        want = kda.stock(stock, ref_kv, projected, metadata)
        if replay is not None:
            replay.replay()
        else:
            verify()
        torch.cuda.synchronize()
        where = f"step {step}"
        # The verify leaves the checkpoint; the windows hold the same raw inputs.
        kda.assert_bitwise(f"{where} conv window", kv[0], ref_kv[0])
        got_rec, want_rec = ours.recoverssm_records, stock.recoverssm_records
        got_kg, want_kg = got_rec.key_gate[:n], want_rec.key_gate[:n]
        kda.assert_bitwise(f"{where} keys", got_kg[..., :128], want_kg[..., :128])
        torch.testing.assert_close(
            got_kg[..., 128:].float(), want_kg[..., 128:].float(),
            rtol=2**-7, atol=1e-6, msg=msg(f"{where} raw gates"),
        )  # fmt: skip
        torch.testing.assert_close(
            got_rec.correction[:n], want_rec.correction[:n],
            rtol=1e-4 * (step + 1), atol=1e-5 * (step + 1),
            msg=msg(f"{where} corrections"),
        )  # fmt: skip
        torch.testing.assert_close(
            out.float(), want.float(), rtol=2e-2, atol=2e-2,
            msg=msg(f"{where} output"),
        )  # fmt: skip

        # Both commit their own records (the same accepted counts).
        accepted = accepted_counts(n, step)
        commit(ours, kv, accepted, checkpoints, starts)
        commit(stock, ref_kv, accepted, checkpoints, starts)
        torch.cuda.synchronize()
        kda.assert_bitwise(f"{where} committed conv window", kv[0], ref_kv[0])
        live = checkpoints.long()
        error = relative(kv[1][live], ref_kv[1][live])
        errors.append(error)
        # Within FP32 reduction order of the corrections, carried by every commit.
        torch.testing.assert_close(
            kv[1], ref_kv[1], rtol=1e-4 * (step + 1), atol=1e-5 * (step + 1),
            msg=msg(f"{where} committed state"),
        )  # fmt: skip
        # Slots of no request are never written.
        others = torch.ones(slots, dtype=torch.bool, device=kda.DEVICE)
        others[live] = False
        kda.assert_bitwise(f"{where} other states", kv[1][others], ref_kv[1][others])
    print(f"n={n} graph={graph}: committed state relative error per step {errors}")
    # FP32 order differences do not compound (a recurrence or record bug grows with
    # every commit): the committed states stay within 1e-4 of stock's (L2).
    assert max(errors) <= 1e-4, errors
