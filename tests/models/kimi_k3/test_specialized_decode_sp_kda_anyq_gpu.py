# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SP decode's KDA record cores at every DSpark verify length (q2..q8, k1..k7),
not only C512's q4: the deferred serving path (records, fold, accepted counts, a
request replaced mid-chain) and the fold-then-verify chain against the stock Triton
verify / commit, re-run from the q4 tests at another q.

Record cores per q (``cutedsl.kda.sp.RECORD_KINDS``): KdaSplit at n1-2, KdaWide from
n3, KdaFlow from n21 (q2..q4) / n18 (q5..q8) at every q (q3 with padded scan vectors; q5..q8 with block-sum
rounds, the two-half scan and the two-slot page ring).
The request counts pick one core each: n1 split, n9 wide, n33 / n64 flow; plus each
q's largest KdaFlow item build at the SP row cap (q5 n192: 24 items per CTA, q6
n160: 20, q7 n136 / q8 n121: 16). The tolerances are the q4 tests' (bitwise except
KdaFlow's chunk order)."""

import sys

import pytest
import torch

from tests.models.kimi_k3 import test_specialized_decode_sp_kda_deferred_path_gpu as path
from tests.models.kimi_k3 import test_specialized_decode_sp_kda_fold_gpu as fold
from tests.models.kimi_k3 import test_specialized_decode_sp_kda_gpu as kda
from tests.models.kimi_k3 import test_specialized_decode_sp_recoverssm_gpu as chain
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda.sp import record_kind

QS = [2, 3, 5, 6, 7, 8]


@pytest.fixture
def verify_length(monkeypatch, request):
    """Re-run the q4 tests' helpers at ``q``: every module's Q."""
    q = request.param
    for module in (kda, path, fold, chain, sys.modules[__name__]):
        if hasattr(module, "Q"):
            monkeypatch.setattr(module, "Q", q)
    return q


@pytest.mark.parametrize("verify_length", QS, indirect=True, ids=lambda q: f"q{q}")
def test_record_kind_per_q(verify_length):
    q = verify_length
    # q5..q8 run KdaFlow from n18 (KdaWide's second CTA wave), q2..q4 from n21.
    first = 18 if q > 4 else 21
    assert [record_kind(kda.HEADS, n, q) for n in (1, 2, 3, first - 1, first, 128, 212, 213)] == [
        "split", "split", "wide", "wide", "flow", "flow", "flow", "wide",
    ]  # fmt: skip


@pytest.mark.parametrize("n", [1, 9, 33, 64], ids=lambda n: f"n{n}")
@pytest.mark.parametrize("verify_length", QS, indirect=True, ids=lambda q: f"q{q}")
def test_deferred_serving_path_any_q(verify_length, n):
    path.test_deferred_serving_path_matches_stock(n, True, False)


# The largest KdaFlow item build each q needs at its SP row cap (SDECK: R <= 968,
# lcm(q, 4) alignment): items per CTA = ceil(24 n / 212).
CAP_N = {2: 212, 3: 212, 5: 192, 6: 160, 7: 136, 8: 121}


@pytest.mark.parametrize("verify_length", QS, indirect=True, ids=lambda q: f"q{q}")
def test_deferred_serving_path_flow_cap(verify_length):
    path.test_deferred_serving_path_matches_stock(CAP_N[verify_length], True, False)


@pytest.mark.parametrize("n", [9, 33], ids=lambda n: f"n{n}")
@pytest.mark.parametrize("verify_length", QS, indirect=True, ids=lambda q: f"q{q}")
def test_fold_then_verify_any_q(verify_length, n):
    fold.test_fold_matches_stock_commit_then_verify(n, True)


@pytest.mark.parametrize("verify_length", QS, indirect=True, ids=lambda q: f"q{q}")
def test_kdaflow_builds_at_every_q(verify_length):
    """KdaFlow builds at every DSpark q (no refusal) and keeps refusing a verify
    length it does not serve (q9): ``Unsupported`` at build, naming KdaFlow."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda import flow

    q = verify_length
    assert q in flow.FLOW_TOKENS
    assert 9 not in flow.FLOW_TOKENS


@pytest.mark.parametrize("verify_length", [2, 3, 5, 6, 7, 8], indirect=True,
                         ids=lambda q: f"q{q}")  # fmt: skip
def test_kdaflow_fold_bitwise_vs_kdawide(verify_length, monkeypatch):
    """The deferred fold of KdaFlow vs KdaWide on the same pending records (0..q
    tokens per request), DCRIT r124 d. Requests with 0 or 1 pending tokens fold bit
    for bit at every q: the recorded-key norm (at q5..q8 from KdaFlow's 32-value
    block-sum rounds) and the single-token commit are KdaWide's. With >= 2 pending
    tokens the multi-token chain associates differently (~1 ulp; the same rates at
    q2 / q4, whose KdaFlow code predates q5..q8): FP32 tolerance. The verify
    (outputs, new corrections) differs in FP32 order and is not compared."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda import sp
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda.sp import SpKdaCore

    q, n = verify_length, 33
    rows, slots = q * n, 2 * n + 2
    kv0 = kda.caches(slots)
    projected, metadata = kda.step(n, slots)
    record_rows = torch.randperm(kda.MAX_REQS, generator=torch.Generator().manual_seed(4))
    record_rows = record_rows[:n].to(torch.int32).to(kda.DEVICE)
    g = torch.Generator().manual_seed(5)
    pending = torch.zeros(kda.MAX_REQS, dtype=torch.int32)
    pending[record_rows.cpu().long()] = torch.randint(0, q + 1, (n,), generator=g,
                                                       dtype=torch.int32)  # fmt: skip
    pending = pending.to(kda.DEVICE)
    records = None

    def no_flush():
        raise AssertionError("a folding core flushed")

    checkpoints = {}
    for kind in ("wide", "flow"):
        monkeypatch.setitem(sp.RECORD_KINDS, kda.HEADS, {"*": f"1-:{kind}"})
        layer = kda.layer()
        layer.deferred_recoverssm = True
        mine = layer.recoverssm_records
        if records is None:  # one random set of pending records for both cores
            gd = torch.Generator(device=kda.DEVICE).manual_seed(6)
            records = (
                torch.randn(mine.key_gate.shape, generator=gd, device=kda.DEVICE)
                .to(mine.key_gate.dtype),
                torch.randn(mine.correction.shape, generator=gd, device=kda.DEVICE)
                * 0.1,
            )
        mine.key_gate.copy_(records[0])
        mine.correction.copy_(records[1])
        kv = tuple(t.clone() for t in kv0)
        core = SpKdaCore(layer, (rows,), records="deferred", ctx=kda.context())
        assert core.kinds[rows] == kind
        core.prepare_caches(list(kv))
        deferred = kda.Module(
            **vars(metadata),
            recoverssm_deferred=kda.Module(
                record_rows=record_rows, pending=pending, flush=no_flush
            ),
        )
        out = torch.empty(rows, kda.DIM, dtype=torch.bfloat16, device=kda.DEVICE)
        core.run(projected, out, caches=list(kv), metadata=deferred)
        torch.cuda.synchronize()
        checkpoints[kind] = kv[1].clone()
    live = metadata.spec_state_indices_tensor[:, 0].long()
    count = pending[record_rows.long()]
    single = live[count <= 1]
    assert single.numel() > 0 and int((count >= 2).sum()) > 0
    kda.assert_bitwise(f"q{q} folded checkpoints KdaFlow vs KdaWide (<= 1 pending)",
                       checkpoints["flow"][single], checkpoints["wide"][single])  # fmt: skip
    torch.testing.assert_close(checkpoints["flow"], checkpoints["wide"], rtol=1e-5,
                               atol=1e-7, msg=f"q{q} folded checkpoints (>= 2 pending)")  # fmt: skip


@pytest.fixture
def conv32(monkeypatch):
    """FP32 KDA conv state (the stock-precision rule: --mamba-cache-dtype float32):
    every helper's caches() returns the conv window in FP32."""
    real = kda.caches

    def caches(slots, seed=1):
        conv, state = real(slots, seed)
        return conv.float(), state

    monkeypatch.setattr(kda, "caches", caches)
    return True


# n33 (every core kind's middle) and each q's largest KdaFlow build at the SP row cap;
# at FP32 conv the q4 20/24-item and q7 / q8 builds run one conv window buffer.
CAP_N32 = {**CAP_N, 4: 200}


@pytest.mark.parametrize("n", ["n33", "cap"])
@pytest.mark.parametrize("verify_length", [2, 3, 4, 5, 6, 7, 8], indirect=True,
                         ids=lambda q: f"q{q}")  # fmt: skip
def test_deferred_serving_path_fp32_conv(verify_length, n, conv32):
    requests = 33 if n == "n33" else CAP_N32[verify_length]
    path.test_deferred_serving_path_matches_stock(requests, True, False)


@pytest.mark.parametrize("verify_length", [4, 8], indirect=True, ids=lambda q: f"q{q}")
def test_fold_then_verify_fp32_conv(verify_length, conv32):
    fold.test_fold_matches_stock_commit_then_verify(33, True)
