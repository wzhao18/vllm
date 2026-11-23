# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SP decode's KDA record cores at every DSpark verify length (q2..q8, k1..k7),
not only C512's q4: the deferred serving path (records, fold, accepted counts, a
request replaced mid-chain) and the fold-then-verify chain against the stock Triton
verify / commit, re-run from the q4 tests at another q.

Record cores per q (``cutedsl.kda.sp.RECORD_KINDS``): KdaSplit at n1-2; KdaWide at
n3-20 and, q2 / q4, KdaFlow from n21; q3 and q5..q8 KdaWide from n3 (KdaFlow
serves q2 / q4 only).
The request counts pick one core each: n1 split, n9 wide, n33 flow (q2 / q4) or
wide (q3, q5..q8). The tolerances are the q4 tests' (bitwise except KdaFlow's chunk
order)."""

import sys

import pytest

from tests.models.kimi_k3 import test_specialized_decode_sp_kda_deferred_path_gpu as path
from tests.models.kimi_k3 import test_specialized_decode_sp_kda_fold_gpu as fold
from tests.models.kimi_k3 import test_specialized_decode_sp_kda_gpu as kda
from tests.models.kimi_k3 import test_specialized_decode_sp_recoverssm_gpu as chain
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda import sp
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda.sp import record_kind
from vllm.models.kimi_k3.nvidia.specialized_decode.shapes import Unsupported

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
    flow = "flow" if q in (2, 4) else "wide"
    assert [record_kind(kda.HEADS, n, q) for n in (1, 2, 3, 20, 21, 128, 212, 213)] == [
        "split", "split", "wide", "wide", flow, flow, flow, "wide",
    ]  # fmt: skip


@pytest.mark.parametrize("n", [1, 9, 33], ids=lambda n: f"n{n}")
@pytest.mark.parametrize("verify_length", QS, indirect=True, ids=lambda q: f"q{q}")
def test_deferred_serving_path_any_q(verify_length, n):
    path.test_deferred_serving_path_matches_stock(n, True, False)


@pytest.mark.parametrize("n", [9, 33], ids=lambda n: f"n{n}")
@pytest.mark.parametrize("verify_length", QS, indirect=True, ids=lambda q: f"q{q}")
def test_fold_then_verify_any_q(verify_length, n):
    fold.test_fold_matches_stock_commit_then_verify(n, True)


@pytest.mark.parametrize("verify_length", [3, 5, 8], indirect=True, ids=lambda q: f"q{q}")
def test_kdaflow_refused_outside_q2_q4(verify_length, monkeypatch):
    """No silent KdaFlow at a verify length it does not serve: forcing it raises
    ``Unsupported`` at build (prepare_caches), naming KdaFlow."""
    q, n = verify_length, 33
    monkeypatch.setitem(sp.RECORD_KINDS, kda.HEADS, "1-:flow")
    ours = kda.layer()
    ours.deferred_recoverssm = True
    core = sp.SpKdaCore(ours, (q * n,), records="triton", ctx=kda.context())
    assert core.kinds[q * n] == "flow"
    with pytest.raises(Unsupported, match="KdaFlow"):
        core.prepare_caches(list(kda.caches(2 * n + 2)))
