# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SP decode's KDA record cores at every DSpark verify length (q2..q8, k1..k7),
not only C512's q4: the deferred serving path (records, fold, accepted counts, a
request replaced mid-chain) and the fold-then-verify chain against the stock Triton
verify / commit, re-run from the q4 tests at another q.

Record cores per q (``cutedsl.kda.sp.RECORD_KINDS``): KdaSplit at n1-2, KdaWide from
n3, KdaFlow from n21 (q2..q4) / n17 (q5..q8) at every q (q3 with padded scan vectors; q5..q8 with block-sum
rounds, the two-half scan and the two-slot page ring).
The request counts pick one core each: n1 split, n9 wide, n33 / n64 flow; plus each
q's largest KdaFlow item build at the SP row cap (q5 n192: 24 items per CTA, q6
n160: 20, q7 n136 / q8 n121: 16). The tolerances are the q4 tests' (bitwise except
KdaFlow's chunk order)."""

import sys

import pytest

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
    # q5..q8 leave KdaWide at its second CTA wave (n17), q2..q4 at n21.
    first = 17 if q > 4 else 21
    assert [record_kind(kda.HEADS, n, q) for n in (1, 2, 3, 16, first - 1, first, 128, 212, 213)] == [
        "split", "split", "wide", "wide", "wide", "flow", "flow", "flow", "wide",
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
