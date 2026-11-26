# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill MegaMoE sub-calls (specialized_prefill.moe).

CPU: the policy (TP1 wide-EP Rubin MegaMoE with more rows than one sub-call), the
EP-uniform sub-call plan from the DP batch sizes, and the wrapper's calls (row
slices, padding slices, output views, one capture of the whole batch, empty slices
on ranks with fewer rows). GPU (SM107): the Rubin MegaMoE output of a row does not
depend on which launch carries it, so sub-calls equal one launch bit for bit.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.config.kernel import KIMI_K3_MEGA_MOE
from vllm.models.kimi_k3.nvidia.specialized_prefill import moe as P
from vllm.platforms import current_platform

C = P.PREFILL_MOE_SUBCALL_TOKENS


def _cfg(backend=KIMI_K3_MEGA_MOE):
    return SimpleNamespace(kernel_config=SimpleNamespace(moe_backend=backend))


def _policy(mnbt=16384, tp=1, sp=False, backend=KIMI_K3_MEGA_MOE):
    return P.prefill_moe_policy(
        _cfg(backend), tp_size=tp, sequence_parallel=sp, max_num_tokens=mnbt,
    )  # fmt: skip


def test_policy_applies_to_tp1_prefill_batches_only():
    p = _policy()
    assert p == P.PrefillMoEPolicy(capacity=C, max_num_tokens=16384) and p.subcalls
    assert _policy(mnbt=32768).capacity == C
    for kw in (
        dict(mnbt=C), dict(tp=4), dict(sp=True),
        dict(backend="deep_gemm_mega_moe"),
    ):  # fmt: skip
        assert _policy(**kw) is None, kw


@pytest.mark.parametrize(
    ("counts", "plan"),
    [
        ([16384] * 4, (2, (C, C))),
        ([16384, 9000, 0, 100], (2, (4121, 2250))),
        ([5000, 5000], (1, (5000,))),
        ([0, 0], (1, (0,))),
        ([24577], (4, (C, C, C, 1))),
    ],
)
def test_subcall_plan_is_ep_uniform(counts, plan):
    assert P._subcall_rows(counts, C) == plan


class _FakeExperts:
    """Stands in for KimiK3MegaMoEExperts: per-row output = f(row), records calls."""

    def __init__(self, capacity, hidden=8):
        self.max_num_tokens = capacity
        self.hidden_size = hidden
        self.prefix = "model.layers.3.mlp.experts"
        self.capture_fn = None
        self.calls = []

    @staticmethod
    def rows(x, w, ids):
        return x * 2 + w.sum(1, keepdim=True) + ids.sum(1, keepdim=True)

    def __call__(self, x, w, ids, is_padding=None, out=None, ep_tokens=None):
        assert x.shape[0] <= self.max_num_tokens
        assert self.capture_fn is None or out is None
        self.calls.append((x.shape[0], ep_tokens, is_padding))
        y = self.rows(x, w, ids)
        if out is None:
            return y
        out.copy_(y)
        return out


def _ctx(monkeypatch, counts, is_padding=None):
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor(counts)),
        is_padding=is_padding,
    )
    monkeypatch.setattr(P, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(P, "get_forward_context", lambda: ctx)
    return ctx


@pytest.mark.parametrize("skip_padding", [False, True])
def test_wrapper_splits_rows_and_matches_one_call(monkeypatch, skip_padding):
    cap = 16
    counts = [40, 23, 0, 33]  # 3 sub-calls on every rank
    n = 23
    pad = torch.zeros(64, dtype=torch.bool)
    pad[20:] = True
    _ctx(monkeypatch, counts, pad if skip_padding else None)
    monkeypatch.setattr(P.envs, "VLLM_MOE_SKIP_PADDING", skip_padding)
    experts = _FakeExperts(cap)
    captured = []
    experts.capture_fn = captured.append
    run = P.PrefillSubcallExperts(experts, cap, dp_size=4)
    g = torch.Generator().manual_seed(0)
    x = torch.randn((n, 8), generator=g)
    w = torch.rand((n, 4), generator=g)
    ids = torch.randint(0, 64, (n, 4), generator=g)
    y = run(x, w, ids)
    torch.testing.assert_close(y, _FakeExperts.rows(x, w, ids), rtol=0, atol=0)
    assert [c[0] for c in experts.calls] == [16, 7, 0]  # the EP-uniform count
    assert [c[1] for c in experts.calls] == list(P._subcall_rows(counts, cap)[1])
    if skip_padding:
        assert [c[2].tolist() for c in experts.calls] == [
            pad[:16].tolist(), pad[16:23].tolist(), []
        ]
    else:
        assert all(c[2] is None for c in experts.calls)
    assert len(captured) == 1 and captured[0] is ids
    assert experts.capture_fn is not None  # restored


def test_wrapper_single_subcall_is_the_stock_call(monkeypatch):
    _ctx(monkeypatch, [10, 16])
    experts = _FakeExperts(16)
    run = P.PrefillSubcallExperts(experts, 16, dp_size=2)
    x, w, ids = torch.ones((10, 8)), torch.ones((10, 4)), torch.ones((10, 4))
    run(x, w, ids)
    assert experts.calls == [(10, None, None)]  # no out / ep_tokens override


def test_wrapper_needs_dp_metadata(monkeypatch):
    ctx = _ctx(monkeypatch, [1])
    ctx.dp_metadata = None
    run = P.PrefillSubcallExperts(_FakeExperts(16), 16, dp_size=2)
    with pytest.raises(RuntimeError, match="DP metadata"):
        run(torch.ones((3, 8)), torch.ones((3, 4)), torch.ones((3, 4)))


def test_wrapper_rejects_mismatched_capacity():
    with pytest.raises(ValueError, match="capacity"):
        P.PrefillSubcallExperts(_FakeExperts(32), 16, dp_size=1)


@pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.is_device_capability((10, 7))),
    reason="Rubin MegaMoE kernels need SM107",
)
def test_rubin_megamoe_rows_are_launch_independent():
    """One launch of 512 rows == the same rows in three uneven launches (bitwise):
    the property that makes 2 x 8K sub-calls equal one 16K launch."""
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import weights as W
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.config import (
        RubinBucket,
        RubinKnobs,
        RubinMoeProblem,
    )
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.runtime import RubinMegaMoE

    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)
    torch.zeros(1, device=dev)  # the CuTe DSL host queries need a current context
    problem = RubinMoeProblem(num_experts=32, quant_kind="nvfp4")
    knobs = RubinKnobs.from_dict(
        {
            "mma_tiler_mnk": [256, 128, 256], "cluster_shape_mn": [2, 1],
            "fallback_cluster_shape_mn": None, "schedule_policy": ["grouped", None],
            "work_id_mode": "grid_stride", "fc2_use_bulk": True, "fc2_tma_stages": 2,
            "epi_flag_batches": [1, 4], "token_in_flag_batch": 1,
            "token_back_mode": "epi_warps", "reduce_topk_in_kernel": False,
        }
    )  # fmt: skip
    cap, n = 512, 512
    bucket = RubinBucket("inference", cap, knobs)
    moe = RubinMegaMoE(problem, None, [bucket])
    try:
        e, h, i = problem.num_experts, problem.hidden, problem.intermediate
        g = torch.Generator(device=dev).manual_seed(0)
        legs = []
        for rows, cols in ((2 * i, h), (h, i)):
            codes, scales = [], []
            for _ in range(e):
                c, s = W.quantize_mxfp4_block32(
                    torch.randn((rows, cols), generator=g, device=dev) * cols**-0.5
                )
                codes.append(c)
                scales.append(s)
            legs += [torch.stack(codes), torch.stack(scales)]
        weights = moe.prepare_weights(*legs, layer_id=0)
        moe.warmup(weights, ids_dtypes=(torch.int32,), weight_dtypes=(torch.float32,))
        x = torch.randn((n, h), generator=g, device=dev).to(torch.bfloat16)
        ids = (
            torch.rand((n, e), generator=g, device=dev)
            .topk(problem.top_k, dim=1)
            .indices.to(torch.int32)
        )
        wts = torch.rand((n, problem.top_k), generator=g, device=dev) + 0.05
        wts = (wts / wts.sum(1, keepdim=True)).float()
        one = torch.empty((n, h), dtype=torch.bfloat16, device=dev)
        moe.forward(x, ids, wts, weights, bucket=bucket, out=one)
        split = torch.empty_like(one)
        for lo, hi in ((0, 100), (100, 357), (357, n)):
            moe.forward(x[lo:hi], ids[lo:hi], wts[lo:hi], weights, bucket=bucket,
                        out=split[lo:hi], ep_tokens=hi - lo)  # fmt: skip
        torch.cuda.synchronize()
        moe.check_route_errors()
        assert torch.equal(one.view(torch.int16), split.view(torch.int16))
    finally:
        moe.destroy()
