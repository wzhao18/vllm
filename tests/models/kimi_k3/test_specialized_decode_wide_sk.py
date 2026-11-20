# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The die-local persistent stream-K token-wide GEMM (``cutedsl.gemm.wide_sk``, the
DSpark drafter's LM head) against the stock BF16 linear's precision: one FP32 sum per
output (K order may differ: the stream-K split adds the later part's partial after the
head's sum) rounded once, so at most one BF16 ulp from an FP32 reference. Repeated
launches must be bitwise identical (deterministic reduction order; the slot claim and
partial flags re-arm for graph replays), with the weight die-homed or where it lies."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(100)),
    reason="tcgen05 2-CTA GEMM targets SM100+",
)


def _ulps(got, ref):
    """Max error in BF16 ulps of the reference, with an FP32-reorder allowance of
    1e-4 of the tensor's magnitude for outputs near cancellation."""
    ref = ref.float()
    ulp = (ref.abs().clamp_min(1e-30).log2().floor() - 7).exp2()
    slack = (got.float() - ref).abs() - 1e-4 * ref.abs().amax()
    return (slack.clamp_min(0) / ulp).max().item()


@pytest.mark.parametrize("place", [True, False])
@pytest.mark.parametrize("rows", [32, 96, 192, 288])
def test_wide_sk(rows, place):
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm import wide_sk

    try:
        _, pairs = wide_sk.die_pairs("cuda")
    except ValueError:
        pytest.skip("die-local GEMM needs a two-die device")
    n, k = 40960, 1024  # the TP4 LM-head shard's N; a short K keeps the test quick
    if not wide_sk.supports(rows, n, k, pairs):
        pytest.skip(f"shape not supported at {pairs} pairs per die")
    torch.manual_seed(rows)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    gemm = wide_sk.PreparedWideSK(w, [rows], place=place).prepare()
    out = torch.empty(rows, n, device="cuda", dtype=torch.bfloat16)
    ref = (x.float() @ w.float().t()).bfloat16()
    first = None
    for _ in range(3):
        out.fill_(float("nan"))
        gemm.run(x, out)
        torch.cuda.synchronize()
        assert _ulps(out, ref) <= 1
        if first is None:
            first = out.clone()
        else:
            assert torch.equal(out, first)


def _pairs_or_skip(wide_sk, rows, n, k):
    try:
        _, pairs = wide_sk.die_pairs("cuda")
    except ValueError:
        pytest.skip("die-local GEMM needs a two-die device")
    if not wide_sk.supports(rows, n, k, pairs, spread=True):
        pytest.skip(f"shape not supported at {pairs} pairs per die")
    return pairs


@pytest.mark.parametrize("rows", [96, 192, 288])
@pytest.mark.parametrize("n,k", [(2048, 1024), (7168, 1536)])
def test_wide_sk_spread(rows, n, k):
    """Many ranges per tile (qkv-like: ~14, o-like: ~5): K-order sums of every range's
    partial, at most one BF16 ulp from FP32, bitwise repeatable."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm import wide_sk

    _pairs_or_skip(wide_sk, rows, n, k)
    torch.manual_seed(rows + n)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    gemm = wide_sk.PreparedWideSK(w, [rows], spread=True).prepare()
    out = torch.empty(rows, n, device="cuda", dtype=torch.bfloat16)
    ref = (x.float() @ w.float().t()).bfloat16()
    first = None
    for _ in range(3):
        out.fill_(float("nan"))
        gemm.run(x, out)
        torch.cuda.synchronize()
        assert _ulps(out, ref) <= 1
        if first is None:
            first = out.clone()
        else:
            assert torch.equal(out, first)


@pytest.mark.parametrize("rows", [96, 288])
def test_wide_sk_silu(rows):
    """gate/up with silu_and_mul in the epilogue: bitwise the same kernel's BF16 gate/up
    output (same tiles, same K order) followed by vLLM's silu_and_mul."""
    import vllm._custom_ops  # noqa: F401
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm import wide_sk

    n, k = 7168, 1024  # the TP4 gate/up shard's N; a short K keeps the test quick
    _pairs_or_skip(wide_sk, rows, n, k)
    torch.manual_seed(rows)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    fused = wide_sk.PreparedWideSK(w, [rows], spread=True, silu=True).prepare()
    plain = wide_sk.PreparedWideSK(wide_sk.silu_rows(w), [rows], spread=True).prepare()
    act = torch.empty(rows, n // 2, device="cuda", dtype=torch.bfloat16)
    h = torch.empty(rows, n, device="cuda", dtype=torch.bfloat16)
    plain.run(x, h)
    h = h.view(rows, n // 128, 2, 64).transpose(1, 2).reshape(rows, n)  # [gate; up]
    ref = torch.empty_like(act)
    torch.ops._C.silu_and_mul(ref, h)
    for _ in range(2):
        act.fill_(float("nan"))
        fused.run(x, act)
        torch.cuda.synchronize()
        assert torch.equal(act, ref)
