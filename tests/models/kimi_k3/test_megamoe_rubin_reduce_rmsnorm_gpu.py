# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rubin MegaMoE fused top-k reduce + latent RMSNorm (vLLM-local E2a, one GPU).

The fused kernel must equal the separate BF16 top-k reduce followed by the stock vLLM
``rms_norm`` kernel: the reduce part bit for bit (same FP32 per-element top-k order,
invalid routes masked), the norm to stock precision (FP32 sum of squares in another
order and the same approximate rsqrt: at most one BF16 ulp, on a small fraction)."""

import pytest
import torch

pytest.importorskip("cutlass")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="Rubin (sm_107) MegaMoE reduce",
)

TOPK, E = 16, 896


def _cute(t, align=16):
    import cutlass.torch as cutlass_torch

    return cutlass_torch.from_dlpack(t, assumed_align=align).mark_layout_dynamic(
        leading_dim=t.dim() - 1
    )


def _compile(hidden, **kw):
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute

    from vllm.models.kimi_k3.nvidia.megamoe_rubin.src.sources.kernel_src.rubin.inference.mega.topk_reduce import (  # noqa: E501
        TopkReduce,
    )
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.src.sources.quant_def import (
        CombineFormat,
    )

    op = TopkReduce(hidden, TOPK, CombineFormat.parse("bf16"), num_experts=E)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = {k: (_cute(v) if isinstance(v, torch.Tensor) else v) for k, v in kw.items()}
    fn = cute.compile(op, combine_sf=None, topk_score=None, stream=stream, **args)
    return lambda **call: fn(
        combine_sf=None,
        topk_score=None,
        stream=stream,
        **{k: (_cute(v) if isinstance(v, torch.Tensor) else v) for k, v in call.items()},
    )


@pytest.mark.parametrize(("n", "hidden"), [(1, 3584), (96, 3584), (128, 3584), (7, 7168)])
def test_fused_reduce_rmsnorm_matches_reduce_then_stock_norm(n, hidden):
    from vllm import _custom_ops as ops

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(n * 7 + hidden)
    combine = (torch.randn((n, TOPK, hidden), device=dev, generator=g) * 0.5).to(
        torch.bfloat16
    )
    ids = torch.randint(0, E, (n, TOPK), device=dev, generator=g, dtype=torch.int32)
    ids[:, -3:] = -1  # invalid routes never write their slot: stale data must be masked
    combine[:, -3:] = float("nan")
    weight = (1 + 0.1 * torch.randn(hidden, device=dev, generator=g)).to(torch.bfloat16)
    eps = 1e-6

    reduced = torch.empty((n, hidden), dtype=torch.bfloat16, device=dev)
    plain = _compile(hidden, combine_quant=combine, reduced_output=reduced, topk_indices=ids)
    plain(combine_quant=combine, reduced_output=reduced, topk_indices=ids)
    fused_out = torch.empty_like(reduced)
    fused = _compile(
        hidden,
        combine_quant=combine,
        reduced_output=fused_out,
        topk_indices=ids,
        rmsnorm_weight=weight,
        rmsnorm_eps=eps,
    )
    fused(
        combine_quant=combine,
        reduced_output=fused_out,
        topk_indices=ids,
        rmsnorm_weight=weight,
        rmsnorm_eps=eps,
    )
    torch.cuda.synchronize()

    # reduce part: the separate kernel equals the FP32 top-k sum over valid routes
    expect = combine[:, : TOPK - 3].float()
    acc = expect[:, 0].clone()
    for k in range(1, TOPK - 3):
        acc = acc + expect[:, k]
    assert torch.equal(reduced, acc.to(torch.bfloat16))

    stock = torch.empty_like(reduced)
    ops.rms_norm(stock, reduced, weight, eps)
    torch.cuda.synchronize()
    a = fused_out.view(torch.int16).int()
    b = stock.view(torch.int16).int()
    ulp = (a - b).abs()
    assert torch.isfinite(fused_out.float()).all()
    assert int(ulp.max()) <= 1, f"max {int(ulp.max())} BF16 ulp vs stock rms_norm"
    assert float((ulp > 0).float().mean()) < 0.01
