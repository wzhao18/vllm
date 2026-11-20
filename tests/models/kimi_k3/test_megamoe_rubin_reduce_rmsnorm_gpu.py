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


def _compile(hidden, polled=False, norm_split=1, **kw):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    if polled:  # GenPhase (src_gp): the self-validating combine plane
        from vllm.models.kimi_k3.nvidia.megamoe_rubin.src_gp.sources.helpers.topk_reduce import (  # noqa: E501
            TopkReduce,
        )
        from vllm.models.kimi_k3.nvidia.megamoe_rubin.src_gp.sources.quant_def import (
            CombineFormat,
        )

        op = TopkReduce(
            hidden,
            TOPK,
            CombineFormat.parse("bf16"),
            num_experts=E,
            ll_encoded=True,
            norm_split=norm_split,
        )
    else:
        from vllm.models.kimi_k3.nvidia.megamoe_rubin.src.sources.kernel_src.rubin.inference.mega.topk_reduce import (  # noqa: E501
            TopkReduce,
        )
        from vllm.models.kimi_k3.nvidia.megamoe_rubin.src.sources.quant_def import (
            CombineFormat,
        )

        op = TopkReduce(hidden, TOPK, CombineFormat.parse("bf16"), num_experts=E)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    def conv(v):
        if isinstance(v, torch.Tensor):
            return _cute(v)
        return cutlass.Float32(v) if isinstance(v, float) else v

    args = {k: conv(v) for k, v in kw.items()}
    fn = cute.compile(op, combine_sf=None, topk_score=None, stream=stream, **args)
    return lambda **call: fn(
        combine_sf=None,
        topk_score=None,
        stream=stream,
        **{k: conv(v) for k, v in call.items()},
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


@pytest.mark.parametrize("split", [1, 2, 4])
@pytest.mark.parametrize(("n", "hidden"), [(1, 3584), (96, 3584), (128, 3584)])
def test_polled_fused_reduce_rmsnorm_matches_polled_reduce_then_stock_norm(
    n, hidden, split
):
    """GenPhase (src_gp) polled plane: cells hold ~bf16 bits (0 = not landed) and
    dropped routes never land. The fused kernel equals the polled reduce + stock
    rms_norm, and clears the landed cells as the polled reduce does, also with a
    token row spread over a cluster of ``split`` CTAs."""
    from vllm import _custom_ops as ops

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(n * 11 + hidden)
    values = (torch.randn((n, TOPK, hidden), device=dev, generator=g) * 0.5).to(
        torch.bfloat16
    )
    ids = torch.randint(0, E, (n, TOPK), device=dev, generator=g, dtype=torch.int32)
    ids[:, -3:] = -1
    plane = (~values.view(torch.int16)).view(torch.bfloat16)
    plane[:, -3:] = 0
    weight = (1 + 0.1 * torch.randn(hidden, device=dev, generator=g)).to(torch.bfloat16)
    eps = 1e-6

    reduced = torch.empty((n, hidden), dtype=torch.bfloat16, device=dev)
    plain_plane = plane.clone()
    kw = dict(combine_quant=plain_plane, reduced_output=reduced, topk_indices=ids)
    _compile(hidden, True, **kw)(**kw)
    fused_plane = plane.clone()
    fused_out = torch.empty_like(reduced)
    kw = dict(
        combine_quant=fused_plane,
        reduced_output=fused_out,
        topk_indices=ids,
        rmsnorm_weight=weight,
        rmsnorm_eps=eps,
    )
    _compile(hidden, True, norm_split=split, **kw)(**kw)
    torch.cuda.synchronize()

    assert not fused_plane.view(torch.int16).any(), "landed cells must be cleared"
    expect = values[:, : TOPK - 3].float()
    acc = expect[:, 0].clone()
    for k in range(1, TOPK - 3):
        acc = acc + expect[:, k]
    assert torch.equal(reduced, acc.to(torch.bfloat16))
    stock = torch.empty_like(reduced)
    ops.rms_norm(stock, reduced, weight, eps)
    torch.cuda.synchronize()
    ulp = (fused_out.view(torch.int16).int() - stock.view(torch.int16).int()).abs()
    assert torch.isfinite(fused_out.float()).all()
    assert int(ulp.max()) <= 1, f"max {int(ulp.max())} BF16 ulp vs stock rms_norm"
    assert float((ulp > 0).float().mean()) < 0.01


@pytest.mark.parametrize("world", [4, 16])
@pytest.mark.parametrize(("n", "hidden"), [(1, 3584), (333, 3584), (64, 7168)])
def test_dedup_fused_reduce_rmsnorm_matches_dedup_reduce_then_stock_norm(
    n, hidden, world
):
    """Prefill dedup combine (rank slots + valid-rank mask): the fused reduce +
    RMSNorm equals the dedup reduce (bit for bit, ascending ranks) followed by the
    stock vLLM ``rms_norm`` to stock precision."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from vllm import _custom_ops as ops
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.src_gp.sources.kernel_src.rubin.inference.mega.topk_reduce_dedup import (  # noqa: E501
        TopkReduceDedup,
    )
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.src_gp.sources.quant_def import (
        CombineFormat,
    )

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(n * 13 + hidden + world)
    slots = (torch.randn((n, world, hidden), device=dev, generator=g) * 0.5).to(
        torch.bfloat16
    )
    mask = torch.randint(1, 1 << world, (n,), device=dev, generator=g, dtype=torch.int32)
    valid = ((mask[:, None] >> torch.arange(world, device=dev)) & 1).bool()
    slots[~valid] = float("nan")  # ranks outside the mask never wrote their slot
    weight = (1 + 0.1 * torch.randn(hidden, device=dev, generator=g)).to(torch.bfloat16)
    eps = 1e-6
    op = TopkReduceDedup(hidden, world, CombineFormat.parse("bf16"))
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    reduced = torch.empty((n, hidden), dtype=torch.bfloat16, device=dev)
    fused_out = torch.empty_like(reduced)
    plain = cute.compile(op, _cute(slots), _cute(mask), _cute(reduced), stream)
    plain(_cute(slots), _cute(mask), _cute(reduced), stream)
    fused = cute.compile(
        op,
        _cute(slots),
        _cute(mask),
        _cute(fused_out),
        stream,
        rmsnorm_weight=_cute(weight),
        rmsnorm_eps=cutlass.Float32(eps),
    )
    fused(
        _cute(slots),
        _cute(mask),
        _cute(fused_out),
        stream,
        rmsnorm_weight=_cute(weight),
        rmsnorm_eps=cutlass.Float32(eps),
    )
    torch.cuda.synchronize()
    acc = torch.zeros((n, hidden), dtype=torch.float32, device=dev)
    for r in range(world):
        acc = torch.where(valid[:, r : r + 1], acc + slots[:, r].float(), acc)
    assert torch.equal(reduced, acc.to(torch.bfloat16))
    stock = torch.empty_like(reduced)
    ops.rms_norm(stock, reduced, weight, eps)
    torch.cuda.synchronize()
    ulp = (fused_out.view(torch.int16).int() - stock.view(torch.int16).int()).abs()
    assert torch.isfinite(fused_out.float()).all()
    assert int(ulp.max()) <= 1, f"max {int(ulp.max())} BF16 ulp vs stock rms_norm"
    assert float((ulp > 0).float().mean()) < 0.01
