# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 SP decode MegaMoE MLP (``runtime.sp_mlp.SpMlp``) on Rubin (compute
capability 10.7), against the stock ``KimiMoE.forward`` (and ``KimiMLP.forward``,
``KimiRoutedOutputTransform.forward``) run on the same stand-in layer: the real
stock methods bound to a layer whose linears are BF16 [N, K] weights at the TP4
per-rank shapes (latent 3584, shared experts 2 x 3072, hidden 7168), whose router
and experts are deterministic stand-ins (the real ones need the EP group):

* every GEMM on "cublas" (the stock kernel): bitwise equal to stock, on both
  stream modes (shared expert on the side stream or after the experts);
* ``sp_policy``'s measured tactics at shard rows covering every kind of the tables
  (warp, split-K, cuBLAS, cuBLASLt): as close to a reference of FP32 GEMMs each
  rounded once to BF16 (the same stock router / experts / norm / activation) as the
  stock layer is (relative L2, within 2x of stock's own distance);
* a stock layer this path cannot reproduce (a quantized linear, an online-MXFP8
  input) raises ``Unsupported`` at build.

One GPU; under a minute with a warm CuTe cache."""

import types

import pytest
import torch

pytest.importorskip("cutlass")

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("needs a Rubin GPU (compute capability 10.7)", allow_module_level=True)

from vllm.model_executor.layers.linear import UnquantizedLinearMethod  # noqa: E402
from vllm.models.kimi_k3.nvidia import model as kimi  # noqa: E402
from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import sp_mlp  # noqa: E402
from vllm.models.kimi_k3.nvidia.specialized_decode.shapes import (  # noqa: E402
    Context,
    Unsupported,
)

DEVICE = torch.device("cuda", 0)
HIDDEN, LATENT, INTER = 7168, 3584, 6144
# Shard rows n: every kind of the measured tables (sp_policy TUNED_KIND), the split-K
# 80-column tiles partial (72) and full (80) and a partial 96-column tile (88); the
# split-K front's 64-row tile from 33 (36); latent up partials at 128 / 160 / 192.
ROWS = (1, 4, 16, 20, 32, 36, 50, 64, 72, 80, 88, 96, 128, 160, 192)


def context():
    torch.cuda.set_device(DEVICE)
    sms = torch.cuda.get_device_properties(DEVICE).multi_processor_count
    return Context(
        arch="rubin", tp=4, rank=0, dcp=4, sms=sms, q=4, max_rows=512,
        max_context=131072, bf16_state=False, dp=4, ep=16, sequence_parallel=True,
        heads_per_rank=24,
    )  # fmt: skip


class Linear:
    """A stock BF16 linear's surface: ``weight``, ``quant_method``, ``__call__``."""

    def __init__(self, n, k, g):
        self.weight = (torch.randn(n, k, device=DEVICE, generator=g) * 0.02).bfloat16()
        self.quant_method = UnquantizedLinearMethod()
        self.bias = None

    def __call__(self, x):
        return torch.nn.functional.linear(x, self.weight), None


class Norm:
    """RMSNorm stand-in with the fields ``_norm_mxfp8`` reads."""

    has_weight, variance_size_override, variance_epsilon = True, None, 1e-6

    def __init__(self, g):
        self.weight = 1 + 0.1 * torch.randn(LATENT, device=DEVICE, generator=g)

    def __call__(self, x):
        v = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(v + self.variance_epsilon) * self.weight).to(
            x.dtype
        )


def silu_and_mul(x):
    """The shared expert's gated activation (a plain function: the stock CustomOp
    needs a vLLM config; the MLP calls whatever ``act_fn`` the layer has)."""
    d = x.shape[-1] // 2
    return (torch.nn.functional.silu(x[..., :d].float()) * x[..., d:].float()).to(
        x.dtype
    )


class Shared(types.SimpleNamespace):
    """The shared expert: the stock ``KimiMLP`` methods bound to stand-in fields."""

    _prefill_situ = None  # KimiMLP class default (prefill SiTU epilogue, unset in decode)
    gate_up_striped = False  # KimiMLP class default (no fused-shared stripes here)

    def __call__(self, x):
        return self.forward(x)


def layer(overlap, seed=0):
    """A stand-in KimiMoE with the stock forward and helpers bound to it."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    shared = Shared(
        gate_up_proj=Linear(2 * INTER, HIDDEN, g),
        down_proj=Linear(HIDDEN, INTER, g),
        act_fn=silu_and_mul,
        shard_sequence_parallel=False,
        gemm_rs_ar=None,
    )
    for name in ("forward", "_gate_up", "_act_mxfp8", "_act_mxfp8_ok", "mxfp8_input"):
        setattr(shared, name, types.MethodType(getattr(kimi.KimiMLP, name), shared))
    router_w = torch.randn(64, HIDDEN, device=DEVICE, generator=g).bfloat16()

    def router(h):  # stand-in for the grouped top-k: (weights, ids)
        probs = (h.float() @ router_w.float().t()).softmax(-1)
        return probs, probs.argmax(-1, keepdim=True)

    mlp = types.SimpleNamespace(
        use_mega_moe=True, moe_token_shard=0, use_latent_moe=True, _packed_moe=None,
        routed_expert_down_proj=Linear(LATENT, HIDDEN, g), shared_experts=shared,
        _shared_overlap=overlap, _router=router,
        routed_output_transform=kimi.KimiRoutedOutputTransform(
            Norm(g), Linear(HIDDEN, LATENT, g)
        ),
        # Experts stand-in: a deterministic latent-space map of the routed rows.
        experts=lambda routed, weights, ids: routed * 2 - routed.flip(0),
        _down_proj_stream=torch.cuda.Stream(),
        _down_proj_events=(torch.cuda.Event(), torch.cuda.Event()),
        _shared_stream=torch.cuda.Stream(),
        _shared_events=(torch.cuda.Event(), torch.cuda.Event()),
        _prefill_subcalls=None,
    )  # fmt: skip
    for name in ("forward", "_maybe_overlap_router_and_down_proj",
                 "_shared_mxfp8_input", "_routed_experts",
                 "_prefill_shared_stream"):  # fmt: skip
        setattr(mlp, name, types.MethodType(getattr(kimi.KimiMoE, name), mlp))
    return mlp


def stock(mlp, h):
    return mlp.forward(h)


@pytest.mark.parametrize("overlap", [False, True])
def test_cublas_everywhere_is_stock_bitwise(overlap, monkeypatch):
    mlp = layer(overlap)
    monkeypatch.setattr(
        sp_mlp, "sp_projection_tactics",
        lambda family, n, k, rows, ctx: dict.fromkeys(rows, ("cublas", None)),
    )  # fmt: skip
    spm = sp_mlp.SpMlp(mlp, ROWS, context(), 1)
    g = torch.Generator(device=DEVICE).manual_seed(5)
    h = torch.randn(max(ROWS), HIDDEN, device=DEVICE, generator=g).bfloat16()
    for n in ROWS:
        assert torch.equal(spm.run(h[:n]), stock(mlp, h[:n])), n


def reference(mlp, h):
    """Each GEMM in FP32 rounded once to BF16 (the beta-add: FP32 GEMM + the BF16
    shared output, one rounding), the stock router / experts / norm / act."""

    def mm(x, linear):
        return (x.float() @ linear.weight.float().t()).bfloat16()

    shared = mlp.shared_experts
    routed = mm(h, mlp.routed_expert_down_proj)
    final = mlp.experts(routed, None, None)
    act = shared.act_fn(mm(h, shared.gate_up_proj))
    shared_out = mm(act, shared.down_proj)
    transform = mlp.routed_output_transform
    x = transform.norm(final)
    up = x.float() @ transform.up_proj.weight.float().t() + shared_out.float()
    return up.bfloat16(), shared_out


@pytest.mark.parametrize("overlap", [False, True])
def test_measured_tactics_match_reference(overlap):
    mlp = layer(overlap, seed=1)
    spm = sp_mlp.SpMlp(mlp, ROWS, context(), 1)
    kinds = {c.tactic[0] for n in ROWS for c in spm.choices(n).values()}
    assert kinds == {"warp", "split_k", "cublas", "lt"}, kinds
    if overlap:  # the side-stream shared expert: torch kinds only
        shared = {
            c.tactic[0]
            for n in ROWS
            for f, c in spm.choices(n).items()
            if f.startswith("shared")
        }
        assert shared <= {"cublas", "lt"}, shared
    g = torch.Generator(device=DEVICE).manual_seed(6)
    h = torch.randn(max(ROWS), HIDDEN, device=DEVICE, generator=g).bfloat16()
    for n in ROWS:
        ref, _ = reference(mlp, h[:n])
        ref = ref.float()
        ours = (spm.run(h[:n]).float() - ref).norm() / ref.norm()
        theirs = (stock(mlp, h[:n]).float() - ref).norm() / ref.norm()
        # BF16 roundings of each GEMM step (propagated through the latent norm):
        # the same order as stock's own distance from the reference
        assert ours <= max(2 * theirs, 2**-8), (n, ours.item(), theirs.item())


def test_graph_replay():
    mlp = layer(True, seed=2)
    spm = sp_mlp.SpMlp(mlp, ROWS, context(), 1)
    h = torch.randn(64, HIDDEN, device=DEVICE).bfloat16()
    want = spm.run(h).clone()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = spm.run(h)
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.equal(got, want)


def test_latent_up_parts():
    """The latent up's FP32 split partials (``parts``) formed as stock's cuBLAS
    beta-add does (BF16 GEMM output, then + residual, rounded): the stock output up to
    the K order (within a BF16 ulp, nearly every element bitwise)."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.attn_res.sp_fused import (
        SplitDelta,
    )

    mlp = layer(True, seed=3)
    spm = sp_mlp.SpMlp(mlp, ROWS, context(), 1)
    assert spm.parts, "no latent up rows on the partials GEMM"
    g = torch.Generator(device=DEVICE).manual_seed(3)
    h = torch.randn(max(ROWS), HIDDEN, device=DEVICE, generator=g).bfloat16()
    for n, split in spm.parts.items():
        want = spm.run(h[:n]).float()
        got = spm.run(h[:n], parts=True)
        assert isinstance(got, SplitDelta) and got.parts == split, n
        p = got.partials.view(split, n, HIDDEN)
        total = p[0].clone()
        for s in range(1, split):
            total += p[s]
        out = (total.bfloat16().float() + got.residual.float()).bfloat16().float()
        def ulp(t):
            return (t.abs().clamp_min(1e-30).log2().floor() - 7).exp2()

        # a K-order ulp of the GEMM output carries through the residual add
        bound = ulp(want) + ulp(total) + 1e-4 * want.abs().amax()
        assert ((out - want).abs() <= bound).all(), n
        assert (out == want).float().mean().item() > 0.99, n


def test_unreproducible_layers_raise():
    ctx = context()
    mlp = layer(False, seed=3)
    mlp.shared_experts.down_proj.quant_method = object()
    with pytest.raises(Unsupported, match="quantized"):
        sp_mlp.SpMlp(mlp, ROWS, ctx, 7)
    mlp = layer(False, seed=3)
    mlp._packed_moe = ("packed",)
    with pytest.raises(Unsupported, match="packed"):
        sp_mlp.SpMlp(mlp, ROWS, ctx, 7)
    mlp = layer(False, seed=3)
    mlp.moe_token_shard = 1
    with pytest.raises(Unsupported, match="not MegaMoE"):
        sp_mlp.SpMlp(mlp, ROWS, ctx, 7)
