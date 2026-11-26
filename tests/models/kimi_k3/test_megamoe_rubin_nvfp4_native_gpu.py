# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rubin MegaMoE on a native ModelOpt NVFP4 checkpoint layer, end to end on one VR200.

The deployed decode path (Kimi K3.1 NVFP4 export, EP32: 28 local experts per GPU) loads
the export's experts as stored (``RubinMegaMoE.prepare_nvfp4_checkpoint_weights``: E2M1
codes, generic E4M3 block-16 scales, per-expert FP32 ``weight_scale_2``, unit input
scales) and runs the NVFP4 W4A4 kernel. The CPU tests prove the re-layout bitwise; this
test runs the kernel itself on weights that are NOT MXFP4-derived (non power-of-two
block scales, per-expert global scales that differ across experts) and compares it with

- the stock ModelOpt NVFP4 MoE math computed from the canonical checkpoint tensors in
  torch (independent of the kernel layout): activations NVFP4 with global scale
  ``1 / input_scale`` = 1, FC1 alpha ``input_scale * weight_scale_2``, SiTU, routing
  weight before the FC2 requantization (the MoE-EP megakernel contract), FC2 alpha,
  BF16 combine;
- the quantization-faithful oracle on the staged bytes and kernel-layout weights.

Shapes (top-16, 3584 x 3072): the EP32 decode bucket (28 experts per GPU, 128-token
capacity) and the EP16 prefill bucket (56 experts per GPU, 2048-token capacity), each on
a shipped NVFP4 knob record of that capacity.
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or torch.cuda.get_device_capability() != (10, 7),
    reason="the Rubin MegaMoE kernel needs a VR200 (sm_107a)",
)

TOP_K = 16
# shape: (local experts, bucket capacity, record filter)
SHAPES = {
    "ep32_decode": (28, 128, dict(max_tokens=128)),
    "ep16_prefill": (56, 2048, dict(world_size=16, kernel_variant="inference",
                                    max_tokens=2048)),
}  # fmt: skip
SITU = dict(situ_beta=4.0, situ_linear_beta=25.0)


def _mods():
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import config as C
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import reference as R
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import weights as W
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.runtime import RubinMegaMoE

    return C, R, W, RubinMegaMoE


def _modelopt_nvfp4(w: torch.Tensor, scale_2: torch.Tensor, W):
    """ModelOpt NVFP4 export of ``w`` (E, R, K): per-expert ``weight_scale_2`` and
    ``e4m3(block_amax / 6 / scale_2)`` block scales, RNE E2M1 codes."""
    codes, sf = W.quantize_nvfp4_block16(w, (1.0 / scale_2).view(-1, 1, 1))
    return codes, sf


def _dequant(codes, sf, scale_2, W):
    return (
        W.unpack_fp4_to_f32(codes)
        * W.e4m3_to_f32(sf).repeat_interleave(W.NV_BLOCK, dim=-1)
        * scale_2.view(-1, 1, 1)
    )


def _checkpoint_layer(problem, W, device, seed):
    """One layer's local experts as the export stores them (w1||w3 fused)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    E, H, inter = problem.num_experts, problem.hidden, problem.intermediate
    std = torch.empty(E, 1, 1, device=device).uniform_(0.01, 0.04, generator=gen)
    w13 = torch.randn(E, 2 * inter, H, generator=gen, device=device) * std
    w2 = torch.randn(E, H, inter, generator=gen, device=device) * std
    # weight_scale_2 = amax / (6 * 448) (ModelOpt), shared by w1 / w3; experts 0..3
    # get the export's power-of-two values (2^-13 / 2^-12) to cover both forms.
    g13 = w13.abs().amax(dim=(1, 2)) / (6.0 * 448.0)
    g2 = w2.abs().amax(dim=(1, 2)) / (6.0 * 448.0)
    g13[:4] = torch.tensor([2.0**-13, 2.0**-12, 2.0**-13, 2.0**-12], device=device)
    g2[:4] = torch.tensor([2.0**-13, 2.0**-13, 2.0**-12, 2.0**-12], device=device)
    q13, s13 = _modelopt_nvfp4(w13, g13, W)
    q2, s2 = _modelopt_nvfp4(w2, g2, W)
    return q13, s13, q2, s2, g13.float().contiguous(), g2.float().contiguous()


def _stock_nvfp4_moe(hidden, ids, wts, layer, problem, W, R):
    """Stock ModelOpt NVFP4 MoE math (unit input scales) from the canonical tensors."""
    q13, s13, q2, s2, g13, g2 = layer
    inter = problem.intermediate
    xq, xsf = W.quantize_nvfp4_block16(hidden.float(), 1.0)
    x = W.unpack_fp4_to_f32(xq) * W.e4m3_to_f32(xsf).repeat_interleave(
        W.NV_BLOCK, dim=-1
    )
    out = torch.zeros_like(x)
    idx = ids.to(torch.int64)
    for e in range(problem.num_experts):
        routed = (idx == e).nonzero(as_tuple=False)
        if routed.shape[0] == 0:
            continue
        t, k = routed[:, 0], routed[:, 1]
        w1 = W.unpack_fp4_to_f32(q13[e]) * W.e4m3_to_f32(s13[e]).repeat_interleave(
            W.NV_BLOCK, dim=-1
        )
        h = (x[t] @ w1.t()) * g13[e]
        act = _situ(R, h, inter) * wts[t, k].float().unsqueeze(-1)
        aq, asf = W.quantize_nvfp4_block16(act, 1.0)
        a = W.unpack_fp4_to_f32(aq) * W.e4m3_to_f32(asf).repeat_interleave(
            W.NV_BLOCK, dim=-1
        )
        w2 = W.unpack_fp4_to_f32(q2[e]) * W.e4m3_to_f32(s2[e]).repeat_interleave(
            W.NV_BLOCK, dim=-1
        )
        term = ((a @ w2.t()) * g2[e]).to(torch.bfloat16).float()
        out.index_add_(0, t, term)
    return out


def _routes(n, experts, device, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    ids = torch.argsort(torch.rand(n, experts, generator=gen, device=device), dim=1)
    ids = ids[:, :TOP_K].to(torch.int32)
    ids[ids == experts - 1] = -1  # one expert never routed; masked routes
    wts = torch.rand(n, TOP_K, generator=gen, device=device)
    wts = wts / wts.sum(dim=1, keepdim=True)
    return ids, wts


def _situ(R, h, inter):
    return R._situ(
        h[:, :inter], h[:, inter:], SITU["situ_beta"], SITU["situ_linear_beta"]
    )


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


@pytest.fixture(scope="module")
def build():
    C, R, W, RubinMegaMoE = _mods()
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    torch.zeros(1, device=device)  # current primary context before the DSL queries it
    entries = C.load_knob_source(C.SHIPPED_KNOBS)["entries"]
    cache = {}

    def get(shape):
        if shape in cache:
            return cache[shape]
        experts, cap, want = SHAPES[shape]
        problem = C.RubinMoeProblem(num_experts=experts, quant_kind="nvfp4")
        # A shipped NVFP4 record of this capacity (its knobs; the EP size is a server
        # property, the tiles and math are the same at EP1).
        records = [
            r
            for r in entries
            if r["dtype"] == "nvfp4" and all(r.get(f) == v for f, v in want.items())
        ]
        assert records, f"no shipped NVFP4 knob record {want}"
        rec = min(records, key=lambda r: r.get("p50_us") or float("inf"))
        bucket = C.RubinBucket(
            rec["kernel_variant"], cap, C.RubinKnobs.from_dict(rec["knobs"])
        )
        moe = RubinMegaMoE(problem, None, [bucket])
        layer = _checkpoint_layer(problem, W, device, seed=7)
        weights = moe.prepare_nvfp4_checkpoint_weights(*layer)
        moe.warmup(weights, ids_dtypes=(torch.int32,))
        cache[shape] = (C, R, W, problem, moe, bucket, layer, weights)
        return cache[shape]

    return get


@pytest.mark.parametrize(
    "shape,n",
    [("ep32_decode", 1), ("ep32_decode", 37), ("ep32_decode", 128),
     ("ep16_prefill", 2048)],
)  # fmt: skip
def test_native_nvfp4_kernel_matches_stock_math(build, shape, n):
    C, R, W, problem, moe, bucket, layer, weights = build(shape)
    E = problem.num_experts
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(100 + n)
    hidden = torch.randn(n, problem.hidden, generator=gen, device=device)
    hidden = hidden.to(torch.bfloat16)
    ids, wts = _routes(n, E, device, seed=200 + n)
    out = torch.full((n, problem.hidden), float("nan"), dtype=torch.bfloat16,
                     device=device)  # fmt: skip
    moe.forward(hidden, ids, wts, weights, bucket=bucket, out=out)
    torch.cuda.synchronize()
    moe.check_route_errors()
    assert bool(torch.isfinite(out).all())

    stock = _stock_nvfp4_moe(hidden, ids, wts, layer, problem, W, R)
    x, xsf, idx, sw = R.stage_inputs_torch(
        hidden, ids, wts, quant_kind="nvfp4", capacity=bucket.max_tokens,
        sf_cols=problem.hidden // W.NV_BLOCK, num_experts=E,
    )  # fmt: skip
    oracle = R.megamoe_oracle(
        x[:n], xsf[:n], idx[:n], sw[:n], weights, quant_kind="nvfp4",
        local_expert_offset=0, hidden=problem.hidden,
        intermediate=problem.intermediate, **SITU,
    )  # fmt: skip
    # The oracle on kernel-layout weights equals the stock math on the canonical
    # export (re-layout + alphas are exact; measured 0.0 on VR200).
    assert _rel(oracle, stock) < 1e-6
    # The kernel: same quantization points, different accumulation order; a rounding
    # flip at the FC2 requantization moves single elements by one E2M1 step
    # (measured 1.7e-3 at n = 1 / 37 / 128).
    assert _rel(out, stock) < 5e-3
    assert _rel(out, oracle) < 5e-3
    # Full-precision sanity (catches a missing / squared alpha the two share).
    w13 = _dequant(layer[0], layer[1], layer[4], W)
    w2 = _dequant(layer[2], layer[3], layer[5], W)
    canon_ref = torch.zeros(n, problem.hidden, device=device)
    for e in range(E):
        t, k = (ids.to(torch.int64) == e).nonzero(as_tuple=True)
        if t.numel() == 0:
            continue
        h = hidden[t].float() @ w13[e].t()
        inter = problem.intermediate
        act = _situ(R, h, inter) * wts[t, k].unsqueeze(-1)
        canon_ref.index_add_(0, t, act @ w2[e].t())
    errs = {
        "oracle_stock": _rel(oracle, stock),
        "kernel_stock": _rel(out, stock),
        "kernel_oracle": _rel(out, oracle),
        "kernel_full": _rel(out, canon_ref),
    }
    print(f"{shape} n={n} rel L2 {errs}")
    # W4A4 activation quantization at two points (measured 0.164).
    assert errs["kernel_full"] < 0.2


def test_native_nvfp4_kernel_uses_per_expert_alphas(build):
    """The kernel reads each expert's own weight_scale_2: rolling the per-expert
    alphas by one expert moves the output far off the stock math."""
    C, R, W, problem, moe, bucket, layer, weights = build("ep32_decode")
    device = torch.device("cuda", 0)
    n = 64
    gen = torch.Generator(device=device).manual_seed(5)
    hidden = torch.randn(n, problem.hidden, generator=gen, device=device)
    hidden = hidden.to(torch.bfloat16)
    ids, wts = _routes(n, problem.num_experts, device, seed=6)
    stock = _stock_nvfp4_moe(hidden, ids, wts, layer, problem, W, R)
    rolled = weights._replace(
        fc1_alpha=weights.fc1_alpha.roll(1).contiguous(),
        fc2_alpha=weights.fc2_alpha.roll(1).contiguous(),
    )
    out = torch.empty((n, problem.hidden), dtype=torch.bfloat16, device=device)
    moe.forward(hidden, ids, wts, rolled, bucket=bucket, out=out)
    torch.cuda.synchronize()
    assert _rel(out, stock) > 0.1
    moe.forward(hidden, ids, wts, weights, bucket=bucket, out=out)
    torch.cuda.synchronize()
    assert _rel(out, stock) < 5e-3


def _stock_trtllm_nvfp4_moe(hidden, ids, wts, layer, problem, W, R):
    """Stock NVFP4 MoE as vLLM ModelOpt -> FlashInfer TRT-LLM-Gen runs it for unit
    input scales: FC2 input ``SiTU(fc1)`` requantized with ``a2_gscale = 1`` and the
    routing weight applied after FC2 (finalize), not before the requantization."""
    q13, s13, q2, s2, g13, g2 = layer
    inter = problem.intermediate

    def dq(codes, sf):
        return W.unpack_fp4_to_f32(codes) * W.e4m3_to_f32(sf).repeat_interleave(
            W.NV_BLOCK, dim=-1
        )

    x = dq(*W.quantize_nvfp4_block16(hidden.float(), 1.0))
    out = torch.zeros_like(x)
    idx = ids.to(torch.int64)
    for e in range(problem.num_experts):
        t, k = (idx == e).nonzero(as_tuple=True)
        if t.numel() == 0:
            continue
        u = _situ(R, (x[t] @ dq(q13[e], s13[e]).t()) * g13[e], inter)
        a = dq(*W.quantize_nvfp4_block16(u, 1.0))
        term = (a @ dq(q2[e], s2[e]).t()) * g2[e] * wts[t, k].float().unsqueeze(-1)
        out.index_add_(0, t, term.to(torch.bfloat16).float())
    return out


def test_fc1_norm_const_keeps_real_scale_wires(build):
    """At the K3.1 MoE input scale (hidden RMS ~0.08, routing weights ~1/16) the
    routing-weighted FC1->FC2 wire is ~1e-3: FlashInfer's fc1_norm_const 1.0 flushes
    its E4M3 block scales to zero, the served constant (nvfp4_fc1_norm_const = 16 at
    top-16) keeps them and matches stock TRT-LLM-Gen NVFP4 MoE accuracy."""
    C, R, W, problem, moe, bucket, layer, weights = build("ep32_decode")
    n_served = C.nvfp4_fc1_norm_const(top_k=TOP_K, **SITU)
    assert n_served == 16.0
    device = torch.device("cuda", 0)
    n = 128
    gen = torch.Generator(device=device).manual_seed(11)
    hidden = (torch.randn(n, problem.hidden, generator=gen, device=device) * 0.08).to(
        torch.bfloat16
    )
    ids, wts = _routes(n, problem.num_experts, device, seed=12)
    w13 = _dequant(layer[0], layer[1], layer[4], W)
    w2 = _dequant(layer[2], layer[3], layer[5], W)
    full = torch.zeros(n, problem.hidden, device=device)
    wire = []
    for e in range(problem.num_experts):
        t, k = (ids.to(torch.int64) == e).nonzero(as_tuple=True)
        if t.numel() == 0:
            continue
        act = _situ(R, hidden[t].float() @ w13[e].t(), problem.intermediate)
        act = act * wts[t, k].unsqueeze(-1)
        wire.append(act)
        full.index_add_(0, t, act @ w2[e].t())
    wire = torch.cat(wire)
    # The regime of the bug: most 16-blocks of the wire below the E4M3 flush at n = 1.
    blocks = wire.abs().reshape(-1, 16).amax(-1)
    assert float((blocks / 6.0 < 2.0**-10).float().mean()) > 0.5

    errs = {}
    for norm in (1.0, n_served):
        _, fc2_alpha, fc1_norm = W.nvfp4_checkpoint_alphas(
            layer[4], layer[5], 1.0, norm
        )
        w = weights._replace(fc2_alpha=fc2_alpha, fc1_norm_const=fc1_norm)
        out = torch.empty((n, problem.hidden), dtype=torch.bfloat16, device=device)
        moe.forward(hidden, ids, wts, w, bucket=bucket, out=out)
        torch.cuda.synchronize()
        errs[norm] = _rel(out, full)
    stock = _rel(_stock_trtllm_nvfp4_moe(hidden, ids, wts, layer, problem, W, R), full)
    print(f"rel L2 vs full precision: n=1 {errs[1.0]:.4f} n=16 {errs[n_served]:.4f} "
          f"stock TRT-LLM-Gen {stock:.4f}")  # fmt: skip
    assert errs[1.0] > 0.3
    assert errs[n_served] < 1.25 * stock + 0.01
    assert errs[n_served] < errs[1.0] / 3
