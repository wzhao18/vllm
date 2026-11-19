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

Sizes are the EP32 per-GPU decode shape (28 experts, top-16, 3584 x 3072).
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or torch.cuda.get_device_capability() != (10, 7),
    reason="the Rubin MegaMoE kernel needs a VR200 (sm_107a)",
)

E_LOCAL = 28  # EP32: 896 experts / 32 ranks
TOP_K = 16
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
    H, inter = problem.hidden, problem.intermediate
    std = torch.empty(E_LOCAL, 1, 1, device=device).uniform_(0.01, 0.04, generator=gen)
    w13 = torch.randn(E_LOCAL, 2 * inter, H, generator=gen, device=device) * std
    w2 = torch.randn(E_LOCAL, H, inter, generator=gen, device=device) * std
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
    for e in range(E_LOCAL):
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


def _routes(n, device, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    ids = torch.argsort(torch.rand(n, E_LOCAL, generator=gen, device=device), dim=1)
    ids = ids[:, :TOP_K].to(torch.int32)
    ids[ids == E_LOCAL - 1] = -1  # one expert never routed; masked routes
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
def setup():
    C, R, W, RubinMegaMoE = _mods()
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    torch.zeros(1, device=device)  # current primary context before the DSL queries it
    problem = C.RubinMoeProblem(num_experts=E_LOCAL, quant_kind="nvfp4")
    # The shipped NVFP4 128-token decode record (its knobs; world size is a server
    # property, the tiles and math are the same at EP1).
    records = [
        r
        for r in C.load_knob_source(C.SHIPPED_KNOBS)["entries"]
        if r["dtype"] == "nvfp4" and r["max_tokens"] == 128
    ]
    assert records, "no NVFP4 128-token knob record shipped"
    rec = records[-1]
    bucket = C.RubinBucket(
        rec["kernel_variant"], 128, C.RubinKnobs.from_dict(rec["knobs"])
    )
    moe = RubinMegaMoE(problem, None, [bucket])
    layer = _checkpoint_layer(problem, W, device, seed=7)
    weights = moe.prepare_nvfp4_checkpoint_weights(*layer)
    moe.warmup(weights, ids_dtypes=(torch.int32,))
    return C, R, W, problem, moe, bucket, layer, weights


@pytest.mark.parametrize("n", [1, 37, 128])
def test_native_nvfp4_kernel_matches_stock_math(setup, n):
    C, R, W, problem, moe, bucket, layer, weights = setup
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(100 + n)
    hidden = torch.randn(n, problem.hidden, generator=gen, device=device)
    hidden = hidden.to(torch.bfloat16)
    ids, wts = _routes(n, device, seed=200 + n)
    out = torch.full((n, problem.hidden), float("nan"), dtype=torch.bfloat16,
                     device=device)  # fmt: skip
    moe.forward(hidden, ids, wts, weights, bucket=bucket, out=out)
    torch.cuda.synchronize()
    moe.check_route_errors()
    assert bool(torch.isfinite(out).all())

    stock = _stock_nvfp4_moe(hidden, ids, wts, layer, problem, W, R)
    x, xsf, idx, sw = R.stage_inputs_torch(
        hidden, ids, wts, quant_kind="nvfp4", capacity=bucket.max_tokens,
        sf_cols=problem.hidden // W.NV_BLOCK, num_experts=E_LOCAL,
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
    for e in range(E_LOCAL):
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
    print(f"n={n} rel L2 {errs}")
    # W4A4 activation quantization at two points (measured 0.164).
    assert errs["kernel_full"] < 0.2


def test_native_nvfp4_kernel_uses_per_expert_alphas(setup):
    """The kernel reads each expert's own weight_scale_2: rolling the per-expert
    alphas by one expert moves the output far off the stock math."""
    C, R, W, problem, moe, bucket, layer, weights = setup
    device = torch.device("cuda", 0)
    n = 64
    gen = torch.Generator(device=device).manual_seed(5)
    hidden = torch.randn(n, problem.hidden, generator=gen, device=device)
    hidden = hidden.to(torch.bfloat16)
    ids, wts = _routes(n, device, seed=6)
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
