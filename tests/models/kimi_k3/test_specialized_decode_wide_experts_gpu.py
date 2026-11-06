# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 specialized decode, wide routed experts (Rubin, TP16 per-rank shapes):
precision against an FP64 reference and batch variance across launch groupings.

The wide engines (``experts_runtime.WideExperts`` -> ``PersistentExperts.prepare_wide``
-> ``experts_wide.RubinWideExpertsKernel`` at 65-80 rows, ``experts_vw.
RubinVWExpertsKernel`` width 16 above) run one TP16 MoE layer's routed experts: MXFP4
weights x MXFP8 activations (per-token 32-element blocks, ``experts.plan_chunk``), SiTU
(beta 4, linear beta 25), the hidden re-quantized to MXFP8 per token and 32-block
(``experts.ExpertsKernel.store_hidden``), FC2 rows staged in BF16, the top-k weighted
sum as an FP32 FMA chain in slot order. The stock arm is FlashInfer TRT-LLM-Gen
``trtllm_fp4_block_scale_routed_moe`` (as ``stock_experts.StockExperts`` and the
kern-moe bench call it) on the same weight bytes.

Logical MXFP4 weights (E2M1 nibbles, E8M0 scales) are converted to the stock layer's
TRT-LLM layout with ``convert_weight_to_mxfp4_moe_kernel_format`` (the bytes both
kernels read in production) and ``place``d like production; the reference reads the
logical weights:

- ``test_wide_experts_reference_band``: single launches of 72 / 128 / 256 / 512 / 768
  rows (the first rows of one 768-row set of q4 verify-like routes); the specialized
  error vs the reference must stay within ``BAND`` x the stock error (max abs, mean
  abs, max / mean row-relative L2).
- ``test_wide_experts_grouping_variance``: the same 768 tokens as one launch, 512 +
  256, 3 x 256, the last 256 alone on a fresh engine, the 768 again (determinism) and
  the 768 in reversed row order; per pair the differing rows, columns, FC2 tiles and
  BF16 ulps. Determinism and the reference band are hard asserts; cross-grouping
  bitwise equality is xfail (strict=False) with the measured difference.

Batch-dependence hypothesis (code reading, see the grouping test's summary for the
measured signature): the FC1 K-split. ``experts.ExpertsKernel.partition``
(experts.py:638-654) cuts every group's FC1 stage list (``tiles x 14 x streams``
stages) evenly over the grid, so where a token's (expert, 128-row) tile chain is cut
into CTA pieces depends on how many streams the whole launch has; the owner sums the
pieces (its TMEM chain, then each peer's FP32 partial in peer order:
``partials`` experts.py:1237-1271, ``RubinVWExpertsKernel.sub_partials``
experts_vw.py:701-737). A different launch size therefore changes the FP32
association of FC1 (an order-only change), which the per-token MXFP8 re-quantization
of the SiTU output (experts.py:1295-1299) turns into a one-E4M3-step change of a hidden
element now and then; that element feeds every FC2 column of the token, so a differing
row differs across most of its 3584 columns (not in single 128-column FC2 tiles).
The token-unit grouping (16-token streams), the plan, the FC2 jobs and the pool zone
(whole jobs) and the finalize (slot order) are per token. Observed on a GPU run before
this test: rows 229 and 991 of 1024 differed between 512 + 512 and 768 + 256 launches.
Rubin (cc 10.7) with CuTe DSL only."""

import pytest

H_LATENT, EXPERTS, TOPK, INTER = 3584, 896, 16, 256
BETA, LINEAR_BETA = 4.0, 25.0
ROWS = (72, 128, 256, 512, 768)
FULL = 768
# "Same band": the specialized error metric may be at most BAND x the stock one.
BAND = 2.0
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def require_rubin():
    torch = pytest.importorskip("torch")
    pytest.importorskip("cutlass")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("Rubin (cc 10.7) wide experts")
    return torch


# --------------------------------------------------------------------------------------
# Weights, routes, reference


def logical_weights(dev, seed=11):
    """One TP16 layer's routed experts as loaded (before the backend's conversion):
    w13 [E, 2I, K/2] (rows [0, I) gate w1, [I, 2I) up w3; E2M1 pairs, low nibble
    first), s13 [E, 2I, K/32] E8M0, w2 [E, H, I/2], s2 [E, H, I/32]. Scales 2^-8..2^-4
    keep FC1 / FC2 outputs O(1) at these K."""
    import torch

    gen = torch.Generator(device=dev).manual_seed(seed)
    u8 = torch.uint8

    def b(shape, lo=0, hi=256):
        return torch.randint(lo, hi, shape, dtype=u8, device=dev, generator=gen)

    return (
        b((EXPERTS, 2 * INTER, H_LATENT // 2)),
        b((EXPERTS, 2 * INTER, H_LATENT // 32), 119, 124),
        b((EXPERTS, H_LATENT, INTER // 2)),
        b((EXPERTS, H_LATENT, INTER // 32), 119, 124),
    )


def kernel_weights(logical):
    """The stock layer's FlashInfer TRT-LLM MXFP4 x MXFP8 layout (what both the stock
    experts and ``PersistentExperts`` read in production), ``place``d on the dies as
    ``runtime.decode`` does; returns ((w13, s13, w2, s2), dies)."""
    from types import SimpleNamespace

    import torch

    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        Mxfp4MoeBackend,
        convert_weight_to_mxfp4_moe_kernel_format,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import (
        experts_runtime as er,
    )

    w13, s13, w2, s2 = (t.clone() for t in logical)
    w13, w2, s13, s2, _, _ = convert_weight_to_mxfp4_moe_kernel_format(
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8, None, w13, w2, s13, s2,
        _cache_permute_indices={},
    )  # fmt: skip
    routed = SimpleNamespace(
        w13_weight=w13.view(torch.uint8), w2_weight=w2.view(torch.uint8)
    )
    dies = er.place(routed)
    return (routed.w13_weight, s13, routed.w2_weight, s2), tuple(dies)


def verify_routes(rows, dev, seed=1007, q=4):
    """q4 verify-like routes (``bench_moe_kern.make_routes`` "req4"): a request's q
    tokens share a 16-expert base, each slot replaced with p 0.25; BF16 top-k weights
    (sum 1), BF16 latent (N(0, 0.25))."""
    import numpy as np
    import torch

    rng = np.random.default_rng(seed)
    ids = np.empty((rows, TOPK), dtype=np.int32)
    for r0 in range(0, rows, q):
        base = rng.choice(EXPERTS, TOPK, replace=False)
        for t in range(r0, min(rows, r0 + q)):
            row = base.copy()
            for s in range(TOPK):
                if rng.random() < 0.25:
                    new = int(rng.integers(EXPERTS))
                    while new in row:
                        new = int(rng.integers(EXPERTS))
                    row[s] = new
            ids[t] = row
    gen = torch.Generator(device=dev).manual_seed(seed)
    w = torch.rand(rows, TOPK, device=dev, generator=gen)
    w = (w / w.sum(1, keepdim=True)).bfloat16()
    latent = (torch.randn(rows, H_LATENT, device=dev, generator=gen) * 0.5).bfloat16()
    return torch.tensor(ids, device=dev), w, latent


def dequant_mxfp4(packed, scales):
    """E2M1 pairs [..., K/2] (low nibble first) x E8M0 [..., K/32] -> FP64 [..., K]."""
    import torch

    table = torch.tensor(E2M1 + tuple(-v for v in E2M1), dtype=torch.float64)
    table = table.to(packed.device)
    values = torch.stack((table[(packed & 15).long()], table[(packed >> 4).long()]), -1)
    values = values.flatten(-2)
    scale = torch.exp2(scales.double() - 127)
    shape = values.shape
    return (values.view(*shape[:-1], -1, 32) * scale[..., None]).view(shape)


def mxfp8_kernel_rule(values):
    """MXFP8 E4M3 round trip of FP32 ``values`` [..., K] in 32-element blocks, as the
    kernel quantizes (``primitives.mx_exponent`` without floor (TP16), ``mx_rescale``,
    cvt.rn.satfinite): biased scale = e(amax) - 8 + (mantissa > 0.75), clamped to
    [0, 254] (0 for a zero / subnormal amax); returns the dequantized FP64 values."""
    import torch

    v = values.float()
    shape = v.shape
    blocks = v.view(*shape[:-1], -1, 32)
    amax = blocks.abs().amax(-1)
    bits = amax.view(torch.int32)
    e, mantissa = bits >> 23, bits & 0x7FFFFF
    scale = (e - 8 + (mantissa > 0x600000).int()).clamp(0, 254)
    scale = torch.where(e == 0, torch.zeros_like(scale), scale)
    rescale_exp = 254 - scale  # mx_rescale: 2^(127 - scale), 0 below a normal
    rescale = torch.where(
        rescale_exp >= 1, torch.exp2(127.0 - scale.float()), torch.zeros_like(amax)
    )
    q = (blocks * rescale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    out = q.double() * torch.exp2(scale.double() - 127)[..., None]
    return out.view(shape)


def situ(gate, up):
    import torch

    g = BETA * torch.tanh(gate / BETA) * torch.sigmoid(gate)
    return g * (LINEAR_BETA * torch.tanh(up / LINEAR_BETA))


def reference(logical, latent, topk_w, ids):
    """Per-token FP64 routed partial [M, 3584]: MXFP8 x (kernel rule), dequantized
    MXFP4 w13 / w2, SiTU, the hidden MXFP8 per token (kernel rule), the top-k sum with
    the BF16 routing weights; no BF16 staging."""
    import torch

    w13, s13, w2, s2 = logical
    rows = latent.shape[0]
    x = mxfp8_kernel_rule(latent.float())
    out = torch.zeros(rows, H_LATENT, dtype=torch.float64, device=latent.device)
    flat = ids.flatten().long()
    token = torch.arange(rows, device=ids.device).repeat_interleave(TOPK)
    weight = topk_w.flatten().double()
    for e in torch.unique(flat).tolist():
        pick = (flat == e).nonzero().flatten()
        tokens = token[pick]
        a = x[tokens] @ dequant_mxfp4(w13[e], s13[e]).T  # [n, 2I]
        h = situ(a[:, :INTER], a[:, INTER:]).float()  # FP32 into the quantizer
        y = mxfp8_kernel_rule(h) @ dequant_mxfp4(w2[e], s2[e]).T  # [n, H]
        out.index_add_(0, tokens, y * weight[pick, None])
    return out


# --------------------------------------------------------------------------------------
# The two arms


def run_wide(weights, dies, dev, latent, topk_w, ids, batches, order=None):
    """``WideExperts`` (TP16, q4) on a fresh ``PersistentExperts`` and scratch: the
    rows ``order`` (default all, in order) in consecutive single launches of
    ``batches`` rows, each behind its plan writer (as the routing would write it);
    returns the BF16 partial rows in ``order``'s order."""
    import torch

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import (
        experts_runtime as er,
    )

    if order is not None:
        latent, topk_w, ids = (t[order].contiguous() for t in (latent, topk_w, ids))
    assert sum(batches) == latent.shape[0]
    assert all(len(er.wide_chunks(b)) == 1 for b in batches)
    buffers = er.scratch(16, dev, rows=max(batches))
    experts = er.PersistentExperts(
        *weights, beta=BETA, linear_beta=LINEAR_BETA, dies=dies, q=4
    )
    bf = torch.bfloat16
    partials = {b: torch.empty(b, H_LATENT, dtype=bf, device=dev) for b in set(batches)}
    wide = er.WideExperts(experts, buffers, partials, planned=True)
    outs, start = [], 0
    for rows in batches:
        end = start + rows
        x = latent[start:end]
        writer = er.compile_wide_plan(rows, dev.index)
        writer(to_cute(x, 16), er.tensor(buffers["plan"]), stream(dev))
        out = wide.run(x, topk_w[start:end], ids[start:end])
        torch.cuda.synchronize()
        outs.append(out[:rows].clone())
        start = end
    return torch.cat(outs)


class Stock:
    """FlashInfer TRT-LLM-Gen MXFP4 x MXFP8 routed experts with SiTU, finalized (the
    kern-moe bench's ``StockExpertsFI.routed``; MXFP8 input via
    ``mxfp8_e4m3_quantize``); no autotune."""

    def __init__(self, weights, dev):
        import torch

        from vllm.model_executor.layers.quantization.utils import flashinfer_utils as fu

        pytest.importorskip("flashinfer")
        if not fu.has_flashinfer_situ_activation():
            pytest.skip("FlashInfer without the SiTU activation: no stock arm")
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation

        self.w13, s13, self.w2, s2 = weights
        self.s13, self.s2 = (s.view(torch.float8_e4m3fn) for s in (s13, s2))
        self.act = fu.activation_to_flashinfer_int(MoEActivation.SITU)
        f32 = dict(dtype=torch.float32, device=dev)
        self.alpha = torch.full((EXPERTS,), BETA, **f32)
        self.beta = torch.full((EXPERTS,), LINEAR_BETA, **f32)
        self.dev = dev

    def __call__(self, latent, topk_w, ids):
        import torch
        from flashinfer import trtllm_fp4_block_scale_routed_moe

        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
            mxfp8_e4m3_quantize,
        )

        x, s = mxfp8_e4m3_quantize(latent)
        out = torch.empty(latent.shape[0], H_LATENT, dtype=torch.bfloat16,
                          device=self.dev)  # fmt: skip
        trtllm_fp4_block_scale_routed_moe(
            topk_ids=(ids, topk_w), routing_bias=None, hidden_states=x,
            hidden_states_scale=s.view(torch.float8_e4m3fn), n_group=None,
            topk_group=None, routed_scaling_factor=None, routing_method_type=1,
            do_finalize=True, enable_pdl=True, output=out,
            gemm1_weights=self.w13, gemm1_weights_scale=self.s13, gemm1_bias=None,
            gemm1_alpha=self.alpha, gemm1_beta=self.beta, gemm1_clamp_limit=None,
            gemm2_weights=self.w2, gemm2_weights_scale=self.s2, gemm2_bias=None,
            output1_scale_scalar=None, output1_scale_gate_scalar=None,
            output2_scale_scalar=None, num_experts=EXPERTS, top_k=TOPK,
            intermediate_size=INTER, local_expert_offset=0, local_num_experts=EXPERTS,
            activation_type=self.act, tune_max_num_tokens=8192,
        )  # fmt: skip
        torch.cuda.synchronize()
        return out


# --------------------------------------------------------------------------------------
# Metrics


def errors(out, ref):
    """max / mean abs error and max / mean row-relative L2 error of ``out`` vs
    ``ref``, plus the per-row relative errors."""
    err = out.double() - ref
    rel = err.norm(dim=1) / ref.norm(dim=1).clamp_min(1e-30)
    return dict(
        max_abs=err.abs().max().item(), mean_abs=err.abs().mean().item(),
        max_rel=rel.max().item(), mean_rel=rel.mean().item(), rows=rel,
    )  # fmt: skip


def fmt(e):
    return (
        f"max_abs {e['max_abs']:.3e} mean_abs {e['mean_abs']:.3e} "
        f"max_rel {e['max_rel']:.3e} mean_rel {e['mean_rel']:.3e}"
    )


def ordered(bf):
    """BF16 bits as a monotone integer (+0 == -0): ulp distance = difference."""
    import torch

    i = bf.view(torch.int16).int()
    return i.where(i >= 0, -(i & 0x7FFF))


def compare(a, b):
    """Rows / columns where BF16 ``a`` and ``b`` differ: a compact dict."""
    ulps = (ordered(a) - ordered(b)).abs()
    differ = ulps > 0
    rows = differ.any(1).nonzero().flatten()
    out = dict(
        rows=rows.tolist(), elements=int(differ.sum()),
        over_1ulp=int((ulps > 1).sum()), max_ulp=int(ulps.max()),
        max_abs=(a.float() - b.float()).abs().max().item(), detail=[],
    )  # fmt: skip
    for r in rows.tolist()[:8]:
        cols = differ[r].nonzero().flatten()
        tiles = sorted({c // 128 for c in cols.tolist()})
        out["detail"].append(
            (r, int(cols.numel()), len(tiles), int(ulps[r].max()),
             (a[r].float() - b[r].float()).abs().max().item())
        )  # fmt: skip
    return out


def describe(name, c):
    head = (
        f"{name}: {len(c['rows'])} rows differ ({c['elements']} elements, "
        f"{c['over_1ulp']} > 1 BF16 ulp, max {c['max_ulp']} ulp, max abs "
        f"{c['max_abs']:.3e})"
    )
    if not c["rows"]:
        return head
    rows = ", ".join(
        f"r{r}: {n} cols / {t} of 28 FC2 tiles, {u} ulp, {d:.2e}"
        for r, n, t, u, d in c["detail"]
    )
    return f"{head}; rows {c['rows'][:16]}; {rows}"


# --------------------------------------------------------------------------------------
# Fixtures and tests


@pytest.fixture(scope="module")
def case():
    torch = require_rubin()
    dev = torch.device("cuda", 0)
    torch.accelerator.set_device_index(dev)
    torch.backends.cuda.matmul.allow_tf32 = False
    logical = logical_weights(dev)
    weights, dies = kernel_weights(logical)
    ids, topk_w, latent = verify_routes(FULL, dev)
    ref = reference(logical, latent, topk_w, ids)
    return dict(dev=dev, weights=weights, dies=dies, inputs=(latent, topk_w, ids),
                ref=ref, cache={})  # fmt: skip


def stock_out(case, rows):
    """The stock arm's partial of the first ``rows`` rows (skips without it)."""
    key = ("stock", rows)
    if key not in case["cache"]:
        if "stock_arm" not in case["cache"]:
            case["cache"]["stock_arm"] = Stock(case["weights"], case["dev"])
        stock = case["cache"]["stock_arm"]
        latent, topk_w, ids = (t[:rows].contiguous() for t in case["inputs"])
        case["cache"][key] = stock(latent, topk_w, ids)
    return case["cache"][key]


def spec_out(case, batches, order=None, tag=None):
    key = ("spec", tag or tuple(batches))
    if key not in case["cache"]:
        rows = sum(batches)
        latent, topk_w, ids = (t[:rows] for t in case["inputs"])
        if order is not None:
            latent, topk_w, ids = case["inputs"]
        case["cache"][key] = run_wide(
            case["weights"], case["dies"], case["dev"], latent, topk_w, ids,
            list(batches), order,
        )  # fmt: skip
    return case["cache"][key]


@pytest.mark.parametrize("rows", ROWS)
def test_wide_experts_reference_band(case, rows):
    """One launch of ``rows`` rows: finite, and the specialized error vs the FP64
    reference within BAND x the stock error on every metric."""
    import torch

    ref = case["ref"][:rows]
    spec = spec_out(case, [rows])
    assert torch.isfinite(spec.float()).all()
    stock = stock_out(case, rows)
    es, ek = errors(spec, ref), errors(stock, ref)
    print(f"\nM{rows} spec  vs ref: {fmt(es)}\nM{rows} stock vs ref: {fmt(ek)}")
    for metric in ("max_abs", "mean_abs", "max_rel", "mean_rel"):
        assert es[metric] <= BAND * ek[metric], (rows, metric, fmt(es), fmt(ek))


class GroupingNotBitwise(AssertionError):
    """Cross-grouping differences inside the reference band (expected xfail)."""


@pytest.mark.xfail(
    strict=False, raises=GroupingNotBitwise,
    reason="wide experts not bitwise across single-launch row counts (FC1 K-split "
    "partition follows the launch's stream count; GPU run: rows 229 / 991 of 1024, "
    "512+512 vs 768+256); measured diff in the raised message",
)  # fmt: skip
def test_wide_experts_grouping_variance(case):
    """The same 768 tokens in other single-launch groupings. Hard: the 768 launch is
    deterministic (a fresh engine twice), and every grouping's rows stay within the
    reference band (each row's relative error <= BAND x the stock 768 launch's worst
    row, or the specialized 768 launch's own worst row without a stock arm). xfail:
    bitwise equality across groupings."""
    import torch

    ref = case["ref"]
    last = torch.arange(512, FULL, device=case["dev"])
    reverse = torch.arange(FULL - 1, -1, -1, device=case["dev"])
    one = spec_out(case, [FULL])
    runs = {
        "768 again": spec_out(case, [FULL], torch.arange(FULL, device=case["dev"]),
                              tag="again"),
        "512+256": spec_out(case, [512, 256]),
        "3x256": spec_out(case, [256, 256, 256]),
        "768 reversed": spec_out(case, [FULL], reverse, tag="reversed")
        .flip(0).contiguous(),
    }  # fmt: skip
    alone = spec_out(case, [256], last, tag="alone")
    stock_lines = []
    try:
        stock = stock_out(case, FULL)
        band_rows = errors(stock, ref)["max_rel"]
        band_from = "stock M768 worst row"
        for rows in (256, 512):  # the stock kernel's own batch variance, for scale
            c = compare(stock[:rows], stock_out(case, rows))
            stock_lines.append(describe(f"stock M768 vs M{rows} (first rows)", c))
    except pytest.skip.Exception:
        band_rows = errors(one, ref)["max_rel"]
        band_from = "spec M768 worst row (no stock arm)"
    pairs = {name: compare(one, out) for name, out in runs.items()}
    pairs["256 alone vs 3x256 (rows 512-767)"] = compare(alone, runs["3x256"][512:])
    pairs["256 alone vs 768 (rows 512-767)"] = compare(alone, one[512:])
    lines = [f"wide experts groupings, band {BAND} x {band_from} ({band_rows:.3e})"]
    lines += [describe(f"768 vs {n}" if " vs " not in n else n, c)
              for n, c in pairs.items()]  # fmt: skip
    lines += stock_lines
    # the reference band on every differing row of every grouping
    outside = []
    for name, out in (*runs.items(), ("768", one), ("256 alone", alone)):
        base = ref[512:] if name == "256 alone" else ref
        rel = errors(out, base)["rows"]
        lines.append(f"  {name}: {fmt(errors(out, base))}")
        bad = (rel > BAND * band_rows).nonzero().flatten().tolist()
        if bad:
            outside.append((name, bad[:8], rel.max().item()))
    for name, c in pairs.items():
        for r, *_ in c["detail"]:
            if name.startswith("256 alone"):
                continue
            mine = errors(one[r : r + 1], ref[r : r + 1])["max_rel"]
            other = errors(runs[name][r : r + 1], ref[r : r + 1])["max_rel"]
            lines.append(f"  row {r} ({name}): rel err 768 {mine:.3e}, {other:.3e}")
    summary = "\n".join(lines)
    print("\n" + summary)
    again = pairs["768 again"]["rows"]
    assert not again, "768-row launch not deterministic\n" + summary
    assert not outside, f"rows outside the reference band: {outside}\n{summary}"
    if any(c["rows"] for c in pairs.values()):
        raise GroupingNotBitwise(summary)
