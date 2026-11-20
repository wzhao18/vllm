# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""specialized_prefill.attn_res_lp (precomputed block logits) on a short
AttnRes call sequence vs the native kimi_k3_attn_res: prefix / block writes
bit-exact, MXFP8 == mxfp8_quantize(the BF16 output) bit for bit, BF16 output
equal up to BF16 rounding of a reordered FP32 sum (and no further from an FP64
reference than the native kernel)."""

import math

import pytest
import torch

from vllm.platforms import current_platform

H = 7168
EPS = 1e-6

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100)
    or not hasattr(torch.ops._C, "kimi_k3_attn_res"),
    reason="native AttnRes requires an SM100-family build",
)


def _calls(nl, bs):
    calls = []
    for l in range(nl):
        write = l % bs == 0
        prev = math.ceil(l / bs)
        calls.append(dict(c=2 * l, post=False, nb=prev,
                          widx=l // bs if write else -1, delta=l > 0, norm=True,
                          quant=l % 2 == 1, bf16=l % 2 == 0))
        calls.append(dict(c=2 * l + 1, post=True, nb=prev + int(write), widx=-1,
                          delta=not write, norm=True, quant=l > 0, bf16=True))
    calls.append(dict(c=2 * nl, post=False, nb=math.ceil(nl / bs), widx=-1,
                      delta=True, norm=False, quant=False, bf16=True))
    return calls


def _ref64(prefix, delta, blocks, nb, w, onw):
    u = prefix.double()
    if delta is not None:
        u = (prefix.float() + delta.float()).bfloat16().double()
    src = [blocks[:, i].double() for i in range(nb)] + [u]
    lg = torch.stack(
        [(v @ w) * torch.rsqrt((v * v).mean(-1) + EPS) for v in src], -1)
    p = torch.softmax(lg, -1)
    acc = sum(p[:, j:j + 1] * v for j, v in enumerate(src))
    if onw is not None:
        acc = acc * torch.rsqrt((acc * acc).mean(-1, keepdim=True) + EPS) * onw.double()
    return acc


REAL_CKPT = "/lustre/share/coreai_comparch_aarwlt/models/nvidia_kimi-k31-pro-nvfp4/hf-1d82032"
PREFIX = "language_model.model."


def _real_attn_res_tensors(nl: int | None = None):
    """(norm, proj) BF16 tensors of the AttnRes calls of layers 0..nl-1 and the
    final AttnRes, in call order, read straight from the real checkpoint."""
    import json
    import os

    from safetensors import safe_open

    if not os.path.isdir(REAL_CKPT):
        pytest.skip(f"real checkpoint {REAL_CKPT} not on this cluster")
    wmap = json.load(open(os.path.join(REAL_CKPT, "model.safetensors.index.json")))
    wmap = wmap["weight_map"]
    if nl is None:
        nl = 1 + max(int(k.split(".")[3]) for k in wmap if "res_norm" in k and ".layers." in k)
    names = []
    for l in range(nl):
        for kind in ("self_attention", "mlp"):
            names.append(f"{PREFIX}layers.{l}.{kind}_res_")
    names.append(f"{PREFIX}output_attn_res_")
    out = []
    for n in names:
        pair = []
        for part in ("norm", "proj"):
            key = f"{n}{part}.weight"
            with safe_open(os.path.join(REAL_CKPT, wmap[key]), "pt") as f:
                pair.append(f.get_tensor(key).reshape(-1))
        out.append(tuple(pair))
    return out


def test_plan_query_is_the_stock_query_exactly():
    """Plan's FP32 query is the stock kernel's (flush-to-zero, as its
    --use_fast_math build: subnormal weights and products are zero) and its
    BF16 terms sum to 2^32 times it exactly, also for the hard values the real
    checkpoint has: subnormal weights, products below 2^-126, products in
    [2^-126, 2^-118) (subnormal low term unscaled) and FP32 norm weights."""
    from vllm.models.kimi_k3.nvidia.specialized_prefill import attn_res_lp

    tiny = torch.finfo(torch.float32).tiny
    g = torch.Generator().manual_seed(1)
    nw = (torch.randn(H, generator=g) * 0.05).bfloat16()
    qk = (torch.randn(H, generator=g) * H**-0.5).bfloat16()
    qk[:8] = torch.tensor([9.18e-41, -3e-39, 1e-37, 2e-36, 5e-35, 0.0, 1e-30, 4e-38]).bfloat16()
    nw[8:12] = torch.tensor([1e-39, 0.03, 0.05, 0.07]).bfloat16()
    qk[8:12] = torch.tensor([0.5, 1e-37, 3e-37, 2e-36]).bfloat16()
    nw32 = torch.randn(H, generator=g) * 0.05  # 24-bit significands: 3 terms
    for weights, nterms in (([(nw, qk)] * 3, 2), ([(nw32, qk)] * 3, 3)):
        plan = attn_res_lp.Plan(weights, [EPS] * 3, 1, 1)
        n, q = weights[0]
        prod = n.float() * q.float()
        ref = torch.where(
            (n.float().abs() < tiny) | (q.float().abs() < tiny) | (prod.abs() < tiny),
            torch.zeros_like(prod), prod)
        assert torch.equal(plan.w32[0], ref)
        assert len(plan.terms) == nterms
        total = sum(t.double() for t in plan.terms)
        assert torch.equal(total, plan.w32.double() * attn_res_lp.QUERY_SCALE)


def test_plan_builds_from_the_real_checkpoint():
    """Every AttnRes query of hf-1d82032 (187 calls), from the safetensors and
    from the parameters vLLM loads them into (RMSNorm weight, BF16 [1, H] proj
    weight through the default weight loader), is an exact BF16 term sum."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    from vllm.models.kimi_k3.nvidia.specialized_prefill import attn_res_lp

    real = _real_attn_res_tensors()
    assert len(real) == 187
    plans = [attn_res_lp.Plan(real, [EPS] * len(real), 12, 8)]
    loaded = []
    with set_current_vllm_config(VllmConfig()):
        for nw, qk in real:
            norm = RMSNorm(H, eps=EPS, dtype=torch.bfloat16)
            default_weight_loader(norm.weight, nw)
            proj = torch.nn.Parameter(
                torch.empty(1, H, dtype=torch.bfloat16), requires_grad=False
            )
            default_weight_loader(proj, qk.reshape(1, H))
            loaded.append((norm.weight.data, proj.data.squeeze(0)))
    plans.append(attn_res_lp.Plan(loaded, [EPS] * len(loaded), 12, 8))
    for plan in plans:
        assert len(plan.terms) == 2
        total = sum(t.double() for t in plan.terms)
        assert torch.equal(total, plan.w32.double() * attn_res_lp.QUERY_SCALE)
    assert torch.equal(plans[0].w32, plans[1].w32)


@pytest.mark.parametrize(
    "rows,dielocal,weights",
    [(1500, False, "synthetic"), (77, False, "synthetic"), (5000, True, "synthetic"),
     (77, True, "synthetic"), (1500, False, "real"), (5000, True, "real")],
)
def test_attn_res_lp_sequence(rows, dielocal, weights):
    """``dielocal``: the production prefill path (die-local block / prefix
    buffers, die-aware token claims, rotating die-local output buffers).
    ``weights`` real: the AttnRes norm / proj weights of layers 0..6 and the
    final AttnRes of hf-1d82032 (subnormal weights and tiny products)."""
    from vllm import _custom_ops as ops
    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act
    from vllm.models.kimi_k3.nvidia.specialized_prefill import (
        attn_res_dielocal,
        attn_res_lp,
    )

    nl, bs = 7, 2
    calls = _calls(nl, bs)
    nbt = math.ceil(nl / bs)
    g = torch.Generator(device="cuda").manual_seed(0)

    def rn(*s, scale=1.0):
        return (torch.randn(*s, device="cuda", generator=g) * scale).bfloat16()

    if weights == "real":
        real = _real_attn_res_tensors(nl)
        nw = [n.cuda() for n, _ in real]
        qk = [q.cuda() for _, q in real]
    else:
        nw = [(1 + rn(H, scale=0.1).float()).bfloat16() for _ in calls]
        qk = [rn(H, scale=H**-0.5) for _ in calls]
    onw = (1 + rn(H, scale=0.1).float()).bfloat16()
    plan = attn_res_lp.Plan(list(zip(nw, qk)), [EPS] * len(calls), bs, nbt)
    prefix0 = rn(rows, H)
    deltas = [rn(rows, H, scale=0.5) for _ in calls]
    attn_out = rn(rows, H)

    ref_blocks = torch.zeros(rows, nbt, H, device="cuda", dtype=torch.bfloat16)
    if dielocal:
        lp_blocks = attn_res_dielocal.block_residual(prefix0, nbt)
        if lp_blocks is None:
            pytest.skip("single-die device")
        assert attn_res_dielocal.kernel_args(lp_blocks)[0] is not None
        lp_blocks.zero_()
    else:
        lp_blocks = ref_blocks.clone()
    ref_prefix, lp_prefix = prefix0.clone(), prefix0.clone()
    attn_res_lp.begin(plan, lp_blocks)
    prev = None  # (output tensor, its value) of the previous call
    for c in calls:
        if c["post"] and not c["delta"]:
            ref_prefix, lp_prefix = attn_out.clone(), attn_out.clone()
        d = deltas[c["c"]] if c["delta"] else None
        o = onw if c["norm"] else None
        r64 = _ref64(ref_prefix, d, ref_blocks, c["nb"],
                     nw[c["c"]].double() * qk[c["c"]].double(), o)
        ref = ops.kimi_k3_attn_res(ref_prefix, d, ref_blocks, nw[c["c"]],
                                   qk[c["c"]], o, c["nb"], c["widx"], EPS, EPS)
        po = attn_res_dielocal.prefix_out(lp_prefix, lp_blocks)
        assert po is None or dielocal
        assert po is not None or not dielocal or c["c"] > 0
        out, qa = attn_res_lp.attn_res(lp_prefix, d, lp_blocks, c["c"], c["nb"],
                                       c["widx"], o, EPS, True, c["quant"],
                                       prefix_out=po)
        if po is not None:
            lp_prefix = po
        torch.accelerator.synchronize()
        # Rotating output buffers: the previous call's output must survive
        # this call (its consumers run before the next-but-one call).
        if prev is not None:
            assert torch.equal(prev[0], prev[1]), c
        prev = (out, out.clone())
        assert torch.equal(lp_prefix, ref_prefix), c
        assert torch.equal(lp_blocks, ref_blocks), c
        rms = r64.pow(2).mean(-1, keepdim=True).sqrt()
        diff = (out.double() - ref.double()).abs()
        assert (diff <= 2.0**-7 * ref.double().abs() + 2.0**-10 * rms).all(), c
        e_lp = ((out.double() - r64).abs() / rms).mean().item()
        e_ref = ((ref.double() - r64).abs() / rms).mean().item()
        assert e_lp <= e_ref * 1.05 + 1e-7, (c, e_lp, e_ref)
        if c["quant"]:
            rq = mxfp8_act.quantize(out)
            assert torch.equal(rq.data.view(torch.uint8), qa.data.view(torch.uint8))
            assert torch.equal(rq.scale.view(torch.uint8), qa.scale.view(torch.uint8))
    assert lp_blocks.data_ptr() not in attn_res_lp._states  # released by the final call
