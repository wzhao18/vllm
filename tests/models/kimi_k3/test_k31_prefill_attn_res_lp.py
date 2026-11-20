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


@pytest.mark.parametrize("rows", [1500, 77])
def test_attn_res_lp_sequence(rows):
    from vllm import _custom_ops as ops
    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act
    from vllm.models.kimi_k3.nvidia.specialized_prefill import attn_res_lp

    nl, bs = 7, 2
    calls = _calls(nl, bs)
    nbt = math.ceil(nl / bs)
    g = torch.Generator(device="cuda").manual_seed(0)

    def rn(*s, scale=1.0):
        return (torch.randn(*s, device="cuda", generator=g) * scale).bfloat16()

    nw = [(1 + rn(H, scale=0.1).float()).bfloat16() for _ in calls]
    qk = [rn(H, scale=H**-0.5) for _ in calls]
    onw = (1 + rn(H, scale=0.1).float()).bfloat16()
    plan = attn_res_lp.Plan(list(zip(nw, qk)), [EPS] * len(calls), bs, nbt)
    prefix0 = rn(rows, H)
    deltas = [rn(rows, H, scale=0.5) for _ in calls]
    attn_out = rn(rows, H)

    ref_blocks = torch.zeros(rows, nbt, H, device="cuda", dtype=torch.bfloat16)
    lp_blocks = ref_blocks.clone()
    ref_prefix, lp_prefix = prefix0.clone(), prefix0.clone()
    attn_res_lp.begin(plan, lp_blocks)
    for c in calls:
        if c["post"] and not c["delta"]:
            ref_prefix, lp_prefix = attn_out.clone(), attn_out.clone()
        d = deltas[c["c"]] if c["delta"] else None
        o = onw if c["norm"] else None
        r64 = _ref64(ref_prefix, d, ref_blocks, c["nb"],
                     nw[c["c"]].double() * qk[c["c"]].double(), o)
        ref = ops.kimi_k3_attn_res(ref_prefix, d, ref_blocks, nw[c["c"]],
                                   qk[c["c"]], o, c["nb"], c["widx"], EPS, EPS)
        out, qa = attn_res_lp.attn_res(lp_prefix, d, lp_blocks, c["c"], c["nb"],
                                       c["widx"], o, EPS, True, c["quant"])
        torch.accelerator.synchronize()
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
