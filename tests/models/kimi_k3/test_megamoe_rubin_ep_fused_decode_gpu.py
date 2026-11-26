# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EP16 / EP64 NVFP4 decode on the EP32 128-token GenPhase decode bucket
(select_rubin_buckets, RUBIN_FUSED_DECODE_EP_SIZES), one VR200.

The bucket runs the src_gp GenPhase kernel with RUBIN_GENPHASE_DECODE_KNOBS at 56 (EP16)
and 14 (EP64) local experts instead of 28. Checks, on a native ModelOpt NVFP4 layer
(test_megamoe_rubin_nvfp4_native_gpu helpers), that it matches the stock NVFP4 MoE math
and the quantization-faithful oracle to the same bound as the EP32 decode bucket, and
that it agrees with the inference-kernel bucket these servers ran before (same
quantization points; only the accumulation order differs)."""

import pytest
import torch

from tests.models.kimi_k3 import test_megamoe_rubin_nvfp4_native_gpu as N

pytestmark = N.pytestmark


@pytest.fixture(scope="module")
def build():
    C, R, W, RubinMegaMoE = N._mods()
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import experts as X

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    torch.zeros(1, device=device)
    entries = C.load_knob_source(C.SHIPPED_KNOBS)["entries"]
    cache = {}

    def get(experts, kind):
        key = (experts, kind)
        if key in cache:
            return cache[key]
        problem = C.RubinMoeProblem(num_experts=experts, quant_kind="nvfp4")
        if kind == "gp128":  # the new bucket
            bucket = C.RubinBucket("genphase", 128, X.RUBIN_GENPHASE_DECODE_KNOBS)
        else:  # the previous EP16 / EP64 decode bucket: inference 256 on the
            # EP16 2048 record's knobs (EP64 borrows EP32's)
            ws = 16 if experts == 56 else 32
            recs = [r for r in entries if r["dtype"] == "nvfp4"
                    and r["world_size"] == ws and r["kernel_variant"] == "inference"
                    and r["max_tokens"] == 2048]  # fmt: skip
            assert recs
            rec = min(recs, key=lambda r: r.get("p50_us") or float("inf"))
            bucket = C.RubinBucket("inference", 256, C.RubinKnobs.from_dict(rec["knobs"]))
        moe = RubinMegaMoE(problem, None, [bucket])
        layer = N._checkpoint_layer(problem, W, device, seed=7)
        weights = moe.prepare_nvfp4_checkpoint_weights(*layer)
        moe.warmup(weights, ids_dtypes=(torch.int32,))
        cache[key] = (C, R, W, problem, moe, bucket, layer, weights)
        return cache[key]

    return get


def _run(state, n, seed):
    C, R, W, problem, moe, bucket, layer, weights = state
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(100 + seed)
    hidden = torch.randn(n, problem.hidden, generator=gen, device=device)
    hidden = hidden.to(torch.bfloat16)
    ids, wts = N._routes(n, problem.num_experts, device, seed=200 + seed)
    out = torch.full((n, problem.hidden), float("nan"), dtype=torch.bfloat16,
                     device=device)  # fmt: skip
    moe.forward(hidden, ids, wts, weights, bucket=bucket, out=out)
    torch.cuda.synchronize()
    moe.check_route_errors()
    return hidden, ids, wts, out


@pytest.mark.parametrize("experts", [56, 14])
@pytest.mark.parametrize("n", [1, 24, 37, 96, 128])
def test_gp128_decode_bucket_matches_stock_and_previous(build, experts, n):
    gp = build(experts, "gp128")
    C, R, W, problem, moe, bucket, layer, weights = gp
    hidden, ids, wts, out = _run(gp, n, n)
    assert bool(torch.isfinite(out).all())
    stock = N._stock_nvfp4_moe(hidden, ids, wts, layer, problem, W, R)
    x, xsf, idx, sw = R.stage_inputs_torch(
        hidden, ids, wts, quant_kind="nvfp4", capacity=bucket.max_tokens,
        sf_cols=problem.hidden // W.NV_BLOCK, num_experts=problem.num_experts,
    )  # fmt: skip
    oracle = R.megamoe_oracle(
        x[:n], xsf[:n], idx[:n], sw[:n], weights, quant_kind="nvfp4",
        local_expert_offset=0, hidden=problem.hidden,
        intermediate=problem.intermediate, **N.SITU,
    )  # fmt: skip
    prev = build(experts, "inf256")
    _, _, _, out_prev = _run(prev, n, n)
    errs = {
        "gp_stock": N._rel(out, stock),
        "gp_oracle": N._rel(out, oracle),
        "prev_stock": N._rel(out_prev, stock),
        "gp_prev": N._rel(out, out_prev),
        "gp_prev_bitwise": bool(torch.equal(out, out_prev)),
        "gp_prev_maxabs": float((out.float() - out_prev.float()).abs().max()),
    }
    print(f"E{experts} n={n} {errs}")
    assert errs["gp_stock"] < 5e-3 and errs["gp_oracle"] < 5e-3
    assert errs["prev_stock"] < 5e-3
    assert errs["gp_prev"] < 5e-3
