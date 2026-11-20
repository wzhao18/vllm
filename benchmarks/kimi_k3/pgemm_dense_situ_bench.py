# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill dense-MLP SiTU-and-mul: precise-tanhf vec8 kernel (ops/situ_and_mul.py)
vs the routed experts' MUFU-light form with BF16 output (mxfp8_act
situ_and_mul_fast_bf16), at the dense layer's shape. Prints us per call, the
HBM rate and the max BF16 ulp distance between the two."""
import json
import sys

import torch


def timed(fn, iters=20, warm=3):
    for _ in range(warm):
        fn()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) * 1000 / iters


def ordered(t):
    i = t.view(torch.int16).int()
    return torch.where(i < 0, -(i & 0x7FFF), i)


def main():
    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act
    from vllm.models.kimi_k3.nvidia.ops.situ_and_mul import situ_and_mul

    cfg = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
    dense = int(cfg.get("intermediate_size", 18432))
    beta = float(cfg.get("situ_beta", cfg.get("swiglu_beta", 4.0)) or 4.0)
    lbeta = cfg.get("situ_linear_beta", 25.0)
    print(json.dumps({"dense_intermediate": dense, "beta": beta, "linear_beta": lbeta,
                      "cfg_keys": [k for k in cfg if "situ" in k or "beta" in k]}))
    for rows, d in ((16384, dense), (8192, dense), (16384, 6144)):
        x = (torch.randn(rows, 2 * d, device="cuda") * 4.0).to(torch.bfloat16)
        t_vec = timed(lambda: situ_and_mul(x, beta, lbeta))
        t_fast = timed(lambda: mxfp8_act.situ_and_mul_fast_bf16(x, beta, lbeta))
        ulp = (ordered(mxfp8_act.situ_and_mul_fast_bf16(x, beta, lbeta))
               - ordered(situ_and_mul(x, beta, lbeta))).abs()
        f, pr = mxfp8_act.situ_and_mul_fast_bf16(x, beta, lbeta).float(), situ_and_mul(x, beta, lbeta).float()
        amax = pr.abs().reshape(rows, -1, 32).amax(-1, keepdim=True)
        bulp = torch.exp2(torch.floor(torch.log2(amax.clamp_min(2.0**-126))) - 7)
        berr = float(((f - pr).abs().reshape(rows, -1, 32) / bulp).max())
        gb = rows * d * 6 / 1e9
        print(json.dumps({"rows": rows, "d": d, "vec8_us": round(t_vec, 1), "fast_us": round(t_fast, 1),
                          "speedup": round(t_vec / t_fast, 2), "fast_TBps": round(gb / t_fast * 1e3, 2),
                          "ulp_max": int(ulp.max()), "block_ulp_err": round(berr, 3), "ulp_frac": float((ulp > 0).float().mean())}), flush=True)


if __name__ == "__main__":
    main()
