#!/usr/bin/env python3
"""B3 interleaved A/B: the deployed shared gate_up path (cuBLASLt MXFP8 GEMM +
situ_and_mul_quantize_auto) vs plain CuTe base tactics and fused
SharedSituGemm tactics, measured round-robin (every variant once per round,
median over rounds) so clock/power drift and co-tenants hit all variants
alike. Real-scale random MXFP8 operands; correctness is pgemm_b3_bench.py.
"""

import argparse
import json
import statistics

import torch

from pgemm_b3_bench import clock_power, kernel_times, real_operands, scaled_mm, tname


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=16384)
    ap.add_argument("--inter", type=int, default=6144)
    ap.add_argument("--k", type=int, default=7168)
    ap.add_argument("--beta", type=float, default=4.0)
    ap.add_argument("--lbeta", type=float, default=25.0)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--fused", type=json.loads, default=[])
    ap.add_argument("--plain", type=json.loads, default=[])
    args = ap.parse_args()

    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm import (
        shared_situ_gemm as ssg,
    )
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm.kda_conv_gemm import (  # noqa: E501
        PlainGemm,
    )

    def tac(t):
        return tuple(tuple(v) if isinstance(v, list) else v for v in t)

    dev = torch.device("cuda")
    m, inter, k = args.m, args.inter, args.k
    n = 2 * inter
    torch.manual_seed(m)
    x, sx, w, sw = real_operands(m, n, k, dev)
    ref = mxfp8_act.Mxfp8Buffer(m, inter, dev)
    out = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
    act = mxfp8_act.Mxfp8Buffer(m, inter, dev)
    wi, swi = w.clone(), sw.clone()
    ssg.interleave_gate_up_rows(wi, swi)

    variants = {
        "ref": lambda: mxfp8_act.situ_and_mul_quantize_auto(
            scaled_mm(x, sx, w, sw), args.beta, args.lbeta, out=ref
        ),
        "cublas": lambda: scaled_mm(x, sx, w, sw),
    }
    for t in args.plain:
        g = PlainGemm(w, sw, tac(t))
        variants["plain_" + tname(tac(t))] = lambda g=g: g(x, sx, out)
    for t in args.fused:
        g = ssg.SharedSituGemm(wi, swi, args.beta, args.lbeta, tactic=tac(t))
        variants["fused_" + tname(tac(t))] = lambda g=g: g(x, sx, act)
    for fn in variants.values():
        fn()
    torch.cuda.synchronize()
    times = {name: [] for name in variants}
    for _ in range(args.rounds):
        for name, fn in variants.items():
            times[name].append(kernel_times(fn)[0])
    rmed = statistics.median(times["ref"]) or 1.0
    for name, fn in variants.items():
        mhz, watt = clock_power(fn, iters=200)
        med = statistics.median(times[name])
        print(json.dumps({"v": name, "med": round(med, 1), "min": round(min(times[name]), 1),
                          "max": round(max(times[name]), 1), "x_ref": round(med / rmed, 3),
                          "mhz": mhz, "w": watt}), flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
