#!/usr/bin/env python3
"""B3 ncu driver: one launch per SharedSituGemm epilogue variant.

Every variant is compiled and launched once (warm-up), then launched once
more, so ``ncu -k regex:kernel_cutlass --launch-skip N --launch-count N``
profiles exactly one launch per variant in one report. Variants differ only
in the tactic flags (``drain``, ``drain+dskel1`` ...).
"""

import argparse
import json

import torch

from pgemm_b3_bench import real_operands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=16384)
    ap.add_argument("--inter", type=int, default=6144)
    ap.add_argument("--k", type=int, default=7168)
    ap.add_argument(
        "--tactic", type=json.loads, default=[[512, 256], [2, 1], 256, 8, "n", 0, 0, ""]
    )
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("flags", nargs="+")
    args = ap.parse_args()

    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm import (
        shared_situ_gemm as ssg,
    )

    dev = torch.device("cuda")
    torch.manual_seed(args.m)
    x, sx, w, sw = real_operands(args.m, 2 * args.inter, args.k, dev)
    ssg.interleave_gate_up_rows(w, sw)
    act = mxfp8_act.Mxfp8Buffer(args.m, args.inter, dev)
    gemms = []
    for flags in args.flags:
        t = list(args.tactic)
        t[7] = flags
        t = tuple(tuple(v) if isinstance(v, list) else v for v in t)
        g = ssg.SharedSituGemm(w, sw, 4.0, 25.0, tactic=t)
        g(x, sx, act)
        gemms.append(g)
    torch.cuda.synchronize()
    for _ in range(args.reps):
        for g in gemms:
            g(x, sx, act)
    torch.cuda.synchronize()
    print("variants", args.flags, flush=True)


if __name__ == "__main__":
    main()
