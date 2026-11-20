# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KDA conv fix-up: blocks per program (FIXUP_BG) sweep, alone and right
after the conv GEMM (in-stream order), bitwise vs the default."""

import argparse
import json

import torch
from pkda_m9_bench import RASTER
from pgemm_conv_bench import kernel_times, operands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="5461,2731")
    ap.add_argument("--nseq", default="3,1,8")
    ap.add_argument("--bgs", default="8,4,2,1")
    ap.add_argument("--blocks", default="512,256")
    ap.add_argument("--p", type=int, default=12288)
    ap.add_argument("--k", type=int, default=7168)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm import conv_fixup as cf
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm.kda_conv_gemm import (  # noqa: E501
        KdaConvGemm,
    )

    p, k = args.p, args.k
    n = 3 * p
    c24 = (2, 4)
    for sl in [int(v) for v in args.seqs.split(",")]:
        m = 3 * sl
        xq, xs, wq, ws = operands(m, n, k)
        cw = (torch.randn(n, 4, device="cuda") * 0.5).contiguous()
        g = KdaConvGemm(wq, ws, cw, p, tactic=((512, 256), c24, 256, *RASTER[c24], 0, 0, ""))
        for ns in [int(v) for v in args.nseq.split(",")]:
            base = m // ns
            lens = [base] * (ns - 1) + [m - base * (ns - 1)]
            qsl = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device="cuda")
            lines = torch.arange(1, ns + 1, dtype=torch.int32, device="cuda")
            has_init = (torch.arange(ns, device="cuda") % 2 == 1)
            st0 = torch.randn(ns + 2, n, 4, device="cuda").to(torch.bfloat16).float()[:, :, :3]
            ref = None
            for blk in [int(v) for v in args.blocks.split(",")]:
                for bg in [int(v) for v in args.bgs.split(",")]:
                    cf.FIXUP_BG = bg
                    cf.FIXUP_BLOCK = blk
                    st = st0.clone()
                    y, _ = g(xq, xs, st, qsl, lines, has_init)
                    torch.cuda.synchronize()
                    row = {"m": m, "nseq": ns, "block": blk, "bg": bg}
                    if ref is None:
                        ref = (y.clone(), st.clone())
                    else:
                        row["bitwise"] = bool(torch.equal(y, ref[0]) and torch.equal(st, ref[1]))
                    side = torch.empty(6 * ((m + 127) // 128) + 6 * ns, n, dtype=torch.bfloat16, device="cuda")
                    side.normal_()
                    out = torch.empty(3, m, p, dtype=torch.bfloat16, device="cuda")

                    def f_fix():
                        cf.conv_fixup(out, side, None, cw, st, qsl, lines, has_init, p, None)

                    def f_both():
                        g(xq, xs, st, qsl, lines, has_init)

                    row["fixup_alone"] = kernel_times(f_fix, iters=args.iters)[0]
                    tot, top = kernel_times(f_both, iters=args.iters)
                    row["gemm_fixup"] = tot
                    row["kernels"] = top[:2]
                    print(json.dumps(row), flush=True)
            cf.FIXUP_BG = None
            cf.FIXUP_BLOCK = 512


if __name__ == "__main__":
    main()
