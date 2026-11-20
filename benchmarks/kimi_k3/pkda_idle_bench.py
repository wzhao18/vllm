# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Idle-SM probe for the KDA in_proj: the c2x4 conv GEMM places 22 clusters =
176 of 212 SMs. Runs a second GEMM (the g rows, c2x1, grid capped to the
leftover SMs) on a side stream concurrently and compares with sequential.

Prints per arm: burst (one call after an idle gap, CUDA events) and
sustained (back-to-back mean) wall time in us.
"""

import argparse
import json
import time

import torch
from pkda_m9_bench import RASTER
from pgemm_conv_bench import operands


def timed(fn, cool, iters):
    fn()
    torch.cuda.synchronize()
    time.sleep(cool)
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    fn()
    e.record()
    torch.cuda.synchronize()
    burst = s.elapsed_time(e) * 1000
    for _ in range(3):
        fn()
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return round(burst, 1), round(s.elapsed_time(e) * 1000 / iters, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=5461)
    ap.add_argument("--p", type=int, default=12288)
    ap.add_argument("--k", type=int, default=7168)
    ap.add_argument("--gcols", default="12288,8192,6144")
    ap.add_argument("--side", default="2x1:18,2x1:16")
    ap.add_argument("--cool", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()

    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm.kda_conv_gemm import (  # noqa: E501
        KdaConvGemm,
        PlainGemm,
    )

    p, k = args.p, args.k
    n = 3 * p
    lens = [args.seq] * 3
    m = sum(lens)
    xq, xs, wq, ws = operands(m, n, k)
    gmax = max(int(c) for c in args.gcols.split(","))
    _, _, wg, wgs = operands(m, gmax, k)
    cw = (torch.randn(n, 4, device="cuda") * 0.5).contiguous()
    qsl = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device="cuda")
    lines = torch.arange(1, 4, dtype=torch.int32, device="cuda")
    has_init = torch.tensor([False, True, False], device="cuda")
    st = torch.randn(5, n, 4, device="cuda").to(torch.bfloat16).float()[:, :, :3]
    c24 = (2, 4)
    conv = KdaConvGemm(wq, ws, cw, p, tactic=((512, 256), c24, 256, *RASTER[c24], 0, 0, ""))
    out_g = torch.empty(m, gmax, dtype=torch.bfloat16, device="cuda")
    s1 = torch.cuda.Stream()
    ev = torch.cuda.Event()

    def plain(ncols, cl, mac):
        w = wg[:ncols]
        sc = wgs  # scale prefix rows match the first ncols rows (128x4 swizzle)
        g = PlainGemm(w, sc, tactic=((512, 256), cl, 256, *RASTER.get(cl, (8, "n")), 0, 0, ""),
                      max_clusters=mac)
        o = out_g[:, :ncols]
        return lambda: g(xq, xs, o)

    def f_conv():
        conv(xq, xs, st, qsl, lines, has_init)

    def conc(fg):
        def f():
            s0 = torch.cuda.current_stream()
            ev.record(s0)
            f_conv()
            s1.wait_event(ev)
            with torch.cuda.stream(s1):
                fg()
            s0.wait_stream(s1)
        return f

    def seq(fg):
        def f():
            f_conv()
            fg()
        return f

    for rnd in range(args.rounds):
        res = {"round": rnd, "conv": timed(f_conv, args.cool, args.iters)}
        for gc in [int(c) for c in args.gcols.split(",")]:
            full = plain(gc, c24, None)
            res[f"g{gc}_c2x4"] = timed(full, args.cool, args.iters)
            res[f"seq_conv+g{gc}"] = timed(seq(full), args.cool, args.iters)
            for sd in args.side.split(","):
                cl_s, mac = sd.split(":")
                cl = tuple(int(v) for v in cl_s.split("x"))
                side = plain(gc, cl, int(mac))
                res[f"g{gc}_{cl_s}m{mac}"] = timed(side, args.cool, args.iters)
                res[f"conc_conv||g{gc}_{cl_s}m{mac}"] = timed(conc(side), args.cool, args.iters)
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
