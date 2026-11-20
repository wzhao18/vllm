# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Router | latent down (4480 x 7168 BF16 -> FP32) on one Rubin GPU: the pinned
cuBLASLt kernel (``ltpin``, today's path) vs the sdec split-K kernel
(``PreparedProjection`` / ``SplitKGemm``) per tactic, with the rotating TMA issuers
(``splitk.ISSUE_LANES``) and with one issuer (``--legacy``).

Each call follows a cold-L2 producer stand-in (``torch.cuda._sleep``) after an L2
flush; the reported time is (producer + GEMM) minus producer alone, median us.
Checks every kernel against an FP64 reference and the split-K variants bitwise
against each other."""

import argparse
import statistics

import torch

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm import (
    projection,
    splitk,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.ltpin import LtPin

TACTICS = {
    32: "64.32.2.9,128.32.2.5,128.32.4.4,64.32.4.6",
    64: "64.64.2.6,128.64.2.4,128.64.4.3",
    96: "64.96.2.5,128.96.2.3,64.96.4.4",
}


def timed(fn, flush, cycles, iters, stream):
    ts = []
    for _ in range(iters):
        flush.sum()
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record(stream)
        torch.cuda._sleep(cycles)
        if fn is not None:
            fn()
        b.record(stream)
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="32,64,96")
    ap.add_argument("--sleep-us", type=float, default=4.0)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--legacy", action="store_true", help="also one TMA issuer")
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.manual_seed(0)
    w = (torch.randn(4480, 7168, device=dev) * 0.02).to(torch.bfloat16)
    flush = torch.ones(1 << 28, dtype=torch.bfloat16, device=dev)
    stream = torch.cuda.current_stream(dev)
    cycles = int(args.sleep_us * 1e-6 * 2.4e9)
    base = timed(None, flush, cycles, args.iters, stream)
    print(f"producer alone {base:.2f} us", flush=True)
    for n in (int(r) for r in args.rows.split(",")):
        x = (torch.randn(n, 7168, device=dev)).to(torch.bfloat16)
        ref = x.double() @ w.double().T
        scale = ref.abs().max().item()
        lt = LtPin("router_latent", 4480, 7168, (n,), out32=True, device=dev)
        y = torch.empty(n, 4480, device=dev, dtype=torch.float32)
        t = timed(lambda: lt.run(w, x, y), flush, cycles, args.iters, stream) - base
        err = (y.double() - ref).abs().max().item() / scale
        print(f"n{n} ltpin {t:.2f} us  rel_err {err:.1e}", flush=True)
        modes = [("rot", splitk.ISSUE_LANES)] + ([("one", 1)] if args.legacy else [])
        for spec in TACTICS[n].split(","):
            tactic = tuple(int(v) for v in spec.split("."))
            outs = {}
            for mode, lanes in modes:
                splitk.ISSUE_LANES = lanes
                projection.compile_projection.cache_clear()
                try:
                    proj = projection.PreparedProjection(
                        w, {n: tactic}, output_dtype=torch.float32
                    ).prepare()
                except Exception as e:  # tactic does not fit / compile error
                    print(f"n{n} sk {spec} {mode}: skip ({type(e).__name__}: "
                          f"{str(e)[:120]})", flush=True)
                    continue
                y = torch.empty(n, 4480, device=dev, dtype=torch.float32)
                t = timed(lambda: proj.run(x, y), flush, cycles, args.iters, stream)
                t -= base
                err = (y.double() - ref).abs().max().item() / scale
                outs[mode] = y.clone()
                gbs = w.numel() * 2 / t / 1e6
                print(f"n{n} sk {spec} {mode} {t:.2f} us ({gbs:.1f} TB/s) "
                      f"rel_err {err:.1e}", flush=True)
            if len(outs) == 2:
                print(f"n{n} sk {spec} rot==one bitwise {torch.equal(*outs.values())}")
            splitk.ISSUE_LANES = modes[0][1]


if __name__ == "__main__":
    main()
