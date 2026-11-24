# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 specialized decode: the DSpark drafter's LM-head shard GEMM at TP16
([vocab / 16, 7168] BF16 shard) on one GPU, cold weights (one weight copy per call in a
CUDA graph), us per call, for every drafter row count (q8: 7 query rows per request):
the BF16-logits GEMV (``cutedsl.verify.lmhead``, <= 8 rows), cuBLAS (F.linear) and the
die-local stream-K kernel (``cutedsl.gemm.wide_sk``, rows padded to 32; ``sk`` with the
shard copied into die-homed halves, ``skip`` with the shard re-homed in place, ``skst``
streamed where it lies). Prints max |d| / max |ref| vs F.linear.

python benchmarks/kernels/bench_kimi_k3_sdec_draft_lm_head_tp16.py [--rows 7,14,28,56]
"""

import argparse
import statistics

import torch
import torch.nn.functional as F


def time_graph(fn, ncopy, reps=10, rounds=5):
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for i in range(ncopy):
            fn(i)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for i in range(ncopy):
            fn(i)
    ts = []
    for _ in range(rounds):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / reps / ncopy)
    del g
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="7,14,28,56,112,224,448")
    ap.add_argument("--n", type=int, default=10240)
    ap.add_argument("--k", type=int, default=7168)
    ap.add_argument("--arms", default="gemv,cublas,sk,skip,skst,sp,spip")
    args = ap.parse_args()
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm import wide_sk
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.verify import lmhead

    dev = torch.device("cuda")
    torch.manual_seed(0)
    n, k = args.n, args.k
    arms = args.arms.split(",")
    _, pairs = wide_sk.die_pairs(dev)
    ncopy = max(2, min(16, int(1.2e9 // (n * k * 2)) + 1))
    print(f"{torch.cuda.get_device_name()} N{n} K{k} pairs/die {pairs} ncopy {ncopy}",
          flush=True)
    ws = [torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(ncopy)]
    for t in [int(r) for r in args.rows.split(",")]:
        rows = -(-t // 32) * 32
        xp = torch.zeros(rows, k, device=dev, dtype=torch.bfloat16)
        xp[:t].normal_()
        x = xp[:t]
        ref = F.linear(x, ws[0]).float()
        res = {}

        def check(tag, out):
            err = ((out[:t].float() - ref).abs().max() / ref.abs().max()).item()
            res[tag + ".err"] = err

        out = torch.empty(rows, n, device=dev, dtype=torch.bfloat16)
        if "cublas" in arms:
            torch.matmul(x, ws[0].t(), out=out[:t])
            check("cublas", out)
            res["cublas"] = time_graph(lambda i: torch.matmul(x, ws[i].t(), out=out[:t]),
                                       ncopy)
        if "gemv" in arms and t <= 8:
            lms = [lmhead.PreparedLmHeadLogits(w, dev) for w in ws]
            check("gemv", lms[0].run(x))
            o = torch.empty(t, n, device=dev, dtype=torch.bfloat16)
            res["gemv"] = time_graph(lambda i: lms[i].run(x, o), ncopy)
            del lms
        for arm, place in (("sk", "copy"), ("skip", "inplace"), ("skst", "none"),
                           ("sp", "copy"), ("spip", "inplace")):
            if arm not in arms:
                continue
            spread = arm.startswith("sp")
            if not wide_sk.supports(rows, n, k, pairs, spread=spread):
                res[arm] = float("nan")
                continue
            if place == "inplace":
                # re-home each copy in place: the die-homed allocation replaces it
                homed = [wide_sk.place_halves(w) for w in ws]
                pk = [wide_sk.PreparedWideSK(h, [rows], place=False, spread=spread).prepare()
                      for h in homed]
            else:
                pk = [wide_sk.PreparedWideSK(w, [rows], place=place == "copy",
                                             spread=spread).prepare()
                      for w in ws]
            out.zero_()
            pk[0].run(xp, out)
            torch.cuda.synchronize()
            check(arm, out)
            res[arm] = time_graph(lambda i, pk=pk: pk[i].run(xp, out), ncopy)
            del pk
            torch.cuda.empty_cache()
        gb = n * k * 2 / 1e6
        line = " ".join(
            f"{a}={v:.2e}" if a.endswith(".err") else f"{a}={v:.2f}us({gb / v:.1f}TB/s)"
            for a, v in res.items()
        )
        print(f"T{t} (pad {rows}): {line}", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
