"""R5-M1 bench: BF16 sm_107 CuTeDSL persistent GEMM (bf16_gemm.Bf16Gemm) vs cuBLAS (torch).

python3 pgemm_r5_bf16.py --shapes 16384:12288:7168,16384:3584:7168 --tactics '256,256,2,1;256,256,2,2'
Tactic: tm,tn,cm,cn[,max_ab]. Prints us/call (CUDA events, back-to-back) and max |diff| vs torch.
"""
import argparse
import json

import torch

from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm import bf16_gemm as G


def timeit(fn, iters):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0 / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="16384:12288:7168")
    ap.add_argument("--tactics", default="256,256,2,1")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.manual_seed(0)
    res = []
    for sh in a.shapes.split(","):
        m, n, k = (int(v) for v in sh.split(":"))
        x = (torch.randn(m, k, device="cuda") * 0.5).to(torch.bfloat16)
        w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
        ref = x @ w.t()
        r = {"shape": sh, "cublas_us": timeit(lambda: torch.matmul(x, w.t()), a.iters)}
        for tac in a.tactics.split(";"):
            f = [int(v) for v in tac.split(",")]
            g = G.Bf16Gemm(((f[0], f[1]), (f[2], f[3]), f[4] if len(f) > 4 and f[4] else None))
            out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
            g(x, w, out)
            torch.cuda.synchronize()
            d = (out.float() - ref.float()).abs()
            ulp = (d / ref.float().abs().clamp(min=1e-3)).max().item()
            r[tac] = {
                "us": timeit(lambda: g(x, w, out), a.iters),
                "max_rel": round(ulp, 5),
                "equal_frac": round((out == ref).float().mean().item(), 4),
            }
        print(json.dumps(r), flush=True)
        res.append(r)
    if a.out:
        with open(a.out, "a") as f:
            for r in res:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
