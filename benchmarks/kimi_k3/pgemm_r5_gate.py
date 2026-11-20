"""R5-M2 bench: DSA g_proj with the sigmoid output gate in the GEMM epilogue (Bf16GateGemm)
vs the deployed cuBLAS g_proj + prefill_glue.gate_sigmoid_mul.

python3 pgemm_r5_gate.py [--m 16384] [--tactics '256,256,2,1'] [--iters 20]
"""
import argparse
import json

import torch

from vllm.models.kimi_k3.nvidia.ops import prefill_glue
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
    ap.add_argument("--m", type=int, default=16384)
    ap.add_argument("--n", type=int, default=12288)
    ap.add_argument("--k", type=int, default=7168)
    ap.add_argument("--tactics", default="256,256,2,1")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.manual_seed(0)
    m, n, k = a.m, a.n, a.k
    x = (torch.randn(m, k, device="cuda") * 0.5).to(torch.bfloat16)
    w = (torch.randn(n, k, device="cuda") * 0.03).to(torch.bfloat16)
    attn = torch.randn(m, n, device="cuda").to(torch.bfloat16)
    ref = attn * (x @ w.t()).sigmoid()
    r = {"m": m, "n": n, "k": k}
    g = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    o = torch.empty_like(g)
    r["cublas_gemm_us"] = timeit(lambda: torch.matmul(x, w.t(), out=g), a.iters)
    r["gate_kernel_us"] = timeit(lambda: o.copy_(prefill_glue.eager_gate_sigmoid_mul(attn, g)), a.iters)
    st = prefill_glue.eager_gate_sigmoid_mul(attn, x @ w.t())
    r["stock_vs_eager_mismatch"] = int((st != ref).sum())
    for tac in a.tactics.split(";"):
        f = [int(v) for v in tac.split(",")]
        gg = G.Bf16GateGemm(
            ((f[0], f[1]), (f[2], f[3]), f[4] if len(f) > 4 and f[4] else None),
            ablate=f[5] if len(f) > 5 else 0,
        )
        out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        gg(x, w, attn, out)
        torch.cuda.synchronize()
        r[tac] = {
            "us": timeit(lambda: gg(x, w, attn, out), a.iters),
            "mismatch": int((out != ref).sum()),
        }
        print(tac, json.dumps(r[tac]), flush=True)
    print(json.dumps(r), flush=True)
    if a.out:
        with open(a.out, "a") as fh:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
