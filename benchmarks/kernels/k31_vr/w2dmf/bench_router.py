# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K3.1 DEP MoE router gate (BF16 [M, 7168] x [896, 7168]^T -> FP32 logits):
cuBLAS (torch.mm out_dtype fp32, the GateLinear tier 6) vs the CuTe DSL
ll_bf16 split-K kernel, alone and with a concurrent 128x(2*3072+latent)x7168
GEMM on a second stream (the packed MoE down/gate_up GEMM it overlaps with).
Times = CUDA graph of 20 back-to-back router calls, median of 7."""

import argparse
import statistics

import torch


def graph_time(fn, reps=20, samples=7):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    out = []
    for _ in range(samples):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b) * 1e3 / reps)
    return statistics.median(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="48,128,224")
    args = p.parse_args()
    from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import ll_bf16_gemm

    torch.manual_seed(0)
    K, N = 7168, 896
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    side_w = torch.randn(2 * 3072 + 3584, K, device="cuda", dtype=torch.bfloat16) * 0.02
    side = torch.cuda.Stream()
    for M in map(int, args.rows.split(",")):
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        ref = x.float() @ w.float().t()
        fns = {
            "cublas_fp32out": lambda: torch.mm(x, w.t(), out_dtype=torch.float32),
            "ll_bf16": lambda: ll_bf16_gemm(x, w),
        }
        for name, fn in fns.items():
            err = ((fn() - ref).abs().max() / ref.abs().max()).item()
            alone = graph_time(fn)

            def both(fn=fn):
                ev = torch.cuda.Event()
                ev.record()
                with torch.cuda.stream(side):
                    ev.wait()
                    torch.mm(x, side_w.t())
                    ev2 = torch.cuda.Event()
                    ev2.record()
                fn()
                ev2.wait()

            conc = graph_time(both)
            side_only = graph_time(lambda: torch.mm(x, side_w.t()))
            print(f"M {M:4d} {name:15s} alone {alone:6.2f} us | with side GEMM {conc:6.2f} us "
                  f"(side alone {side_only:6.2f}) | max rel err {err:.1e}")


if __name__ == "__main__":
    main()
