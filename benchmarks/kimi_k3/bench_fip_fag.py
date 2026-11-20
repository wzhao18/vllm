# SPDX-License-Identifier: Apache-2.0
"""Persistent KDA prefill: stock gate projection vs the in-kernel one (FAG).

A: g1 = f_b_proj(f_a) (torch BF16 GEMM) then the persistent core reads g1.
B: the persistent core reads f_a and the f_b_proj weight and forms g1 with a
   tcgen05 MMA per chunk (FP32 accumulate, BF16 round as the stock GEMM).

Reports output / final-state differences (B vs A; only accumulation order
differs inside the K=128 dot product) and CUDA-graph medians of the core
alone (A_core, B) and of A including its GEMM (A_gemm+core).

  python3 benchmarks/kimi_k3/bench_fip_fag.py --lens 5461,5461,5462 8192,8192
"""
import argparse
import json

import torch

import vllm.models.kimi_k3.nvidia.specialized_prefill.kda_persistent.forward as fwd_mod
from vllm.models.kimi_k3.nvidia.specialized_prefill import kda as sp_kda

D = 128


def graph_of(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    gph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gph):
        fn()
    return gph


def time_graph(gph, iters):
    ts = []
    for _ in range(iters):
        e0, e1 = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        e0.record()
        gph.replay()
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def one_shape(a, lens_s):
    lens = [int(x) for x in lens_s.split(",")]
    torch.manual_seed(0)
    T, N, H = sum(lens), len(lens), 96
    bf = torch.bfloat16
    offs = [0, *torch.tensor(lens).cumsum(0).tolist()]
    cu_cpu = torch.tensor(offs, dtype=torch.int32)
    cu = cu_cpu.cuda()
    q, k = (torch.randn(1, T, H, D, device="cuda").to(bf) for _ in range(2))
    v = (torch.randn(1, T, H, D, device="cuda") * 0.5).to(bf)
    beta = torch.randn(1, T, H, device="cuda").to(bf)
    A_log = torch.randn(H, device="cuda") * 0.5
    dt_bias = torch.randn(H * D, device="cuda") * 0.1
    # f_a lives inside a wider projection row (as in_proj's packed output)
    wide = torch.randn(T, D + 72, device="cuda").to(bf)
    f_a = wide[:, 40 : 40 + D]
    w_fb = (torch.randn(H * D, D, device="cuda") * D**-0.5).to(bf)
    s0 = torch.randn(N, H, D, D, device="cuda") * 0.1

    def gemm():
        return torch.matmul(f_a, w_fb.t())

    g1 = gemm().view(1, T, H, D)

    def run_a():
        return sp_kda.persistent_prefill(
            q, k, v, g1, beta, A_log, dt_bias, -5.0, s0, cu, cu_cpu
        )

    def run_a_full():
        g1.copy_(gemm().view(1, T, H, D))
        return run_a()

    def run_b():
        return sp_kda.persistent_prefill(
            q, k, v, None, beta, A_log, dt_bias, -5.0, s0, cu, cu_cpu,
            f_a=f_a, f_b_weight=w_fb,
        )

    oa, sa = run_a()
    ob, sb = run_b()
    torch.cuda.synchronize()
    do = (oa.float() - ob.float()).abs()
    ds = (sa - sb).abs()
    ref = oa.float().abs().max().item()
    print(f"lens={lens_s} out maxdiff {do.max().item():.3e} (|out|max {ref:.2e}) "
          f"mismatch {(do > 0).float().mean().item():.2e}; state maxdiff "
          f"{ds.max().item():.3e} bitequal out={torch.equal(oa, ob)} "
          f"state={torch.equal(sa, sb)}", flush=True)
    gs = {"A_core": graph_of(run_a), "gemm": graph_of(gemm), "B_fag": graph_of(run_b)}
    def run_bg():
        return sp_kda.persistent_prefill(
            q, k, v, g1, beta, A_log, dt_bias, -5.0, s0, cu, cu_cpu,
            f_a=f_a, f_b_weight=w_fb,
        )

    for var in a.dev:
        fwd_mod._DEV = json.loads(var)
        gs["B" + var] = graph_of(run_bg if fwd_mod._DEV.get("FAGP_", 0) & 16 else run_b)
        fwd_mod._DEV = {}
    med = {n: [] for n in gs}
    names = list(gs)
    for r in range(a.rounds):
        for n in (names if r % 2 == 0 else names[::-1]):
            med[n].append(time_graph(gs[n], a.iters))
    for n in names:
        m = sorted(med[n])
        print(f"lens={lens_s} {n:16s} graph us median-of-rounds {m[len(m) // 2]:.1f} "
              f"rounds {' '.join(f'{x:.1f}' for x in med[n])}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", nargs="+", default=["5461,5461,5462"])
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--dev", nargs="*", default=[], help="timing-only B variants (_DEV json)")
    a = ap.parse_args()
    for lens_s in a.lens:
        one_shape(a, lens_s)


if __name__ == "__main__":
    main()
