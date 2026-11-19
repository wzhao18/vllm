# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A/B: cuDNN frontend FROST KDA prefill engine vs FlashKDA at the deployed shape.

DEP16 F16K: 3 sequences x ~5.4K fresh tokens, 96 heads, head_dim 128, FP32
state, safe gate (lower_bound * sigmoid(exp(A_log) * (g + dt_bias))), q/k L2
norm and beta sigmoid in kernel. Checks FROST output and final state against
FlashKDA (rel-rms), then times the core (CUPTI kernel sums per call).

usage: python bench_kda_frost_ab.py [--tokens 16384] [--seqs 3] [--iters 10]
       [--impls stock,split,frost] [--batch-invariant]
"""

import argparse
import time
from collections import defaultdict

import torch

D = 128


def rel_rms(a, b):
    a, b = a.float(), b.float()
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp_min(1e-30)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--seqs", type=int, default=3)
    ap.add_argument("--heads", type=int, default=96)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--impls", default="stock,split,frost")
    ap.add_argument("--batch-invariant", action="store_true")
    ap.add_argument("--plan", default="kda_frost")
    args = ap.parse_args()

    import vllm._flashkda_C  # noqa: F401
    from vllm.models.kimi_k3.nvidia.ops import kda_segscan

    kda_segscan.load_extension()
    from cudnn.linear_attention.ops import kimi_delta_attention

    dev = torch.device("cuda")
    torch.manual_seed(0)
    T, N, H = args.tokens, args.seqs, args.heads
    lens = [T // N] * N
    lens[-1] += T - sum(lens)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device=dev)
    bf = torch.bfloat16
    q = torch.randn(1, T, H, D, dtype=bf, device=dev)
    k = torch.randn(1, T, H, D, dtype=bf, device=dev)
    v = torch.randn(1, T, H, D, dtype=bf, device=dev) * 0.5
    g = torch.randn(1, T, H, D, dtype=bf, device=dev) * 0.5
    beta = torch.randn(1, T, H, dtype=bf, device=dev)
    A_log = torch.randn(H, dtype=torch.float32, device=dev) * 0.5
    dt_bias = torch.randn(H, D, dtype=torch.float32, device=dev) * 0.1
    lb = -5.0
    init_state = torch.zeros(N, H, D, D, dtype=torch.float32, device=dev)
    ws = torch.empty(torch.ops._flashkda_C.get_workspace_size(T, H, N), dtype=torch.uint8, device=dev)
    outs, states = {}, {}

    def mk_flashkda(op_split):
        out = torch.empty(1, T, H, D, dtype=bf, device=dev)
        fs = torch.empty_like(init_state)
        outs[op_split], states[op_split] = out, fs

        def run():
            a = (q, k, v, g, beta, D**-0.5, out, ws, A_log, dt_bias, lb, init_state, fs, cu, None, None)
            if op_split == "split":
                torch.ops.k3_flashkda_split.fwd(*a, 3)
            else:
                torch.ops._flashkda_C.fwd(*a)

        return run

    q3, k3, v3, g3 = (x.view(T, H, D) for x in (q, k, v, g))
    b2 = beta.view(T, H).contiguous()

    def run_frost():
        o, fs = kimi_delta_attention(
            q3, k3, v3, g3, b2, cu, scale=D**-0.5, initial_state=init_state,
            output_final_state=True, use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=True, safe_gate=True, gate_lower_bound=lb,
            a_log=A_log, dt_bias=dt_bias, plan_name=args.plan,
            batch_invariant=args.batch_invariant,
        )[:2]
        outs["frost"], states["frost"] = o, fs

    runs = {}
    for name in args.impls.split(","):
        runs[name] = run_frost if name == "frost" else mk_flashkda(name)
    for name, fn in runs.items():
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        print(f"{name}: first call {time.time() - t0:.1f} s", flush=True)

    ref = "stock" if "stock" in outs else None
    if ref:
        for name in outs:
            if name == ref:
                continue
            so, ro = states[name], states[ref]
            print(
                f"check {name} vs {ref}: out rel-rms {rel_rms(outs[name].view(T, H, D), outs[ref].view(T, H, D)):.3e}"
                f" state rel-rms {rel_rms(so, ro):.3e} state^T rel-rms {rel_rms(so.transpose(-1, -2), ro):.3e}"
                f" nan {torch.isnan(outs[name]).any().item()}"
            )

    from torch.profiler import ProfilerActivity, profile

    for name, fn in runs.items():
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        h0 = time.perf_counter()
        for _ in range(args.iters):
            fn()
        host = (time.perf_counter() - h0) * 1e6 / args.iters
        e1.record()
        torch.cuda.synchronize()
        wall = e0.elapsed_time(e1) * 1e3 / args.iters
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(args.iters):
                fn()
            torch.cuda.synchronize()
        acc, cnt = defaultdict(float), defaultdict(int)
        for e in prof.events():
            if e.device_type.name == "CUDA" and e.device_time_total > 0:
                acc[e.name] += e.device_time_total
                cnt[e.name] += 1
        rows = sorted(((t / args.iters, cnt[n] // args.iters, n) for n, t in acc.items()), reverse=True)
        tot = sum(r[0] for r in rows)
        print(f"impl={name} T={T} N={N} H={H}: kernels {tot:.1f} us, event wall {wall:.1f} us, host issue {host:.1f} us/call")
        for us, c, n in rows[:6]:
            print(f"  {us:8.1f} us x{c} {n[:110]}")


if __name__ == "__main__":
    main()
