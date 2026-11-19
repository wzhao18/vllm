# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FROST KDA prefill with the short conv fused into its q/k/v loads (ops/frost_kda)
vs qkv_conv + cuDNN FROST: bitwise output / final state / conv state, kernel time.

usage: bench_frost_conv.py [--tokens 16384] [--seqs 2] [--heads 96] [--iters 10]
"""

import argparse
import time
from collections import defaultdict

import torch

D = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--seqs", type=int, default=2)
    ap.add_argument("--heads", type=int, default=96)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--pad-cols", type=int, default=256)
    ap.add_argument("--no-time", action="store_true")
    args = ap.parse_args()

    from cudnn.linear_attention.ops import kimi_delta_attention

    from vllm.models.kimi_k3.nvidia.ops import kda_conv
    from vllm.models.kimi_k3.nvidia.ops.frost_kda.launcher import frost_conv_plan

    dev = torch.device("cuda")
    torch.manual_seed(0)
    T, N, H = args.tokens, args.seqs, args.heads
    HD = H * D
    lens = [T // N] * N
    lens[-1] += T - sum(lens)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device=dev)
    bf = torch.bfloat16
    ld = 3 * HD + args.pad_cols
    x_full = torch.randn(T, ld, dtype=bf, device=dev)
    qkv = x_full[:, : 3 * HD]
    conv_w = (torch.randn(3 * HD, 4, dtype=torch.float32, device=dev) * 0.5).contiguous()
    lines = N + 5
    state0 = torch.randn(lines, 3 * HD, 3, dtype=bf, device=dev).float()
    idx = torch.tensor([(5 + 3 * i) % lines for i in range(N)], dtype=torch.int32, device=dev)
    has_init = torch.tensor([i % 2 == 0 for i in range(N)], dtype=torch.bool, device=dev)
    g = torch.randn(T, H, D, dtype=bf, device=dev) * 0.5
    beta = torch.randn(T, H, dtype=bf, device=dev)
    A_log = torch.randn(H, dtype=torch.float32, device=dev) * 0.5
    dt_bias = torch.randn(H, D, dtype=torch.float32, device=dev) * 0.1
    lb = -5.0
    init_state = torch.randn(N, H, D, D, dtype=torch.float32, device=dev) * 0.1

    plan = frost_conv_plan(T, N, H, D, lb, dev)
    p = getattr(plan, "plan", None)
    print(f"T={T} N={N} H={H}: fused plan supported={plan.supported}", flush=True)
    if not plan.supported:
        return

    st_ref = state0.clone()

    def run_ref():
        planes = kda_conv.qkv_conv(qkv, conv_w, st_ref, cu, idx, has_init, HD)
        q, k, v = (t.view(T, H, D) for t in planes.unbind(0))
        o, fs = kimi_delta_attention(
            q, k, v, g, beta, cu, scale=D**-0.5, initial_state=init_state,
            output_final_state=True, use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=True, safe_gate=True, gate_lower_bound=lb,
            a_log=A_log, dt_bias=dt_bias, plan_name="kda_frost",
        )[:2]
        return o, fs

    st_fused = state0.clone()
    slot = torch.where(has_init, idx, torch.full_like(idx, -1))

    def run_fused():
        o, fs = plan(qkv, g, beta, A_log, dt_bias, init_state, cu, conv_w, st_fused, slot)
        kda_conv.qkv_conv_state(qkv, st_fused, cu, idx, has_init)
        return o, fs

    t0 = time.time()
    o_ref, fs_ref = run_ref()
    torch.cuda.synchronize()
    print(f"ref first call {time.time() - t0:.1f} s", flush=True)
    t0 = time.time()
    o, fs = run_fused()
    torch.cuda.synchronize()
    print(f"fused first call {time.time() - t0:.1f} s", flush=True)

    def cmp(name, a, b):
        a, b = a.float(), b.float()
        eq = torch.equal(a, b)
        diff = (a - b).abs()
        rel = (diff.pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp_min(1e-30)).item()
        bad = torch.nonzero(diff.view(-1) > 0)
        first = bad[0].item() if bad.numel() else -1
        print(f"check {name}: bitwise {eq} rel-rms {rel:.3e} max {diff.max().item():.3e} "
              f"mismatches {bad.numel()} first {first} nan {torch.isnan(a).any().item()}", flush=True)
        return eq

    ok = cmp("out", o.view(T, H, D), o_ref.view(T, H, D))
    if not ok:
        d = (o.float().view(T, H, D) - o_ref.float().view(T, H, D)).abs().amax(dim=(1, 2))
        rows = torch.nonzero(d > 0).view(-1)
        print(f"  mismatching token rows: {rows.numel()}, first {rows[:8].tolist()}", flush=True)
        dh = (o.float().view(T, H, D) - o_ref.float().view(T, H, D)).abs().amax(dim=(0, 2))
        print(f"  mismatching heads: {torch.nonzero(dh > 0).view(-1)[:12].tolist()}", flush=True)
    cmp("final_state", fs, fs_ref)
    cmp("conv_state", st_fused, st_ref)

    if args.no_time:
        return
    from torch.profiler import ProfilerActivity, profile

    for name, fn in (("ref", run_ref), ("fused", run_fused)):
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
        print(f"impl={name} T={T} N={N} H={H}: kernels {tot:.1f} us, event wall {wall:.1f} us, host {host:.1f} us/call")
        for us, c, n in rows[:6]:
            print(f"  {us:8.1f} us x{c} {n[:100]}")


if __name__ == "__main__":
    main()
