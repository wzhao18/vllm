# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KDA prefill core with the q/k conv fused into the persistent kernel's prep
(specialized_prefill/kda_persistent, FUSED_CONV_) vs qkv_conv + the persistent
kernel, at DEP16 F16K (3 x 5461 tokens, 96 heads).

fused = q/k history + conv-state update (torch) + V-only qkv_conv + fip(conv=...)
ref   = qkv_conv (Q|K|V) + fip
Checks output / final state / conv state vs ref, then times both (CUPTI sums).

usage: python bench_fip_conv.py [--tokens 16384] [--seqs 3] [--init 0|1]
"""

import argparse
import time
from collections import defaultdict

import torch

D = 128


def rel_rms(a, b):
    a, b = a.float(), b.float()
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp_min(1e-30)).item()


def kernels(fn, iters):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    acc, cnt = defaultdict(float), defaultdict(int)
    for e in prof.events():
        if e.device_type.name == "CUDA" and e.device_time_total > 0:
            acc[e.name] += e.device_time_total
            cnt[e.name] += 1
    return sorted(((t / iters, cnt[n] // iters, n) for n, t in acc.items()), reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--seqs", type=int, default=3)
    ap.add_argument("--heads", type=int, default=96)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--init", type=int, default=1)
    ap.add_argument("--lens", default="")
    args = ap.parse_args()

    from vllm.models.kimi_k3.nvidia.specialized_prefill.kda_persistent import forward as fip
    from vllm.models.kimi_k3.nvidia.ops import kda_conv

    kda_conv._load_qkv_conv()
    dev = torch.device("cuda")
    torch.manual_seed(0)
    T, N, H = args.tokens, args.seqs, args.heads
    lens = [T // N] * N
    lens[-1] += T - sum(lens)
    if args.lens:
        lens = [int(x) for x in args.lens.split(",")]
        T, N = sum(lens), len(lens)
    P = H * D
    offs = [0] + torch.tensor(lens).cumsum(0).tolist()
    cu = torch.tensor(offs, dtype=torch.int32, device=dev)
    bf = torch.bfloat16
    proj = torch.randn(T, 4 * P + 256, dtype=bf, device=dev)  # in_proj-like rows
    x = proj[:, : 3 * P]
    weight = (torch.randn(3 * P, 4, device=dev) * 0.5).float()
    lines = N + 2
    state0 = torch.randn(lines, 3 * P, 3, device=dev) * 0.5
    cidx = torch.arange(1, N + 1, dtype=torch.int32, device=dev)
    has_init = torch.tensor([bool(args.init) and i % 2 == 0 for i in range(N)], device=dev)
    g = torch.randn(1, T, H, D, dtype=bf, device=dev) * 0.5
    beta = torch.randn(1, T, H, dtype=bf, device=dev)
    A_log = torch.randn(H, dtype=torch.float32, device=dev) * 0.5
    dt_bias = torch.randn(H, D, dtype=torch.float32, device=dev) * 0.1
    init_state = torch.randn(N, H, D, D, dtype=torch.float32, device=dev) * 0.05
    lb = -5.0
    plan = fip.make_plan(offs, H, dev)
    wg = fip.conv_weight_groups(weight, H)
    res = {}

    def conv_op(xx, w, st, out):
        torch.ops.k3_kda_conv.qkv_conv(xx, w, st, cu, cidx, has_init, out, -1,
                                       kda_conv._qkv_conv_warps(xx.shape[1], T))

    st_ref = state0.clone()
    planes = torch.empty(3, T, P, dtype=bf, device=dev)
    o_ref = torch.empty(1, T, H, D, dtype=bf, device=dev)
    fs_ref = torch.empty_like(init_state)

    def run_ref():
        st_ref.copy_(state0)
        conv_op(x, weight, st_ref, planes)
        fip.fwd(planes[0].view(1, T, H, D), planes[1].view(1, T, H, D), planes[2].view(1, T, H, D),
                g, beta, D**-0.5, o_ref, A_log, dt_bias, lb, initial_state=init_state,
                final_state=fs_ref, cu_seqlens=cu, plan=plan)

    st_f = state0.clone()
    vbuf = torch.empty(T, P, dtype=bf, device=dev)
    vplanes = vbuf.as_strided((3, T, P // 3), (P // 3, P, 1))
    o_f = torch.empty(1, T, H, D, dtype=bf, device=dev)
    fs_f = torch.empty_like(init_state)

    def run_fused():
        st_f.copy_(state0)
        hist = fip.conv_qk_history(x, st_f, cidx, has_init, cu, H)
        conv_op(x[:, 2 * P:], weight[2 * P:], st_f[:, 2 * P:], vplanes)
        fip.fwd(x[:, :P].view(1, T, H, D), x[:, P:2 * P].view(1, T, H, D), vbuf.view(1, T, H, D),
                g, beta, D**-0.5, o_f, A_log, dt_bias, lb, initial_state=init_state,
                final_state=fs_f, cu_seqlens=cu, plan=plan, conv=(wg, hist))

    for name, fn in (("ref", run_ref), ("fused", run_fused)):
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        print(f"{name}: first call {time.time() - t0:.1f} s", flush=True)
    print(f"CHECK fused vs ref: out {rel_rms(o_f, o_ref):.3e} state {rel_rms(fs_f, fs_ref):.3e} "
          f"conv-state equal {torch.equal(st_f, st_ref)} v equal {torch.equal(vbuf, planes[2])} "
          f"nan {torch.isnan(o_f).any().item()}", flush=True)
    for s_ in range(N):
        a, b = offs[s_], offs[s_ + 1]
        print(f"  seq {s_} [{a},{b}) init {bool(has_init[s_])}: out {rel_rms(o_f[:, a:b], o_ref[:, a:b]):.3e}"
              f" first-32 {rel_rms(o_f[:, a:a + 32], o_ref[:, a:a + 32]):.3e}")
    for name, fn in (("ref", run_ref), ("fused", run_fused)):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(args.iters):
            fn()
        e1.record()
        torch.cuda.synchronize()
        rows = kernels(fn, args.iters)
        print(f"TIME {name}: wall {e0.elapsed_time(e1) * 1e3 / args.iters:.1f} us, "
              f"kernels {sum(r[0] for r in rows):.1f} us", flush=True)
        for us, c, n in rows[:7]:
            print(f"  {us:8.1f} us x{c} {n[:90]}")


if __name__ == "__main__":
    main()
