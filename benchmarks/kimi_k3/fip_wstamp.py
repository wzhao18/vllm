# SPDX-License-Identifier: Apache-2.0
"""Wait anatomy of the persistent prefill KDA kernel (dev probe).

Times the production build, then a TPROBE_ == 2 build that accumulates
clock64 deltas around every chain wait of the MMA warp and of compute warp 0,
and prints per-step means (cycles) over all CTAs.

usage: python fip_wstamp.py [--lens 5461,5461,5462] [--iters 20]
"""

import argparse

import torch

import vllm.models.kimi_k3.nvidia.specialized_prefill.kda_persistent.forward as kp

D = 128
NAMES = {
    0: "mma_total",
    1: "mma_steps",
    2: "mma_wait_state_inp(pass)",
    3: "mma_wait_OEMPTY",
    4: "mma_wait_u_inp(residual)",
    5: "mma_wait_u2_inp(convert)",
    8: "cw_wait_QK(prep)",
    9: "cw_wait_VFULL",
    10: "cw_wait_OOUT(MMA1)",
    11: "cw_wait_U2ACC(MMA3)",
    12: "cw_wait_FIN(MMA4a)",
}


def make_inputs(lens, dev):
    torch.manual_seed(0)
    T, N, H = sum(lens), len(lens), 96
    bf = torch.bfloat16
    q, k, g = (torch.randn(1, T, H, D, dtype=bf, device=dev) for _ in range(3))
    v = torch.randn(1, T, H, D, dtype=bf, device=dev) * 0.5
    beta = torch.randn(1, T, H, dtype=bf, device=dev)
    A_log = torch.randn(H, dtype=torch.float32, device=dev) * 0.5
    dt_bias = torch.randn(H, D, dtype=torch.float32, device=dev) * 0.1
    h0 = torch.randn(N, H, D, D, dtype=torch.float32, device=dev) * 0.05
    offs = [0] + torch.tensor(lens).cumsum(0).tolist()
    plan = kp.make_plan(offs, H, dev)
    out, fs = torch.empty_like(q), torch.empty_like(h0)
    return (q, k, v, g, beta, A_log, dt_bias, h0, plan, out, fs)


def run(x):
    q, k, v, g, beta, A_log, dt_bias, h0, plan, out, fs = x
    kp.fwd(q, k, v, g, beta, D**-0.5, out, A_log, dt_bias, -5.0,
           initial_state=h0, final_state=fs, cu_seqlens=None, plan=plan)


def timeit(x, iters):
    for _ in range(3):
        run(x)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        run(x)
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", default="5461,5461,5462")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--mma1w", type=int, default=1)
    ap.add_argument("--ab", action="store_true", help="ABAB time MMA1W 0 vs 1, no stamps")
    args = ap.parse_args()
    dev = torch.device("cuda")
    lens = [int(s) for s in args.lens.split(",")]
    x = make_inputs(lens, dev)
    if args.ab:
        outs = {}
        for rep_i in range(2):
            for m in (0, 1):
                kp._MMA1W = m
                t = timeit(x, args.iters)
                outs.setdefault(m, x[9].clone())
                print(f"lens={args.lens} AB rep{rep_i} MMA1W={m} prod_us={t:.1f}")
        print(f"MMA1W out bitequal={torch.equal(outs[0], outs[1])} "
              f"maxdiff={(outs[0].float() - outs[1].float()).abs().max().item():.3e}")
        return
    kp._MMA1W = args.mma1w
    t0 = timeit(x, args.iters)
    ref = x[9].clone()
    kp._TPROBE = 2
    kp._COMPILED.clear()
    t2 = timeit(x, args.iters)
    same = torch.equal(ref, x[9])
    tp = kp.LAST_TPROBE[0].view(-1, 64).cpu().double()
    live = tp[:, 1] > 0
    tp = tp[live]
    steps = tp[:, 1]
    print(f"lens={args.lens} prod_us={t0:.1f} stamp_us={t2:.1f} out_bitequal={same} ctas={int(live.sum())}")
    print(f"clk/us (mma_total / stamp kernel) = {tp[:, 0].max().item() / t2:.0f}")
    for i, n in NAMES.items():
        if i == 1:
            print(f"  {n:28s} mean {steps.mean().item():8.1f} max {steps.max().item():8.0f}")
            continue
        per = tp[:, i] / steps
        print(f"  {n:28s} clk/step mean {per.mean().item():8.0f} p10 {per.quantile(0.1).item():8.0f} p90 {per.quantile(0.9).item():8.0f}")


if __name__ == "__main__":
    main()
