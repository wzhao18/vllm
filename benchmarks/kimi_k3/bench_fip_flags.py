"""Persistent KDA compile-flag sweep at F16K (dev tool): times each variant of
_launch_forward's constexpr options and checks it against the default.

usage: python bench_fip_flags.py [--lens 5461,5461,5462]
"""

import argparse
import itertools

import torch

from vllm.models.kimi_k3.nvidia.specialized_prefill.kda_persistent import forward as kp

D = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", default="5461,5461,5462")
    ap.add_argument("--heads", type=int, default=96)
    args = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(0)
    lens = [int(x) for x in args.lens.split(",")]
    T, N, H = sum(lens), len(lens), args.heads
    offs = [0] + torch.tensor(lens).cumsum(0).tolist()
    bf = torch.bfloat16
    q, k, g = (torch.randn(1, T, H, D, dtype=bf, device=dev) for _ in range(3))
    v = torch.randn(1, T, H, D, dtype=bf, device=dev) * 0.5
    beta = torch.randn(1, T, H, dtype=bf, device=dev)
    A_log = torch.randn(H, dtype=torch.float32, device=dev) * 0.5
    dt_bias = torch.randn(H, D, dtype=torch.float32, device=dev) * 0.1
    h0 = torch.randn(N, H, D, D, dtype=torch.float32, device=dev) * 0.05
    plan = kp.make_plan(offs, H, dev)
    (cu32, soff, schain, spt0, sptn, ssrc, sdst, G, mid, mfl, has_split, full) = plan
    fixed = N == 1
    base = dict(gate2=1 if fixed else 0, has_split=has_split, beta_tma=True,
                qk_rowpair=4 if fixed else 8, restore_tail=False, reg_mode=2,
                full_chunks=full, lower_bound_m5=True, approximate_split=False,
                gate_roll=not fixed, partial_tma=not fixed)
    space = dict(reg_mode=[0, 1, 2], gram_w9=[0, 1], prep_peel=[0, 1],
                 rcp_decor=[True, False], rf_hoist=[True, False])
    out = torch.empty_like(q)
    fs = torch.empty_like(h0)
    ref = None
    s0, s1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    variants = [{}] + [{kk: vv} for kk, vals in space.items() for vv in vals
                       if vv != base.get(kk, {"gram_w9": 0, "prep_peel": 0, "rcp_decor": True,
                                                  "rf_hoist": True}.get(kk))]
    variants += [dict(zip(("reg_mode", "gram_w9"), c)) for c in itertools.product([0, 1], [1])]
    for var in variants:
        kw = dict(base, **var)

        def fn():
            if has_split:
                mfl.zero_()
            kp._launch_forward(q, k, v, g, beta, D**-0.5, out, A_log, dt_bias, -5.0,
                               initial_state=h0, final_state=fs, cu_seqlens=cu32,
                               sched=(soff, schain, spt0, sptn, ssrc, sdst, G, mid, mfl, 1), **kw)

        try:
            fn()
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            print(f"{var}: FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
            continue
        if ref is None:
            ref = (out.clone(), fs.clone())
        d_o = (out.float() - ref[0].float()).abs().max().item()
        d_s = (fs - ref[1]).abs().max().item()
        for _ in range(3):
            fn()
        s0.record()
        for _ in range(10):
            fn()
        s1.record()
        torch.cuda.synchronize()
        print(f"{var}: {s0.elapsed_time(s1) * 100:.1f} us  max|d| out {d_o:.2e} state {d_s:.2e}",
              flush=True)


if __name__ == "__main__":
    main()
