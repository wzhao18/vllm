"""Persistent KDA chain-step A/B (dev tool): times the production fip route
against kernel probe variants selected through forward-module globals.

usage: python bench_fip_wy.py [--lens 5461,5461,5462] [--probe WYPROBE=1]
"""

import argparse
import os
import threading

import torch

from vllm.models.kimi_k3.nvidia.specialized_prefill.kda_persistent import forward as kp

D = 128


def run(lens, H, probes):
    dev = torch.device("cuda")
    torch.manual_seed(0)
    T, N = sum(lens), len(lens)
    offs = [0] + torch.tensor(lens).cumsum(0).tolist()
    bf = torch.bfloat16
    q, k, g = (torch.randn(1, T, H, D, dtype=bf, device=dev) for _ in range(3))
    v = torch.randn(1, T, H, D, dtype=bf, device=dev) * 0.5
    beta = torch.rand(1, T, H, dtype=bf, device=dev)
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
    out = torch.empty_like(q)
    fs = torch.empty_like(h0)
    s0, s1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ref = None
    variants = [("old", {}, {"wy": False})]
    variants += [(n, glb, {"wy": False}) for n, glb in probes]
    variants += [("wy", {}, {"wy": not fixed}), ("wy+tf4", {}, {"wy": not fixed, "tf4": not fixed})]
    for name, glb, kw in variants:
        saved = {kk: getattr(kp, kk) for kk in glb}
        for kk, vv in glb.items():
            setattr(kp, kk, vv)

        def fn():
            if has_split:
                mfl.zero_()
            kp._launch_forward(q, k, v, g, beta, D**-0.5, out, A_log, dt_bias, -5.0,
                               initial_state=h0, final_state=fs, cu_seqlens=cu32,
                               sched=(soff, schain, spt0, sptn, ssrc, sdst, G, mid, mfl, 1),
                               **dict(base, **kw))

        dog = threading.Timer(300, lambda: (print(f"lens={lens} {name}: HANG", flush=True),
                                           os._exit(3)))
        dog.start()
        try:
            fn()
            torch.cuda.synchronize()
            if ref is None:
                ref = (out.clone(), fs.clone())
            rel_o = ((out.float() - ref[0].float()).norm() / ref[0].float().norm()).item()
            rel_s = ((fs - ref[1]).norm() / ref[1].norm()).item()
            for _ in range(3):
                fn()
            s0.record()
            for _ in range(10):
                fn()
            s1.record()
            torch.cuda.synchronize()
            print(f"lens={lens} {name}: {s0.elapsed_time(s1) * 100:.1f} us  "
                  f"rel out {rel_o:.2e} state {rel_s:.2e}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"lens={lens} {name}: FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
        dog.cancel()
        for kk, vv in saved.items():
            setattr(kp, kk, vv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", default="5461,5461,5462;8192,8192")
    ap.add_argument("--heads", type=int, default=96)
    ap.add_argument("--probe", action="append", default=[])
    args = ap.parse_args()
    probes = []
    for p in args.probe:
        kvs = dict(x.split("=") for x in p.split(","))
        probes.append((p, {"_" + kk: int(vv) for kk, vv in kvs.items()}))
    for ls in args.lens.split(";"):
        run([int(x) for x in ls.split(",")], args.heads, probes)


if __name__ == "__main__":
    main()
