"""Clock-stamp anatomy of the fip BT32 chain step (compute warpgroup, TPROBE_=2 build).

Per slot (CTA), warp 0 accumulates SM clock64 deltas per phase over all chain steps:
  qk   wait MB_QK (prep stage ready)
  pass state master pass (TMEM ld S windows, decay, st INP bf16 + st S; incl. MB_OFIN wait)
  mma1 wait MB_VFULL/MB_OOUT (U leg of MMA1 done)
  res  residual pass (ld U, (v-U)beta, st; WY: u/ld pass)
  mma3 wait MB_U2ACC (MMA3 done; WY: none)
  vst  vnew pass (ld, cvt, st MMA4 operand; TF4: none)
  mma4 wait MB_FIN (MMA4 done)
usage: python fip_stamps.py [--lens 5461,5461,5462] [--variants old,wy,wy+tf4]
"""
import argparse
import os
import threading

import torch

from vllm.models.kimi_k3.nvidia.specialized_prefill.kda_persistent import forward as kp

D = 128
PH = ["qk", "pass", "mma1", "res", "mma3", "vst", "mma4"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", default="5461,5461,5462")
    ap.add_argument("--variants", default="old,wy,wy+tf4")
    args = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(0)
    lens = [int(x) for x in args.lens.split(",")]
    H = 96
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
    kws = {"old": {"wy": False}, "wy": {"wy": not fixed}, "wy+tf4": {"wy": not fixed, "tf4": not fixed}}
    for name in args.variants.split(","):
        for tp in (0, 2):
            kp._TPROBE = tp
            s0, s1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

            def fn():
                if has_split:
                    mfl.zero_()
                kp._launch_forward(q, k, v, g, beta, D**-0.5, out, A_log, dt_bias, -5.0,
                                   initial_state=h0, final_state=fs, cu_seqlens=cu32,
                                   sched=(soff, schain, spt0, sptn, ssrc, sdst, G, mid, mfl, 1),
                                   **dict(base, **kws[name]))

            dog = threading.Timer(300, lambda: (print(f"{name}: HANG", flush=True), os._exit(3)))
            dog.start()
            try:
                fn()
                torch.cuda.synchronize()
                s0.record()
                for _ in range(10):
                    fn()
                s1.record()
                torch.cuda.synchronize()
            finally:
                dog.cancel()
            us = s0.elapsed_time(s1) * 100
            if tp == 0:
                print(f"lens={lens} {name}: kernel {us:.1f} us (no probe)", flush=True)
                continue
            tpb = kp.LAST_TPROBE[0].view(-1, 64)[:G].double().cpu()
            n = tpb[:, 48].clamp(min=1)
            per = tpb[:, 40:47] / n[:, None]          # clk per step per slot (last launch)
            tot = per.sum(1)
            med = per.median(0).values
            parts = " ".join(f"{p} {m:.0f}" for p, m in zip(PH, med.tolist()))
            print(f"lens={lens} {name}: probe kernel {us:.1f} us | per step clk (median slot): {parts} | "
                  f"sum {tot.median().item():.0f} (min {tot.min().item():.0f} max {tot.max().item():.0f}) | "
                  f"steps/slot median {tpb[:, 48].median().item():.0f}", flush=True)


if __name__ == "__main__":
    main()
