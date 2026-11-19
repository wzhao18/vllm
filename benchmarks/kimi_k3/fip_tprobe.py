"""Persistent KDA piece timeline at F16K (dev tool): per-piece globaltimer
start/end from the kernel's TPROBE build -> makespan, per-chunk step time,
handoff waits and idle tails per CTA.

usage: python fip_tprobe.py [--lens 5461,5461,5462]
"""

import argparse

import torch

from vllm.models.kimi_k3.nvidia.specialized_prefill.kda_persistent import forward as kp

D = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", default="5461,5461,5462")
    args = ap.parse_args()
    dev = torch.device("cuda")
    kp._TPROBE = 1
    torch.manual_seed(0)
    lens = [int(x) for x in args.lens.split(",")]
    T, N, H = sum(lens), len(lens), 96
    offs = [0] + torch.tensor(lens).cumsum(0).tolist()
    bf = torch.bfloat16
    q, k, g = (torch.randn(1, T, H, D, dtype=bf, device=dev) for _ in range(3))
    v = torch.randn(1, T, H, D, dtype=bf, device=dev) * 0.5
    beta = torch.randn(1, T, H, dtype=bf, device=dev)
    A_log = torch.randn(H, dtype=torch.float32, device=dev) * 0.5
    dt_bias = torch.randn(H, D, dtype=torch.float32, device=dev) * 0.1
    h0 = torch.randn(N, H, D, D, dtype=torch.float32, device=dev) * 0.05
    plan = kp.make_plan(offs, H, dev)
    out, fs = torch.empty_like(q), torch.empty_like(h0)
    for _ in range(4):
        kp.fwd(q, k, v, g, beta, D**-0.5, out, A_log, dt_bias, -5.0,
               initial_state=h0, final_state=fs, cu_seqlens=None, plan=plan)
    torch.cuda.synchronize()
    tp = kp.LAST_TPROBE[0].view(-1, 32, 2).cpu()
    soff, sptn, ssrc = plan[1].cpu(), plan[4].cpu(), plan[5].cpu()
    G = plan[7]
    t0 = tp[:, :, 0][tp[:, :, 0] > 0].min().item()
    ends, busy, waits, steps, first = [], [], [], [], []
    for s in range(G):
        n = soff[s + 1] - soff[s]
        st = tp[s, :n, 0] - t0
        en = tp[s, :n, 1] - t0
        ends.append(en.max().item())
        first.append(st.min().item())
        busy.append((en - st).sum().item())
        for i in range(n):
            ch = (sptn[soff[s] + i].item() + 31) // 32
            steps.append((en[i] - st[i]).item() / ch)
            if i > 0:
                waits.append((st[i] - en[i - 1]).item())
    mk = max(ends)
    steps.sort()
    print(f"G={G} makespan {mk / 1e3:.1f} us, mean CTA end {sum(ends) / G / 1e3:.1f}, "
          f"min end {min(ends) / 1e3:.1f}, max first start {max(first) / 1e3:.1f} us")
    print(f"busy/CTA mean {sum(busy) / G / 1e3:.1f} us; per-chunk32 ns p10/p50/p90 "
          f"{steps[len(steps) // 10]:.0f}/{steps[len(steps) // 2]:.0f}/{steps[9 * len(steps) // 10]:.0f}")
    if waits:
        waits.sort()
        print(f"gaps between pieces: n={len(waits)} p50 {waits[len(waits) // 2] / 1e3:.2f} "
              f"max {waits[-1] / 1e3:.2f} us sum {sum(waits) / 1e3:.1f} us; consumer pieces "
              f"{int((ssrc >= 0).sum())}")


if __name__ == "__main__":
    main()
