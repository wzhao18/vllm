"""Per-CTA piece timeline of the persistent KDA prefill kernel (TPROBE_=1).

For each shape: makespan, mean/p50/p90 slot end, per-slot busy time, waits
for producer pieces, and the step time (us per 32-token chunk) per slot.

  python3 benchmarks/kimi_k3/kda_dev/bench_fip_tail.py --lens 5461,5461,5462
"""
import argparse
import statistics as st

import torch

import vllm.models.kimi_k3.nvidia.specialized_prefill.kda_persistent.forward as fm
from vllm.models.kimi_k3.nvidia.specialized_prefill import kda as sp_kda

D = 128


def one(lens_s, reps):
    lens = [int(x) for x in lens_s.split(",")]
    torch.manual_seed(0)
    T, N, H = sum(lens), len(lens), 96
    bf = torch.bfloat16
    offs = [0, *torch.tensor(lens).cumsum(0).tolist()]
    cu_cpu = torch.tensor(offs, dtype=torch.int32)
    cu = cu_cpu.cuda()
    q, k, g = (torch.randn(1, T, H, D, device="cuda").to(bf) for _ in range(3))
    v = (torch.randn(1, T, H, D, device="cuda") * 0.5).to(bf)
    beta = torch.randn(1, T, H, device="cuda").to(bf)
    A_log = torch.randn(H, device="cuda") * 0.5
    dt_bias = torch.randn(H * D, device="cuda") * 0.1
    cache0 = torch.randn(N, H, D, D, device="cuda") * 0.1
    plan = fm.make_plan(offs, H, torch.device("cuda"))
    soff = plan[1].tolist()
    sptn = plan[4].tolist()
    ssrc = plan[5].tolist()
    G = plan[7]
    fm._TPROBE = 1
    res = []
    for _ in range(reps + 2):
        cache = cache0.clone()
        sp_kda.persistent_prefill(q, k, v, g, beta, A_log, dt_bias, -5.0, cache, cu, cu_cpu)
        torch.cuda.synchronize()
        res.append(fm.LAST_TPROBE[0].view(-1, 32, 2).cpu())
    fm._TPROBE = 0
    for tp in res[2:]:
        t0 = tp[:G][tp[:G] > 0].min().item()
        ends, busy, waits, steps = [], [], [], []
        for s in range(G):
            n = soff[s + 1] - soff[s]
            e = (tp[s, n - 1, 1].item() - t0) / 1e3
            b = sum((tp[s, i, 1] - tp[s, i, 0]).item() for i in range(n)) / 1e3
            gaps = sum(
                (tp[s, i, 0] - tp[s, i - 1, 1]).item() for i in range(1, n)
            ) / 1e3
            ch = sum((sptn[soff[s] + i] + 31) // 32 for i in range(n))
            ends.append(e)
            busy.append(b)
            waits.append(gaps + (tp[s, 0, 0].item() - t0) / 1e3)
            steps.append(b / ch)
        ends_s = sorted(ends)
        nsplit = sum(1 for x in ssrc if x >= 0)
        print(
            f"lens={lens_s} G={G} pieces={len(sptn)} consumers={nsplit} "
            f"makespan={max(ends):.1f} mean_end={st.mean(ends):.1f} "
            f"p50={ends_s[len(ends) // 2]:.1f} p10={ends_s[len(ends) // 10]:.1f} "
            f"busy_mean={st.mean(busy):.1f} busy_max={max(busy):.1f} "
            f"wait_mean={st.mean(waits):.2f} wait_max={max(waits):.2f} "
            f"us_per_chunk mean={st.mean(steps):.3f} min={min(steps):.3f} "
            f"max={max(steps):.3f}",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", nargs="+", default=["5461,5461,5462"])
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()
    for lens_s in a.lens:
        one(lens_s, a.reps)


if __name__ == "__main__":
    main()
