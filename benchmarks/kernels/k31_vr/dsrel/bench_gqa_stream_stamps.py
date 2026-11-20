# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase stamps of GqaWindowStream (the drafter's window attention streamed over a
fixed grid) at TP4 q3: bench_gqa_attention_int.py's fused arm on the stream launch
shape with the kernel's development ``debug`` stamps (globaltimer per CTA), one GPU.

Per batch: the per-layer time, then from the last call's stamps the kernel span, the
CTA start spread, and per-CTA medians of: PDL wait (s1 - s0), setup (s2 - s1), piece
0 loop (s4 - s2) and merge (s5 - s4), piece 1 prep + loop (s6 - s5) and merge
(s7 - s6), the end (s11 - s9); merger poll (s10) and end quantiles.

  python3 bench_gqa_stream_stamps.py --batches 32,64,96 --shape 0/-1/6/0
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bench_gqa_attention_int as bench  # noqa: E402

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft import (  # noqa: E402
    gqa_attention as ga,
)

DBG = None


def _patch():
    orig = ga.compile_window_stream

    def patched(*args, **kwargs):
        kwargs["debug"] = DBG.data_ptr()
        return orig(*args, **kwargs)

    ga.compile_window_stream = patched


def _report(b):
    s = DBG.view(-1, 16).cpu()
    ok = s[:, 0] > 0
    s = s[ok].double()
    n = s.shape[0]
    if n == 0:
        print(f"B{b}: no stamps", flush=True)
        return
    t0 = s[:, 0].min()
    span = (s[:, 11].max() - t0) / 1e3

    def med(x):
        return float(x.median()) / 1e3

    ph = {
        "wait": med(s[:, 1] - s[:, 0]),
        "setup": med(s[:, 2] - s[:, 1]),
        "p0loop": med(s[:, 4] - s[:, 2]),
        "p0merge": med(s[:, 5] - s[:, 4]),
        "p1loop": med(s[:, 6] - s[:, 5]),
        "p1merge": med(s[:, 7] - s[:, 6]),
        "p2": med(s[:, 9] - s[:, 7]),
        "end": med(s[:, 11] - s[:, 9]),
    }
    merger = s[:, 10] > 0
    poll = med(s[merger, 10] - s[merger, 6]) if bool(merger.any()) else 0.0
    # co-residency: per SM, the most CTAs whose [entry, end] overlap
    entry, end, sm = s[:, 14], s[:, 11], s[:, 13].long()
    most = 0
    for one in sm.unique().tolist():
        rows = (sm == one).nonzero().flatten().tolist()
        for i in rows:
            live = sum(1 for j in rows if entry[j] <= entry[i] < end[j])
            most = max(most, live)
    land_q = torch.quantile(entry - entry.min(), torch.tensor([0.1, 0.5, 0.9],
                                                              dtype=torch.float64)) / 1e3
    print(f"B{b}: co-resident CTAs per SM max {most}, SMs used {sm.unique().numel()}, "
          f"entry q10/50/90 " + "/".join(f"{x:.1f}" for x in land_q.tolist()),
          flush=True)
    work = (s[:, 11] - s[:, 1]) / 1e3
    after = (s[:, 1] - s[:, 1].min()) / 1e3
    order = torch.argsort(s[:, 11], descending=True)[:6].tolist()
    print(f"B{b}: work (end - wait) q50/q90/max {float(work.median()):.2f}/"
          f"{float(torch.quantile(work, 0.9)):.2f}/{float(work.max()):.2f} | wait-pass "
          f"q50/q90/max {float(after.median()):.2f}/{float(torch.quantile(after, 0.9)):.2f}"
          f"/{float(after.max()):.2f}", flush=True)
    for i in order:
        r = s[i]
        d = lambda a, b: float(r[b] - r[a]) / 1e3  # noqa: E731
        mg = d(6, 10) if r[10] > 0 else -1.0
        print(f"   slow cta {i}: pass {float(r[1] - s[:, 1].min()) / 1e3:.2f} setup "
              f"{d(1, 2):.2f} p0loop {d(2, 4):.2f} p0merge {d(4, 5):.2f} p1loop {d(5, 6):.2f}"
              f" p1merge {d(6, 7):.2f} poll {mg:.2f} stages {int(r[12])}", flush=True)
    q = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
    endq = torch.quantile(s[:, 11] - t0, q) / 1e3
    startq = torch.quantile(s[:, 0] - t0, q) / 1e3
    print(
        f"B{b}: CTAs {n} span {span:.2f} us | start q10/50/90 "
        + "/".join(f"{x:.1f}" for x in startq.tolist())
        + " | per-CTA median "
        + " ".join(f"{k} {v:.2f}" for k, v in ph.items())
        + f" | stages med {float(s[:, 12].median()):.0f} max {float(s[:, 12].max()):.0f}"
        + f" | merger p1->poll {poll:.2f} | end q10/50/90 "
        + "/".join(f"{x:.1f}" for x in endq.tolist()),
        flush=True,
    )


def main():
    global DBG
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="32,64,96")
    ap.add_argument("--ctx", default="150000")
    ap.add_argument("--shape", default="0/-1/6/0")
    args, rest = ap.parse_known_args()
    DBG = torch.zeros(8192 * 16, dtype=torch.int64, device="cuda")
    _patch()
    for b in [int(x) for x in args.batches.split(",")]:
        DBG.zero_()
        sys.argv = ["bench", "--tp", "4", "--q", "3", "--batches", str(b), "--arms",
                    "fused", "--no-context", "--ctx", args.ctx, "--shape",
                    args.shape] + rest  # fmt: skip
        bench.main()
        torch.cuda.synchronize()
        _report(b)


if __name__ == "__main__":
    main()
