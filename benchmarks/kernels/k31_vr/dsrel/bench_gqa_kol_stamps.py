# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-CTA phase stamps of GqaWindowKol (keys-on-lanes drafter GQA) in the
bench_gqa_attention_int.py micro (TP4 q3, cold KV): span, start spread, setup, loop,
epilogue, waits; ``--skeleton`` runs the data-movement skeleton (no MMA / softmax).

  python3 bench_gqa_kol_stamps.py --batches 24,32,48 [--skeleton] [--ring 5]
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bench_gqa_attention_int as bench  # noqa: E402
import bench_gqa_umma_stamps as stamps  # noqa: E402

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft import (  # noqa: E402
    gqa_attention as ga,
)

OPTS = {}


def _patch_kols():
    orig = ga.compile_window_kols

    def patched(*args, **kwargs):
        kwargs["debug"] = stamps.DBG.data_ptr()
        return orig(*args, **kwargs)

    ga.compile_window_kols = patched


def _report_kols(b):
    s = stamps.DBG.view(2, -1, 16)[0].cpu()
    s = s[s[:, 0] > 0].double()
    if s.shape[0] == 0:
        print(f"B{b}: no stamps", flush=True)
        return
    t0 = s[:, 0].min()

    def q(col, base=2):
        v = (s[:, col] - s[:, base]) / 1e3
        v = v[s[:, col] > 0]
        if v.numel() == 0:
            return "-"
        return f"{float(v.median()):.2f}/{float(v.max()):.2f}"

    print(f"B{b} KOLS: CTAs {s.shape[0]} span {float((s[:, 11].max() - t0) / 1e3):.2f} us | "
          f"tape {q(2, 0)} | after tape (median/max): prep0 {q(4)} PV end {q(3)} "
          f"softmax end {q(10)} last epi start {q(5)} merge end {q(6)} end {q(11)}", flush=True)


def _patch():
    orig = ga.compile_window_kol

    def patched(*args, **kwargs):
        kwargs["debug"] = stamps.DBG.data_ptr()
        kwargs.update(OPTS)
        return orig(*args, **kwargs)

    ga.compile_window_kol = patched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="24,32,48")
    ap.add_argument("--ctx", default="150000")
    ap.add_argument("--ring", type=int, default=5)
    ap.add_argument("--fill", type=int, default=1)
    ap.add_argument("--skeleton", action="store_true")
    ap.add_argument("--ahead", type=int, default=0, help="launch shape ahead bits (1: refill warp)")
    ap.add_argument("--trace", action="store_true", help="per-stage events of request 0 KV head 1")
    ap.add_argument("--kols", action="store_true", help="GqaWindowKols (key groups -3)")
    args, rest = ap.parse_known_args()
    stamps.DBG = torch.zeros(2 * 8192 * 16, dtype=torch.int64, device="cuda")
    OPTS["fill"] = args.fill
    OPTS["skeleton"] = args.skeleton
    trace = torch.zeros(48 * 16, dtype=torch.int64, device="cuda")
    if args.trace:
        OPTS["trace"] = trace.data_ptr()
    _patch()
    _patch_kols()
    shape = f"1/-{3 if args.kols else 2}/{args.ring}/{args.ahead}"
    for b in [int(x) for x in args.batches.split(",")]:
        stamps.DBG.zero_()
        trace.zero_()
        sys.argv = ["bench", "--tp", "4", "--q", "3", "--batches", str(b), "--arms",
                    "fused", "--no-context", "--ctx", args.ctx, "--shape", shape] + rest
        bench.main()
        torch.cuda.synchronize()
        if args.kols:
            _report_kols(b)
            continue
        stamps._report(b)
        s = stamps.DBG.view(2, -1, 16)[0].cpu()
        s = s[s[:, 0] > 0].double()
        if s.shape[0]:
            print(f"B{b}: rescales/CTA median {float(s[:, 9].median()):.0f} max "
                  f"{float(s[:, 9].max()):.0f} | sready wait {float((s[:, 8] / s[:, 15].clamp(min=1)).median()):.2f}"
                  f" pready {float((s[:, 6] / s[:, 15].clamp(min=1)).median()):.2f} of loop cycles",
                  flush=True)
        if args.trace:
            _trace(b, trace.view(48, 16).cpu().double(), args.ring)


def _trace(b, e, ring):
    """Median per-stage phases (cycles) of the traced CTA over stages 2..n-2."""
    n = int((e[:, 0] > 0).sum())
    if n < 6:
        print(f"B{b}: trace has {n} stages", flush=True)
        return
    rows = range(2, n - 3)

    def med(f):
        v = sorted(f(t) for t in rows)
        return v[len(v) // 2]

    parts = {
        "period": lambda t: (e[t + 2, 0] - e[t, 0]) / 2,
        "waitS": lambda t: e[t, 1] - e[t, 0],
        "ld": lambda t: e[t, 2] - e[t, 1],
        "math": lambda t: e[t, 3] - e[t, 2],
        "vote": lambda t: e[t, 4] - e[t, 3],
        "exp": lambda t: e[t, 5] - e[t, 4],
        "waitPbuf": lambda t: e[t, 11] - e[t, 5],
        "sts": lambda t: e[t, 12] - e[t, 11],
        "fence+arrive": lambda t: e[t, 13] - e[t, 12],
        "S landed->issued": lambda t: e[t, 7] - e[t, 6],
        "S issued->softmax sees": lambda t: e[t, 1] - e[t, 7],
        "P ready->PV issued": lambda t: e[t, 9] - e[t, 8],
        "P arrive->PV wait done": lambda t: e[t, 8] - e[t, 13],
        "refill issued->landed": lambda t: e[t + ring, 6] - e[t, 10] if t + ring < n else 0,
    }
    print(f"B{b} trace cycles: " + " ".join(f"{k} {med(f):.0f}" for k, f in parts.items()),
          flush=True)


if __name__ == "__main__":
    main()
