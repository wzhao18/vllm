# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase stamps of the deployed GqaWindowUmma (the release drafter's window attention)
at the deployed launch shapes: bench_gqa_attention_int.py's fused arm with the kernel's
development ``debug`` stamps on (globaltimer + clock64 per CTA), one VR200 GPU.

Per batch: the per-layer time of the bench, then from the last call's stamps the kernel
span (last CTA end - first CTA start), the CTA start spread (wave), and per-CTA medians
of the phases: PDL wait (s1 - s0), setup to the first stage (s2 - s1), the stage loop
(s3 - s2), the merge (s10 - s3), the store (s11 - s10); TMA-full wait / P-ready wait as
a share of the loop cycles. This calibrates the GQA model (bytes / placement rate +
fixed phases) against today's kernel.

  python3 bench_gqa_umma_stamps.py --batches 96 --shape 1/0/6/0 --fill 2
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


FILL = 0


def _patch():
    orig = ga.compile_window_umma

    def patched(*args, **kwargs):
        kwargs["debug"] = DBG.data_ptr()
        kwargs["fill"] = FILL
        return orig(*args, **kwargs)

    ga.compile_window_umma = patched


def _report(b):
    s = DBG.view(2, -1, 16)[0].cpu()
    ok = s[:, 0] > 0
    s = s[ok].double()
    n = s.shape[0]
    if n == 0:
        print(f"B{b}: no stamps (not GqaWindowUmma at this batch?)", flush=True)
        return
    t0 = s[:, 0].min()
    span = (s[:, 11].max() - t0) / 1e3
    spread = (s[:, 0].max() - t0) / 1e3

    def med(x):
        return float(x.median())

    ph = {
        "pdl": med(s[:, 1] - s[:, 0]) / 1e3,
        "setup": med(s[:, 2] - s[:, 1]) / 1e3,
        "loop": med(s[:, 3] - s[:, 2]) / 1e3,
        "merge": med(s[:, 10] - s[:, 3]) / 1e3,
        "store": med(s[:, 11] - s[:, 10]) / 1e3,
    }
    loopmax = float((s[:, 3] - s[:, 2]).max()) / 1e3
    endq = torch.quantile(s[:, 11] - t0, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)) / 1e3
    stages = med(s[:, 12])
    cyc = s[:, 15].clamp(min=1)
    full = med(s[:, 4] / cyc)
    pready = med(s[:, 6] / cyc)
    print(
        f"B{b}: CTAs {n} span {span:.2f} us start-spread {spread:.2f} | per-CTA median "
        + " ".join(f"{k} {v:.2f}" for k, v in ph.items())
        + f" | loop max {loopmax:.2f} stages {stages:.0f} | end q10/50/90 "
        + "/".join(f"{x:.1f}" for x in endq.tolist())
        + f" | wait full {full:.2f} pready {pready:.2f} pv {med(s[:, 5] / cyc):.2f} sm-sready {med(s[:, 8] / cyc):.2f} of loop cycles",
        flush=True,
    )


def main():
    global DBG
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="32,64,96")
    ap.add_argument("--ctx", default="150000")
    ap.add_argument("--fill", type=int, default=0, help="first-ring issue placement (GqaWindowUmma fill)")
    args, rest = ap.parse_known_args()
    DBG = torch.zeros(2 * 8192 * 16, dtype=torch.int64, device="cuda")
    global FILL
    FILL = args.fill
    _patch()
    for b in [int(x) for x in args.batches.split(",")]:
        DBG.zero_()
        sys.argv = ["bench", "--tp", "4", "--q", "3", "--batches", str(b), "--arms", "fused",
                    "--no-context", "--ctx", args.ctx] + rest
        bench.main()
        torch.cuda.synchronize()
        _report(b)


if __name__ == "__main__":
    main()
