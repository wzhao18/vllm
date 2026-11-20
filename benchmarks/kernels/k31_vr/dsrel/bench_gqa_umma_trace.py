# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-stage event trace of one GqaWindowUmma CTA (request ``--req``, KV head 1) at
the deployed launch shapes: bench_gqa_attention_int.py's fused arm with the kernel's
development ``trace`` on (clock64 per event, one VR200 GPU). Prints, over the steady
stages, the median cycles of each step of the producer (S issue, P-ready wait, PV,
refill) and of softmax warp 0, and the cross-warp latencies (S MMA issue -> S ready
seen, P ready -> seen, refill issue -> data seen), which locate the stage loop's
critical path.

  python3 bench_gqa_umma_trace.py --batches 64,96 --shape 1/0/6/0
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

TR = None
REQ = 0
RING = 6


def _patch():
    orig = ga.compile_window_umma

    def patched(*args, **kwargs):
        global RING
        kwargs["trace"] = TR.data_ptr()
        kwargs["trace_req"] = REQ
        RING = args[4]
        return orig(*args, **kwargs)

    ga.compile_window_umma = patched


def _med(x):
    x = sorted(x)
    return x[len(x) // 2] if x else float("nan")


def _report(b, ghz):
    e = TR.view(48, 16).cpu().tolist()
    n = 0
    while n < 48 and e[n][8] > 0:
        n += 1
    if n < 12:
        print(f"B{b}: {n} traced stages (not GqaWindowUmma?)", flush=True)
        return
    ts = range(4, n - 6)

    def d(a, bb, off=0):
        return _med([e[t + off][bb] - e[t][a] for t in ts])

    rows = {
        "P wait full": d(0, 1),
        "P wait sfree": d(1, 2),
        "P S issue": d(2, 3),
        "P wait pready": d(4, 5),
        "P PV issue": d(5, 6),
        "P period": _med([e[t + 1][4] - e[t][4] for t in ts]),
        "S wait sready": d(8, 9),
        "S ld S": d(9, 10),
        "S math": d(10, 11),
        "S wait P buf": d(11, 12),
        "S st P+arrive": d(12, 13),
        "S period": _med([e[t + 1][8] - e[t][8] for t in ts]),
        "x S issued->seen": d(3, 9),
        "x pready->seen(w0)": d(13, 5),
        "x pready w3-w0": d(13, 14),
        "x PV issued->pvdone seen": _med([e[t][7] - e[t + 1][6] for t in ts]),
        "x refill->data seen": _med([e[t + RING][1] - e[t][7] for t in ts if t + RING < n]),
    }
    print(f"B{b}: {n} stages traced (req {REQ}); median cycles (ns at {ghz} GHz):", flush=True)
    for k, v in rows.items():
        print(f"  {k:26s} {v:8.0f} cyc {v / ghz:8.1f} ns", flush=True)


def main():
    global TR, REQ
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="64,96")
    ap.add_argument("--req", type=int, default=-1, help="traced request (-1: B // 2)")
    ap.add_argument("--ghz", type=float, default=2.4)
    args, rest = ap.parse_known_args()
    TR = torch.zeros(48 * 16, dtype=torch.int64, device="cuda")
    _patch()
    for b in [int(x) for x in args.batches.split(",")]:
        REQ = b // 2 if args.req < 0 else args.req
        TR.zero_()
        sys.argv = ["bench", "--tp", "4", "--q", "3", "--batches", str(b), "--arms",
                    "fused", "--no-context"] + rest
        bench.main()
        torch.cuda.synchronize()
        _report(b, args.ghz)


if __name__ == "__main__":
    main()
