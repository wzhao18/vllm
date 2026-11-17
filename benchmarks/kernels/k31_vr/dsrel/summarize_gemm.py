# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize bench_draft_gemm.py's chosen_tp*.csv: per linear and row band the
geometric means of the production choice and of cuBLAS (us), and the per-step sum of
the release drafter's projections vs the preview drafter's at matching rows (TP16:
5 x (qkv + o_proj) + context K/V vs 5 x (qkv_a + q_b + o_proj) + context K/V; the
MLP and context shard are the same shapes in both). Pure python.

    python summarize_gemm.py <chosen_tp16.csv>
"""

import csv
import math
import sys
from collections import defaultdict


def geo(vals):
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def main(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    data = defaultdict(dict)  # linear -> rows -> (spec, cublas, chosen)
    for r in rows:
        try:
            spec, cub = float(r["spec_us"]), float(r["cublas_us"])
        except ValueError:
            continue
        data[r["linear"]][int(r["rows"])] = (spec, cub, r["chosen"])
    print(
        f"{'linear':<11} {'band':<8} {'n':>3} {'choice us':>9} {'cuBLAS us':>9} ratio"
    )
    for linear, by_rows in data.items():
        bands = defaultdict(list)
        for n, v in by_rows.items():
            bands[r_band(n)].append(v)
        for band in sorted(bands, key=lambda b: int(b.split("-")[0].rstrip("+"))):
            vals = bands[band]
            s, c = geo([v[0] for v in vals]), geo([v[1] for v in vals])
            print(
                f"{linear:<11} {band:<8} {len(vals):>3} {s:9.2f} {c:9.2f} {s / c:.3f}"
            )
    # Per-step projection time, release vs preview, at the query rows / context rows
    # of matching batches (k7: R = 7B, C = 8B; k3: R = 3B, C = 4B).
    print("\nper-step projections (5 layers' attention in/out + context K/V), us")
    print(f"{'k':>2} {'B':>4} {'release':>9} {'preview':>9} {'release cuBLAS':>15}")
    for k, batches in ((7, (1, 2, 4, 8, 16, 32, 64)), (3, (2, 8, 16, 40, 80, 160))):
        for b in batches:
            q, c = b * k, b * (k + 1)
            try:
                rel = 5 * (data["qkv"][q][0] + data["o_proj"][q][0])
                rel += data["ctx_kv"][c][0]
                cub = 5 * (data["qkv"][q][1] + data["o_proj"][q][1])
                cub += data["ctx_kv"][c][1]
                pv = 5 * (data["pv_qkv_a"][q][0] + data["pv_q_b"][q][0])
                pv += 5 * data["pv_o_proj"][q][0] + data["pv_ctx_kv"][c][0]
            except KeyError:
                continue
            print(f"{k:>2} {b:>4} {rel:9.1f} {pv:9.1f} {cub:15.1f}")


def r_band(n):
    for top, name in ((8, "1-8"), (16, "9-16"), (32, "17-32"), (64, "33-64"),
                      (128, "65-128"), (256, "129-256"), (640, "257-640")):  # fmt: skip
        if n <= top:
            return name
    return "641+"


if __name__ == "__main__":
    main(sys.argv[1])
