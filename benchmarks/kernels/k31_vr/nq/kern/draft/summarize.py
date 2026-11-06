# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Merge the drafter kernel benches' CSVs under a results dir into SUMMARY.txt
content (stdout): per CSV the row / WORSE counts and worst ratios, the combined WORSE
list (specialized / stock > 1.0), failed / missing rows, and the GEMM TUNE bands.
Pure python (no torch)."""

import csv
import glob
import math
import os
import sys


def load(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def num(v):
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def main(root):
    # an earlier run's results set aside by inner.sh (gemm.prev.*), and the GEMM
    # bench's resume log (raw.csv), are not result tables
    files = sorted(f for f in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
                   if ".prev." not in os.path.relpath(f, root)
                   and os.path.basename(f) != "raw.csv")  # fmt: skip
    worse, missing, refs = [], [], []
    print(f"drafter kernel bench summary: {root}")
    for f in files:
        name = os.path.relpath(f, root)
        if name.endswith("gemm_tune_bands.csv"):
            continue
        rows = load(f)
        if name.startswith("dcp_single" + os.sep):
            # reference: new-geometry core vs old-geometry core, not specialized vs
            # stock (never WORSE / MISSING)
            refs += [(name, r) for r in rows]
            continue
        ratios = [(num(r.get("ratio")), r) for r in rows]
        have = [(x, r) for x, r in ratios if x is not None]
        bad = [(x, r) for x, r in have if x > 1.0]
        worst = max(have, key=lambda t: t[0]) if have else None
        best = min(have, key=lambda t: t[0]) if have else None
        print(f"\n{name}: {len(rows)} rows, {len(have)} with both arms, "
              f"{len(bad)} WORSE"
              + (f"; ratio min {best[0]:.3f} ({best[1]['case']} r{best[1]['rows']}) "
                 f"max {worst[0]:.3f} ({worst[1]['case']} r{worst[1]['rows']})"
                 if have else ""))  # fmt: skip
        for x, r in bad:
            worse.append((name, x, r))
        for x, r in ratios:
            if x is None:
                missing.append((name, r))
    print(f"\n==== WORSE (specialized / stock > 1.0): {len(worse)}")
    for name, x, r in sorted(worse, key=lambda t: -t[1]):
        print(f"WORSE {x:6.3f} {r['case']:<36} rows {r['rows']:>5} spec {r['spec_us']} "
              f"stock {r['stock_us']} chosen {r['chosen']} {r.get('labels', '')} [{name}]")
    print(f"\n==== no ratio (failed, dropped to stock, or one arm missing): {len(missing)}")
    for name, r in missing[:200]:
        print(f"MISSING {r['case']:<36} rows {r['rows']:>5} spec {r['spec_us'] or '-'} "
              f"stock {r['stock_us'] or '-'} {r['chosen']} {r.get('note', '')} [{name}]")
    if refs:
        print(f"\n==== REFERENCE tokenspeed core, new 96h causal / old 64h non-causal: "
              f"{len(refs)} rows")
        for name, r in refs:
            print(f"REF {r['case']:<36} rows {r['rows']:>5} new96 {r.get('new96_us') or '-'} "
                  f"old64 {r.get('old64_us') or '-'} new/old {r.get('new_over_old') or '-'}")
    tunes = sorted(glob.glob(os.path.join(root, "**", "gemm_tune_bands.csv"), recursive=True))
    tunes = [f for f in tunes if ".prev." not in os.path.relpath(f, root)]
    if tunes:
        print("\n==== GEMM TUNE: best tactic per (linear, row band), geo-mean us vs cuBLAS")
        for f in tunes:
            for r in load(f):
                print(f"TUNE {r['linear']:<12} band {r['band']:<8} best {r['best_tactic']:<16} "
                      f"{r['geo_us']:>7} us cuBLAS {r['cublas_geo_us']:>7} ratio {r['ratio']} "
                      f"| table {r['table_tactic']} {r['table_geo_us']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
