# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tables, WORSE list and CSV of the nq DSA kernel benches (pure Python, no torch).

    python3 summarize.py <results dir or *.jsonl ...> [--csv out.csv] [--arms arms.csv]
    python3 summarize.py --selftest

Per (bench, case, q, rows): the "spec" arm (the code's policy at those rows), the
"stock" arm, ratio = spec / stock, the chosen tactic (the spec arm), the fastest
specialized arm (spec or tactic) and its ratio, the previous-configuration ("ref":
nvfp4_ds_mla) and "alt" numbers, and the failed arms. Derived rows (bench "chain"):

  core+value   rows <= 64, "bal" core: served chain = trtllm_fp8 + value spec arm
               vs the DSA sparse core chain = dsac policy (else fastest) variant +
               DcpValue(pushed[_spread] by dsa_core SPREAD); (1-GPU core with local
               pushes + 4+12-rank value)
  front+value  rows > 64: WideDsa (mlaquery + value spec) vs S2-DSA (stock_front +
               stock_a2a), the stages that differ (the indexer front and the shared
               indexer / sparse cores excluded; stock arms APPROXIMATE, see the
               bench docstrings)

WORSE: every row whose ratio > 1.0 (spec slower than stock). BEST!=POLICY: rows where
a specialized tactic beats the policy's arm by more than --margin (default 3%).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

SPREAD_ROWS = range(17, 65)  # dsa_core.decode.SPREAD = "17-64:1"

# Previous numbers in the code comments (us per call), compared with this run:
# dsa_kernels/dsa_core/decode.py (Round 3 Step 0a, one SM107 GPU, DCP16 rank shapes,
# q8, CUPTI: the trtllm-gen chain / 1x16 / 2x16s2 / 4x16s2 / 8x16s4; DcpValue push +
# merge vs DcpValue(pushed) at M8 / M16).
_CORE_ARMS = ("trtllm_fp8", "dsac_1x16", "dsac_2x16s2", "dsac_4x16s2", "dsac_8x16s4")
_PREV_CORE = {
    ("bal", 8): (13.47, 9.48, 9.45, 8.73, 10.43),
    ("x256", 8): (13.54, 12.75, 10.59, 15.84, 14.18),
    ("x384", 8): (13.61, 17.47, 12.98, 20.45, 25.50),
    ("x600", 8): (14.74, 26.13, 17.69, 25.20, 32.30),
    ("x2048", 8): (17.17, 78.56, 43.72, 49.28, 47.25),
    ("bal", 16): (14.34, 9.65, 9.71, 16.25, 17.66),
    ("x256", 16): (14.44, 12.85, 10.69, 17.70, 21.10),
    ("x384", 16): (14.65, 17.56, 13.07, 21.21, 29.72),
    ("x600", 16): (15.71, 26.22, 17.78, 25.34, 34.05),
}
PREVIOUS = {("core", case, 8, rows): dict(zip(_CORE_ARMS, us))
            for (case, rows), us in _PREV_CORE.items()}  # fmt: skip
PREVIOUS[("core", "bal", 8, 24)] = {"trtllm_fp8": 14.98, "dsac_1x16": 9.83,
                                    "dsac_2x16s2": 16.19}  # fmt: skip
PREVIOUS[("core", "x384", 8, 24)] = {"trtllm_fp8": 15.25, "dsac_1x16": 17.64,
                                     "dsac_2x16s2": 16.99}  # fmt: skip
PREVIOUS[("core", "x600", 8, 24)] = {"trtllm_fp8": 15.91, "dsac_1x16": 26.30,
                                     "dsac_2x16s2": 17.89}  # fmt: skip
PREVIOUS[("value", "agentx", 8, 8)] = {"bf16": 8.96, "pushed": 7.00}
PREVIOUS[("value", "agentx", 8, 16)] = {"bf16": 11.82, "pushed": 9.59}


def load(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "**", "*.jsonl"), recursive=True))
        else:
            files += sorted(glob.glob(p))
    recs = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if "arm" in rec and "bench" in rec:
                    recs.append(rec)
    return recs, files


def group(recs):
    """(bench, case, q, rows) -> {arm: record} (the last record of an arm wins)."""
    out = defaultdict(dict)
    for r in recs:
        out[(r["bench"], r.get("case"), r.get("q"), r.get("rows"))][r["arm"]] = r
    return out


def ok(rec):
    return rec is not None and rec.get("status") == "ok" and rec.get("us") is not None


def summarize_key(key, arms, margin):
    bench, case, q, rows = key
    row = dict(bench=bench, case=case, q=q, rows=rows, spec_arm="", spec_us=None,
               stock_arm="", stock_us=None, ratio=None, best_arm="", best_us=None,
               best_ratio=None, ref_arm="", ref_us=None, alt_arm="", alt_us=None,
               note="", failed="", flags="")  # fmt: skip
    hang = arms.pop("*", None)  # a run died inside this key: the later arms missing
    if hang is not None:
        row["flags"] = "HANG "
    spec = next((r for r in arms.values() if r.get("policy") and ok(r)), None)
    stock = next((r for r in arms.values() if r.get("role") == "stock" and ok(r)), None)
    specialized = [r for r in arms.values()
                   if r.get("role") in ("spec", "tactic") and ok(r)]  # fmt: skip
    if spec is not None and spec.get("role") == "stock":
        specialized.append(spec)  # (the policy is the stock op at these rows)
    best = min(specialized, key=lambda r: r["us"], default=None)
    for kind in ("ref", "alt"):
        r = next((r for r in arms.values() if r.get("role") == kind and ok(r)), None)
        if r is not None:
            row[kind + "_arm"], row[kind + "_us"] = r["arm"], r["us"]
    if spec is not None:
        row["spec_arm"], row["spec_us"] = spec["arm"], spec["us"]
        row["note"] = spec.get("note", "")
    if stock is not None:
        row["stock_arm"], row["stock_us"] = stock["arm"], stock["us"]
        if "APPROX" in (stock.get("note") or ""):
            row["flags"] += "STOCK~ "
    if best is not None:
        row["best_arm"], row["best_us"] = best["arm"], best["us"]
    if spec is not None and stock is not None and stock["us"] > 0:
        row["ratio"] = round(spec["us"] / stock["us"], 4)
    if best is not None and stock is not None and stock["us"] > 0:
        row["best_ratio"] = round(best["us"] / stock["us"], 4)
    if spec is not None and best is not None and best["us"] < spec["us"] * (1 - margin):
        row["flags"] += "BEST!=POLICY "
    if row["ratio"] is not None and row["ratio"] > 1.0:
        row["flags"] += "WORSE "
    if spec is None:
        row["flags"] += "NO-SPEC "
    if stock is None:
        row["flags"] += "NO-STOCK "
    failed = [f"{a}({r.get('status')})" for a, r in arms.items() if not ok(r)]
    if hang is not None:
        failed.append("HANG(arms after the last record)")
    row["failed"] = " ".join(sorted(failed))
    row["flags"] = row["flags"].strip()
    return row


def derived(groups, margin):
    """The chain rows (see the module docstring)."""
    rows_out = []
    keys = {(k[2], k[3]) for k in groups}
    for q, rows in sorted(keys, key=lambda t: (t[0] or 0, t[1] or 0)):
        core = groups.get(("core", "bal", q, rows), {})
        value = groups.get(("value", "agentx", q, rows), {})
        front = groups.get(("front", "agentx", q, rows), {})
        vspec = next((r for r in value.values() if r.get("policy") and ok(r)), None)
        if rows is not None and rows <= 64 and core and value:
            served_core = core.get("trtllm_fp8")
            dsac = {a: r for a, r in core.items() if a.startswith("dsac_") and ok(r)}
            pol = next((r for r in dsac.values() if r.get("policy")), None)
            pick = pol or min(dsac.values(), key=lambda r: r["us"], default=None)
            pushed = value.get("pushed_spread" if rows in SPREAD_ROWS else "pushed")
            if ok(served_core) and ok(vspec) and pick is not None and ok(pushed):
                stock_us = served_core["us"] + vspec["us"]
                spec_us = pick["us"] + pushed["us"]
                flags = "WORSE" if spec_us > stock_us else ""
                if pol is None:
                    flags = (flags + " NO-CORE-POLICY(best variant)").strip()
                rows_out.append(dict(
                    bench="chain", case="core+value", q=q, rows=rows,
                    spec_arm=f"{pick['arm']}+{pushed['arm']}", spec_us=round(spec_us, 3),
                    stock_arm=f"trtllm_fp8+{vspec['arm']}", stock_us=round(stock_us, 3),
                    ratio=round(spec_us / stock_us, 4), best_arm="", best_us=None,
                    best_ratio=None, ref_arm="", ref_us=None, alt_arm="", alt_us=None,
                    note="1-GPU core (local pushes) + 4+12-rank value", failed="",
                    flags=flags))  # fmt: skip
        if rows is not None and rows > 64 and front and value:
            fspec, fstock = front.get("mlaquery"), front.get("stock_front")
            vstock = value.get("stock_a2a")
            if ok(fspec) and ok(fstock) and ok(vspec) and ok(vstock):
                spec_us = fspec["us"] + vspec["us"]
                stock_us = fstock["us"] + vstock["us"]
                ref = front.get("mlaquery_nvfp4")
                rows_out.append(dict(
                    bench="chain", case="front+value", q=q, rows=rows,
                    spec_arm=f"mlaquery+{vspec['arm']}", spec_us=round(spec_us, 3),
                    stock_arm="stock_front+stock_a2a", stock_us=round(stock_us, 3),
                    ratio=round(spec_us / stock_us, 4), best_arm="", best_us=None,
                    best_ratio=None, ref_arm="mlaquery_nvfp4+value" if ok(ref) else "",
                    ref_us=round(ref["us"] + vspec["us"], 3) if ok(ref) else None,
                    alt_arm="", alt_us=None,
                    note="WideDsa vs S2-DSA, differing stages only", failed="",
                    flags=("WORSE " if spec_us > stock_us else "") + "STOCK~"))  # fmt: skip
    return rows_out


COLS = ("bench", "case", "q", "rows", "spec_arm", "spec_us", "stock_arm", "stock_us",
        "ratio", "best_arm", "best_us", "best_ratio", "ref_arm", "ref_us", "alt_arm",
        "alt_us", "flags", "failed", "note")  # fmt: skip


def fmt(v, width, digits=2):
    if v is None or v == "":
        return "-".rjust(width)
    if isinstance(v, float):
        return f"{v:.{digits}f}".rjust(width)
    return str(v).rjust(width)


def print_table(rows, out=sys.stdout):
    head = (f"{'bench':>6} {'case':>11} {'q':>2} {'rows':>5} {'spec_us':>9} "
            f"{'stock_us':>9} {'ratio':>6}  {'chosen tactic':<20} {'best':<20} "
            f"{'best/st':>7} {'ref_us':>8} {'alt_us':>8}  flags / failed")  # fmt: skip
    print(head, file=out)
    last = None
    for r in rows:
        sect = (r["bench"], r["case"])
        if last is not None and sect != last:
            print("", file=out)
        last = sect
        print(f"{r['bench']:>6} {str(r['case']):>11} {fmt(r['q'], 2)} {fmt(r['rows'], 5)} "
              f"{fmt(r['spec_us'], 9)} {fmt(r['stock_us'], 9)} {fmt(r['ratio'], 6, 3)}  "
              f"{(r['spec_arm'] or '-'):<20} {(r['best_arm'] or '-'):<20} "
              f"{fmt(r['best_ratio'], 7, 3)} {fmt(r['ref_us'], 8)} {fmt(r['alt_us'], 8)}  "
              f"{r['flags']} {r['failed']}".rstrip(), file=out)  # fmt: skip


def build_rows(recs, margin):
    groups = group(recs)
    order = {"core": 0, "value": 1, "front": 2}
    keys = sorted(groups, key=lambda k: (order.get(k[0], 9), str(k[1]), k[2] or 0,
                                         k[3] or 0))  # fmt: skip
    rows = [summarize_key(k, groups[k], margin) for k in keys]
    return rows + derived(groups, margin)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--arms", default=None, help="every record, flattened")
    ap.add_argument("--margin", type=float, default=0.03)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    recs, files = load(args.paths)
    print(f"{len(recs)} records from {len(files)} files")
    rows = build_rows(recs, args.margin)
    print_table(rows)
    worse = [r for r in rows if "WORSE" in r["flags"]]
    print(f"\nWORSE (ratio > 1.0: specialized slower than stock): {len(worse)}")
    for r in worse:
        print(f"  {r['bench']} {r['case']} q{r['q']} M{r['rows']}: {r['spec_arm']} "
              f"{r['spec_us']} vs {r['stock_arm']} {r['stock_us']} (x{r['ratio']})"
              f"{' [stock approximate]' if 'STOCK~' in r['flags'] else ''}")  # fmt: skip
    better = [r for r in rows if "BEST!=POLICY" in r["flags"]]
    print(f"\nBEST!=POLICY (a tactic beats the policy by > {args.margin:.0%}): "
          f"{len(better)}")  # fmt: skip
    for r in better:
        print(f"  {r['bench']} {r['case']} q{r['q']} M{r['rows']}: policy "
              f"{r['spec_arm']} {r['spec_us']} vs {r['best_arm']} {r['best_us']}")
    groups = group(recs)
    print("\nPREVIOUS (numbers in the code comments, us) vs now:")
    for key, prev in sorted(PREVIOUS.items(), key=lambda kv: str(kv[0])):
        now = groups.get(key, {})
        cells = []
        for arm, us in prev.items():
            r = now.get(arm)
            cells.append(f"{arm} {us:.2f}->" + (f"{r['us']:.2f}" if ok(r) else "-"))
        print(f"  {key[0]} {key[1]} q{key[2]} M{key[3]}: " + ", ".join(cells))
    bad = [r for r in rows if r["failed"] or "NO-" in r["flags"] or "HANG" in r["flags"]]
    print(f"\nINCOMPLETE (failed / hung arms, missing spec or stock): {len(bad)}")
    for r in bad:
        print(f"  {r['bench']} {r['case']} q{r['q']} M{r['rows']}: {r['flags']} "
              f"{r['failed']}")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in COLS})
        print(f"\nCSV: {args.csv}")
    if args.arms:
        keys = ["bench", "case", "q", "rows", "arm", "role", "policy", "us", "status",
                "raw_us", "sub_us", "per_rank_us", "err", "note"]  # fmt: skip
        with open(args.arms, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in recs:
                w.writerow({k: r.get(k) for k in keys})
        print(f"arms CSV: {args.arms}")
    return 0


def selftest():
    """Synthetic records through every path (no GPU)."""
    import tempfile

    recs = []

    def rec(bench, case, q, rows, arm, us, role, policy=False, status="ok", note=""):
        recs.append(dict(bench=bench, case=case, q=q, rows=rows, arm=arm, us=us,
                         role=role, policy=policy, status=status, note=note))  # fmt: skip

    rec("core", "bal", 8, 8, "trtllm_fp8", 13.5, "stock")
    rec("core", "bal", 8, 8, "dsac_2x16s2", 9.4, "spec", True)
    rec("core", "bal", 8, 8, "dsac_4x16s2", 8.7, "tactic")
    rec("core", "bal", 8, 8, "prims_nvfp4", 16.0, "ref")
    rec("core", "bal", 4, 4, "trtllm_fp8", 12.0, "stock", True)
    rec("core", "bal", 4, 4, "dsac_1x16", 9.0, "tactic")
    rec("core", "x2048", 8, 8, "trtllm_fp8", 17.2, "stock")
    rec("core", "x2048", 8, 8, "dsac_2x16s2", 43.7, "spec", True)
    rec("value", "agentx", 8, 8, "bf16", 9.0, "spec", True)
    rec("value", "agentx", 8, 8, "proj", 9.5, "tactic")
    rec("value", "agentx", 8, 8, "pushed", 7.0, "part")
    rec("value", "agentx", 8, 8, "stock_a2a", 20.0, "stock", note="APPROX: a2a")
    rec("value", "agentx", 4, 1024, "proj", 60.0, "spec", True)
    rec("value", "agentx", 4, 1024, "bf16", 80.0, "tactic")
    rec("value", "agentx", 4, 1024, "bf16_g18", None, "tactic", status="fail")
    rec("value", "agentx", 4, 1024, "stock_a2a", 55.0, "stock", note="APPROX: a2a")
    rec("front", "agentx", 4, 1024, "mlaquery", 30.0, "spec", True)
    rec("front", "agentx", 4, 1024, "stock_front", 45.0, "stock", note="APPROX")
    rec("front", "agentx", 4, 1024, "mlaquery_nvfp4", 31.0, "ref")
    recs.append(dict(bench="value", case="agentx", q=8, rows=2048, arm="*", us=None,
                     role="part", status="hang", err="begun"))  # fmt: skip
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.jsonl")
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        csv_path = os.path.join(d, "s.csv")
        main([d, "--csv", csv_path, "--arms", os.path.join(d, "a.csv")])
        loaded, _ = load([d])
        rows = {(r["bench"], r["case"], r["q"], r["rows"]): r for r in build_rows(loaded, 0.03)}
        assert rows[("core", "bal", 8, 8)]["ratio"] == round(9.4 / 13.5, 4)
        assert "BEST!=POLICY" in rows[("core", "bal", 8, 8)]["flags"]
        assert rows[("core", "bal", 4, 4)]["ratio"] == 1.0
        assert rows[("core", "bal", 4, 4)]["best_arm"] == "dsac_1x16"
        assert "WORSE" in rows[("core", "x2048", 8, 8)]["flags"]
        assert "WORSE" in rows[("value", "agentx", 4, 1024)]["flags"]
        assert "bf16_g18(fail)" in rows[("value", "agentx", 4, 1024)]["failed"]
        assert rows[("value", "agentx", 8, 2048)]["flags"].startswith("HANG")
        chain = rows[("chain", "core+value", 8, 8)]
        assert chain["spec_us"] == 16.4 and chain["stock_us"] == 22.5, chain
        fv = rows[("chain", "front+value", 4, 1024)]
        assert fv["spec_us"] == 90.0 and fv["stock_us"] == 100.0, fv
        with open(csv_path) as f:
            assert len(list(csv.DictReader(f))) == len(rows)
    print("summarize selftest OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
