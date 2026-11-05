"""nq_summary.py <results dir> [--tag <point-drafter-ctx[-x...]>] [--csv out.csv]: per-row A/B table of the nq step bench.

Reads every <tag>-<i><arm>.log in the dir (nq_run1.sh logs; works on partial / crashed runs, the json is only
written at the end). Per verify row M (= B*q): each arm's run medians (us), ls/stock (mean of run medians), the ls
process's coverage line for M ('Kimi K3 decode coverage', runtime/decode.py: specialized families or 'stock (why)')
and the drafter fast paths the ls process installed for B (fused Markov batches, Lamport DCP / context rows, LM head).
Flags per row: WORSE (ls >= stock), MISSING(<arm>) (a requested row an arm did not measure: crash, timeout, skip),
UNCOVERED (no ls coverage line for the row: not a captured specialized shape), XCAP (above the point's production
limit: that process ran with raised max-num-seqs / capture cap, nq_ab.sh part 'hi'); per point: whole-step stock
rows, family stock seams. --tag T reads the point's two process tags T (rows <= production limit) and T-hi.
"""
import argparse
import csv
import glob
import os
import re
import statistics

FWD = re.compile(r"^\[forward\] q(\d+)_B(\d+)_M(\d+)_(\S+): .*median ([0-9.]+) us")
COV = re.compile(r"Kimi K3 decode coverage M(\d+)\(q(\d+)\): (.*)$")
DRAFT = {
    "markov": re.compile(r"fused \w+ Markov sampling for draft batches \(([^)]*)\)"),
    # 'Lamport DCP attention (causal|non-causal) for (rows) rows' (any annotation between)
    "dcp": re.compile(r"Lamport DCP attention\b.*? for \(([^)]*)\) rows"),
    "ctx": re.compile(r"sharded context step for \(([^)]*)\) rows"),
    "lm": re.compile(r"LM-head GEMV for draft batches \(([^)]*)\)"),
    "lmw": re.compile(r"evict-first LM-head GEMM for \(([^)]*)\)"),
}
ERR = re.compile(r"(\w+Error|AssertionError|CUDA error|NCCL error|out of memory)[^\n]*")


def ints(text):
    return {int(x) for x in re.findall(r"\d+", text)}


def parse(path):
    run = {"path": path, "fwd": {}, "cov": {}, "draft": {}, "err": "", "rc": None, "meta": {}}
    with open(path, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("NQRUN "):
                run["meta"] = dict(kv.split("=", 1) for kv in line[6:].split() if "=" in kv)
            elif line.startswith("NQEND "):
                run["rc"] = int(re.search(r"rc=(-?\d+)", line).group(1))
            elif m := FWD.match(line):
                q, b, rows, ctx, med = m.groups()
                run["fwd"][(int(rows), ctx)] = (int(b), float(med))
            elif m := COV.search(line):
                run["cov"][int(m.group(1))] = m.group(3).strip()
            else:
                for key, pat in DRAFT.items():
                    if (m := pat.search(line)) and key not in run["draft"]:
                        run["draft"][key] = ints(m.group(1))
                if not run["err"] and (m := ERR.search(line)) and "Error:" in line:
                    run["err"] = m.group(0)[:160]
    return run


def seams(body):
    if body.startswith("stock"):
        return []
    fams = re.split(r"\s(?=\w+=)", body)
    return [f for f in fams if "=stock" in f]


def drafter_flags(run, b, k):
    d = run["draft"]
    out = []
    if b in d.get("markov", ()):
        out.append("markov")
    if b * k in d.get("dcp", ()):
        out.append("dcp")
    if b * (k + 1) in d.get("ctx", ()):  # context rows are B*(K+1) (runtime/draft.py)
        out.append("ctx")
    if b in d.get("lm", ()) or b in d.get("lmw", ()):
        out.append("lm")
    return ",".join(out) or "stock"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--tag", default="")
    ap.add_argument("--csv", default="")
    a = ap.parse_args()
    logs = sorted(glob.glob(os.path.join(a.dir, f"{a.tag}*.log")))
    groups = {}
    for p in logs:
        m = re.match(r"(.*)-(\d+)(ls|stock)\.log$", os.path.basename(p))
        if m and (not a.tag or m.group(1) in (a.tag, f"{a.tag}-hi")):
            groups.setdefault(m.group(1), []).append((int(m.group(2)), m.group(3), parse(p)))
    if not groups:
        print(f"== {a.tag or a.dir}: MISSING: no run logs")
    rows_out = []
    for tag, runs in sorted(groups.items()):
        runs.sort()
        meta = runs[0][2]["meta"]
        q = int(meta.get("q", "0") or 0)
        k = q - 1
        print(f"== {tag}: point {meta.get('point')} q{q} drafter {meta.get('drafter')} ctx {meta.get('ctx')} "
              f"extra {meta.get('extra') or '-'} overlay {meta.get('ov')}")
        for i, arm, r in runs:
            state = f"rc={r['rc']}" if r["rc"] is not None else "running/killed"
            print(f"   run {i}{arm}: {state}, {len(r['fwd'])} cases" + (f", {r['err']}" if r["err"] else ""))
        arms = sorted({arm for _, arm, _ in runs})
        try:
            prodlimit = int(meta.get("prodlimit", ""))
        except ValueError:
            prodlimit = None
        # every requested case (rows x ctx of the NQRUN line), measured or not
        asked = {(int(m), c) for _, _, r in runs for m in r["meta"].get("rows", "").split(",") if m.isdigit()
                 for c in r["meta"].get("ctx", "").split(",") if c}
        keys = sorted(asked | {key for _, _, r in runs for key in r["fwd"]})
        ls_runs = [r for _, arm, r in runs if arm == "ls"]
        cov = next((r["cov"] for r in ls_runs if r["cov"]), {})
        draft_run = next((r for r in ls_runs if r["draft"]), None)
        print(f"   {'rows':>6} {'B':>4} {'ctx':>7} {'stock us':>19} {'ls us':>19} {'ls/st':>6}  ls coverage | drafter")
        worse, stock_rows, seam_rows, missing, uncovered = [], [], [], [], []
        for rows, ctx in keys:
            med = {"stock": [], "ls": []}
            b = None
            for _, arm, r in runs:
                if (rows, ctx) in r["fwd"]:
                    b, us = r["fwd"][(rows, ctx)]
                    med[arm].append(us)
            st = statistics.mean(med["stock"]) if med["stock"] else None
            ls = statistics.mean(med["ls"]) if med["ls"] else None
            ratio = ls / st if st and ls else None
            if b is None and q:
                b = rows // q
            body = cov.get(rows, "?")
            dflag = drafter_flags(draft_run, b, k) if draft_run and b else "?"
            flag = ""
            if ratio is not None and ratio >= 1.0:
                flag = " WORSE"
                worse.append(f"M{rows}/{ctx}")
            gone = [arm for arm in arms if not med[arm]]
            if gone:
                flag += f" MISSING({','.join(gone)})"
                missing.append(f"M{rows}/{ctx}:{','.join(gone)}")
            if "ls" in arms and rows not in cov:
                flag += " UNCOVERED"
                if rows not in uncovered:
                    uncovered.append(rows)
            if prodlimit and rows > prodlimit:
                flag += " XCAP"
            if body.startswith("stock") and rows not in stock_rows:
                stock_rows.append(rows)
            if (s := seams(body)) and (seam := f"M{rows}: {' '.join(s)}") not in seam_rows:
                seam_rows.append(seam)
            fmt = lambda v: "/".join(f"{x:.0f}" for x in v) or "-"
            print(f"   M{rows:<5} {b or 0:>4} {ctx:>7} {fmt(med['stock']):>19} {fmt(med['ls']):>19} "
                  f"{(f'{ratio:.3f}' if ratio else '-'):>6}{flag}  {body} | {dflag}")
            rows_out.append([tag, meta.get("point"), q, meta.get("drafter"), ctx, meta.get("extra"), rows, b,
                             st and round(st, 1), ls and round(ls, 1), ratio and round(ratio, 4), body, dflag])
        print(f"   WORSE (ls >= stock): {worse or 'none'}")
        print(f"   MISSING (requested, not measured): {missing or 'none'}")
        print(f"   UNCOVERED (no ls coverage line): {uncovered or 'none'}")
        if prodlimit and any(r > prodlimit for r, _ in keys):
            print(f"   XCAP: rows above the production limit {prodlimit} ran with raised max-num-seqs / capture cap")
        print(f"   whole-step stock in ls: {stock_rows or 'none'}")
        print(f"   family stock seams in ls: {'; '.join(seam_rows) or 'none'}")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tag", "point", "q", "drafter", "ctx", "extra", "rows", "B", "stock_us", "ls_us",
                        "ls_over_stock", "ls_coverage", "ls_drafter"])
            w.writerows(rows_out)


if __name__ == "__main__":
    main()
