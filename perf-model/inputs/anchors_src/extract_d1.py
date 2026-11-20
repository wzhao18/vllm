"""Extract D1 (TP16+DCP16 sdec) decode step anchors from e2e engine logs.
step = aiperf mean ITL x engine-logged mean AL; B = mean Running per decode engine over the steady window
(30-90% of busy log intervals); step_ms_logs = Running x AL / generation throughput (busy-assumption, biased at B~1)."""
import glob, gzip, json, os, re, statistics, sys
ROOT = sys.argv[2] if len(sys.argv) > 2 else "results"   # run from the repo root: python3 k31vr/solmodel/inputs/anchors_src/extract_d1.py k31vr/solmodel/inputs/anchors_d1.json
SWEEPS = ["k31vr-frozen-sweep-20261114", "k31vr-release-draft-sweep-20261117"]
re_eng = re.compile(r"\[loggers.py:\d+\] Engine (\d+): .*Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs")
re_spec = re.compile(r"Mean acceptance length: ([\d.]+).*Per-position acceptance rate: ([\d., ]+?), Avg")
out = []
for sw in SWEEPS:
    pts = json.load(open(os.path.join(ROOT, sw, "points.json")))["points"]
    for p in pts:
        if "D TP16+DCP16" not in p["topology"]:
            continue
        logs = sorted(glob.glob(os.path.join(ROOT, sw, "runs", p["run"], "agentx", "logs", "*_decode_w*.out*")))
        engines = []
        for lf in logs:
            op = gzip.open if lf.endswith(".gz") else open
            iv = []
            pending = None
            with op(lf, "rt", errors="replace") as f:
                for line in f:
                    m = re_eng.search(line)
                    if m:
                        pending = (float(m.group(2)), int(m.group(3)))
                        continue
                    m = re_spec.search(line)
                    if m and pending:
                        al = float(m.group(1)); k = len(m.group(2).split(","))
                        iv.append((pending[0], pending[1], al, k))
                        pending = None
            if len(iv) < 20:
                continue
            busy = [x for x in iv if x[1] > 0 and x[0] > 0]
            n = len(busy)
            win = busy[int(0.3 * n): int(0.9 * n)]          # skip primers / warmup and the tail
            if not win:
                continue
            steps = sum(g * 10.0 / (r * al) for g, r, al, k in win)
            step_ms = len(win) * 10.0 / steps * 1e3
            engines.append(dict(log=os.path.basename(lf), B=statistics.mean(x[1] for x in win),
                                AL=sum(x[0] for x in win) / sum(x[0] / x[2] for x in win),
                                k=statistics.mode(x[3] for x in win), step_ms=step_ms, n=len(win)))
        if not engines:
            continue
        B = statistics.mean(e["B"] for e in engines)
        aj = glob.glob(os.path.join(ROOT, sw, "runs", p["run"], "agentx", "logs", "agentic", "conc_*",
                                    "aiperf_artifacts", "profile_export_aiperf.json"))
        itl = json.load(open(aj[0]))["inter_token_latency"]["avg"] if aj else None
        out.append(dict(sweep=sw[:-9], label=p["label"], run=p["run"], engines=len(engines), k=engines[0]["k"],
                        B=round(B, 2), r=round(B / 16, 4), AL=round(statistics.mean(e["AL"] for e in engines), 3),
                        itl_ms=round(itl, 3) if itl else None,
                        step_ms=round(itl * statistics.mean(e["AL"] for e in engines), 2) if itl else None,
                        step_ms_logs=round(statistics.mean(e["step_ms"] for e in engines), 2)))
json.dump(out, open(sys.argv[1], "w"), indent=1)
for o in out:
    print(o)
