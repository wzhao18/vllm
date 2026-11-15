#!/usr/bin/env python3
"""update_points.py: merge runs/<label>-a<job>/metrics.json (+ gsm8k/score.json, meta.json) into points.json, regenerate the
README points table (between the POINTS markers) and the two plots. Chosen flags live in meta.json (decided per the README rule)."""
import glob, json, os, re, subprocess, sys
H = os.path.dirname(os.path.abspath(__file__))
pts = []
for r in sorted(glob.glob(f"{H}/runs/*/metrics.json")):
    d = os.path.dirname(r); m = json.load(open(r)); meta = {}
    if os.path.exists(f"{d}/meta.json"): meta = json.load(open(f"{d}/meta.json"))
    g = json.load(open(f"{d}/gsm8k/score.json")) if os.path.exists(f"{d}/gsm8k/score.json") else {}
    pts.append(dict(label=meta.get("label", m["label"]), run=os.path.basename(d), topology=meta.get("topology", ""),
                    chosen=bool(meta.get("chosen", False)), note=meta.get("note", ""), gsm8k_strict=g.get("strict"),
                    **{k: m[k] for k in ("gpus", "ok", "err", "span_min", "tput_per_gpu", "p90_intvty", "p90_e2e_norm_intvty", "p90_e2el_s", "p90_ttft_s", "p50_ttft_s")}))
pts.sort(key=lambda p: -p["p90_intvty"])
json.dump({"points": pts}, open(f"{H}/points.json", "w"), indent=1)
rows = ["| point | topology (GPUs) | AgentX run | TPS/GPU | P90 intvty | P90 E2E-norm | P90 E2E (s) | P90 TTFT (s) | gsm8k strict | chosen | note |",
        "|---|---|---|---|---|---|---|---|---|---|---|"]
for p in pts:
    rows.append(f"| {p['label']} | {p['topology']} ({p['gpus']}) | {p['run']} | {p['tput_per_gpu']:,.0f} | {p['p90_intvty']:.1f} | {p['p90_e2e_norm_intvty']:.1f} | "
                f"{p['p90_e2el_s']:.1f} | {p['p90_ttft_s']:.2f} | {p['gsm8k_strict'] if p['gsm8k_strict'] is not None else 'pending'} | {'**yes**' if p['chosen'] else 'no'} | {p['note']} |")
rd = open(f"{H}/README.md").read()
rd = re.sub(r"<!-- POINTS -->.*<!-- /POINTS -->", "<!-- POINTS -->\n" + "\n".join(rows) + "\n<!-- /POINTS -->", rd, flags=re.S)
open(f"{H}/README.md", "w").write(rd)
subprocess.check_call([sys.executable, f"{H}/plot_pareto4.py"])
print(f"{len(pts)} points")
