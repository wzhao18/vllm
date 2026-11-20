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
                    chosen=bool(meta.get("chosen", False)), note=meta.get("note", ""), gsm8k_strict=g.get("strict"), gsm8k_real_al=g.get("real_al"), k=g.get("k"),
                    **{k: m[k] for k in ("gpus", "ok", "err", "span_min", "tput_per_gpu", "p90_intvty", "p90_e2e_norm_intvty", "p90_e2el_s", "p90_ttft_s", "p50_ttft_s")}))
pts.sort(key=lambda p: -p["p90_intvty"])
json.dump({"points": pts}, open(f"{H}/points.json", "w"), indent=1)
old = {q["label"]: q for q in json.load(open(f"{H}/../k31vr-frozen-sweep-20261114/points.json"))["points"]}
def d(p, k):
    q = old.get(p["label"]); return f"{100 * (p[k] / q[k] - 1):+.1f}%" if q and q.get(k) else "-"
cmp = ["| point | TPS/GPU (preview -> release) | P90 intvty (preview -> release) | P90 TTFT s (preview -> release) | gsm8k strict | real AL (k) |",
       "|---|---|---|---|---|---|"]
for p in pts:
    q = old.get(p["label"], {})
    cmp.append(f"| {p['label']} | {q.get('tput_per_gpu', 0):,.0f} -> {p['tput_per_gpu']:,.0f} ({d(p, 'tput_per_gpu')}) | "
               f"{q.get('p90_intvty', 0):.1f} -> {p['p90_intvty']:.1f} ({d(p, 'p90_intvty')}) | {q.get('p90_ttft_s', 0):.2f} -> {p['p90_ttft_s']:.2f} | "
               f"{q.get('gsm8k_strict')} -> {p['gsm8k_strict'] if p['gsm8k_strict'] is not None else 'pending'} | {p['gsm8k_real_al'] or '-'} ({p['k'] or '-'}) |")
rows = ["| point | topology (GPUs) | AgentX run | TPS/GPU | P90 intvty | P90 E2E-norm | P90 E2E (s) | P90 TTFT (s) | gsm8k strict | chosen | note |",
        "|---|---|---|---|---|---|---|---|---|---|---|"]
for p in pts:
    rows.append(f"| {p['label']} | {p['topology']} ({p['gpus']}) | {p['run']} | {p['tput_per_gpu']:,.0f} | {p['p90_intvty']:.1f} | {p['p90_e2e_norm_intvty']:.1f} | "
                f"{p['p90_e2el_s']:.1f} | {p['p90_ttft_s']:.2f} | {p['gsm8k_strict'] if p['gsm8k_strict'] is not None else 'pending'} | {'**yes**' if p['chosen'] else 'no'} | {p['note']} |")
rd = re.sub(r"<!-- CMP -->.*<!-- /CMP -->", "<!-- CMP -->\n" + "\n".join(cmp) + "\n<!-- /CMP -->", open(f"{H}/README.md").read(), flags=re.S)
rd = re.sub(r"<!-- POINTS -->.*<!-- /POINTS -->", "<!-- POINTS -->\n" + "\n".join(rows) + "\n<!-- /POINTS -->", rd, flags=re.S)
open(f"{H}/README.md", "w").write(rd)
subprocess.check_call([sys.executable, f"{H}/plot_pareto4.py"])
print(f"{len(pts)} points")
