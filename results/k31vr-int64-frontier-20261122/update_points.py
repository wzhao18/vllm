#!/usr/bin/env python3
"""update_points.py: rebuild points.json from runs/*/{metrics.json, metrics_win10_60.json, meta.json, gsm8k/score.json} and
print the README metrics table (both windows). Then run plot_pareto4.py for the PNGs."""
import glob, json, os
H = os.path.dirname(os.path.abspath(__file__))
pts = []
for r in sorted(glob.glob(f"{H}/runs/*/metrics.json")):
    d = os.path.dirname(r); m = json.load(open(r)); w = json.load(open(f"{d}/metrics_win10_60.json")); meta = json.load(open(f"{d}/meta.json"))
    s = json.load(open(f"{d}/gsm8k/score.json")) if os.path.exists(f"{d}/gsm8k/score.json") else {}
    p = dict(label=meta["label"], run=os.path.basename(d), topology=meta["topology"], chosen=meta.get("chosen", True), note=meta.get("note", ""),
             build=meta.get("build"), gsm8k_strict=s.get("strict"), gsm8k_real_al=s.get("real_al"), k=s.get("k"))
    p.update({k: v for k, v in m.items() if k not in ("label", "out")})
    p["win10_60"] = w["end"]; p["win10_60_start"] = w["start"]
    pts.append(p)
pts.sort(key=lambda p: p["tput_per_gpu"])
json.dump(dict(points=pts), open(f"{H}/points.json", "w"), indent=1)
print("| point | window | n ok | TPS/GPU | P90 intvty | P90 E2E-norm | P90 E2E (s) | P90 TTFT (s) | P50 TTFT (s) | client hit |")
print("|---|---|---|---|---|---|---|---|---|---|")
for p in pts:
    w = p["win10_60"]
    print(f"| {p['label']} | full profiling ({p['span_min']} min) | {p['ok']:,} | {p['tput_per_gpu']:,.0f} | {p['p90_intvty']:.1f} | {p['p90_e2e_norm_intvty']:.1f} | {p['p90_e2el_s']:.1f} | {p['p90_ttft_s']:.2f} | {p['p50_ttft_s']:.2f} | |")
    print(f"| {p['label']} | minutes 10-60 (by end) | {w['n']:,} | {w['tput_per_gpu']:,.0f} | {w['p90_intvty']:.1f} | {w['p90_e2e_norm_intvty']:.1f} | {w['p90_e2el_s']:.1f} | {w['p90_ttft_s']:.2f} | {w['p50_ttft_s']:.2f} | {w['hit_pct']:.2f}% |")
