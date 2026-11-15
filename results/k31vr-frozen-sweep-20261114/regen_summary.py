#!/usr/bin/env python3
"""regen_summary.py: rebuild the chosen-set line and the VR200-vs-GB300 table in README.md from points.json."""
import json, re
P = json.load(open('points.json'))['points']; G = json.load(open('gb300_sa_k31.json'))['points']
f, best = [], -1
for p in sorted(G, key=lambda p: -p['p90_intvty']):
    if p['tput_per_gpu'] > best: f.append(p); best = p['tput_per_gpu']
f = sorted(f, key=lambda p: p['p90_intvty'])
def gb(x):
    for a, b in zip(f, f[1:]):
        if a['p90_intvty'] <= x <= b['p90_intvty']:
            t = (x - a['p90_intvty']) / (b['p90_intvty'] - a['p90_intvty']); return a['tput_per_gpu'] + t * (b['tput_per_gpu'] - a['tput_per_gpu'])
ch = sorted([p for p in P if p['chosen']], key=lambda p: -p['p90_intvty'])
rows = ["| point | TPS/GPU | P90 intvty | TTFT p90 (s) | GB300 front at same intvty | VR200 / GB300 per GPU | per MW (x 2.12/3.3) |", "|---|---|---|---|---|---|---|"]
for p in ch:
    g = gb(p['p90_intvty'])
    rows.append(f"| {p['label']} | {p['tput_per_gpu']:,.0f} | {p['p90_intvty']:.0f} | {p['p90_ttft_s']:.2f} | " + (f"{g:,.0f} | {p['tput_per_gpu']/g:.2f}x | {p['tput_per_gpu']/g*2.12/3.3:.2f}x |" if g else "beyond GB300 range (max x 450) | - | - |"))
s = open('README.md').read()
s = re.sub(r"\| point \| TPS/GPU \| P90 intvty \| TTFT p90 \(s\) \| GB300.*?\n\n", "\n".join(rows) + "\n\n", s, flags=re.S)
names = ", ".join(p['label'] for p in ch)
s = re.sub(r"- \*\*Final chosen set[^\n]*", f"- **Final chosen set ({len(ch)} points, frozen code b68d67e7e, K3.1 golden AL, watermark 0.05; user selection 2026-11-15 + xPyD search):** {names}. All other measured runs are valid but unselected (grey); reasons in the points table.", s)
open('README.md', 'w').write(s); print(names); print("\n".join(rows))
