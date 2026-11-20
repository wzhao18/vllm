# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-kernel duration stats from step_p prof kernels.json files.
Usage: python3 pidx_kern_stats.py <pattern> <kernels.json>..."""
import json
import re
import sys

pat = re.compile(sys.argv[1], re.I)
for f in sys.argv[2:]:
    d = json.load(open(f))
    rows = d if isinstance(d, list) else d.get("kernels", d.get("events", []))
    if isinstance(rows, dict):
        rows = list(rows.values())
    hits = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        name = str(r.get("name", r.get("kernel", "")))
        if pat.search(name):
            dur = r.get("dur", r.get("dur_us", r.get("us", 0)))
            hits.setdefault(name[60:150] + " calls=" + str(r.get("calls")), []).append(float(dur))
    print(f.split("/")[-1], "rows", len(rows), "keys", list(rows[0].keys())[:8] if rows else None)
    for k, v in hits.items():
        v.sort()
        print(f"  {k}: n={len(v)} min={v[0]:.0f} med={v[len(v)//2]:.0f} max={v[-1]:.0f} sum={sum(v):.0f}")
