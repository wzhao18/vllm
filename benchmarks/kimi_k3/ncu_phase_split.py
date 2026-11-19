"""Split an ncu --print-source sass CSV into phases at the globaltimer reads
(the kernel's stamp() markers) and print executed warp instructions per phase.

usage: ncu_phase_split.py sass.csv [n_ctas]
"""

import csv
import sys

rows = list(csv.reader(open(sys.argv[1])))
hi = next(i for i, r in enumerate(rows) if "Source" in r)
hdr = rows[hi]
src = hdr.index("Source")
ex = hdr.index("Instructions Executed")
nct = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
phase, acc, ops = 0, [0.0], [{}]
for r in rows[hi + 1:]:
    if len(r) <= max(src, ex):
        continue
    s = r[src]
    try:
        e = float(r[ex] or 0)
    except ValueError:
        continue
    if "GLOBALTIMER" in s.upper():
        phase += 1
        acc.append(0.0)
        ops.append({})
    acc[phase] += e
    op = s.split()[0] if s.split() else "?"
    if op.startswith("@"):
        op = s.split()[1]
    op = op.split(".")[0]
    ops[phase][op] = ops[phase].get(op, 0.0) + e
tot = sum(acc)
print(f"total {tot / nct:.0f} warp-instr per CTA; {len(acc)} segments (split at globaltimer reads)")
for i, a in enumerate(acc):
    if a / tot < 0.005:
        continue
    top = sorted(ops[i].items(), key=lambda kv: -kv[1])[:6]
    print(f"  seg {i:2d}: {a / nct:7.0f} ({100 * a / tot:4.1f}%)  " +
          " ".join(f"{k}:{v / nct:.0f}" for k, v in top))
