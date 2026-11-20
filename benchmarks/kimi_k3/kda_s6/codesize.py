"""Code footprint per role from an ncu SASS source CSV (addresses + exec counts).

usage: codesize.py SASS_CSV
Roles split at SETMAXREG marks (as sass_roles.py); stubs = after last EXIT.
Reports static bytes, executed bytes (instructions with exec>0) and the
hot bytes (instructions executed >= 1% of the role's max count).
"""
import csv
import sys

rows = list(csv.reader(open(sys.argv[1])))
hi = next(i for i, r in enumerate(rows) if "Source" in r and "Address" in r)
hdr = rows[hi]
ai = hdr.index("Address")
ei = hdr.index("Instructions Executed")
si = hdr.index("Source")
d = []
for r in rows[hi + 1:]:
    if len(r) <= max(ei, si, ai):
        continue
    try:
        ex = float(r[ei] or 0)
        ad = int(r[ai], 16) if r[ai].startswith("0x") else int(r[ai])
    except ValueError:
        continue
    t = r[si].split()
    op = t[1] if t and t[0].startswith("@") and len(t) > 1 else (t[0] if t else "?")
    d.append((ad, op, ex))
d.sort()
n = len(d)
print("instructions", n, "bytes", n * 16, "addr span", d[-1][0] - d[0][0] + 16)
marks = [i for i, x in enumerate(d) if "SETMAXREG" in x[1]]
exits = [i for i, x in enumerate(d) if x[1] == "EXIT"]
print("SETMAXREG at", marks[:12], "EXIT at", exits[:12])
cuts = sorted(set([0] + marks + ([exits[-1] + 1] if exits else []) + [n]))
tot_ex = sum(x[2] for x in d)
for a, b in zip(cuts[:-1], cuts[1:]):
    seg = d[a:b]
    if not seg:
        continue
    ex = sum(x[2] for x in seg)
    mx = max(x[2] for x in seg)
    nz = sum(1 for x in seg if x[2] > 0)
    hot = sum(1 for x in seg if x[2] >= 0.01 * mx and x[2] > 0)
    hot10 = sum(1 for x in seg if x[2] >= 0.10 * mx and x[2] > 0)
    print(f"[{a:6d},{b:6d}) static {len(seg)*16/1024:7.1f} KiB exec>0 {nz*16/1024:7.1f} KiB "
          f"hot1% {hot*16/1024:7.1f} KiB hot10% {hot10*16/1024:7.1f} KiB "
          f"exec {100*ex/tot_ex:5.1f}%  first {seg[0][1]}")
