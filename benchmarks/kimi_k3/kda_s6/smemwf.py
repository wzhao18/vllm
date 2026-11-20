"""Shared-memory wavefronts per role and per opcode from an ncu SASS source CSV.

usage: smemwf.py SASS_CSV [CHUNKS] [TOPN]
"""
import collections
import csv
import sys

rows = list(csv.reader(open(sys.argv[1])))
CH = float(sys.argv[2]) if len(sys.argv) > 2 else 49248.0
TOP = int(sys.argv[3]) if len(sys.argv) > 3 else 25
hi = next(i for i, r in enumerate(rows) if "Source" in r and "Address" in r)
hdr = rows[hi]
ai, ei, si = hdr.index("Address"), hdr.index("Instructions Executed"), hdr.index("Source")
wi = hdr.index("L1 Wavefronts Shared")
wid = hdr.index("L1 Wavefronts Shared Ideal")
wx = hdr.index("L1 Wavefronts Shared Excessive")


def f(x):
    try:
        return float(x or 0)
    except ValueError:
        return 0.0


d = []
for r in rows[hi + 1:]:
    if len(r) <= max(ei, si, ai, wi, wid, wx):
        continue
    try:
        ad = int(r[ai], 16)
    except ValueError:
        continue
    t = r[si].split()
    op = t[1] if t and t[0].startswith("@") and len(t) > 1 else (t[0] if t else "?")
    d.append((ad, op, f(r[ei]), f(r[wi]), f(r[wid]), f(r[wx]), " ".join(t)))
d.sort()
marks = [i for i, x in enumerate(d) if "SETMAXREG" in x[1]]
exits = [i for i, x in enumerate(d) if x[1] == "EXIT"]
cuts = sorted(set([0] + marks + ([exits[-1] + 1] if exits else []) + [len(d)]))
names = ["pre", "compute", "epilogue", "wg2", "prep", "stubs", "x", "y"]
T = sum(x[3] for x in d)
print(f"total shared wavefronts {T/1e6:.2f}M = {T/CH:.0f}/chunk; ideal {sum(x[4] for x in d)/CH:.0f}; excessive {sum(x[5] for x in d)/CH:.0f}")
role = {}
for k, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
    seg = d[a:b]
    w = sum(x[3] for x in seg)
    print(f"{names[k]:9s} wf/chunk {w/CH:7.0f} ideal {sum(x[4] for x in seg)/CH:7.0f} excess {sum(x[5] for x in seg)/CH:6.0f}")
    for x in seg:
        role[x[0]] = names[k]
by = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
for x in d:
    if x[3] <= 0:
        continue
    key = (role[x[0]], x[1])
    by[key][0] += x[3]
    by[key][1] += x[5]
    by[key][2] += x[2]
print("-- per role/opcode (wf/chunk, excess/chunk, inst/chunk)")
for k, v in sorted(by.items(), key=lambda kv: -kv[1][0])[:TOP]:
    print(f"{k[0]:9s} {k[1]:28s} wf {v[0]/CH:7.1f} exc {v[1]/CH:6.1f} inst {v[2]/CH:7.1f}")
print("-- top instructions by excessive wavefronts")
for x in sorted(d, key=lambda x: -x[5])[:12]:
    print(f"{role[x[0]]:9s} 0x{x[0]:x} {x[6][:60]:60s} wf {x[3]/CH:6.1f} exc {x[5]/CH:6.1f} inst {x[2]/CH:5.1f}")
