"""Per-role / poll / hot-block warp-instruction counts from an ncu SASS source CSV.

usage: sass_roles.py SASS_CSV [CHUNKS]   (CHUNKS = BT32 chunk-heads, f16k 49248)
Roles are split at the USETMAXREG instructions (compute alloc, epilogue alloc,
WG2 dealloc 24, prep dealloc/alloc); the tail after the last EXIT holds the
out-of-line mbarrier retry stubs (SYNCS / NANOSLEEP / BRA)."""
import collections
import csv
import sys

rows = list(csv.reader(open(sys.argv[1])))
CH = float(sys.argv[2]) if len(sys.argv) > 2 else 49248.0
hi = next(i for i, r in enumerate(rows) if "Source" in r and "Address" in r)
hdr = rows[hi]
ei = hdr.index("Instructions Executed")
si = hdr.index("Source")
d = []
for r in rows[hi + 1:]:
    if len(r) <= max(ei, si):
        continue
    try:
        ex = float(r[ei] or 0)
    except ValueError:
        continue
    t = r[si].split()
    op = t[1] if t and t[0].startswith("@") and len(t) > 1 else (t[0] if t else "?")
    d.append((op, ex, " ".join(t)))
tot = sum(x[1] for x in d)
marks = [i for i, x in enumerate(d) if "SETMAXREG" in x[0]]
exits = [i for i, x in enumerate(d) if x[0] == "EXIT"]
bounds = [0] + marks + [exits[-1] + 1 if exits else len(d), len(d)]
names = ["setup", "compute", "epilogue", "wg2", "prep", "polls/tail"]
print(f"instr {len(d)} executed/chunk {tot / CH:.0f}; SETMAXREG at {marks}; last EXIT {exits[-1:]}")
for k in range(len(bounds) - 1):
    a, b = bounds[k], bounds[k + 1]
    s = sum(x[1] for x in d[a:b])
    nm = names[k] if k < len(names) else f"r{k}"
    ops = collections.Counter()
    for x in d[a:b]:
        ops[x[0].split(".")[0]] += x[1]
    top = " ".join(f"{o}:{v / CH:.0f}" for o, v in ops.most_common(8))
    print(f"{nm:10s} [{a:5d},{b:5d}) {s / CH:6.0f}/chunk {100 * s / tot:5.1f}% | {top}")
poll = sum(x[1] for x in d if x[0].startswith(("SYNCS", "USYNCS", "NANOSLEEP")))
print(f"SYNCS+NANOSLEEP everywhere: {poll / CH:.0f}/chunk ({100 * poll / tot:.1f}%)")
