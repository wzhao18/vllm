"""Per-role warp stall samples from an ncu SASS source CSV; where no_inst lands.

usage: stalls.py SASS_CSV ROLE_FOR_DETAIL
"""
import csv
import sys

rows = list(csv.reader(open(sys.argv[1])))
want = sys.argv[2] if len(sys.argv) > 2 else "compute"
hi = next(i for i, r in enumerate(rows) if "Source" in r and "Address" in r)
hdr = rows[hi]
ai, si = hdr.index("Address"), hdr.index("Source")
st = [(i, h) for i, h in enumerate(hdr) if h.startswith("stall_") and "Not Issued" not in h]


def f(x):
    try:
        return float(x or 0)
    except ValueError:
        return 0.0


d = []
for r in rows[hi + 1:]:
    if len(r) <= max(ai, si, st[-1][0]):
        continue
    try:
        ad = int(r[ai], 16)
    except ValueError:
        continue
    t = r[si].split()
    op = t[1] if t and t[0].startswith("@") and len(t) > 1 else (t[0] if t else "?")
    d.append((ad, op, {h: f(r[i]) for i, h in st}, " ".join(t)))
d.sort()
marks = [i for i, x in enumerate(d) if "SETMAXREG" in x[1]]
exits = [i for i, x in enumerate(d) if x[1] == "EXIT"]
cuts = sorted(set([0] + marks + ([exits[-1] + 1] if exits else []) + [len(d)]))
names = ["pre", "compute", "epilogue", "wg2", "prep", "stubs", "x", "y"]
role = [None] * len(d)
for k, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
    for j in range(a, b):
        role[j] = names[k]
keys = [h for _, h in st]
tot = {n: {h: 0.0 for h in keys} for n in names}
for j, x in enumerate(d):
    for h in keys:
        tot[role[j]][h] += x[2][h]
for n in names:
    s = sum(tot[n].values())
    if s <= 0:
        continue
    top = sorted(tot[n].items(), key=lambda kv: -kv[1])[:7]
    print(f"{n:9s} samples {s:9.0f} " + " ".join(f"{h[6:]}={100*v/s:.0f}%" for h, v in top))
# no_inst detail for the chosen role: top instructions + previous op
idx = [j for j in range(len(d)) if role[j] == want]
ni = sorted(idx, key=lambda j: -d[j][2]["stall_no_inst"])[:14]
sw = sum(d[j][2]["stall_no_inst"] for j in idx)
print(f"-- {want}: no_inst samples {sw:.0f}; top instructions (prev op):")
for j in ni:
    print(f"  {d[j][2]['stall_no_inst']:6.0f} 0x{d[j][0]:x} {d[j][3][:50]:50s} prev {d[j-1][3][:34]}")
