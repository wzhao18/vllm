"""Top SASS lines of an ncu --print-source sass CSV by stall samples, plus
the per-opcode totals of executed instructions and stall samples."""
import collections
import csv
import sys

rows = list(csv.reader(open(sys.argv[1])))
hi = next(i for i, r in enumerate(rows) if "Source" in r)
hdr = rows[hi]


def col(*names):
    for n in names:
        for i, h in enumerate(hdr):
            if h == n:
                return i
    return None


ci = col("Warp Stall Sampling (All Samples)", "Warp Stall Sampling (All Cycles)")
si = col("Source")
ei = col("Instructions Executed")
data = []
for r in rows[hi + 1:]:
    if len(r) <= max(ci, si):
        continue
    try:
        data.append((float(r[ci] or 0), r[si].strip(), float(r[ei] or 0) if ei is not None else 0.0))
    except ValueError:
        pass
tot = sum(d[0] for d in data) or 1
tex = sum(d[2] for d in data) or 1
print(f"total stall samples {tot:.0f}, executed warp instr {tex:.0f}")
by = collections.defaultdict(lambda: [0.0, 0.0])
for s, src, e in data:
    op = src.split()[0] if src else "?"
    if op.startswith("@"):
        op = src.split()[1]
    op = op.split(".")[0]
    by[op][0] += s
    by[op][1] += e
print("by opcode (stall %, exec %):")
for op, (s, e) in sorted(by.items(), key=lambda x: -x[1][0])[:22]:
    print(f"  {op:10s} {100 * s / tot:5.1f}% {100 * e / tex:5.1f}%")
print("top lines:")
for d in sorted(data, reverse=True)[:25]:
    print(f"{100 * d[0] / tot:5.1f}%  exec {d[2]:>9.0f}  {d[1][:90]}")
