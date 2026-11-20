"""Stall reasons of a SASS index range of an ncu source CSV.

usage: sass_stall.py SASS_CSV LO HI [N]  -> totals by reason, then the N
instructions with most stall samples (index, reason split, text)."""
import collections
import csv
import sys

rows = list(csv.reader(open(sys.argv[1])))
hi = next(i for i, r in enumerate(rows) if "Source" in r and "Address" in r)
hdr = rows[hi]
ci = hdr.index("Warp Stall Sampling (All Samples)")
si = hdr.index("Source")
rc = [(i, h) for i, h in enumerate(hdr) if h.startswith("stall_") and "Not Is" not in h]
d = [r for r in rows[hi + 1:] if len(r) > max(ci, si)]
lo, h2 = int(sys.argv[2]), int(sys.argv[3])
N = int(sys.argv[4]) if len(sys.argv) > 4 else 12
seg = d[lo:h2]
tot_all = sum(float(r[ci] or 0) for r in d) or 1
tot = sum(float(r[ci] or 0) for r in seg) or 1
agg = collections.Counter()
for r in seg:
    for i, h in rc:
        try:
            agg[h[6:]] += float(r[i] or 0)
        except ValueError:
            pass
print(f"[{lo},{h2}) {100 * tot / tot_all:.1f}% of all samples: " + " ".join(
    f"{k}={100 * v / tot:.0f}%" for k, v in agg.most_common(8)))
for k, r in sorted(enumerate(seg), key=lambda kr: -float(kr[1][ci] or 0))[:N]:
    rs = sorted(((h[6:], float(r[i] or 0)) for i, h in rc), key=lambda x: -x[1])[:2]
    print(f"{lo + k:5d} {100 * float(r[ci] or 0) / tot:4.1f}% " + " ".join(
        f"{a}={b:.0f}" for a, b in rs) + " | " + " ".join(r[si].split())[:56])
