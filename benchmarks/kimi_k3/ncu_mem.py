"""Top SASS lines of an ncu sass CSV by L1 shared wavefronts and global sectors."""
import csv
import sys

rows = list(csv.reader(open(sys.argv[1])))
hi = next(i for i, r in enumerate(rows) if "Source" in r)
h = rows[hi]
ix = {n: h.index(n) for n in ("Source", "L1 Wavefronts Shared", "L1 Wavefronts Shared Excessive",
                              "L2 Theoretical Sectors Global", "L1 Tag Requests Global",
                              "Instructions Executed")}


def f(r, n):
    try:
        return float(r[ix[n]] or 0)
    except (ValueError, IndexError):
        return 0.0


data = [r for r in rows[hi + 1:] if len(r) > max(ix.values())]
for key in ("L1 Wavefronts Shared", "L2 Theoretical Sectors Global", "L1 Tag Requests Global"):
    tot = sum(f(r, key) for r in data) or 1
    print(f"== {key}: total {tot:.3e}")
    for r in sorted(data, key=lambda r: -f(r, key))[:12]:
        print(f"  {100 * f(r, key) / tot:5.1f}%  exc {f(r, 'L1 Wavefronts Shared Excessive'):9.0f} "
              f"exec {f(r, 'Instructions Executed'):9.0f}  {r[ix['Source']].strip()[:70]}")
