"""Summarize an ncu source CSV: totals per stall column and the top lines."""
import csv
import sys

rows = list(csv.reader(open(sys.argv[1], errors="replace")))
hi = next(i for i, r in enumerate(rows) if any("Sampling" in c for c in r))
h = rows[hi]
body = [r for r in rows[hi + 1:] if len(r) == len(h)]
def num(x):
    try:
        return float(x.replace(",", ""))
    except ValueError:
        return 0.0
sc = [i for i, c in enumerate(h) if "Sampling" in c or "stall" in c.lower()]
tot = {h[i]: sum(num(r[i]) for r in body) for i in sc}
allc = next(i for i in sc if "All" in h[i])
T = tot[h[allc]] or 1
print("columns:", len(h), "lines:", len(body), "all samples:", int(T))
for k, v in sorted(tot.items(), key=lambda x: -x[1])[:14]:
    print(f"  {v / T * 100:5.1f}%  {k[:70]}")
src = next((i for i, c in enumerate(h) if c.strip() in ("Source", "#Source")), 1)
lno = 0
top = sorted(body, key=lambda r: -num(r[allc]))[:int(sys.argv[2]) if len(sys.argv) > 2 else 28]
for r in top:
    rs = sorted(((num(r[i]), h[i]) for i in sc if i != allc and "Not" not in h[i]), reverse=True)[:3]
    print(f"{num(r[allc]) / T * 100:5.1f}% L{r[0][:6]:>6} {r[src].strip()[:62]:62s} " + " ".join(f"{n[:18]}={v / T * 100:.1f}" for v, n in rs))
