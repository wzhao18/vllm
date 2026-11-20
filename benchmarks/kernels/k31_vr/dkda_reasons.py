# D-KDA: per-source-line stall reasons from an ncu mixed source CSV (--print-source cuda,sass --csv).
# usage: python3 dkda_reasons.py mix.csv N   (top N lines by stall samples, each with its top reasons)
import collections
import csv
import sys

num = lambda x: float(x.replace(",", "")) if x.strip() not in ("", "-") else 0.0
cur, key, hdr = "", None, None
agg = collections.defaultdict(lambda: collections.Counter())
text = {}
for r in csv.reader(open(sys.argv[1])):
    if not r:
        continue
    if r[0] == "File Path":
        cur = r[1].rsplit("/", 1)[-1]
        continue
    if r[0] == "Line No":
        hdr = r
        cols = [i for i, h in enumerate(r) if h.startswith("stall_") and "Not Issued" not in h]
        si, ii = r.index("Warp Stall Sampling (All Samples)"), r.index("Instructions Executed")
        continue
    if hdr is None or len(r) <= ii:
        continue
    if r[0].strip().isdigit():
        key = (cur, int(r[0]))
        text[key] = r[1].strip()[:50]
        continue
    if key and r[2].startswith("0x"):
        a = agg[key]
        a["_s"] += num(r[si])
        a["_i"] += num(r[ii])
        for i in cols:
            a[hdr[i][6:]] += num(r[i])
S = sum(v["_s"] for v in agg.values())
for k, v in sorted(agg.items(), key=lambda x: -x[1]["_s"])[: int(sys.argv[2])]:
    top = sorted(((c, n) for n, c in v.items() if not n.startswith("_")), reverse=True)[:4]
    print(f"{k[0][:12]:12s} {k[1]:5d} s{100 * v['_s'] / S:5.1f}% | " + " ".join(f"{n}={100 * c / max(v['_s'], 1):.0f}%" for c, n in top) + f" | {text.get(k, '')}")
