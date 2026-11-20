# D-KDA: shared-memory wavefronts (total / excessive = bank conflicts) per source line, ncu mixed CSV.
# usage: python3 dkda_smemlines.py mix.csv N
import collections
import csv
import sys

num = lambda x: float(x.replace(",", "")) if x.strip() not in ("", "-") else 0.0
cur, key, hdr = "", None, None
agg = collections.defaultdict(lambda: [0.0, 0.0, 0.0])
text = {}
for r in csv.reader(open(sys.argv[1])):
    if not r:
        continue
    if r[0] == "File Path":
        cur = r[1].rsplit("/", 1)[-1]
        continue
    if r[0] == "Line No":
        hdr = r
        wi, xi, ii = r.index("L1 Wavefronts Shared"), r.index("L1 Wavefronts Shared Excessive"), r.index("Instructions Executed")
        continue
    if hdr is None or len(r) <= wi:
        continue
    if r[0].strip().isdigit():
        key = (cur, int(r[0]))
        text[key] = r[1].strip()[:50]
        continue
    if key and r[2].startswith("0x"):
        a = agg[key]
        a[0] += num(r[wi])
        a[1] += num(r[xi])
        a[2] += num(r[ii])
W = sum(v[0] for v in agg.values())
X = sum(v[1] for v in agg.values())
print(f"total wavefronts {W:.0f} excessive {X:.0f}")
for k, v in sorted(agg.items(), key=lambda x: -x[1][0])[: int(sys.argv[2])]:
    print(f"{k[0][:12]:12s} {k[1]:5d} wf {100 * v[0] / W:5.1f}% exc {v[1] / max(v[0], 1) * 100:4.0f}% instr {v[2]:.0f} | {text.get(k, '')}")
