"""Per-source-line executed instructions and stall samples from
`ncu -i r.ncu-repz --page source --csv --print-source cuda,sass` (needs a
-lineinfo build): argv[1] = that CSV, argv[2] = file name filter."""
import collections
import csv
import sys

pat = sys.argv[2] if len(sys.argv) > 2 else ""
acc = collections.defaultdict(lambda: [0.0, 0.0, ""])
fname, line, hdr = "", None, None
for r in csv.reader(open(sys.argv[1])):
    if not r:
        continue
    if r[0] == "File Path":
        fname = r[1]
        continue
    if r[0] == "Line No":
        hdr = r
        ex = hdr.index("Instructions Executed")
        st = hdr.index("Warp Stall Sampling (All Samples)")
        continue
    if hdr is None or len(r) <= ex:
        continue
    if r[0].strip():
        line = (fname, r[0].strip())
        acc[line][2] = r[1].strip()
    try:
        e, s = float(r[ex] or 0), float(r[st] or 0)
    except ValueError:
        continue
    if line is not None:
        acc[line][0] += e
        acc[line][1] += s
te = sum(v[0] for v in acc.values()) or 1
ts = sum(v[1] for v in acc.values()) or 1
print(f"executed {te:.3e}, stall samples {ts:.0f}")
for k, (e, s, src) in sorted(acc.items(), key=lambda kv: -kv[1][0])[:40]:
    if pat in k[0] or e / te > 0.02:
        print(f"  {k[0].split('/')[-1][:14]}:{k[1]:>4} ex {100 * e / te:4.1f}% st {100 * s / ts:4.1f}%  {src[:72]}")
