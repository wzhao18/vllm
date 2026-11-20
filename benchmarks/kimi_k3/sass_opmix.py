"""SASS opcode mix of an ncu --page source --csv --print-source sass export:
share of warp instructions executed and of stall samples per opcode.
Usage: python3 sass_opmix.py <csv> [n]"""
import csv, sys, collections
rows = list(csv.reader(open(sys.argv[1])))
hi = next(i for i, r in enumerate(rows) if "Source" in r)
hdr = rows[hi]
ci = hdr.index("Instructions Executed")
si = hdr.index("Source")
wi = hdr.index("Warp Stall Sampling (All Samples)")
tot = collections.Counter()
st = collections.Counter()
T = 0
S = 0
for r in rows[hi + 1:]:
    if len(r) != len(hdr):
        continue
    src = r[si].strip()
    toks = src.split()
    if not toks:
        continue
    op = toks[1] if toks[0].startswith("@") and len(toks) > 1 else toks[0]
    op = op.split(".")[0]
    try:
        n = float(r[ci] or 0)
        s = float(r[wi] or 0)
    except ValueError:
        continue
    tot[op] += n
    st[op] += s
    T += n
    S += s
print("total", T, "samples", S)
for op, n in tot.most_common(int(sys.argv[2]) if len(sys.argv) > 2 else 25):
    print(f"{op:10s} {100*n/T:5.1f}% instr {100*st[op]/max(S,1):5.1f}% stall")
