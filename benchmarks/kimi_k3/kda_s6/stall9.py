"""Per-range stall table + per-line stall sites from an ncu cuda,sass source export (P-KDA2-S6 r9).

usage: python3 stall9.py <cs.csv> <ranges file> <source-file suffix> [RANGE REASON [N]]
  ranges file: "<first> <last> <name>" per line (line numbers of the source file).
Prints per range: executed warp-instr and stall samples by reason (absolute, so two kernels can be
subtracted), then optionally the top-N lines of RANGE by stall REASON (e.g. long_sb) with source text.
"""
import collections
import csv
import sys

REASONS = ["stall_barrier", "stall_long_sb", "stall_short_sb", "stall_mio", "stall_wait",
           "stall_no_inst", "stall_math", "stall_sleep", "stall_not_selected", "stall_selected",
           "stall_branch_resolving", "stall_dispatch", "stall_lg", "stall_membar", "stall_misc",
           "stall_drain", "stall_tex"]


def num(x):
    try:
        return float(x.replace(",", ""))
    except ValueError:
        return 0.0


path, rng, suf = sys.argv[1], sys.argv[2], sys.argv[3]
want = sys.argv[4] if len(sys.argv) > 4 else None
wreason = ("stall_" + sys.argv[5]) if len(sys.argv) > 5 else None
topn = int(sys.argv[6]) if len(sys.argv) > 6 else 15
ranges = []
for ln in open(rng):
    p = ln.split()
    if len(p) >= 3 and p[0].isdigit():
        ranges.append((int(p[0]), int(p[1]), " ".join(p[2:])))
ie = collections.Counter()
st = collections.defaultdict(collections.Counter)
per_line = collections.Counter()
text = {}
hdr = None
cur = -1
fname = ""
for r in csv.reader(open(path, errors="replace")):
    if not r:
        continue
    if r[0] in ("File Path", "File Name"):
        fname = r[1] if len(r) > 1 else ""
        continue
    if r[0] == "Line No":
        hdr = {n: i for i, n in reversed(list(enumerate(r)))}
        i_addr = r.index("Address")
        i_src = hdr.get("Source", 1)
        continue
    if hdr is None:
        continue
    if r[0].strip():
        try:
            cur = int(r[0]) if fname.endswith(suf) else -1
        except ValueError:
            cur = -1
        if cur >= 0 and cur not in text and len(r) > i_src:
            text[cur] = r[i_src].strip()[:70]
    if len(r) <= i_addr or not r[i_addr].strip():
        continue
    name = next((nm for a, b, nm in ranges if a <= cur <= b), "other")
    ie[name] += num(r[hdr["Instructions Executed"]])
    for k in REASONS:
        if k in hdr:
            st[name][k] += num(r[hdr[k]])
    if want and name == want and wreason in hdr:
        per_line[cur] += num(r[hdr[wreason]])
tot_s = sum(sum(c.values()) for c in st.values())
tot_i = sum(ie.values())
print(f"total warp-instr {tot_i / 1e6:.2f}M  stall samples {tot_s:.0f}")
cols = [k for k in REASONS if any(st[n][k] for n in st)]
print("range".ljust(20) + "instr_M".rjust(8) + "samples".rjust(9) + "".join(k[6:][:8].rjust(9) for k in cols))
for name in [n for _, _, n in ranges] + ["other"]:
    if name not in ie:
        continue
    print(name[:20].ljust(20) + f"{ie[name] / 1e6:8.2f}" + f"{sum(st[name].values()):9.0f}"
          + "".join(f"{st[name][k]:9.0f}" for k in cols))
if want:
    print(f"top lines of {want} by {wreason}:")
    for ln, v in per_line.most_common(topn):
        print(f"  L{ln:5d} {v:8.0f}  {text.get(ln, '')}")
