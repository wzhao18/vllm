"""Dynamic SASS opcode mix per kernel.py line range from an ncu source export (cuda,sass CSV).

  python3 opmix_rng.py <cs.csv> <ranges file> <chunks> [range-name ...]
Prints executed warp instructions per chunk by opcode (top 18) for each named range (all ranges if none named).
"""
import collections
import csv
import sys


def num(x):
    try:
        return float(x.replace(",", ""))
    except ValueError:
        return 0.0


path, rng, chunks = sys.argv[1], sys.argv[2], float(sys.argv[3])
want = set(sys.argv[4:])
ranges = []
for ln in open(rng):
    p = ln.split()
    if len(p) >= 3 and p[0].isdigit():
        ranges.append((int(p[0]), int(p[1]), " ".join(p[2:])))
mix = collections.defaultdict(collections.Counter)
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
        continue
    if hdr is None:
        continue
    if r[0].strip():
        try:
            cur = int(r[0]) if fname.endswith("kernel.py") else -1
        except ValueError:
            cur = -1
    if len(r) <= i_addr + 1 or not r[i_addr].strip():
        continue
    name = next((nm for a, b, nm in ranges if a <= cur <= b), "other")
    toks = r[i_addr + 1].replace("@", " @").split()
    toks = [t for t in toks if not t.startswith("@")]
    op = toks[0].split(".")[0] if toks else "?"
    mix[name][op] += num(r[hdr["Instructions Executed"]])
for name, c in mix.items():
    if want and name not in want:
        continue
    tot = sum(c.values())
    print(f"== {name}: {tot / chunks:.0f} warp-instr/chunk")
    print("   " + "  ".join(f"{op} {v / chunks:.0f}" for op, v in c.most_common(18)))
