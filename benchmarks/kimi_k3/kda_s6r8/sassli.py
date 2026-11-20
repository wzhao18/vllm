"""Per-source-line map of local-memory / generic / global-store SASS ops in an nvdisasm -g(i) dump.

  python3 sassli.py <file.sass> [ops]   (ops default: LDL,STL,STG,LD,ST)
"""
import collections
import re
import sys

path = sys.argv[1]
ops = (sys.argv[2] if len(sys.argv) > 2 else "LDL,STL,STG,LD,ST").split(",")
loc_re = re.compile(r'//## File "([^"]+)", line (\d+)(.*)')
inl_re = re.compile(r'inlined at "([^"]+)", line (\d+)')
ins_re = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_]*)((?:\.[A-Z0-9_]+)*)\s*([^;]*);")
cur = "?"
n_ins = 0
by_op = collections.defaultdict(collections.Counter)
example = {}
for ln in open(path, errors="replace"):
    m = loc_re.search(ln)
    if m:
        chain = [f"{m.group(1).rsplit('/', 1)[-1]}:{m.group(2)}"]
        for f, l in inl_re.findall(m.group(3)):
            chain.append(f"{f.rsplit('/', 1)[-1]}:{l}")
        cur = "<".join(chain[:3])
        continue
    m = ins_re.search(ln)
    if not m:
        continue
    n_ins += 1
    op = m.group(2)
    if op in ops:
        by_op[op][cur] += 1
        example.setdefault((op, cur), (m.group(2) + m.group(3) + " " + m.group(4).strip())[:60])
print(f"instructions {n_ins}")
for op in ops:
    c = by_op.get(op)
    if not c:
        print(f"{op}: 0")
        continue
    print(f"{op}: {sum(c.values())} static")
    for k, v in c.most_common(14):
        print(f"  {v:4d}  {k:45s} {example[(op, k)]}")
