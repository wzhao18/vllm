# SPDX-License-Identifier: Apache-2.0
"""Summarise tcgen05 MMA issue code in an nvdisasm -c listing: for each
UTC*MMA instruction, the number of SASS instructions since the previous
UTC* instruction and the opcode histogram of the gap (dev diagnostic for the
persistent prefill KDA MMA warp)."""
import collections
import re
import sys

ins_re = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]+)")
lines = []
for line in open(sys.argv[1]):
    m = ins_re.search(line)
    if m:
        lines.append((int(m.group(1), 16), m.group(3), line.strip()))
print("instructions", len(lines))
utc = [i for i, (_, op, _) in enumerate(lines) if op.startswith("UTC")]
print("UTC* instructions", len(utc))
ops = collections.Counter(lines[i][1].split(".")[0] for i in utc)
print("UTC opcodes", dict(ops))
# Clusters of MMAs: runs of UTC* instructions with gaps < 200.
prev = None
cluster = []
clusters = []
for i in utc:
    if prev is not None and i - prev > 200:
        clusters.append(cluster)
        cluster = []
    cluster.append(i)
    prev = i
if cluster:
    clusters.append(cluster)
for c in clusters:
    span = c[-1] - c[0] + 1
    hist = collections.Counter(lines[j][1].split(".")[0] for j in range(c[0], c[-1] + 1))
    top = ", ".join(f"{k}:{v}" for k, v in hist.most_common(12))
    print(f"cluster @{lines[c[0]][0]:#x} n_utc={len(c)} span={span} :: {top}")
if len(sys.argv) > 2:
    lo, hi = (int(x, 16) for x in sys.argv[2].split(":"))
    for addr, op, text in lines:
        if lo <= addr <= hi:
            print(text[:150])
