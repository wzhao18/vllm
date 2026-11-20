"""Variant with fewer prep instances (STAGES = N, warps past 12 + 4N idle).

usage: mkstages.py SRC DST N
Used to test whether fewer concurrent prep code streams (instruction-fetch
pressure) speed up the full kernel despite lower prep throughput.
"""
import sys

src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
s = open(src).read()
reps = [
    ("STAGES = 5\n", f"STAGES = {n}\n", 1),
    ("    if warp >= 12:\n", "    if (warp >= 12) & (warp < 12 + 4 * STAGES):\n", 1),
]
for a, b, c in reps:
    if s.count(a) != c:
        sys.exit(f"pattern count {s.count(a)} != {c}: {a!r}")
    s = s.replace(a, b)
open(dst, "w").write(s)
print("wrote", dst)
