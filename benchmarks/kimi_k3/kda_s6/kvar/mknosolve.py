"""Timing-only upper-bound probe (P-KDA2-S6 r9): remove the instance solve (phase 6, warp 0) and the off-diagonal
combine (phase 7, warp 0) from an n4-lineage kernel; warps 1-3 keep the Ki restore, warp 0 keeps the restore tail.
Numerics wrong by design. Measures how much of the prep instance latency W (and of the kernel) the warp-0 serial
solve section costs: the upper bound of moving it to a dedicated solver role.

usage: mknosolve.py IN OUT
"""
import sys

src = open(sys.argv[1]).read()
lines = src.split("\n")
p6 = next(i for i, l in enumerate(lines) if "# -- phase 6: solve" in l)
p7 = next(i for i, l in enumerate(lines) if "# -- phase 7: off-diag combine" in l)
i6 = next(i for i in range(p6, p7) if lines[i] == "                    if plw == 0:")
lines[i6] = "                    if plw == 99:"
i7 = next(i for i in range(p7, len(lines)) if lines[i] == "                    if plw == 0:")
i7e = next(i for i in range(i7 + 1, len(lines)) if lines[i] == "                    else:")
lines[i7] = "                    if plw == 99:"
lines[i7e] = "                    elif plw != 0:"
open(sys.argv[2], "w").write("\n".join(lines))
print("mknosolve ok", sys.argv[2], i6 + 1, i7 + 1, i7e + 1)
