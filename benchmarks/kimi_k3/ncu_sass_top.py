# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Top SASS instructions by warp-stall samples from an `ncu --page source --csv
--print-source sass` export, with the opcode mix of executed instructions.

  python3 ncu_sass_top.py <csv> [top]
"""

import csv
import sys
from collections import Counter


def main():
    path, top = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 25
    rows = list(csv.reader(open(path)))
    hi = next(i for i, r in enumerate(rows) if r and r[0] == "Address")
    h = rows[hi]
    col = {n: h.index(n) for n in h}
    data = [r for r in rows[hi + 1:] if len(r) == len(h)]

    def num(r, n):
        try:
            return float(r[col[n]] or 0)
        except ValueError:
            return 0.0

    stall = "Warp Stall Sampling (All Samples)"
    tot = sum(num(r, stall) for r in data) or 1
    exe = "Instructions Executed"
    ops, ops_stall = Counter(), Counter()
    for r in data:
        op = r[col["Source"]].split()[0] if r[col["Source"]].split() else "?"
        if op.startswith("@"):
            op = r[col["Source"]].split()[1]
        op = op.split(".")[0]
        ops[op] += num(r, exe)
        ops_stall[op] += num(r, stall)
    te = sum(ops.values()) or 1
    print(f"total executed warp-instr {te:.3g}, stall samples {tot:.3g}")
    print("opcode mix (executed %, stall %):",
          ", ".join(f"{o} {100 * c / te:.1f}/{100 * ops_stall[o] / tot:.1f}"
                    for o, c in ops.most_common(14)))
    data.sort(key=lambda r: -num(r, stall))
    for r in data[:top]:
        print(f"{100 * num(r, stall) / tot:5.1f}% {r[col['Address']]:>6} "
              f"exe {num(r, exe):9.0f}  {r[col['Source']][:90]}")


if __name__ == "__main__":
    main()
