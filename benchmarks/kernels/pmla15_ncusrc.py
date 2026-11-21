# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P-MLA-2CTA round 5: aggregate an ncu report's source page (CuTe DSL line info) by
source line and by method of sparse_mla_h8.py: instructions executed and warp stall
samples. Usage: python3 pmla15_ncusrc.py <ncu binary> <report.ncu-repz> <sparse_mla_h8.py>"""
import csv
import io
import os
import subprocess
import sys


def num(x):
    try:
        return float(x.replace(",", ""))
    except ValueError:
        return 0.0


def main():
    ncu, rep, src = sys.argv[1:4]
    out = subprocess.run([ncu, "-i", rep, "--page", "source", "--csv", "--print-source",
                          "cuda,sass"], capture_output=True, text=True).stdout
    lines = open(src).read().split("\n")
    meth = {}
    cur = "?"
    for i, ln in enumerate(lines, 1):
        s = ln.strip()
        if s.startswith("def "):
            cur = s[4:s.index("(")]
        meth[i] = cur
    agg = {}
    fname = "?"
    for r in csv.reader(io.StringIO(out)):
        if not r:
            continue
        if r[0] == "File Path":
            fname = os.path.basename(r[1])
            continue
        if not r[0].isdigit() or len(r) < 8 or r[2] != "-":
            continue
        k = (fname, int(r[0]))
        a = agg.setdefault(k, [0.0, 0.0])
        a[0] += num(r[7])
        a[1] += num(r[4])
    ti = sum(v[0] for v in agg.values()) or 1.0
    ts = sum(v[1] for v in agg.values()) or 1.0
    bym = {}
    for (f, ln), (ins, stl) in agg.items():
        key = meth.get(ln, "?") if f == os.path.basename(src) else f
        m = bym.setdefault(key, [0.0, 0.0])
        m[0] += ins
        m[1] += stl
    print(f"total warp instructions executed {ti:.4g}, stall samples {ts:.4g}")
    print("by method / file (instr %, stall %):")
    for m, (ins, stl) in sorted(bym.items(), key=lambda x: -x[1][0])[:16]:
        print(f"  {m:26s} {100 * ins / ti:5.1f}% {100 * stl / ts:5.1f}%")
    h8 = os.path.basename(src)
    print("top h8 lines by instructions:")
    for (f, ln), (ins, stl) in sorted(agg.items(), key=lambda x: -x[1][0])[:14]:
        t = lines[ln - 1].strip()[:64] if f == h8 and ln <= len(lines) else f
        print(f"  {f[:14]}:{ln:5d} {100 * ins / ti:5.1f}% stall {100 * stl / ts:5.1f}% {t}")
    print("top lines by stall samples:")
    for (f, ln), (ins, stl) in sorted(agg.items(), key=lambda x: -x[1][1])[:14]:
        t = lines[ln - 1].strip()[:64] if f == h8 and ln <= len(lines) else f
        print(f"  {f[:14]}:{ln:5d} stall {100 * stl / ts:5.1f}% instr {100 * ins / ti:5.1f}% {t}")


if __name__ == "__main__":
    main()
