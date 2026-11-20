# SPDX-License-Identifier: Apache-2.0
"""Summarize an ncu export of the persistent KDA kernel (P-KDA2 dev).

usage: python3 ncu_agg.py <cs.csv> <ranges file> [top]
<cs.csv> = ncu -i R --page source --csv --print-source cuda,sass (needs
CUTE_DSL_LINEINFO=1 at compile).  The ranges file holds lines "start end name"
over kernel.py line numbers; the table sums executed warp instructions and
stall samples per range, lists the top lines and the SASS opcode mix per range.
"""
import collections
import csv
import re
import sys


def num(x):
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return 0.0


def main():
    path, rng = sys.argv[1], sys.argv[2]
    top = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    ranges = []
    for ln in open(rng):
        p = ln.split()
        if len(p) >= 3 and p[0].isdigit():
            ranges.append((int(p[0]), int(p[1]), " ".join(p[2:])))
    per_line = collections.Counter()
    stall_line = collections.Counter()
    ops_rng = collections.defaultdict(collections.Counter)
    src_text = {}
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
            hdr = r
            i_addr = r.index("Address")
            i_sass = i_addr + 1
            i_ie = r.index("Instructions Executed")
            i_st = r.index("Warp Stall Sampling (All Samples)")
            continue
        if hdr is None:
            continue
        if r[0].strip():
            try:
                cur = int(r[0]) if fname.endswith("kernel.py") else -1
            except ValueError:
                cur = -1
            if cur >= 0:
                src_text.setdefault(cur, r[1].strip()[:72])
        if len(r) > i_ie and r[i_addr].strip():
            ie = num(r[i_ie])
            st = num(r[i_st])
            per_line[cur] += ie
            stall_line[cur] += st
            m = re.match(r"\s*(@!?U?P\w+\s+)?([A-Z0-9_]+)(\.[A-Z0-9_.]+)?", r[i_sass])
            op = (m.group(2) + (m.group(3) or "")[:10]) if m else "?"
            name = "other"
            for a, b, nm in ranges:
                if a <= cur <= b:
                    name = nm
                    break
            ops_rng[name][op] += ie
    tot = sum(per_line.values())
    tst = sum(stall_line.values())
    print(f"total_warp_instr={tot:.0f} stall_samples={tst:.0f} (line -1 = non-kernel.py {per_line[-1]:.0f})")
    for a, b, name in ranges + [(-1, -1, "other")]:
        s = sum(v for k, v in per_line.items() if (a <= k <= b) or (name == "other" and k == -1))
        ss = sum(v for k, v in stall_line.items() if (a <= k <= b) or (name == "other" and k == -1))
        mix = ", ".join(f"{o} {100 * v / max(s, 1):.0f}%" for o, v in ops_rng[name].most_common(7))
        print(f"R {name:20s} {s:11.0f} {100 * s / max(tot, 1):5.1f}% st {100 * ss / max(tst, 1):5.1f}% | {mix}")
    print("TOP lines:")
    for li, v in per_line.most_common(top):
        print(f" L{li:5d} {v:10.0f} {100 * v / max(tot, 1):4.1f}% st {100 * stall_line[li] / max(tst, 1):4.1f}% | {src_text.get(li, '')[:60]}")


if __name__ == "__main__":
    main()
