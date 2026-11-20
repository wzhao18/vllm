# SPDX-License-Identifier: Apache-2.0
"""Per-range stall-reason table from an ncu source export (cuda,sass CSV).

usage: python3 stall_rng.py <cs.csv> <ranges file>
Counts only SASS rows (rows with an Address) under kernel.py source lines, so
the CUDA-line summary rows are not double counted.  Prints, per range, the
executed warp instructions and the share of each stall reason (all samples).
"""
import collections
import csv
import sys

REASONS = ["stall_barrier", "stall_long_sb", "stall_short_sb", "stall_mio",
           "stall_wait", "stall_no_inst", "stall_math", "stall_sleep",
           "stall_not_selected", "stall_selected", "stall_branch_resolving",
           "stall_dispatch", "stall_lg", "stall_membar", "stall_misc",
           "stall_drain"]


def num(x):
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return 0.0


def main():
    path, rng = sys.argv[1], sys.argv[2]
    ranges = []
    for ln in open(rng):
        p = ln.split()
        if len(p) >= 3 and p[0].isdigit():
            ranges.append((int(p[0]), int(p[1]), " ".join(p[2:])))
    ie = collections.Counter()
    st = collections.defaultdict(collections.Counter)
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
        if len(r) <= i_addr or not r[i_addr].strip():
            continue
        name = "other"
        for a, b, nm in ranges:
            if a <= cur <= b:
                name = nm
                break
        ie[name] += num(r[hdr["Instructions Executed"]])
        for k in REASONS:
            if k in hdr:
                st[name][k] += num(r[hdr[k]])
    if len(sys.argv) > 3 and sys.argv[3] == "--op":
        # executed count of SASS ops with a prefix, per kernel.py line
        pre = sys.argv[4]
        per = collections.Counter()
        cur = -1
        fname = ""
        for r in csv.reader(open(path, errors="replace")):
            if not r or r[0] in ("File Path", "File Name", "Line No"):
                if r and r[0] in ("File Path", "File Name"):
                    fname = r[1] if len(r) > 1 else ""
                continue
            if r[0].strip():
                try:
                    cur = int(r[0]) if fname.endswith("kernel.py") else -1
                except ValueError:
                    cur = -1
            if len(r) <= i_addr + 1 or not r[i_addr].strip():
                continue
            sass = r[i_addr + 1].strip()
            if sass.split(" ")[-1:] and any(
                tok.startswith(pre) for tok in sass.replace("@", " ").split()[:2]
            ):
                per[cur] += num(r[hdr["Instructions Executed"]])
        tot_p = sum(per.values())
        print(f"{pre}: total executed {tot_p:.0f}")
        for ln, v in per.most_common(16):
            nm = next((n for a, b, n in ranges if a <= ln <= b), "other")
            print(f"L{ln:5d} {v:12.0f} {nm}")
        return
    if len(sys.argv) > 4:
        # top lines of one range by one stall reason: <range name> <reason>
        want, reason = sys.argv[3], "stall_" + sys.argv[4]
        per = collections.Counter()
        cur = -1
        fname = ""
        src = {}
        for r in csv.reader(open(path, errors="replace")):
            if not r or r[0] in ("File Path", "File Name", "Line No"):
                if r and r[0] in ("File Path", "File Name"):
                    fname = r[1] if len(r) > 1 else ""
                continue
            if r[0].strip():
                try:
                    cur = int(r[0]) if fname.endswith("kernel.py") else -1
                except ValueError:
                    cur = -1
                if cur >= 0 and not r[i_addr].strip():
                    src.setdefault(cur, r[1].strip()[:70])
            if len(r) <= i_addr or not r[i_addr].strip():
                continue
            if any(a <= cur <= b and nm == want for a, b, nm in ranges):
                per[(cur, r[i_addr + 1].strip()[:28])] += num(r[hdr[reason]])
        tot_r = sum(per.values()) or 1.0
        for (ln, sass), v in per.most_common(14):
            print(f"L{ln:5d} {100 * v / tot_r:5.1f}% {sass:28s} | {src.get(ln, '')}")
        return
    tot = sum(ie.values()) or 1.0
    print("range                 instr%  samples/instr  " +
          " ".join(k.replace("stall_", "")[:8] for k in REASONS[:10]))
    for name in [nm for _, _, nm in ranges] + ["other"]:
        s = sum(st[name].values())
        if ie[name] == 0 and s == 0:
            continue
        sh = " ".join(f"{100 * st[name][k] / max(s, 1):8.1f}" for k in REASONS[:10])
        print(f"{name:20s} {100 * ie[name] / tot:6.1f} {s / max(ie[name], 1) * 1e3:10.3f}  {sh}")


if __name__ == "__main__":
    main()
