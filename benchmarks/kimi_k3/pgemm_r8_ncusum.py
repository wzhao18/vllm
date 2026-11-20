#!/usr/bin/env python3
"""Summarize ncu --page raw --csv files (one per ledger row) into one JSON line per kernel.

    python3 pgemm_r8_ncusum.py <dir with <row>.csv> [out.jsonl]

Per kernel: duration, cycles, DRAM / L2 / XBAR->L1 bytes, and the L2-read-per-delivered-byte
factor (lts read bytes / xbar2l1tex read bytes: < 1 means multicast delivery).
"""

import csv
import glob
import json
import os
import re
import sys

WANT = [
    r"^gpu__time_duration\.sum$",
    r"^sm__cycles_active\.avg$",
    r"^sm__cycles_elapsed\.avg$",
    r"^gpc__cycles_elapsed\.max$",
    r"^dram__bytes_read\.sum$",
    r"^dram__bytes_write\.sum$",
    r"^lts__t_sectors_op_read\.sum$",
    r"^lts__t_sectors_op_write\.sum$",
    r"^lts__t_sectors_srcunit_tex_op_read\.sum$",
    r"^lts__t_bytes\.sum$",
    r"^l1tex__m_xbar2l1tex_read_bytes\.sum$",
    r"^l1tex__m_l1tex2xbar_write_bytes\.sum$",
    r"^launch__grid_size$",
    r"^launch__cluster_dim_[xyz]$",
    r"^launch__registers_per_thread$",
    r"^sm__throughput\.avg\.pct_of_peak_sustained_elapsed$",
    r"pipe_tc.*pct_of_peak_sustained_active$",
    r"tensor.*pct_of_peak_sustained_active$",
    r"^smsp__cycles_active\.avg$",
]


def num(v):
    try:
        return float(v.replace(",", ""))
    except (ValueError, AttributeError):
        return v


def main():
    d = sys.argv[1]
    out = open(sys.argv[2], "w") if len(sys.argv) > 2 else None
    for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
        row = os.path.basename(f)[:-4]
        lines = [ln for ln in open(f, errors="replace") if ln.startswith('"')]
        if len(lines) < 3:
            print(json.dumps({"row": row, "error": "no kernels"}))
            continue
        rd = list(csv.reader(lines))
        hdr, units, data = rd[0], rd[1], rd[2:]
        cols = [i for i, h in enumerate(hdr) if any(re.search(w, h) for w in WANT)]
        kcol = hdr.index("Kernel Name") if "Kernel Name" in hdr else None
        for r in data:
            rec = {"row": row, "kernel": r[kcol][:80] if kcol is not None else "?"}
            for i in cols:
                u = units[i]
                v = num(r[i])
                rec[hdr[i] + ("[" + u + "]" if u else "")] = v
            rb = rec.get("lts__t_sectors_op_read.sum[sector]")
            xb = rec.get("l1tex__m_xbar2l1tex_read_bytes.sum[byte]") or rec.get(
                "l1tex__m_xbar2l1tex_read_bytes.sum[Mbyte]"
            )
            if isinstance(rb, float) and isinstance(xb, float) and xb:
                rec["l2rd_per_xbar"] = round(32 * rb / xb, 3)
            print(json.dumps(rec))
            if out:
                out.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
