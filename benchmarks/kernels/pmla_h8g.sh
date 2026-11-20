#!/bin/bash
# P-MLA M10 (real change): h8g = h8 with an LDGSTS row producer (2 or 1 warps, one 576 B
# row per lane, 36 x 16 B cp.async.cg, per-lane cp.async.mbarrier.arrive.noinc) instead of
# 80 TMA gather4 ops per stage from 9 warps. Pre-registered: bitwise equal to h8;
# gather-only arm 1210 -> <= 800 us; full F16K L78K <= A - 100 us (pred A - 150).
# usage: pmla_h8g.sh <run dir name> [nlw ...]
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/$1
shift
mkdir -p $R
timeout 1500 python3 benchmarks/kernels/pmla_h8g_ab.py "$@" > $R/ab.log 2>&1
echo "ab rc=$?" > $R/summary.txt
grep -E "nlw=|Error|error|Traceback|spill" $R/ab.log | head -12 | cut -c1-300 >> $R/summary.txt
tail -4 $R/ab.log | cut -c1-300 >> $R/summary.txt
echo DONE >> $R/summary.txt
