#!/bin/bash
# P-MLA: cost of the consumer-side proxy fence of the cp.async gather. cpa 3 (production:
# per-lane noinc arrive, QK/PV warps fence.proxy.async after KVFULL / OREADY) vs 4 (no
# consumer fence, timing only) vs 5 (lagged producer-side fences), full and gather-only.
# usage: pmla_fence.sh <run>
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/$1
mkdir -p $R
Z=0:0:0:0:0:0:0:0:0:0
G=0:0:1:1:0:0:0:0:0:0
timeout 1500 python3 benchmarks/kernels/pmla10_check.py --nocheck $Z:4 $Z:5 $G:3 $G:4 $G:5 > $R/ab.log 2>&1
echo "ab rc=$?" > $R/summary.txt
grep -E "F16K|Error|error|Traceback|assert" $R/ab.log | grep -v Warning | cut -c1-200 | tail -20 >> $R/summary.txt
echo DONE >> $R/summary.txt
