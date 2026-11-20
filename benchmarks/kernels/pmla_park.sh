#!/bin/bash
# P-MLA M17: in-kernel cp.async KV gather with a minimal consumer (park: softmax / QK / Q
# warps idle, PV warp frees each stage once it lands) vs gather-only with the full protocol
# (no math, no MMA, no store) vs full. Timing only.
# usage: pmla_park.sh <run>
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/$1
mkdir -p $R
timeout 1200 python3 benchmarks/kernels/pmla10_check.py --nocheck 1:0:1:1:0:0:0:0:0:0:0:0:0:0:0:1 1:0:1:1 > $R/ab.log 2>&1
echo "ab rc=$?" > $R/summary.txt
grep -E "phys F16K|Error|error|Traceback|assert" $R/ab.log | grep -v Warning | cut -c1-120 | tail -12 >> $R/summary.txt
echo DONE >> $R/summary.txt
