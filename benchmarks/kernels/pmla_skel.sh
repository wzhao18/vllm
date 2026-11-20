#!/bin/bash
# P-MLA M15: protocol skeleton of the cp.async h8 form (timing only).
# usage: pmla_skel.sh <run>
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/$1
mkdir -p $R
timeout 1500 python3 benchmarks/kernels/pmla10_check.py --nocheck 0:1:1:1 1:1:1:1 0:1:1:0 1:0:1:1 0:0:1:0 > $R/ab.log 2>&1
echo "ab rc=$?" > $R/summary.txt
grep -E "F16K|Error|error|Traceback|assert" $R/ab.log | grep -v Warning | cut -c1-200 | tail -24 >> $R/summary.txt
echo DONE >> $R/summary.txt
