#!/bin/bash
# P-MLA M14: P proxy fence only in the softmax warps that wrote P (default) vs every softmax
# warp (fpad = 1, pre-M14): bitwise + F16K L78K ABBA full and gather-only (no math, no MMA:
# no softmax fences at all in the default).
# usage: pmla_fpad.sh <run>
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/$1
mkdir -p $R
Z=0:0:0:0:0:0:0:0:0:0:0:0:0:0
timeout 1500 python3 benchmarks/kernels/pmla10_check.py $Z:1 0:0:1:1:0:0:0:0:0:0:0:0:0:0:0 0:0:1:1:0:0:0:0:0:0:0:0:0:0:1 > $R/ab.log 2>&1
echo "ab rc=$?" > $R/summary.txt
grep -E "equal|F16K|Error|error|Traceback|assert" $R/ab.log | grep -v Warning | cut -c1-200 | tail -24 >> $R/summary.txt
echo DONE >> $R/summary.txt
