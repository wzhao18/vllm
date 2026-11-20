#!/bin/bash
# usage: pmla_split.sh <run>
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/$1
mkdir -p $R
timeout 1500 python3 benchmarks/kernels/pmla_split.py > $R/ab.log 2>&1
echo "ab rc=$?" > $R/summary.txt
grep -E "equal|F16K|Error|error|Traceback|assert" $R/ab.log | grep -v Warning | cut -c1-200 | tail -20 >> $R/summary.txt
echo DONE >> $R/summary.txt
