#!/bin/bash
# P-MLA M11 variants: SASS facts + bitwise + F16K L78K ABBA of osm 0 (default) / 1 / 2 / 3.
# usage: pmla_m11b.sh <run>
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/$1
mkdir -p $R
Z=0:0:0:0:0:0:0:0:0:0:0:0:0:0
timeout 900 python3 benchmarks/kernels/pmla_sass.py $R/sass $Z:0 $Z:1 $Z:2 $Z:3 > $R/sass.log 2>&1
echo "sass rc=$?" > $R/summary.txt
grep -E "sass_instr|failed|no cubin" $R/sass.log | cut -c1-260 >> $R/summary.txt
timeout 1500 python3 benchmarks/kernels/pmla10_check.py $Z:1 $Z:2 $Z:3 > $R/ab.log 2>&1
echo "ab rc=$?" >> $R/summary.txt
grep -E "equal|F16K|Error|error|Traceback" $R/ab.log | grep -v Warning | cut -c1-200 | tail -40 >> $R/summary.txt
echo DONE >> $R/summary.txt
