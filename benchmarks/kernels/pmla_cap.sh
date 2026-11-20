#!/bin/bash
# P-MLA: cp.async in-flight stage cap CP_CAP 3/4/5/6 (NS 6, ring below 227 KiB): full and
# gather-only (no math, no MMA, no store) F16K L78K, one process per cap (timing only).
# usage: pmla_cap.sh <run>
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/$1
mkdir -p $R
: > $R/summary.txt
for c in 4 5 6 3; do
  timeout 900 python3 benchmarks/kernels/pmla10_check.py --nocheck --cap $c 1:0:1:1 > $R/cap$c.log 2>&1
  echo "cap $c rc=$?" >> $R/summary.txt
  grep -E "CP_CAP|F16K|Error|Traceback" $R/cap$c.log | grep -v Warning | cut -c1-120 >> $R/summary.txt
done
echo DONE >> $R/summary.txt
