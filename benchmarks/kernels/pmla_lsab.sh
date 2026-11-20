#!/bin/bash
# P-MLA E1 (S-chain lever check): usage: pmla_lsab.sh <run dir name>
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/$1
mkdir -p $R
cd /k31ov 2>/dev/null || true
timeout 1500 python3 benchmarks/kernels/pmla_lsab.py > $R/ab.log 2>&1
echo "ab rc=$?" > $R/summary.txt
grep -E "bitwise|qkh0|Error|error|Traceback" $R/ab.log | head -12 | cut -c1-300 >> $R/summary.txt
tail -3 $R/ab.log | cut -c1-300 >> $R/summary.txt
echo DONE >> $R/summary.txt
