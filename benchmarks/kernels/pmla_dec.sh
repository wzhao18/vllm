#!/bin/bash
# P-MLA decomposition of the production h8 form (cp.async gather, pre-M11 softmax order =
# osm 1): full / no gather (MMA + softmax) / gather only (no math, no MMA) / neither
# gather nor MMA, F16K L78K physical + logical ABBA (timing only, no bitwise).
# usage: pmla_dec.sh <run>
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/$1
mkdir -p $R
O=0:0:0:0:0:0:0:0:0:0:0:0:0:0:1
timeout 1500 python3 benchmarks/kernels/pmla10_check.py --nocheck $O 0:1:0:0:0:0:0:0:0:0:0:0:0:0:1 \
    0:0:1:1:0:0:0:0:0:0:0:0:0:0:1 0:1:0:1:0:0:0:0:0:0:0:0:0:0:1 > $R/ab.log 2>&1
echo "ab rc=$?" > $R/summary.txt
grep -E "F16K|Error|error|Traceback" $R/ab.log | grep -v Warning | cut -c1-200 | tail -20 >> $R/summary.txt
echo DONE >> $R/summary.txt
