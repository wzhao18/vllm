#!/bin/bash
# P-MLA M11 (softmax order): bitwise + F16K L78K ABBA of the M11 default vs the pre-M11
# order (knob osm=1), physical and logical entries, then ptxas/occupancy facts.
# usage: pmla_m11.sh <run> [extra knob lists]
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/$1
shift
mkdir -p $R
OSM=0:0:0:0:0:0:0:0:0:0:0:0:0:0:1
timeout 1500 python3 benchmarks/kernels/pmla10_check.py $OSM "$@" > $R/ab.log 2>&1
echo "ab rc=$?" > $R/summary.txt
grep -E "equal|ABBA|phys F16K|logical F16K|median|us|Error|error|Traceback" $R/ab.log \
    | grep -v Warning | cut -c1-220 | tail -40 >> $R/summary.txt
echo DONE >> $R/summary.txt
