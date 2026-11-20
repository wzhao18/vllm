#!/bin/bash
# P-MLA generic microbench runner: pmla_py.sh <run dir name> <bench.py> [args...]
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/$1
shift
mkdir -p $R
timeout 1500 python3 "$@" > $R/ab.log 2>&1
echo "rc=$?" > $R/summary.txt
grep -v -E "^\s*$|Warning|warn" $R/ab.log | tail -n 14 | cut -c1-300 >> $R/summary.txt
echo DONE >> $R/summary.txt
