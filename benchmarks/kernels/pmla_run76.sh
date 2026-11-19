#!/bin/bash
# P-MLA run 76: mixed gather4 + LDG row gather probe (lead s1), F16K, h6 stage shape.
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/run76
mkdir -p $R
timeout 900 python3 benchmarks/kernels/pmla_gather_mix.py --tokens 16384 --json $R/mix.json > $R/mix.log 2>&1
echo "rc=$?" >> $R/mix.log
echo DONE > $R/DONE
