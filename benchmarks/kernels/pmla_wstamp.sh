#!/bin/bash
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/$1
mkdir -p $R
timeout 600 python3 benchmarks/kernels/pmla_wstamp.py --tokens 16384 --qmag 0.15 --loaders "" --kernels h8 --no-base --h4-knobs 0:0:0:0:1:0:1 --json $R/w.json > $R/ab.log 2>&1
echo "rc=$?" > $R/summary.txt
grep -E "^u[0-9]|^sstamp|Error|error" $R/ab.log | cut -c1-260 >> $R/summary.txt
echo DONE >> $R/summary.txt
