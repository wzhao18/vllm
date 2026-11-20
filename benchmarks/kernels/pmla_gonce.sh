#!/bin/bash
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/$1
mkdir -p $R
timeout 900 python3 benchmarks/kernels/pmla_gonce.py > $R/ab.log 2>&1; echo "ab rc=$?" > $R/summary.txt
grep -E "bitwise|full|gonce|Error|error" $R/ab.log | cut -c1-200 | tail -6 >> $R/summary.txt
timeout 600 python3 benchmarks/kernels/pmla_wstamp.py --tokens 16384 --qmag 0.15 --loaders "" --kernels h8 --no-base --h4-knobs 0:2:0:0:1:0:1 --json $R/w.json > $R/st.log 2>&1; echo "st rc=$?" >> $R/summary.txt
grep -E "^u[0-9]|Error" $R/st.log | cut -c1-260 >> $R/summary.txt
grep -E "^h8 0:2" $R/st.log | grep -v sstamp | cut -c1-120 >> $R/summary.txt; echo DONE >> $R/summary.txt
