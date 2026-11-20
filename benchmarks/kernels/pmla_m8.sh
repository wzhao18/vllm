#!/bin/bash
# P-MLA M8: LDG row-producer warps sweep (nldg:lq) vs int12 h8, bitwise + ABBA F16K L78K, then the h8 tests.
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/$1; mkdir -p $R
shift
timeout 1500 python3 benchmarks/kernels/pmla_m8_ab.py benchmarks/kernels/pmla_patches/sparse_mla_h8_prev.py "$@" 2>&1 | grep -vE "Warning|warn" > $R/ab.log
echo "ab rc=$?" > $R/summary.txt; grep -E "T=|Error|error" $R/ab.log | head -30 | cut -c1-250 >> $R/summary.txt
timeout 600 python3 -m pytest -q -x --noconftest tests/models/kimi_k3/test_specialized_prefill_sparse_mla.py -k "sol_matches or h8 or two_short" > $R/tests.log 2>&1
echo "tests rc=$?" >> $R/summary.txt; tail -3 $R/tests.log | cut -c1-200 >> $R/summary.txt; echo DONE >> $R/summary.txt; cat $R/summary.txt
