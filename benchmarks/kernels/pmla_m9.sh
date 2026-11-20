#!/bin/bash
# P-MLA M9: SMEM slot-index preload vs int12 h8: bitwise + ABBA (T=300, F16K L78K), then the h8 tests.
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/$1; mkdir -p $R
timeout 900 python3 benchmarks/kernels/pmla_m7_ab.py benchmarks/kernels/pmla_patches/sparse_mla_h8_int12.py 0.15 1.0 2>&1 | grep -vE "Warning|warn" > $R/ab.log
echo "ab rc=$?" > $R/summary.txt; grep -E "T=|Error|error" $R/ab.log | head -20 | cut -c1-250 >> $R/summary.txt
timeout 600 python3 -m pytest -q -x --noconftest tests/models/kimi_k3/test_specialized_prefill_sparse_mla.py > $R/tests.log 2>&1
echo "tests rc=$?" >> $R/summary.txt; tail -3 $R/tests.log | cut -c1-200 >> $R/summary.txt; echo DONE >> $R/summary.txt
