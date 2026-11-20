#!/bin/bash
# P-MLA M7 (real change): 4 dedicated RoPE LDG warps (64 instead of 80 gather4 ops per stage).
# A = int12 h8 (pmla_patches/sparse_mla_h8_prev.py), B = this tree. Pre-registered: bitwise
# equal; F16K L78K microbench >= 65 us faster (predicted ~ -120 us: G 949 -> ~780 clk/stage).
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/run134
mkdir -p $R
timeout 1200 python3 benchmarks/kernels/pmla_m7_ab.py benchmarks/kernels/pmla_patches/sparse_mla_h8_prev.py 0.15 1.0 > $R/ab.log 2>&1
echo "ab rc=$?" > $R/summary.txt
grep -E "qmag|Error|error|spill" $R/ab.log | head -12 | cut -c1-250 >> $R/summary.txt
timeout 600 python3 -m pytest -q -x --noconftest tests/models/kimi_k3/test_specialized_prefill_sparse_mla.py -k "sol_matches or h8 or two_short" > $R/tests.log 2>&1
echo "tests rc=$?" >> $R/summary.txt
tail -3 $R/tests.log | cut -c1-200 >> $R/summary.txt
echo DONE >> $R/summary.txt
