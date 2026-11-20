#!/bin/bash
# P-MLA run 118: h8 logical v3 hang hunt (small cases with timeouts) + F16K h8 / h8L.
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/run118
mkdir -p $R
python3 benchmarks/kernels/pmla_dbg_logical.py > $R/dbg.log 2>&1
cp $R/dbg.log $R/summary.txt
timeout 400 python3 benchmarks/kernels/benchmark_kimi_k3_prefill_sparse_mla.py --tokens 16384 --qmag 0.15 --loaders "" --kernels h8,h8L --h4-knobs "" > $R/f16k.log 2>&1
echo "bench rc=$?" >> $R/summary.txt; grep -E "^h8|^remap|Error" $R/f16k.log | head -6 | cut -c1-300 >> $R/summary.txt
echo DONE > $R/DONE
