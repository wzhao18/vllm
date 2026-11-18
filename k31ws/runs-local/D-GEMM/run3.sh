#!/bin/bash
# D-GEMM run 3: SplitKGemmG quick correctness, then the full sweep if it compiles.
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/D-GEMM
cd $R
python3 $R/g_sweep.py $R/g_quick.json quick 2>&1 | grep -v -E "Warning|deprecated|jit-cache|flashinfer-cubin"
if grep -q '"err": "' $R/g_quick.json; then echo "QUICK FAILED"; exit 1; fi
python3 $R/g_sweep.py $R/g_full.json 2>&1 | grep -v -E "Warning|deprecated|jit-cache|flashinfer-cubin"
