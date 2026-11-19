#!/bin/bash
# P-MLA run 89: h6 (d848184be7) decomposition F16K: no store, +no math, +no gather, +no mma (separate processes, timeouts).
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/run89
mkdir -p $R
for kn in 1 1:0:1 1:1 1:0:0:1 1:1:1; do timeout 300 python3 benchmarks/kernels/benchmark_kimi_k3_prefill_sparse_mla.py --tokens 16384 --qmag 0.15 --loaders "" --kernels h6 --no-base --h4-knobs $kn > $R/k_${kn//:/_}.log 2>&1; echo "$kn rc=$? $(grep -E "^h6" $R/k_${kn//:/_}.log | head -1)" >> $R/summary.txt; done
echo DONE > $R/DONE
