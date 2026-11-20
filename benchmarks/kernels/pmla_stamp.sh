#!/bin/bash
# P-MLA run 145: h8 sstamp timelines: full, MMA-only (no gather), MMA-only no-math.
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/$1
mkdir -p $R
for kn in 0:0:0:0:1 0:1:0:0:1 0:1:1:0:1; do timeout 400 python3 benchmarks/kernels/benchmark_kimi_k3_prefill_sparse_mla.py --tokens 16384 --qmag 0.15 --loaders "" --kernels h8 --no-base --h4-knobs $kn --json $R/k_${kn//:/_}.json > $R/k_${kn//:/_}.log 2>&1; echo "$kn rc=$?" >> $R/summary.txt; grep -E "^h8" $R/k_${kn//:/_}.log | cut -c1-1200 >> $R/summary.txt; done
echo DONE >> $R/summary.txt
