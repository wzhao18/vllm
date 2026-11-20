#!/bin/bash
# P-MLA run 98: h8 TMA op-rate test: skip 0-3 of the 5 gather4 ops per quad (timing only), full and no store+no math.
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/run98
mkdir -p $R
for kn in 0:0:0:0:0:0:0:0:0:0 0:0:0:0:0:0:0:0:0:1 0:0:0:0:0:0:0:0:0:2 0:0:0:0:0:0:0:0:0:3 1:0:1:0:0:0:0:0:0:0 1:0:1:0:0:0:0:0:0:1 1:0:1:0:0:0:0:0:0:2 1:0:1:0:0:0:0:0:0:3 1:0:1:0:0:0:0:0:5:2; do
  timeout 300 python3 benchmarks/kernels/benchmark_kimi_k3_prefill_sparse_mla.py --tokens 16384 --qmag 0.15 --loaders "" --kernels h8 --no-base --h4-knobs $kn > $R/k_${kn//:/_}.log 2>&1
  echo "$kn rc=$? $(grep -E "^h8" $R/k_${kn//:/_}.log | head -1)" >> $R/summary.txt
done
echo DONE > $R/DONE
