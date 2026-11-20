#!/bin/bash
# P-MLA run 106: h8 gather-warp count slope (gw knob: only the first gw of 9 gather warps gather).
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/run106
mkdir -p $R
for kn in 0:0:0:0:0:0:0:0:0:0:0:9 0:0:0:0:0:0:0:0:0:0:0:8 0:0:0:0:0:0:0:0:0:0:0:6 1:0:1:0:0:0:0:0:0:0:0:9 1:0:1:0:0:0:0:0:0:0:0:8 1:0:1:0:0:0:0:0:0:0:0:6 1:0:1:0:0:0:0:0:0:0:0:4; do
  timeout 300 python3 benchmarks/kernels/benchmark_kimi_k3_prefill_sparse_mla.py --tokens 16384 --qmag 0.15 --loaders "" --kernels h8 --no-base --h4-knobs $kn > $R/k_${kn//:/_}.log 2>&1
  echo "$kn rc=$? $(grep -E "^h8" $R/k_${kn//:/_}.log | head -1)" >> $R/summary.txt
done
echo DONE > $R/DONE
