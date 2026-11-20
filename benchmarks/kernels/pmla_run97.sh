#!/bin/bash
# P-MLA run 97: h8 F16K sensitivity to the KV footprint (L2 residency): ctx 72900 / 20000 / 4000, 1 vs 3 layer pools.
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/run97
mkdir -p $R
for cfg in "72900 3" "72900 1" "20000 1" "4000 1"; do set -- $cfg; for kn in 0 1:0:1; do timeout 300 python3 benchmarks/kernels/benchmark_kimi_k3_prefill_sparse_mla.py --tokens 16384 --ctx $1 --layers $2 --qmag 0.15 --loaders "" --kernels h8 --no-base --h4-knobs $kn > $R/c$1_l$2_${kn//:/_}.log 2>&1; echo "ctx $1 layers $2 knobs $kn rc=$? $(grep -E "^h8" $R/c$1_l$2_${kn//:/_}.log | head -1)" >> $R/summary.txt; done; done
echo DONE > $R/DONE
