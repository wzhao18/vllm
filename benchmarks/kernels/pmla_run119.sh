#!/bin/bash
# P-MLA run 119: where does h8 logical lose ~125 us vs h8? lglob (counts from global), no_store / no_math skeletons for h8 and h8L.
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/run119
mkdir -p $R
timeout 600 python3 benchmarks/kernels/benchmark_kimi_k3_prefill_sparse_mla.py --tokens 16384 --qmag 0.15 --loaders "" --kernels h8,h8L --h4-knobs "1,1:0:1" --h8l-knobs "0:0:0:0:0:0:0:0:0:0:0:0:1,1,1:0:1" > $R/f16k.log 2>&1
echo "bench rc=$?" > $R/summary.txt; grep -E "^h8|Error" $R/f16k.log | head -10 | cut -c1-300 >> $R/summary.txt
echo DONE > $R/DONE
