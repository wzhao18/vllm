#!/bin/bash
# P-MLA run 74: h6 (loader unroll=1) decomposition at F16K: full, no store, no softmax math, no MMA, no gather,
# no math+MMA, and the full shared-memory stamp timeline (CTA 0, subtiles 16-23).
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/run74
mkdir -p $R
timeout 1200 python3 benchmarks/kernels/benchmark_kimi_k3_prefill_sparse_mla.py --tokens 16384 --qmag 0.15 --loaders "" --kernels h6 --no-base --h4-knobs 0,1,1:0:1,1:0:0:1,1:1,1:0:1:1,1:1:1:1,0:0:0:0:1 --json $R/f16k.json > $R/f16k.log 2>&1
echo "bench rc=$?" >> $R/f16k.log
echo DONE > $R/DONE
