#!/bin/bash
# P-MLA run 79: h6 sparse per-head O rescale (ballot masks): tests, F16K base + stamps.
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/run79
mkdir -p $R
timeout 400 python3 -m pytest --noconftest -q -x tests/models/kimi_k3/test_specialized_prefill_sparse_mla.py -k "stock or h6" > $R/test.log 2>&1
echo "tests: $(tail -1 $R/test.log)" > $R/summary.txt
timeout 900 python3 benchmarks/kernels/benchmark_kimi_k3_prefill_sparse_mla.py --tokens 16384 --qmag 0.15 --loaders "" --kernels h6 --no-base --h4-knobs 0,0:0:0:0:1 --json $R/f16k.json > $R/f16k.log 2>&1
echo "bench rc=$?" >> $R/summary.txt; grep "^h6" $R/f16k.log >> $R/summary.txt
echo DONE > $R/DONE
