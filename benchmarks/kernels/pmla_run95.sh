#!/bin/bash
# P-MLA run 95: h8 gather issue spread over lanes (parts per quad 1/2/3/5): tests + F16K.
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/run95
mkdir -p $R
timeout 400 python3 -m pytest --noconftest -q -x tests/models/kimi_k3/test_specialized_prefill_sparse_mla.py -k "h8" > $R/test.log 2>&1
echo "tests: $(tail -1 $R/test.log)" > $R/summary.txt
grep -E "Error|error|assert" $R/test.log | head -5 | cut -c1-300 >> $R/summary.txt
for kn in 0:0:0:0:0:0:0:0:1 0:0:0:0:0:0:0:0:2 0:0:0:0:0:0:0:0:3 0:0:0:0:0:0:0:0:5 1:0:1:0:0:0:0:0:2 1:0:1:0:0:0:0:0:3 1:0:1:0:0:0:0:0:5; do timeout 300 python3 benchmarks/kernels/benchmark_kimi_k3_prefill_sparse_mla.py --tokens 16384 --qmag 0.15 --loaders "" --kernels h8 --no-base --h4-knobs $kn > $R/k_${kn//:/_}.log 2>&1; echo "$kn rc=$? $(grep -E "^h8" $R/k_${kn//:/_}.log | head -1)" >> $R/summary.txt; done
echo DONE > $R/DONE
