#!/bin/bash
# P-MLA run 85: h7 split-row softmax: tests, F16K h6 vs h7.
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/run85
mkdir -p $R
timeout 300 python3 -m pytest --noconftest -q -x tests/models/kimi_k3/test_specialized_prefill_sparse_mla.py -k "h7" > $R/test.log 2>&1
echo "tests: $(tail -1 $R/test.log)" > $R/summary.txt
grep -E "Error|error|assert" $R/test.log | head -5 | cut -c1-300 >> $R/summary.txt
timeout 600 python3 benchmarks/kernels/benchmark_kimi_k3_prefill_sparse_mla.py --tokens 16384 --qmag 0.15 --loaders "" --kernels h6,h7 --no-base --h4-knobs 0 --json $R/f16k.json > $R/f16k.log 2>&1
echo "bench rc=$?" >> $R/summary.txt; grep -E "^h[67]" $R/f16k.log >> $R/summary.txt
echo DONE > $R/DONE
