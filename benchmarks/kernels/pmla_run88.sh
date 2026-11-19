#!/bin/bash
# P-MLA run 88: h6 L2 prefetch of slot rows one token ahead.
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/run88
mkdir -p $R
timeout 300 python3 -m pytest --noconftest -q -x tests/models/kimi_k3/test_specialized_prefill_sparse_mla.py -k "h6 or h7 or stock" > $R/test.log 2>&1
echo "tests: $(tail -1 $R/test.log)" > $R/summary.txt
grep -E "Error|error|assert" $R/test.log | head -5 | cut -c1-300 >> $R/summary.txt
timeout 600 python3 benchmarks/kernels/benchmark_kimi_k3_prefill_sparse_mla.py --tokens 16384 --qmag 0.15 --loaders "" --kernels h6,h7 --no-base --h4-knobs 0,0:0:0:0:1 --json $R/f16k.json > $R/f16k.log 2>&1
echo "bench rc=$?" >> $R/summary.txt; grep -E "^h[67]" $R/f16k.log >> $R/summary.txt
echo DONE > $R/DONE
