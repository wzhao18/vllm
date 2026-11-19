#!/bin/bash
# P-MLA run 75: dispatch (sparse_mla.sparse_mla_prefill -> h6): tests, and h6 vs trtllm-gen over chunk sizes.
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/run75
mkdir -p $R
timeout 400 python3 -m pytest --noconftest -q -x tests/models/kimi_k3/test_specialized_prefill_sparse_mla.py -k "stock or h6" > $R/test.log 2>&1
echo "tests: $(tail -1 $R/test.log)" > $R/summary.txt
for F in 1024 2048 4096 8192; do
  timeout 600 python3 benchmarks/kernels/benchmark_kimi_k3_prefill_sparse_mla.py --tokens $F --qmag 0.15 --loaders "" --kernels h6 --no-base --h4-knobs 0 --json $R/f$F.json > $R/f$F.log 2>&1
  echo "F$F rc=$? $(grep -o "'attn_trtllm': [0-9.]*, 'attn_h6_0': [0-9.]*" $R/f$F.log | tail -1)" >> $R/summary.txt
done
echo DONE > $R/DONE
