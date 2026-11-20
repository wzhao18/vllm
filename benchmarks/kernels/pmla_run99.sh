#!/bin/bash
# P-MLA run 99: h8 KV ring depth sensitivity (NS 4/5/6) and RoPE by cp.async (cpa knob): tests + F16K.
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/run99
mkdir -p $R
timeout 400 python3 -m pytest --noconftest -q -x tests/models/kimi_k3/test_specialized_prefill_sparse_mla.py -k "h8" > $R/test.log 2>&1
echo "tests: $(tail -1 $R/test.log)" > $R/summary.txt
for ns in 6 5 4; do for kn in 0 1:0:1; do
  timeout 300 python3 benchmarks/kernels/benchmark_kimi_k3_prefill_sparse_mla.py --tokens 16384 --qmag 0.15 --loaders "" --kernels h8 --no-base --h8-ns $ns --h4-knobs $kn > $R/ns${ns}_${kn//:/_}.log 2>&1
  echo "ns $ns $kn rc=$? $(grep -E "^h8" $R/ns${ns}_${kn//:/_}.log | head -1)" >> $R/summary.txt
done; done
for kn in 0:0:0:0:0:0:0:0:0:0:1 1:0:1:0:0:0:0:0:0:0:1; do
  timeout 300 python3 benchmarks/kernels/benchmark_kimi_k3_prefill_sparse_mla.py --tokens 16384 --qmag 0.15 --loaders "" --kernels h8 --no-base --h4-knobs $kn > $R/k_${kn//:/_}.log 2>&1
  echo "$kn rc=$? $(grep -E "^h8" $R/k_${kn//:/_}.log | head -1)" >> $R/summary.txt
done
echo DONE > $R/DONE
