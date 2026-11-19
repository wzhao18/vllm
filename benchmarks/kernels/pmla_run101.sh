#!/bin/bash
# P-MLA run 101: h8 with epilogue warps 12-15 (5 gather warps): tests, F16K, decomposition, ncu.

R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/run101
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws
mkdir -p $R
timeout 400 python3 -m pytest --noconftest -q -x tests/models/kimi_k3/test_specialized_prefill_sparse_mla.py -k "h8 or h6" > $R/test.log 2>&1
echo "tests: $(tail -1 $R/test.log)" > $R/summary.txt
grep -E "Error|error|assert" $R/test.log | head -5 | cut -c1-300 >> $R/summary.txt
for kn in 0 1 1:0:1 0:0:0:0:0:0:0:0:1 0:0:0:0:0:0:0:0:3; do
  timeout 300 python3 benchmarks/kernels/benchmark_kimi_k3_prefill_sparse_mla.py --tokens 16384 --qmag 0.15 --loaders "" --kernels h8 --no-base --h4-knobs $kn > $R/k_${kn//:/_}.log 2>&1
  echo "$kn rc=$? $(grep -E "^h8" $R/k_${kn//:/_}.log | head -1)" >> $R/summary.txt
done
export CUTE_DSL_LINEINFO=1 CUTE_DSL_CACHE_DIR=$R/cute_cache
timeout 1500 $K/tools/nsight/bin/ncu --set full --import-source yes -k regex:SparseMlaPrefillH8 -c 1 -s 1 -f -o $R/h8 \
  python3 benchmarks/kernels/benchmark_kimi_k3_prefill_sparse_mla.py --tokens 16384 --qmag 0.15 --loaders "" --kernels h8 --iters 2 --layers 1 --no-base --h4-knobs 0 > $R/ncu.log 2>&1
echo "ncu rc=$?" >> $R/summary.txt
$K/tools/nsight/bin/ncu -i $R/h8.ncu-repz --page details > $R/details.txt 2>&1
$K/tools/nsight/bin/ncu -i $R/h8.ncu-repz --page source --csv --print-source sass > $R/sass_src.csv 2>&1
echo DONE > $R/DONE
