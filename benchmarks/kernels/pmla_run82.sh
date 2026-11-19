#!/bin/bash
# P-MLA run 82: ncu of h6 (unroll=1 loader, sparse rescale): stalls per SASS, icache.
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/run82
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws
mkdir -p $R
export CUTE_DSL_LINEINFO=1 CUTE_DSL_CACHE_DIR=$R/cute_cache
timeout 1500 $K/tools/nsight/bin/ncu --set full --import-source yes -k regex:SparseMlaPrefillH6 -c 1 -s 1 -f -o $R/h6 \
  python3 benchmarks/kernels/benchmark_kimi_k3_prefill_sparse_mla.py --tokens 16384 --qmag 0.15 --loaders "" --kernels h6 --iters 2 --layers 1 --no-base --h4-knobs 0 > $R/ncu.log 2>&1
echo "ncu rc=$?" >> $R/ncu.log
$K/tools/nsight/bin/ncu -i $R/h6.ncu-repz --page details > $R/details.txt 2>&1
$K/tools/nsight/bin/ncu -i $R/h6.ncu-repz --page source --csv --print-source sass > $R/sass_src.csv 2>&1

grep -E "No Instruction|Stall|Issue Slots|Warp Cycles|Eligible|Active Warps" $R/details.txt | head -40 > $R/summary.txt
echo DONE > $R/DONE
