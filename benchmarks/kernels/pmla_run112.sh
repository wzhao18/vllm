#!/bin/bash
# P-MLA run 112: h8 logical mode (index remap fused into the gather): tests, F16K h8 vs h8L vs stock remap.
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/run112
mkdir -p $R
timeout 500 python3 -m pytest --noconftest -q -x tests/models/kimi_k3/test_specialized_prefill_sparse_mla.py -k "logical or h8" > $R/test.log 2>&1
echo "tests: $(tail -1 $R/test.log)" > $R/summary.txt
grep -E "Error|error|assert" $R/test.log | head -8 | cut -c1-300 >> $R/summary.txt
timeout 500 python3 benchmarks/kernels/benchmark_kimi_k3_prefill_sparse_mla.py --tokens 16384 --qmag 0.15 --loaders "" --kernels h8,h8L,remap --h4-knobs "" > $R/f16k.log 2>&1
echo "bench rc=$?" >> $R/summary.txt; grep -E "^h8|^remap|Error" $R/f16k.log | head -8 | cut -c1-300 >> $R/summary.txt
echo DONE > $R/DONE
