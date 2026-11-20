#!/bin/bash
# P-MLA run 114: h8 logical v2 (slot producer) + QK halves knob (qkh): tests, F16K h8 / h8L / remap, qkh full and no-gather skeleton.
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/run114
mkdir -p $R
timeout 600 python3 -m pytest --noconftest -q tests/models/kimi_k3/test_specialized_prefill_sparse_mla.py -k "logical or h8" > $R/test.log 2>&1
echo "tests: $(tail -1 $R/test.log)" > $R/summary.txt
grep -E "^FAILED|Error:|assert " $R/test.log | head -8 | cut -c1-300 >> $R/summary.txt
timeout 500 python3 benchmarks/kernels/benchmark_kimi_k3_prefill_sparse_mla.py --tokens 16384 --qmag 0.15 --loaders "" --kernels h8,h8L,remap --h4-knobs "0:0:0:0:0:0:0:0:0:0:0:1,0:1,0:1:0:0:0:0:0:0:0:0:0:1" > $R/f16k.log 2>&1
echo "bench rc=$?" >> $R/summary.txt; grep -E "^h8|^remap|Error" $R/f16k.log | head -10 | cut -c1-300 >> $R/summary.txt
echo DONE > $R/DONE
