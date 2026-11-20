#!/bin/bash
# P-MLA run 124: token-end barrier race fix (h6/h8): regression test, dbg mixes (physical + logical), F16K h8 / h8L, test file.
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/run124
mkdir -p $R
timeout 600 python3 -m pytest --noconftest -q tests/models/kimi_k3/test_specialized_prefill_sparse_mla.py -k "two_short" > $R/t_short.log 2>&1
echo "two_short rc=$? $(tail -1 $R/t_short.log)" > $R/summary.txt
python3 benchmarks/kernels/pmla_dbg_logical.py > $R/dbg.log 2>&1
cut -c1-160 $R/dbg.log >> $R/summary.txt
timeout 400 python3 benchmarks/kernels/benchmark_kimi_k3_prefill_sparse_mla.py --tokens 16384 --qmag 0.15 --loaders "" --kernels h6,h8,h8L --h4-knobs "" > $R/f16k.log 2>&1
echo "bench rc=$?" >> $R/summary.txt; grep -E "^h[68]|Error" $R/f16k.log | head -6 | cut -c1-200 >> $R/summary.txt
timeout 1500 python3 -m pytest --noconftest -q tests/models/kimi_k3/test_specialized_prefill_sparse_mla.py -k "h6 or h8 or logical or two_short or trtllm" > $R/test.log 2>&1
echo "tests rc=$? $(tail -1 $R/test.log)" >> $R/summary.txt
grep -E "^FAILED" $R/test.log | head -5 | cut -c1-200 >> $R/summary.txt
echo DONE > $R/DONE
