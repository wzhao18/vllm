#!/bin/bash
# P-MLA run 122: h8 logical (prefix count, all-lane slot wait): hang re-check per case (240 s), F16K h8 / h8L / remap, test file.
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/run122
mkdir -p $R
python3 benchmarks/kernels/pmla_dbg_logical.py full_short > $R/dbg_fs.log 2>&1; echo "full_short rc=$? $(grep -E 'full_short|Error' $R/dbg_fs.log | tail -3 | tr '\n' ' ')" > $R/summary.txt
timeout 400 python3 benchmarks/kernels/benchmark_kimi_k3_prefill_sparse_mla.py --tokens 16384 --qmag 0.15 --loaders "" --kernels h8,h8L,remap --h4-knobs "" > $R/f16k.log 2>&1
echo "bench rc=$?" >> $R/summary.txt; grep -E "^h8|^remap|Error" $R/f16k.log | head -6 | cut -c1-300 >> $R/summary.txt
python3 benchmarks/kernels/pmla_dbg_logical.py > $R/dbg.log 2>&1
cut -c1-200 $R/dbg.log >> $R/summary.txt
timeout 900 python3 -m pytest --noconftest -q tests/models/kimi_k3/test_specialized_prefill_sparse_mla.py -k "logical or (h8 and not qkh)" > $R/test.log 2>&1
echo "tests rc=$? $(tail -1 $R/test.log)" >> $R/summary.txt
echo DONE > $R/DONE
