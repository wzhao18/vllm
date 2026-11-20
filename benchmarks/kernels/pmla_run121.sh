#!/bin/bash
# P-MLA run 121: h8 logical: hang localization (token mixes per CTA) + xphys (gather warps on the physical loop, producer still running) + prefix count.
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/run121
mkdir -p $R
python3 benchmarks/kernels/pmla_dbg_logical.py > $R/dbg.log 2>&1
cp $R/dbg.log $R/summary.txt
timeout 500 python3 benchmarks/kernels/benchmark_kimi_k3_prefill_sparse_mla.py --tokens 16384 --qmag 0.15 --loaders "" --kernels h8,h8L --h4-knobs "" --h8l-knobs "0:0:0:0:0:0:0:0:0:0:0:0:0:1,1:0:1:0:0:0:0:0:0:0:0:0:0:1" > $R/f16k.log 2>&1
echo "bench rc=$?" >> $R/summary.txt; grep -E "^h8|Error" $R/f16k.log | head -8 | cut -c1-300 >> $R/summary.txt
echo DONE > $R/DONE
