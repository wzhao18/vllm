#!/bin/bash
# P-MLA r7j: M41c in-situ copier ladder in LOGICAL (tconv, smem indices = the in-step path) vs physical (global index rotation), kernel clock stamps
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/r7j
mkdir -p $R
cd /k31ov
Z=0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0
timeout 1500 python3 benchmarks/kernels/pmla17b_skel.py $R/res.json \
  PC0=p/0:0:0:0:$Z:1 LC0=l/0:0:0:0:$Z:1 LC1=l/0:0:0:0:$Z:2 LA=l/ PA=p/ > $R/log 2>&1
echo "rc=$?" >> $R/log
grep -E 'SUMMARY|Error|error|Traceback|bitwise' $R/log | head -20
