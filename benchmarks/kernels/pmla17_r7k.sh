#!/bin/bash
# P-MLA r7k: M42 logical (in-step path) full kernel + M22-C pacing 8/12/16 ns (knob pace), bitwise vs production, kernel clock stamps
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/r7k
mkdir -p $R
cd /k31ov
Z=0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0
timeout 1500 python3 benchmarks/kernels/pmla17b_skel.py $R/res.json \
  LA=l/ LP8=l/$Z:8 LP12=l/$Z:12 LP16=l/$Z:16 LP4=l/$Z:4 > $R/log 2>&1
echo "rc=$?" >> $R/log
grep -E 'SUMMARY|Error|error|Traceback|bitwise' $R/log | head -20
