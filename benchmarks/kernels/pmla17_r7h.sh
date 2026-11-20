#!/bin/bash
# P-MLA r7h: M41 in-situ copier ladder (coordinator 09:40): C0 copiers alone (warps 10,12-18), C1 copiers on 12-19 (2/SMSP),
# C2 = C0 without REG_LOW dealloc; + production full / skeleton; kernel clock64/globaltimer stamps (knob clk)
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/r7h
mkdir -p $R
cd /k31ov
Z=0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0
timeout 1500 python3 benchmarks/kernels/pmla17b_skel.py $R/res.json \
  C0=p/0:0:0:0:$Z:1 C1=p/0:0:0:0:$Z:2 C2=p/0:0:0:0:$Z:3 SK=p/1:0:1:1 FA=p/ > $R/log 2>&1
echo "rc=$?" >> $R/log
grep -E 'SUMMARY|Error|error|Traceback' $R/log | head -20
