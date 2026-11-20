#!/bin/bash
# P-MLA r7l: M43 stage release timing: PV issue distance pvd 1 (stage freed one subtile earlier) / 2 (production) / 3, logical full, bitwise
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/r7l
mkdir -p $R
cd /k31ov
Z=0:0:0:0:0:0:0:0:0:0:0:0:0:0
timeout 1500 python3 benchmarks/kernels/pmla17b_skel.py $R/res.json \
  LA=l/ LV1=l/$Z:1 LV3=l/$Z:3 LA2=l/ > $R/log 2>&1
echo "rc=$?" >> $R/log
grep -E 'SUMMARY|Error|error|Traceback|bitwise' $R/log | head -20
