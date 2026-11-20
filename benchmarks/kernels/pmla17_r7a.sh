#!/bin/bash
# P-MLA r7a: M34 h8 producer wait-style arms (skeleton + full + logical), F16K L78K, 1 GPU
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/r7a
mkdir -p $R
cd /k31ov
Z=0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0
timeout 2400 python3 benchmarks/kernels/pmla17_skel.py $R/res.json \
  SA=p/1:0:1:1 SB1=p/1:0:1:1:$Z:1 SB2=p/1:0:1:1:$Z:2 SB4=p/1:0:1:1:$Z:4 SB5=p/1:0:1:1:$Z:5 SE=p/1:1:1:1 \
  FA=p/ FB1=p/0:0:0:0:$Z:1 FB2=p/0:0:0:0:$Z:2 FB5=p/0:0:0:0:$Z:5 FS8=p/0:0:0:0:$Z:8 FS32=p/0:0:0:0:$Z:32 \
  LA=l/ LB2=l/0:0:0:0:$Z:2 > $R/log 2>&1
echo "rc=$?" >> $R/log
grep -E 'SUMMARY|bitwise|Error|error' $R/log | head -30
