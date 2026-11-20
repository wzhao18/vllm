#!/bin/bash
# P-MLA r7n: M44b event-paced (eph 3 deadlocks: dropped) copier halves (eph: warps pw>=4 wait S-free of subtile g-eph before stage g), logical full, bitwise
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/r7n
mkdir -p $R
cd /k31ov
timeout 1500 python3 benchmarks/kernels/pmla17b_skel.py $R/res.json \
  LA=l/ LE1=l/0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:1 LE2=l/0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:2 > $R/log 2>&1
echo "rc=$?" >> $R/log
grep -E 'SUMMARY|Error|error|Traceback|bitwise' $R/log | head -20
