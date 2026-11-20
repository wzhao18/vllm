#!/bin/bash
# P-KDA2 M5: skel6 (2 chains/SM, BT64, INP in smem, bf16 U legs). usage: run_m5.sh <srcdir> <outdir>
S=$1; O=$2; mkdir -p $O
NVCC=$(command -v nvcc || ls /usr/local/cuda/bin/nvcc)
F="-O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a"
$NVCC $F -Xptxas -v -o $O/skel6 $S/skel6.cu -lcuda > $O/build6.log 2>&1 || { echo BUILD6_FAIL; grep -m8 error $O/build6.log; exit 1; }
grep -E "spill|registers" $O/build6.log | head -2
B=$O/skel6
timeout 60 $B 58 80 3 | grep -E "RESULT|TIMEOUT|CU|CUDA" || echo "SMOKE_FAIL rc=$?"
# 16 ideal prep + HBM (gate), 80 ideal no HBM, 0 real image copy; attribution 18 no out, 20 no pass, 24 no MMA
for r in 1 2; do for fl in 16 80 0; do timeout 90 $B 58 $fl 7 | grep -E "RESULT|TIMEOUT" || echo "FAIL fl=$fl rc=$?"; done; done
for fl in 18 20 24 92; do timeout 90 $B 58 $fl 7 | grep -E "RESULT|TIMEOUT" || echo "FAIL fl=$fl rc=$?"; done
for pf in "0 31" "16 31"; do set -- $pf; timeout 90 $B 58 $1 7 $2 6 | grep -E "RESULT|TIMEOUT" || echo "FAIL pfill rc=$?"; done
nvidia-smi --query-gpu=index,clocks.sm,power.draw,clocks_throttle_reasons.active --format=csv,noheader | head -1
