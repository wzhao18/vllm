#!/bin/bash
# P-KDA2 M3'' (skel5: U double buffer 256, packed out stores 512, flag-8 fix) + C1 math calibration (skel2c).
# usage: run_m3.sh <srcdir> <outdir>
S=$1; O=$2; mkdir -p $O
NVCC=$(command -v nvcc || ls /usr/local/cuda/bin/nvcc)
F="-O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a"
$NVCC $F -Xptxas -v -o $O/skel5 $S/skel5.cu -lcuda > $O/build5.log 2>&1 || { echo BUILD5_FAIL; grep -m8 error $O/build5.log; }
grep -E "spill|registers" $O/build5.log | head -2
B=$O/skel5
timeout 60 $B 116 80 3 | grep -E "RESULT|TIMEOUT|CU|CUDA" || echo "SMOKE_FAIL rc=$?"
for r in 1 2; do for fl in 16 272 528 784 80 336; do timeout 90 $B 116 $fl 7 | grep -E "RESULT|TIMEOUT" || echo "FAIL fl=$fl rc=$?"; done; done
# attribution on the M3'' config (784 = ideal prep + udb + packed): +2 no out, +4 no pass, +8 no MMA (deadlock fix), +64 no HBM
for fl in 786 788 792 848 24 0 768; do timeout 90 $B 116 $fl 7 | grep -E "RESULT|TIMEOUT" || echo "FAIL fl=$fl rc=$?"; done
nvidia-smi --query-gpu=index,clocks.sm,power.draw,clocks_throttle_reasons.active --format=csv,noheader | head -1
echo C1_START
bash $S/run_c1.sh $S $O/c1
