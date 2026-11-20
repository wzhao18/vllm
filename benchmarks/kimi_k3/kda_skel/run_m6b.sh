#!/bin/bash
# P-KDA2 M6: 4-lane TMA issue (flag 2048) on skel5 (1 chain) and skel6 (2 chains). usage: run_m6b.sh <srcdir> <outdir>
S=$1; O=$2; mkdir -p $O
NVCC=$(command -v nvcc || ls /usr/local/cuda/bin/nvcc)
F="-O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a"
for k in 5 6; do $NVCC $F -Xptxas -v -o $O/skel$k $S/skel$k.cu -lcuda > $O/build$k.log 2>&1 || { echo BUILD${k}_FAIL; grep -m5 error $O/build$k.log; }; grep -E "spill" $O/build$k.log | head -1; done
for r in 1 2; do
  for fl in 16 2064 2072 24 2128; do timeout 90 $O/skel5 116 $fl 5 | grep -E "RESULT|STAMPS|TIMEOUT" | sed 's/ | HBM.*//' || echo "FAIL5 fl=$fl"; done
  for fl in 16 2064 2048 80; do timeout 90 $O/skel6 58 $fl 5 | grep -E "RESULT|TIMEOUT" | sed 's/ | HBM.*//' || echo "FAIL6 fl=$fl"; done
done
