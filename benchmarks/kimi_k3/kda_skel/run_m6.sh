#!/bin/bash
# P-KDA2: skel5 sync-floor attribution (stamps) + M5 skel6 (2 chains/SM). usage: run_m6.sh <srcdir> <outdir>
S=$1; O=$2; mkdir -p $O
NVCC=$(command -v nvcc || ls /usr/local/cuda/bin/nvcc)
F="-O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a"
$NVCC $F -Xptxas -v -o $O/skel5 $S/skel5.cu -lcuda > $O/build5.log 2>&1 || { echo BUILD5_FAIL; grep -m8 error $O/build5.log; }
grep -E "spill|registers" $O/build5.log | head -2
for fl in 16 24 88 94 30; do timeout 90 $O/skel5 116 $fl 5 | grep -E "RESULT|STAMPS|TIMEOUT" | sed 's/ | HBM.*//' || echo "FAIL fl=$fl"; done
timeout 90 $O/skel5 58 94 5 | grep -E "RESULT|TIMEOUT" | sed 's/ | HBM.*//'
echo M5_START
bash $S/run_m5.sh $S $O/m5
