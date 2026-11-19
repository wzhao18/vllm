#!/bin/bash
# Build + run the dual-chain sync skeleton matrix (1 GPU). usage: run_skel2.sh <srcdir> <outdir> [quick]
S=$1; O=$2; mkdir -p $O
NVCC=$(command -v nvcc || ls /usr/local/cuda/bin/nvcc)
$NVCC -O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a -o $O/skel2 $S/skel2.cu -lcuda > $O/build.log 2>&1 || { tail -30 $O/build.log; exit 1; }
$NVCC -O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a -Xptxas -v -o /dev/null $S/skel2.cu -lcuda 2>&1 | grep -E "registers|spill" | head -4
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm --format=csv,noheader
B=$O/skel2
if [ "$3" = quick ]; then
  timeout 60 $B 1 2 2 116 0 3; exit $?
fi
for mode in 0 1; do
  for cfg in "1 1 232" "1 2 232" "2 1 116" "2 2 116"; do
    timeout 60 $B $mode $cfg 0 5 | grep -E "RESULT|alloc|TIMEOUT|CUDA|CU " | tail -3
  done
done
# attribution at the dual-chain config, die-local: 1 no HBM loads, 2 no out stores, 4 no state pass, 8 no MMAs
for fl in 1 2 4 8 3 12 15; do timeout 60 $B 1 2 2 116 $fl 5 | grep -E "RESULT|TIMEOUT" ; done
for fl in 1 4 8; do timeout 60 $B 1 1 2 232 $fl 5 | grep -E "RESULT|TIMEOUT" ; done
