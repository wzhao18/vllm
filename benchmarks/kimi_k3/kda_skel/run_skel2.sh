#!/bin/bash
# Build + run the dual-chain sync skeleton matrix (1 GPU). usage: run_skel2.sh <srcdir> <outdir> [quick|prep]
S=$1; O=$2; mkdir -p $O
NVCC=$(command -v nvcc || ls /usr/local/cuda/bin/nvcc)
$NVCC -O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a -o $O/skel2 $S/skel2.cu -lcuda > $O/build.log 2>&1 || { tail -30 $O/build.log; exit 1; }
B=$O/skel2
if [ "$3" = quick ]; then
  timeout 60 $B 1 2 2 116 0 3; exit $?
fi
if [ "$3" = rep ]; then
  $NVCC -O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a -Xptxas -v -o /dev/null $S/skel2.cu -lcuda 2>&1 | grep -E "spill|registers" | head -4
  nvidia-smi --query-gpu=index,name,clocks.sm,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | head
  for r in 1 2; do for fl in 0 16 32 48 20 24 4 8; do timeout 60 $B 1 2 2 116 $fl 7 | grep -E "RESULT|TIMEOUT|CU" ; done; done
  for fl in 0 16 20; do timeout 60 $B 1 1 2 232 $fl 7 | grep RESULT; done
  for fl in 0 16; do timeout 60 $B 1 2 1 116 $fl 7 | grep RESULT; done
  nvidia-smi --query-gpu=index,clocks.sm,power.draw,clocks_throttle_reasons.active --format=csv,noheader
  exit 0
fi
if [ "$3" = prep ]; then
  # ideal-prep chain structure (flag 16) and its attribution
  for fl in 32 48 50 52 36 0 16; do timeout 60 $B 1 2 2 116 $fl 5 | grep -E "RESULT|TIMEOUT|CU" ; done
  for fl in 32 48 52; do timeout 60 $B 1 2 1 116 $fl 5 | grep -E "RESULT|TIMEOUT" ; done
  for fl in 32 48; do timeout 60 $B 1 1 2 232 $fl 5 | grep -E "RESULT|TIMEOUT" ; done
  exit 0
fi
for mode in 0 1; do
  for cfg in "1 1 232" "1 2 232" "2 1 116" "2 2 116"; do
    timeout 60 $B $mode $cfg 0 5 | grep -E "RESULT|SMs|TIMEOUT|CUDA|CU " | tail -3
  done
done
# attribution at the dual-chain config, die-local: 1 no HBM loads, 2 no out stores, 4 no state pass, 8 no MMAs, 16 no prep copy
for fl in 1 2 4 8 3 12 15 16; do timeout 60 $B 1 2 2 116 $fl 5 | grep -E "RESULT|TIMEOUT" ; done
for fl in 1 4 8 16; do timeout 60 $B 1 1 2 232 $fl 5 | grep -E "RESULT|TIMEOUT" ; done
