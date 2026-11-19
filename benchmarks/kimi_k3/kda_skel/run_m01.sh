#!/bin/bash
# P-KDA2 M0 (skeleton calibration incl. math fill) + M1 (tcgen05 tf32/f16 MMA rate probe). usage: run_m01.sh <srcdir> <outdir>
S=$1; O=$2; mkdir -p $O
NVCC=$(command -v nvcc || ls /usr/local/cuda/bin/nvcc)
F="-O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a"
$NVCC $F -o $O/mma_rate $S/mma_rate.cu -lcuda > $O/build_rate.log 2>&1 || { tail -20 $O/build_rate.log; exit 1; }
$NVCC $F -Xptxas -v -o $O/skel2 $S/skel2.cu -lcuda > $O/build.log 2>&1 || { tail -20 $O/build.log; exit 1; }
grep -E "spill" $O/build.log | sort | uniq -c | head -4
nvidia-smi --query-gpu=index,name,clocks.sm,power.draw --format=csv,noheader | head -2
timeout 120 $O/mma_rate | grep -E "RATE|CUDA"
B=$O/skel2
# M0 calibration: today's design (BT32, 1 chain/SM) with fip's prep / epilogue instruction counts (34 / 8 iters)
for r in 1 2; do
  for pf in "0 0" "34 8"; do timeout 90 $B 1 1 2 232 0 7 $pf | grep -E "RESULT|TIMEOUT|CU"; done
done
for pf in "17 4" "51 12"; do timeout 90 $B 1 1 2 232 0 7 $pf | grep -E "RESULT|TIMEOUT"; done
# same fill on the 2-chain BT32 structure (design point reference)
for pf in "0 0" "34 8"; do timeout 90 $B 1 2 2 116 0 7 $pf | grep -E "RESULT|TIMEOUT"; done
nvidia-smi --query-gpu=index,clocks.sm,power.draw,clocks_throttle_reasons.active --format=csv,noheader | head -2
