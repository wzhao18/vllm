#!/bin/bash
# P-KDA2 M0 (BT32 skeleton calibration with math fill) + M1 (tcgen05 rate probe) + M2/M3/M4 previews (BT64 skel3).
# usage: run_m012.sh <srcdir> <outdir>
S=$1; O=$2; mkdir -p $O
NVCC=$(command -v nvcc || ls /usr/local/cuda/bin/nvcc)
F="-O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a"
$NVCC $F -o $O/mma_rate $S/mma_rate.cu -lcuda > $O/build_rate.log 2>&1 || { echo BUILD_RATE_FAIL; grep -m5 error $O/build_rate.log; }
$NVCC $F -Xptxas -v -o $O/skel2 $S/skel2.cu -lcuda > $O/build2.log 2>&1 || { echo BUILD2_FAIL; grep -m5 error $O/build2.log; }
$NVCC $F -Xptxas -v -o $O/skel3 $S/skel3.cu -lcuda > $O/build3.log 2>&1 || { echo BUILD3_FAIL; grep -m8 error $O/build3.log; }
echo "skel2 spills:"; grep -E "spill" $O/build2.log | sort | uniq -c | head -3
echo "skel3 regs/spills:"; grep -E "spill|registers" $O/build3.log | head -4
nvidia-smi --query-gpu=index,name,clocks.sm,power.draw --format=csv,noheader | head -1
[ -x $O/mma_rate ] && timeout 120 $O/mma_rate | grep -E "RATE|CUDA"
B=$O/skel3
if [ -x $B ]; then
  # M2: BT64 2 chains/SM, ideal prep, 58 steps (F16K volume); descw 1 = 15-bit smem descriptor, 0 = 14-bit
  for r in 1 2; do timeout 90 $B 2 58 16 7 0 0 1 | grep -E "RESULT|TIMEOUT|CU|CUDA"; done
  timeout 90 $B 2 58 16 7 0 0 0 | grep -E "RESULT|TIMEOUT|CU|CUDA"
  # attribution: 4 no pass, 8 no MMA, 2 no out, 64 no HBM
  for fl in 20 24 18 80 30; do timeout 90 $B 2 58 $fl 7 | grep -E "RESULT|TIMEOUT"; done
  timeout 90 $B 1 116 16 7 | grep -E "RESULT|TIMEOUT"
  # M3 preview: real image writes
  for r in 1 2; do timeout 90 $B 2 58 0 7 | grep -E "RESULT|TIMEOUT"; done
  # M4 preview: + math fill (fip mix 68/16, S4 target 51/16)
  for pf in "68 16" "51 16"; do timeout 90 $B 2 58 0 7 $pf | grep -E "RESULT|TIMEOUT"; done
fi
B=$O/skel2
if [ -x $B ]; then
  # M0 calibration: today's design (BT32, 1 chain/SM) with fip's prep / epilogue instruction counts (34 / 8 iters per 32 tok)
  for r in 1 2; do for pf in "0 0" "34 8"; do timeout 90 $B 1 1 2 232 0 7 $pf | grep -E "RESULT|TIMEOUT|CU"; done; done
  for pf in "17 4" "51 12"; do timeout 90 $B 1 1 2 232 0 7 $pf | grep -E "RESULT|TIMEOUT"; done
  for pf in "0 0" "34 8"; do timeout 90 $B 1 2 2 116 0 7 $pf | grep -E "RESULT|TIMEOUT"; done
fi
nvidia-smi --query-gpu=index,clocks.sm,power.draw,clocks_throttle_reasons.active --format=csv,noheader | head -1
