#!/bin/bash
# P-KDA2 M0 (BT32 calibration: fip prep/chain math as background issue load on 16 idle warps = fip's 20 prep warps,
# epilogue math in-line) + M1 (unrolled tcgen05 rate probe) + M2/M3/M4 previews (BT64 skel3). usage: <srcdir> <outdir>
S=$1; O=$2; mkdir -p $O
NVCC=$(command -v nvcc || ls /usr/local/cuda/bin/nvcc)
F="-O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a"
$NVCC $F -o $O/mma_rate $S/mma_rate.cu -lcuda > $O/build_rate.log 2>&1 || { echo BUILD_RATE_FAIL; grep -m5 error $O/build_rate.log; }
$NVCC $F -Xptxas -v -o $O/skel2 $S/skel2.cu -lcuda > $O/build2.log 2>&1 || { echo BUILD2_FAIL; grep -m5 error $O/build2.log; }
$NVCC $F -Xptxas -v -o $O/skel3 $S/skel3.cu -lcuda > $O/build3.log 2>&1 || { echo BUILD3_FAIL; grep -m8 error $O/build3.log; }
echo "skel2:"; grep -E "spill" $O/build2.log | sort | uniq -c | head -3
echo "skel3:"; grep -E "spill|registers" $O/build3.log | head -2
$NVCC $F -cubin -o $O/skel3.cubin $S/skel3.cu > /dev/null 2>&1 && cuobjdump -sass $O/skel3.cubin > $O/skel3.sass 2>/dev/null; echo "skel3 SASS UTCMMA/UTCHMMA count: $(grep -c -E 'UTC.*MMA' $O/skel3.sass 2>/dev/null) R2UR: $(grep -c R2UR $O/skel3.sass 2>/dev/null)"
[ -x $O/mma_rate ] && timeout 120 $O/mma_rate | grep -E "RATE|CUDA"
B=$O/skel3
if [ -x $B ]; then
  for r in 1 2; do timeout 90 $B 2 58 16 7 0 0 1 | grep -E "RESULT|TIMEOUT|CU|CUDA"; done
  timeout 90 $B 2 58 16 7 0 0 0 | grep -E "RESULT|TIMEOUT|CU|CUDA"
  for fl in 20 24 18 80 30; do timeout 90 $B 2 58 $fl 7 | grep -E "RESULT|TIMEOUT"; done
  timeout 90 $B 1 116 16 7 | grep -E "RESULT|TIMEOUT"
  for r in 1 2; do timeout 90 $B 2 58 0 7 | grep -E "RESULT|TIMEOUT"; done
  for pf in "68 16" "51 16" "34 16"; do timeout 90 $B 2 58 0 7 $pf | grep -E "RESULT|TIMEOUT"; done
fi
B=$O/skel2
if [ -x $B ]; then
  for r in 1 2; do for pf in "0 0 0" "0 8 10" "0 8 20"; do timeout 90 $B 1 1 2 232 0 7 $pf | grep -E "RESULT|TIMEOUT|CU"; done; done
fi
nvidia-smi --query-gpu=index,clocks.sm,power.draw,clocks_throttle_reasons.active --format=csv,noheader | head -1
