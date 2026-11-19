#!/bin/bash
# run_calib.sh <srcdir> <outdir>: M1 calibration of today's KDA prefill layer (F16K 3x5461, F8K 3x2731), 1 GPU.
# conv + o_norm passes (kda_calib.cu) and the fip-structure core (P-KDA2 skel2.cu, 1 chain/SM, steps = fip LPT span).
S=$1; O=$2; mkdir -p $O
NVCC=$(command -v nvcc || ls /usr/local/cuda/bin/nvcc)
echo "== host $(hostname) gpus=$CUDA_VISIBLE_DEVICES $(date)"
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem --format=csv,noheader | head -1
$NVCC -O3 -std=c++17 -Xptxas -v -gencode arch=compute_107a,code=sm_107a -o $O/kda_calib $S/kda_calib.cu > $O/calib.build.log 2>&1 \
  || { grep -m20 -i -A2 error $O/calib.build.log; }
grep -E "registers|spill" $O/calib.build.log | head -4
$NVCC -O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a -o $O/skel2 $S/skel2.cu -lcuda > $O/skel2.build.log 2>&1 || tail -5 $O/skel2.build.log
for rep in 1 2; do
  echo "== rep $rep F16K"; timeout 300 $O/kda_calib 3 5461 15
  # fip span F16K: 288 chains x (171 + 3) chunk32 / 212 = 237 steps; F8K: 288 x (86 + 3) / 212 = 121
  for m in 0 1; do timeout 120 $O/skel2 $m 1 2 237 0 7 | grep -E "RESULT|TIMEOUT|CU"; done
  timeout 120 $O/skel2 0 1 2 237 16 7 | grep -E "RESULT|TIMEOUT|CU"
  echo "== rep $rep F8K"; timeout 300 $O/kda_calib 3 2731 15
  timeout 120 $O/skel2 0 1 2 121 0 7 | grep -E "RESULT|TIMEOUT|CU"
done
echo "== END $(date)"
