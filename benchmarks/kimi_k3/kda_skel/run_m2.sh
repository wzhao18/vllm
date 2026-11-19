#!/bin/bash
# run_m2.sh <srcdir> <outdir>: M1 conv recalibration (kda_calib) + M2/M3 SOL skeleton suite (kda_skel "m2": chain
# latency surrogate, KDA-pass SIMT load, o_norm role) for build specs NCW:CHUNK:NSLOT (BUILDS, default 8:64:1 8:32:2).
S=$1; O=$2; mkdir -p $O
NVCC=$(command -v nvcc || ls /usr/local/cuda/bin/nvcc)
echo "== host $(hostname) gpus=$CUDA_VISIBLE_DEVICES $(date)"
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem --format=csv,noheader | head -1
$NVCC -O3 -std=c++17 -Xptxas -v -gencode arch=compute_107a,code=sm_107a -o $O/kda_calib $S/kda_calib.cu > $O/calib.build.log 2>&1 \
  || grep -m20 -i -A2 error $O/calib.build.log
grep -E "registers|spill" $O/calib.build.log | head -4
[ -n "$SKIPCAL" ] || for n in "3 5461" "3 2731"; do timeout 200 $O/kda_calib $n 15 | grep CALIB; done
for B in ${BUILDS:-8:64:1 8:32:2}; do
  IFS=: read W CH NS <<< "$B"; X=$O/kda_skel_${W}_${CH}_${NS}
  $NVCC -O3 -std=c++17 -DNCW=$W -DCHUNK=$CH -DNSLOT=$NS -DNGRP=1 -Xptxas -v -gencode arch=compute_107a,code=sm_107a -o $X \
    $S/kda_skel.cu -lcuda > $X.build.log 2>&1 || { grep -m20 -i -A2 error $X.build.log; continue; }
  grep -E "registers|spill" $X.build.log | head -2
  for n in "3 5461" "3 2731"; do echo "== NCW=$W CHUNK=$CH NSLOT=$NS T=$n"; timeout 300 $X $n 11 ${SUITE:-m2} | grep -E "SKEL|SPAN|FAIL|WATCH"; done
done
echo "== END $(date)"
