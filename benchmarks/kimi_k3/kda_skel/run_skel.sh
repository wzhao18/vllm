#!/bin/bash
# run_skel.sh <outdir>: build kda_skel.cu for sm_107a per build spec "NCW:CHUNK:NSLOT" (BUILDS, default
# "8:32:2 8:64:1 8:32:1") and run F16K (3x5461) and F8K (3x2731).
OUT=$1; shift; D=$(cd "$(dirname "$0")" && pwd); mkdir -p $OUT
echo "== host $(hostname) gpus=$CUDA_VISIBLE_DEVICES $(date)"
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem --format=csv,noheader | head -1
NVCC=/usr/local/cuda/bin/nvcc
for B in ${BUILDS:-8:32:2 8:64:1 8:32:1}; do
  IFS=: read W CH NS <<< "$B"; X=$OUT/kda_skel_${W}_${CH}_${NS}
  $NVCC -O3 -std=c++17 -DNCW=$W -DCHUNK=$CH -DNSLOT=$NS -gencode arch=compute_107a,code=sm_107a -o $X $D/kda_skel.cu \
    -lcuda > $X.build.log 2>&1 || { grep -m20 -i -A2 error $X.build.log; continue; }
  echo "== NCW=$W CHUNK=$CH NSLOT=$NS F16K"; timeout 300 $X 3 5461 ${ITERS:-15}
  echo "== NCW=$W CHUNK=$CH NSLOT=$NS F8K"; timeout 300 $X 3 2731 ${ITERS:-15}
done
echo "== END $(date)"
