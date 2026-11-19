#!/bin/bash
# run_skel.sh <outdir>: build kda_skel.cu for sm_107a (consumer warps per chain slot NCWS, default "4 8") and run
# F16K (3x5461) and F8K (3x2731).
OUT=$1; shift; D=$(cd "$(dirname "$0")" && pwd); mkdir -p $OUT
echo "== host $(hostname) gpus=$CUDA_VISIBLE_DEVICES $(date)"
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem --format=csv,noheader | head -1
NVCC=/usr/local/cuda/bin/nvcc
for W in ${NCWS:-4 8}; do
  $NVCC -O3 -std=c++17 -DNCW=$W -gencode arch=compute_107a,code=sm_107a -o $OUT/kda_skel$W $D/kda_skel.cu -lcuda \
    > $OUT/build$W.log 2>&1 || { grep -m20 -i -A2 error $OUT/build$W.log; exit 2; }
  echo "== NCW=$W F16K"; timeout 300 $OUT/kda_skel$W 3 5461 ${ITERS:-15}
  echo "== NCW=$W F8K"; timeout 300 $OUT/kda_skel$W 3 2731 ${ITERS:-15}
done
echo "== END $(date)"
