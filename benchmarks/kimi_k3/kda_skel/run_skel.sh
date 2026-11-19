#!/bin/bash
# run_skel.sh <outdir> [nseq seqlen iters]...: build kda_skel.cu for sm_107a and run F16K (3x5461) and F8K (3x2731).
OUT=$1; shift; D=$(cd "$(dirname "$0")" && pwd); mkdir -p $OUT
echo "== host $(hostname) gpus=$CUDA_VISIBLE_DEVICES $(date)"
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem --format=csv,noheader | head -1
NVCC=/usr/local/cuda/bin/nvcc
$NVCC -O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a -o $OUT/kda_skel $D/kda_skel.cu -lcuda > $OUT/build.log 2>&1 \
  || { grep -m20 -i -A2 error $OUT/build.log; exit 2; }
echo "== F16K"; timeout 300 $OUT/kda_skel 3 5461 ${ITERS:-15}
echo "== F8K"; timeout 300 $OUT/kda_skel 3 2731 ${ITERS:-15}
echo "== END $(date)"
