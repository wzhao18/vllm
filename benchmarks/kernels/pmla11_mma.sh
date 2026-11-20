#!/bin/bash
# pmla11_mma.sh <outdir> [nsub]: build + run the exact h8 tensor-pipe mix probe (1 GPU)
OUT=$1; D=$(cd "$(dirname "$0")" && pwd); mkdir -p $OUT
echo "== host $(hostname) gpus=$CUDA_VISIBLE_DEVICES" > $OUT/log
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a -Xptxas -v -o $OUT/pmla11m $D/pmla11_mma.cu > $OUT/build.log 2>&1 || { grep -m20 error $OUT/build.log; exit 1; }
timeout 300 $OUT/pmla11m ${2:-4096} >> $OUT/log 2>&1; echo "== rc=$?" >> $OUT/log; cat $OUT/log
