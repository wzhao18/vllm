#!/bin/bash
# pmla16_rt.sh <outdir>: build + run the P-MLA round-6 layout tensor-pipe floor probe (1 GPU)
OUT=$1; D=$(cd "$(dirname "$0")" && pwd); mkdir -p $OUT
echo "== host $(hostname) gpus=$CUDA_VISIBLE_DEVICES $(date +%H:%M:%S)" > $OUT/log
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a -Xptxas -v -o $OUT/pmla16 $D/pmla16_rt.cu > $OUT/build.log 2>&1 || { grep -m20 -i error $OUT/build.log; exit 1; }
grep -E "spill|Used" $OUT/build.log | head -4 >> $OUT/log
timeout 300 $OUT/pmla16 >> $OUT/log 2>&1; echo "== rc=$? $(date +%H:%M:%S)" >> $OUT/log; cat $OUT/log
