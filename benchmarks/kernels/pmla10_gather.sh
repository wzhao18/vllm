#!/bin/bash
# pmla9_gather.sh <outdir>: build + run the LDGSTS rate-law probe (1 GPU)
OUT=$1; D=$(cd "$(dirname "$0")" && pwd); mkdir -p $OUT
echo "== host $(hostname) gpus=$CUDA_VISIBLE_DEVICES" > $OUT/log
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a -Xptxas -v -o $OUT/pmla10g $D/pmla10_gather.cu -lcuda > $OUT/build.log 2>&1 || { grep -m20 error $OUT/build.log; exit 1; }; grep -E "spill|Used" $OUT/build.log | head -8 >> $OUT/log
timeout 300 $OUT/pmla10g >> $OUT/log 2>&1; echo "== rc=$?" >> $OUT/log; cat $OUT/log
