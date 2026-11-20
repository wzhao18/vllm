#!/bin/bash
# pmla13_rt.sh <outdir>: build + run the h8 single-S round-trip probe (1 GPU)
OUT=$1; D=$(cd "$(dirname "$0")" && pwd); mkdir -p $OUT
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a -Xptxas -v -o $OUT/pmla13 $D/pmla13_rt.cu > $OUT/build.log 2>&1 || { grep -m20 error $OUT/build.log; exit 1; }
grep -E "spill|Used" $OUT/build.log | head -4 > $OUT/log; timeout 300 $OUT/pmla13 >> $OUT/log 2>&1; echo "== rc=$?" >> $OUT/log; cat $OUT/log
