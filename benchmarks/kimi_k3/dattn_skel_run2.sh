#!/bin/bash
# dattn_skel_run2.sh <outdir> <src.cu>: isolation run of the sparse-MLA skeleton (MODE bit0 synthetic indices,
# bit1 no Q load / partial-O epilogue) with per-tile traces; one GPU.
O=$1; SRC=$2; mkdir -p $O
echo "== host $(hostname) gpus=$CUDA_VISIBLE_DEVICES $(date)"
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,clocks.mem --format=csv,noheader | head -1
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a -o $O/skel $SRC -lcuda > $O/build.log 2>&1 \
  || { grep -m20 -i -A2 error $O/build.log; exit 1; }
for m in 0 1 2 3; do
  for c in "96 1 8 4" "96 2 8 4" "96 1 4 8"; do timeout 120 $O/skel $c 0.8 6 448 576 0 0 $m || echo "case [$c m$m] rc=$?"; done
done
ISRC=$(dirname $SRC)/dattn_skel_idx.cu
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a -o $O/skel_idx $ISRC -lcuda > $O/build_idx.log 2>&1 \
  || grep -m20 -i -A2 error $O/build_idx.log
for c in "96 516 24 2" "96 516 12 4" "96 516 48 1" "32 516 24 2"; do timeout 120 $O/skel_idx $c 6 || echo "idx case [$c] rc=$?"; done
echo "== END $(date)"
