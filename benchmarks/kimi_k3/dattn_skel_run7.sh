#!/bin/bash
# dattn_skel_run7.sh <outdir> <src.cu>: sparse-MLA skeleton, stage barrier split (NB) x box form (MODE 4 no swizzle,
# MODE 8 whole-row 576 B gather4 box) x issuers; one GPU.
O=$1; SRC=$2; mkdir -p $O
echo "== host $(hostname) gpus=$CUDA_VISIBLE_DEVICES $(date)"
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a -o $O/skel $SRC -lcuda > $O/build.log 2>&1 \
  || { grep -m20 -i -A2 error $O/build.log; exit 1; }
run() { timeout 120 $O/skel $1 0.8 6 448 576 0 0 $2 $3 | grep -E "SKEL|RESULT|TRACE cta 0" || echo "case [$*] rc=$?"; }
for m in "40 8" "40 32" "32 8" "32 32" "41 32" "40 1"; do run "96 1 8 4" $m; done
run "96 1 4 8" 40 32; run "96 1 8 8" 40 32; run "96 2 8 4" 40 32; run "64 1 8 4" 40 32; run "32 1 8 4" 40 32
echo "== END $(date)"
