#!/bin/bash
# dattn_skel_run10.sh <outdir> <src.cu>: sparse-MLA skeleton, stage barrier split (NB) x box form (MODE 4 no swizzle,
# MODE 8 whole-row 576 B gather4 box) x issuers; one GPU.
O=$1; SRC=$2; mkdir -p $O
echo "== host $(hostname) gpus=$CUDA_VISIBLE_DEVICES $(date)"
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a -o $O/skel $SRC -lcuda > $O/build.log 2>&1 \
  || { grep -m20 -i -A2 error $O/build.log; exit 1; }
run() { timeout 120 $O/skel $1 0.8 6 448 576 0 0 $2 $3 $4 | grep -E "SKEL|RESULT|TRACE cta 0" || echo "case [$*] rc=$?"; }
for m in "202 32 22" "74 32 22" "168 1 22" "40 1 22" "128 1 22"; do run "96 1 8 4" $m; done
run "96 1 8 4" 202 32 16; run "32 1 8 4" 202 32 22
echo "== END $(date)"
