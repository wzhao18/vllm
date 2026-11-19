#!/bin/bash
# dattn_skel_run.sh <outdir> <src.cu> [CASES...]: build the sparse-MLA skeleton for sm_107a and run a case sweep.
# CASE = "NREQ CL W L cfrac iters nmin nmax LW LP" (see dattn_skel_mla.cu). One GPU.
O=$1; SRC=$2; shift 2; mkdir -p $O
echo "== host $(hostname) gpus=$CUDA_VISIBLE_DEVICES $(date)"
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,clocks.mem --format=csv,noheader | head -1
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a -o $O/skel $SRC -lcuda > $O/build.log 2>&1 \
  || { grep -m20 -i -A2 error $O/build.log; exit 1; }
if [ $# -eq 0 ]; then
  set -- "96 1 4 6 0.8 10 448 576 0 0" "96 2 4 6 0.8 10 448 576 0 0" "96 4 4 6 0.8 10 448 576 0 0" \
         "96 4 4 8 0.8 10 448 576 0 0" "96 2 8 4 0.8 10 448 576 0 0" \
         "96 1 4 6 0.8 10 448 576 6 1" "96 1 4 6 0.8 10 448 576 6 2" "96 2 4 6 0.8 10 448 576 6 1" "96 4 4 6 0.8 10 448 576 6 1" \
         "96 4 4 6 0.6 10 448 576 0 0" "96 4 4 6 0.9 10 448 576 0 0" "64 4 4 6 0.8 10 448 576 0 0" "32 4 4 6 0.8 10 448 576 0 0" \
         "64 1 4 6 0.8 10 448 576 6 1" "32 1 4 6 0.8 10 448 576 6 1"
fi
for c in "$@"; do timeout 120 $O/skel $c || echo "case [$c] rc=$?"; done
# CL=1, 32 issuers per SM (hw15c recipe)
for c in "96 1 4 8 0.8 10 448 576 0 0" "96 1 8 4 0.8 10 448 576 0 0" "96 2 4 8 0.8 10 448 576 0 0"; do timeout 120 $O/skel $c || echo "case [$c] rc=$?"; done
# indexer skeleton (dattn_skel_idx.cu next to SRC): NREQ NBLK THREADS DEPTH
ISRC=$(dirname $SRC)/dattn_skel_idx.cu
if [ -f $ISRC ]; then
  /usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a -o $O/skel_idx $ISRC -lcuda > $O/build_idx.log 2>&1 \
    || grep -m20 -i -A2 error $O/build_idx.log
  for c in "96 516 16 2" "96 516 32 1" "96 516 8 4" "64 516 16 2" "32 516 16 2"; do timeout 120 $O/skel_idx $c || echo "idx case [$c] rc=$?"; done
fi
echo "== END $(date)"
