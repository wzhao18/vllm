#!/bin/bash
# dattn_skel2_run4.sh <outdir> <src.cu>: D-ATTN M3c: deeper per-lane rings (S 3-4) with 16-24 issuers + indexer span skeleton; one GPU.
O=$1; SRC=$2; mkdir -p $O
echo "== host $(hostname) gpus=$CUDA_VISIBLE_DEVICES $(date)"
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a -o $O/skel2 $SRC -lcuda > $O/build.log 2>&1 || { grep -m20 -i -A2 error $O/build.log; exit 1; }
r() { timeout 120 $O/skel2 "$@" | grep -E "SKEL2|MISMATCH|error|QMAX|bad" | cut -c1-400 || echo "case [$*] rc=$?"; }
echo "-- M3 in-order"; r 3 4 4 2 0.8 10 16; r 3 4 4 4 0.8 10 16; r 3 4 4 6 0.8 10 16; r 3 6 4 4 0.8 10 24; r 3 8 4 3 0.8 10 32; r 3 4 2 6 0.8 10 8
echo "-- M3b OOO"; r 11 4 4 4 0.8 10 16; r 11 4 4 4 0.8 10 8; r 11 8 4 3 0.8 10 16; r 11 4 4 6 0.8 10 16
echo "-- M2/static refs"; r 1 4 4 4 0.8 10 16; r 65 4 4 4 0.8 10 16; r 1 8 4 3 0.8 10 32; r 7 4 4 4 0.8 10 16; r 15 4 4 4 0.8 10 16
ISRC=$(dirname $SRC)/dattn_skel_idx.cu; /usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a -o $O/skel_idx $ISRC -lcuda > $O/build_idx.log 2>&1 || grep -m5 error $O/build_idx.log
echo "-- indexer key stream (in-kernel span)"; for c in "96 516 24 2" "96 516 12 4" "96 516 48 1"; do timeout 120 $O/skel_idx $c 6 | grep -E "SKEL-IDX|SPAN|RESULT" | cut -c1-200 || echo "idx [$c] rc=$?"; done
echo "== END $(date)"
