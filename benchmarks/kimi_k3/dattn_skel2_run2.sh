#!/bin/bash
# dattn_skel2_run2.sh <outdir> <src.cu>: D-ATTN M3b (out-of-order chunk consumer) + SW128 5-op MMA form; one GPU.
O=$1; SRC=$2; mkdir -p $O
echo "== host $(hostname) gpus=$CUDA_VISIBLE_DEVICES $(date)"
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a -Xptxas -v -o $O/skel2 $SRC -lcuda > $O/build.log 2>&1 || { grep -m20 -i -A2 error $O/build.log; exit 1; }
r() { timeout 120 $O/skel2 "$@" | grep -E "SKEL2|TRACE|MISMATCH|error|QMAX|bad" | cut -c1-1500 || echo "case [$*] rc=$?"; }
echo "-- ref M2/M3"; r 1 8 4 2; r 3 8 4 2 0.8 10 32
echo "-- M3b OOO"; for g in 4 8 16 32; do r 11 8 4 2 0.8 10 $g; done; r 11 4 8 2 0.8 10 8; r 11 16 2 2 0.8 10 8; r 11 16 4 1 0.8 10 8; r 27 8 4 2 0.8 10 8
echo "-- SW128 5-op form"; r 33 8 4 2; r 33 16 4 1; r 43 8 4 2 0.8 10 8; r 43 16 4 1 0.8 10 8; r 43 16 4 1 0.8 10 16
echo "-- no dedup / warm"; r 10 8 4 2 0.8 10 8; r 15 8 4 2 0.8 10 8; r 47 8 4 2 0.8 10 8
echo "== END $(date)"
