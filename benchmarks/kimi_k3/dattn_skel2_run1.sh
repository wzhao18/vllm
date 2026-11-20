#!/bin/bash
# dattn_skel2_run1.sh <outdir> <src.cu>: D-ATTN milestones M1 (hw15c form, per-row lists), M2 (request-union dedup),
# M3 (tile assembly + consumer warp) for the R384 sparse-MLA gather; one GPU.
O=$1; SRC=$2; mkdir -p $O
echo "== host $(hostname) gpus=$CUDA_VISIBLE_DEVICES $(date)"
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a -Xptxas -v -o $O/skel2 $SRC -lcuda > $O/build.log 2>&1 || { grep -m20 -i -A2 error $O/build.log; exit 1; }
r() { timeout 120 $O/skel2 "$@" | grep -E "SKEL2|MISMATCH|error|QMAX|bad" || echo "case [$*] rc=$?"; }
echo "-- M1 no dedup"; for c in "8 4 2" "32 1 2" "4 8 2" "16 2 2" "10 4 2" "16 4 1" "8 8 1"; do r 0 $c; done; r 4 8 4 2
echo "-- M2 dedup"; for c in "8 4 2" "32 1 2" "10 4 2" "16 4 1"; do r 1 $c; done; r 1 8 4 2 0.5; r 5 8 4 2
echo "-- M3 tiles"; r 3 8 4 2 0.8 10 32; r 3 8 4 2 0.8 10 8; r 3 16 4 1 0.8 10 64; r 3 16 4 1 0.8 10 16; r 3 10 4 2 0.8 10 40; r 2 8 4 2 0.8 10 32
echo "== END $(date)"
