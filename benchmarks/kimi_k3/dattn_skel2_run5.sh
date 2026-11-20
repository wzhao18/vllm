#!/bin/bash
# dattn_skel2_run5.sh <outdir> <src.cu>: D-ATTN M3e: consumer releases slots through a volatile smem flag the issuer polls (bit8) vs mbarrier; one GPU.
O=$1; SRC=$2; mkdir -p $O
echo "== host $(hostname) gpus=$CUDA_VISIBLE_DEVICES $(date)"
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a -o $O/skel2 $SRC -lcuda > $O/build.log 2>&1 || { grep -m20 -i -A2 error $O/build.log; exit 1; }
r() { timeout 120 $O/skel2 "$@" | grep -E "SKEL2|MISMATCH|error|QMAX|bad" | sed -E "s/rows [0-9]+ .keys [0-9]+, 0.45x. 50.2 MB //" | cut -c1-200 || echo "case [$*] rc=$?"; }
echo "-- refs"; r 1 4 4 4 0.8 10 16; r 3 4 4 2 0.8 10 16
echo "-- flag release"; r 259 4 4 2 0.8 10 16; r 259 4 4 4 0.8 10 16; r 267 4 4 4 0.8 10 16; r 259 8 4 2 0.8 10 32; r 267 8 4 2 0.8 10 8; r 263 4 4 4 0.8 10 16
echo "== END $(date)"
