#!/bin/bash
# dattn_skel2_run3.sh <outdir> <src.cu>: D-ATTN M3 root cause: issue/consume traces, static no-consumer, spin-wait issuers; one GPU.
O=$1; SRC=$2; mkdir -p $O
echo "== host $(hostname) gpus=$CUDA_VISIBLE_DEVICES $(date)"
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a -Xptxas -v -o $O/skel2 $SRC -lcuda > $O/build.log 2>&1 || { grep -m20 -i -A2 error $O/build.log; exit 1; }
r() { timeout 120 $O/skel2 "$@" | grep -E "SKEL2|TRACE|ISSUE|MISMATCH|error|QMAX|bad" | cut -c1-900 || echo "case [$*] rc=$?"; }
echo "-- M2 ref + static no-consumer (bit6)"; r 1 8 4 2; r 67 8 4 2; r 83 8 4 2; r 71 8 4 2; r 87 8 4 2
echo "-- M3 traced in-order / OOO, cold+warm"; r 19 8 4 2 0.8 10 32; r 27 8 4 2 0.8 10 32; r 23 8 4 2 0.8 10 32; r 31 8 4 2 0.8 10 8
echo "-- spin-wait issuers (bit7)"; r 131 8 4 2 0.8 10 32; r 139 8 4 2 0.8 10 8; r 135 8 4 2 0.8 10 32
echo "-- S1 32 lanes / fewer lanes"; r 65 8 4 1; r 3 8 4 1 0.8 10 32; r 65 4 4 2; r 3 4 4 2 0.8 10 16
echo "== END $(date)"
