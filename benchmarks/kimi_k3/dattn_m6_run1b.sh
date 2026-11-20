#!/bin/bash
# dattn_m6_run1a.sh <outdir> <srcdir>: D-ATTN M6-1a (v2: indices staged in smem, one tile of loads ahead): MMA-consumable SW128 KV tiles via coalesced LDG + st.shared
# (dattn_skel3_ldg.cu) vs the gather4 SW128 form; union/per-row, cold/warm, R384 (+R256/R128 best); one GPU.
O=$1; D=$2; mkdir -p $O
echo "== host $(hostname) gpus=$CUDA_VISIBLE_DEVICES $(date)"
NV="/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a"
$NV -Xptxas -v -o $O/skel3 $D/dattn_skel3_ldg.cu > $O/build.log 2>&1 || { grep -m20 -i -A2 error $O/build.log; exit 1; }
grep -E "registers|spill" $O/build.log | sed -E 's/ptxas info *: //' | sort -u | head -6
r() { timeout 120 $O/skel3 "$@" | grep -E "SKEL3|error|bad" | sed -E "s/^SKEL3 //" | cut -c1-200 || echo "case [$*] rc=$?"; }
for M in 1 5 4 0; do echo "-- MODE $M"; r $M 16 8 2; r $M 16 8 3; done
echo "-- R256 / R128 union cold, per-row warm"; for N in 64 32; do r 1 16 8 2 0.8 10 $N; r 4 16 8 2 0.8 10 $N; done
echo "== END $(date)"
