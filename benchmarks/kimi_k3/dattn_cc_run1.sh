#!/bin/bash
# dattn_cc_run1.sh <outdir> <srcdir>: D-ATTN M6a step 1 compute-chain MMA skeleton (1 GPU)
O=$1; D=$2; mkdir -p $O
echo "== host $(hostname) gpus=$CUDA_VISIBLE_DEVICES $(date)"
NV="/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a"
$NV -o $O/skel_cc $D/dattn_skel_cc.cu > $O/build_cc.log 2>&1 || { grep -m12 -i -A2 error $O/build_cc.log; exit 1; }
CUDA_VISIBLE_DEVICES=0 timeout 120 $O/skel_cc 256 2>&1 | tail -12
CUDA_VISIBLE_DEVICES=0 timeout 120 $O/skel_cc 64 2>&1 | grep -E "^CC" | cut -c1-60,61-130
echo "== END $(date)"
