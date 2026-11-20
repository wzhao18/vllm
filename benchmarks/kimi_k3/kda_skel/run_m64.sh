#!/bin/bash
# P-KDA2: tcgen05 M64 vs M128 rate probe (d_v-split 2-chain decision). usage: run_m64.sh <srcdir> <outdir>
S=$1; O=$2; mkdir -p $O
NVCC=$(command -v nvcc || ls /usr/local/cuda/bin/nvcc)
$NVCC -O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a -o $O/mma_rate2 $S/mma_rate2.cu > $O/buildr.log 2>&1 || { echo BUILDR_FAIL; grep -m5 error $O/buildr.log; exit 1; }
timeout 120 $O/mma_rate2 | grep RATE
