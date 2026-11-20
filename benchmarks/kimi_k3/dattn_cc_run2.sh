#!/bin/bash
# dattn_cc_run2.sh <outdir> <srcdir>: harness check: hwdb mma2cta.cu vs dattn_skel_cc.cu + SASS local-memory check
O=$1; D=$2; mkdir -p $O
NV="/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a"
$NV -o $O/mma2cta $D/mma2cta.cu > $O/b1.log 2>&1 || grep -m5 error $O/b1.log
$NV -Xptxas -v -o $O/skel_cc $D/dattn_skel_cc.cu > $O/b2.log 2>&1; grep -E "registers|spill|lmem" $O/b2.log | head -4
cuobjdump -sass $O/skel_cc | grep -cE "LDL|STL" ; cuobjdump -sass $O/skel_cc | grep -E "UTCQMMA|UTCHMMA|UTCMMA|UTC.*MMA" | head -3
CUDA_VISIBLE_DEVICES=0 timeout 120 $O/mma2cta | cut -c1-150
CUDA_VISIBLE_DEVICES=0 timeout 120 $O/skel_cc 1024 | grep -E "ref|km  |tsr" | cut -c1-120
echo "== END $(date)"
