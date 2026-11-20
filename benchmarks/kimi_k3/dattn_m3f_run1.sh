#!/bin/bash
# dattn_m3f_run1.sh <outdir> <srcdir>: D-ATTN M3f die-homed KV pages (skel2 MODE bit 1024) vs striped, R384 union (1 GPU)
O=$1; D=$2; mkdir -p $O
echo "== host $(hostname) gpus=$CUDA_VISIBLE_DEVICES $(date)"
NV="/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a"
$NV -o $O/skel2 $D/dattn_skel2_mla.cu -lcuda > $O/build2.log 2>&1 || { grep -m12 -i -A2 error $O/build2.log; exit 1; }
r() { CUDA_VISIBLE_DEVICES=0 timeout 120 $O/skel2 "$@" | grep -E "SKEL2|BRK|DIE|MISMATCH|rror|bad|QMAX" | sed -E "s/rows [0-9]+ .keys [0-9]+, //" | cut -c1-230 || echo "case [$*] rc=$?"; }
for c in "4 4 4" "8 4 2" "16 2 4"; do
  r 1 $c 0.8 10; r 1025 $c 0.8 10; r 5 $c 0.8 10; r 1029 $c 0.8 10
done
echo "-- breakdown stamps striped / die"
r 513 4 4 4 0.8 10 16; r 1537 4 4 4 0.8 10 16
echo "-- coupled (M3e form) striped / die"
r 267 4 4 4 0.8 10 16; r 1291 4 4 4 0.8 10 16
echo "-- compute-chain MMA skeleton (M6a-1 repeat)"
[ -f $D/dattn_skel_cc.cu ] && $NV -o $O/skel_cc $D/dattn_skel_cc.cu > $O/build_cc.log 2>&1 && CUDA_VISIBLE_DEVICES=0 timeout 60 $O/skel_cc 256 | grep -E "^CC|ERR"
echo "== END $(date)"
