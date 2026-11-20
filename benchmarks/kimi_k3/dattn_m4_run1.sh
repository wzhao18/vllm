#!/bin/bash
# dattn_m4_run1.sh <outdir> <srcdir>: D-ATTN M4 (DcpValue pull skeleton, 4 GPUs) + M5b (indexer issuers spread over
# warps, 1 GPU) + M3f (sparse MLA gather fixed-term breakdown stamps, 1 GPU). Needs 4 GPUs.
O=$1; D=$2; mkdir -p $O
echo "== host $(hostname) gpus=$CUDA_VISIBLE_DEVICES $(date)"
NV="/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a"
$NV -o $O/skel_dv $D/dattn_skel_dv.cu -lcuda > $O/build_dv.log 2>&1 || grep -m12 -i -A2 error $O/build_dv.log
$NV -o $O/skel_idx $D/dattn_skel_idx.cu -lcuda > $O/build_idx.log 2>&1 || grep -m12 -i -A2 error $O/build_idx.log
$NV -o $O/skel2 $D/dattn_skel2_mla.cu -lcuda > $O/build2.log 2>&1 || grep -m12 -i -A2 error $O/build2.log
echo "-- M3f breakdown (1 GPU)"
r() { CUDA_VISIBLE_DEVICES=0 timeout 120 $O/skel2 "$@" | grep -E "SKEL2|BRK|MISMATCH|error|bad" | sed -E "s/rows [0-9]+ .keys [0-9]+, 0.45x. 50.2 MB //" | cut -c1-220 || echo "case [$*] rc=$?"; }
r 513 4 4 4 0.8 10 16; r 513 4 4 2 0.8 10 16; r 517 4 4 4 0.8 10 16; r 512 4 4 4 0.8 10 16
echo "-- M5b indexer issuers per warp (1 GPU): NREQ NBLK TH D iters LPW"
for c in "96 516 12 4 6 32" "96 516 512 2 6 1" "96 516 512 1 6 2" "96 516 256 2 6 2" "96 516 384 3 6 1" "96 516 256 4 6 1" "96 516 512 1 6 1"; do
  CUDA_VISIBLE_DEVICES=0 timeout 120 $O/skel_idx $c | grep -E "SKEL-IDX|SPAN" | cut -c1-200 || echo "idx [$c] rc=$?"; done
echo "-- M4 DcpValue skeleton (4 GPUs)"
timeout 900 $O/skel_dv v 2>&1 | grep -E "^V |sms|error|ERR|TIMEOUT|DONE" | cut -c1-200
echo "== END $(date)"
