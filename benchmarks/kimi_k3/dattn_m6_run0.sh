#!/bin/bash
# dattn_m6_run0.sh <outdir> <srcdir>: D-ATTN M6-0 data floors for the M6 design choice: per-row vs request-union gather,
# cold vs L2-warm (warm = L2-prefetch proxy), whole-row vs SW128 MMA form, R384/R256/R128; one GPU.
O=$1; D=$2; mkdir -p $O
echo "== host $(hostname) gpus=$CUDA_VISIBLE_DEVICES $(date)"
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a -o $O/skel2 $D/dattn_skel2_mla.cu -lcuda > $O/build.log 2>&1 || { grep -m20 -i -A2 error $O/build.log; exit 1; }
r() { timeout 120 $O/skel2 "$@" | grep -E "SKEL2|error|QMAX|bad" | sed -E "s/^SKEL2 //" | cut -c1-230 || echo "case [$*] rc=$?"; }
for N in 96 64 32; do
  echo "-- NREQ $N: union cold / warm, per-row cold / warm"
  r 1 4 4 4 0.8 10 16 $N; r 5 4 4 4 0.8 10 16 $N; r 0 4 4 4 0.8 10 16 $N; r 4 4 4 4 0.8 10 16 $N
done
echo "-- SW128 MMA form R384: union cold, per-row warm"
r 33 4 4 4 0.8 10 16 96; r 36 4 4 4 0.8 10 16 96
echo "== END $(date)"
