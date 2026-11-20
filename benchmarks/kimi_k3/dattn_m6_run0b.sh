#!/bin/bash
# dattn_m6_run0b.sh <outdir> <srcdir>: D-ATTN M6-0b: MMA-consumable SW128 gather form (5 ops per quad) vs issuing lanes:
# does more issuers (op concurrency) bring SW128 to the whole-row rate? R384, one GPU.
O=$1; D=$2; mkdir -p $O
echo "== host $(hostname) gpus=$CUDA_VISIBLE_DEVICES $(date)"
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a -o $O/skel2 $D/dattn_skel2_mla.cu -lcuda > $O/build.log 2>&1 || { grep -m20 -i -A2 error $O/build.log; exit 1; }
r() { timeout 120 $O/skel2 "$@" | grep -E "SKEL2|error|QMAX|bad" | sed -E "s/^SKEL2 //; s/cfrac=0.80 //; s/rows [0-9]+ .keys [0-9]+, //" | cut -c1-200 || echo "case [$*] rc=$?"; }
echo "-- whole-row refs: union cold, per-row warm"; r 1 4 4 4 0.8 10 16; r 4 4 4 4 0.8 10 16
for M in 33 37 36 32; do
  echo "-- SW128 MODE $M"; r $M 4 4 4 0.8 10 16; r $M 8 4 2 0.8 10 16; r $M 4 8 2 0.8 10 16; r $M 8 8 1 0.8 10 16; r $M 16 4 1 0.8 10 16
done
echo "== END $(date)"
