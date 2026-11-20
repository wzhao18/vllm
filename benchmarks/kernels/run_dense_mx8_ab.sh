#!/bin/bash
# Interleaved A/B of DenseMx8 between two overlay trees on one GPU (per-process
# placement noise is +-15%: compare medians of interleaved processes).
#   OVA=<tree A> OVB=<tree B> REPS=3 N= K= ROWS= COPIES= run_dense_mx8_ab.sh <variant list>
REL=benchmarks/kernels/bench_kimi_k3_sdec_dense_mx8.py
for R in ${ROWS:-384}; do for v in $(echo "$*" | tr ',' ' '); do
  for rep in $(seq ${REPS:-3}); do for side in A B; do
    T=$OVA; [ $side = B ] && T=$OVB
    out=$(cd $T && PYTHONPATH=$T timeout 150 python3 $T/$REL --rows $R --n ${N:-12448} \
          --k ${K:-7168} --copies ${COPIES:-4} --variants $v 2>&1)
    ref=$(echo "$out" | grep -oE "scaled_mm [0-9.]+" | awk '{print $2}')
    t=$(echo "$out" | grep -E "^  $v: [0-9.]+ us/call" | awk '{print $2}')
    ok=$(echo "$out" | grep -oE "mismatch [0-9]+" | head -1)
    echo "AB N${N:-12448} R$R $v $side rep$rep ours $t ref $ref $ok"
  done; done
done; done
echo DONE
