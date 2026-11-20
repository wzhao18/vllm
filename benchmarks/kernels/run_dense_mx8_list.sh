#!/bin/bash
# DenseMx8 variants from one comma/space separated list, one process each (a fault poisons the context).
# Shape: N (weight rows, default KDA in_proj 12448) x K (default 7168); ROWS token counts.
B=/k31ov/benchmarks/kernels/bench_kimi_k3_sdec_dense_mx8.py
for R in ${ROWS:-384}; do for v in $(echo "$*" | tr ',' ' '); do
  echo "=== N${N:-12448} K${K:-7168} R$R $v"
  timeout 150 python3 $B --rows $R --n ${N:-12448} --k ${K:-7168} --copies ${COPIES:-4} --variants $v 2>&1 | grep -E "^  |^R|rel|FAILED|Error" | grep -v "Search for" | head -8
done; done
echo DONE
