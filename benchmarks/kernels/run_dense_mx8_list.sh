#!/bin/bash
# DenseMx8 variants from one comma/space separated list, one process each (a fault poisons the context).
B=/k31ov/benchmarks/kernels/bench_kimi_k3_sdec_dense_mx8.py
for R in ${ROWS:-384}; do for v in $(echo "$*" | tr ',' ' '); do
  echo "=== R$R $v"
  timeout 150 python3 $B --rows $R --copies 4 --variants $v 2>&1 | grep -E "^  |^R|rel|FAILED|Error" | grep -v "Search for" | head -8
done; done
echo DONE
