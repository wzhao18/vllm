#!/bin/bash
# DenseMx8 variants, one process each (a fault poisons the context); 4 weight copies (> L2) for timing.
B=/k31ov/benchmarks/kernels/bench_kimi_k3_sdec_dense_mx8.py
for v in "$@"; do
  echo "=== $v"
  timeout 120 python3 $B --rows 384 --copies 4 --variants $v 2>&1 | grep -E "^  |^R|rel|FAILED|Error|DONE" | grep -v "Search for" | head -12
done
