#!/bin/bash
# Isolation run for DenseMx8: each variant in its own process (a fault poisons the context).
B=/k31ov/benchmarks/kernels/bench_kimi_k3_sdec_dense_mx8.py
for v in "$@"; do
  echo "=== $v"
  timeout 90 python3 $B --rows 384 --copies 1 --variants $v 2>&1 | grep -E "^  |^R|rel|FAILED|Error|DONE" | grep -v "Search for" | head -12
done
if false && command -v compute-sanitizer >/dev/null; then
  echo "=== sanitizer c0s4k1"
  timeout 300 compute-sanitizer --print-limit 5 python3 $B --rows 384 --copies 1 --variants c0s4k1 2>&1 | grep -E "=====|Invalid|Error|at 0x|by thread|Address" | head -30
else
  echo "no compute-sanitizer"
fi
