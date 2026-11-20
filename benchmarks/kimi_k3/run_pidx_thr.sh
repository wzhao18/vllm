#!/bin/bash
# Threshold sampling A/B (SAMPLES, MARGIN) of the selective prefill indexer at P3-16. Usage: run_pidx_thr.sh "S:M"...
for c in "$@"; do
  echo "== samples ${c%%:*} margin ${c##*:}"
  python3 benchmarks/kimi_k3/bench_pidx_fused.py --sel --iters 10 --samples ${c%%:*} --margin ${c##*:} 2>&1 | grep -E "check_sel|perf_sel|Error|Traceback" | cut -c1-1500
done
