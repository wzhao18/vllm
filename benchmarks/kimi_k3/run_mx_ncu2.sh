#!/bin/bash
# D-GEMM: ncu sections (SOL, memory tables, warp stalls, compute pipes) for vendored sm107
# MXFP8 tactics, one cold launch per arm; full CSV per arm under OUT, filtered summary on stdout.
# usage: run_mx_ncu2.sh OUT N K ROWS ARM [ARM ...]
OUT=$1; N=$2; K=$3; R=$4; shift 4
mkdir -p "$OUT"
for A in "$@"; do
  F="$OUT/ncu_${R}_$(echo "$A" | tr ':x' '_-').csv"
  ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Tables \
    --section WarpStateStats --section ComputeWorkloadAnalysis --section SchedulerStats \
    -k regex:Sm107 --launch-skip 6 --launch-count 1 --csv --page details \
    python3 benchmarks/kimi_k3/bench_mx_ncu.py --n $N --k $K --rows $R --arm $A > "$F" 2>&1
  echo "=== $A ($F)"
  grep -E '^"[0-9]' "$F" | awk -F'","' '{print $(NF-3)": "$(NF-1)" "$(NF-2)}' | tr -d '"' \
    | grep -E -i 'Duration|Throughput|Busy|Stall|Pipe|Shared|Bank|Cycles Active|Issue|Elapsed Cycles|Hit Rate' | head -45
done
