#!/bin/bash
# ncu of the PUBLISH score kernel (+ CandSelect) at the P3-16 shape: SOL, stalls, pipes, hot SASS.
# Usage: run_pidx_ncu2.sh <out prefix>
O=$1; mkdir -p $(dirname $O)
export CUTE_DSL_LINEINFO=1 CUTE_DSL_CACHE_DIR=$(dirname $O)/cute-lineinfo CUTE_DSL_ARCH=sm_107a
ncu --set full --import-source yes -k regex:"IndexerScore|CandSelect" -s 3 -c 3 -f -o $O \
  python3 benchmarks/kimi_k3/bench_pidx_fused.py --ncu 2>&1 | grep -v "jit-cache" | tail -3
for s in SpeedOfLight WarpStateStats SchedulerStats ComputeWorkloadAnalysis; do
  ncu -i $O.ncu-repz --page details --section $s -k regex:IndexerScore 2>&1 \
    | grep -v "^\s*$\|----" | grep -v "OPT \|INF \|   [A-Z][a-z].*\.\s*$" | cut -c1-120 | head -40
done
ncu -i $O.ncu-repz --page raw -k regex:IndexerScore --csv \
  --metrics regex:sm__inst_executed_pipe_.*_pct_of_peak_sustained_active$,regex:smsp__pcsamp_warps_issue_stalled_.*[a-z]$ \
  2>/dev/null > $O.raw.csv
python3 benchmarks/kimi_k3/ncu_top_metrics.py $O.raw.csv 30
ncu -i $O.ncu-repz --page source --csv --print-source sass -k regex:IndexerScore --launch-skip 1 \
  > $O.sass.csv 2>/dev/null; wc -l $O.sass.csv
