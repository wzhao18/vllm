#!/bin/bash
# ncu of the selective path kernels (IndexerScore sample + PUBLISH, CandSelect) at the P3-16 shape.
# Usage: run_pidx_ncu.sh <out prefix>
O=$1; mkdir -p $(dirname $O)
export CUTE_DSL_LINEINFO=1 CUTE_DSL_CACHE_DIR=$(dirname $O)/cute-lineinfo CUTE_DSL_ARCH=sm_107a
ncu --set full --import-source yes -k regex:"IndexerScore|CandSelect" -s 3 -c 3 -f -o $O python3 benchmarks/kimi_k3/bench_pidx_fused.py --ncu 2>&1 | grep -v "jit-cache" | tail -5
for s in "--section SpeedOfLight" "--section WarpStateStats" "--section SchedulerStats" "--section LaunchStats"; do ncu -i $O.ncu-repz --page details $s 2>&1 | grep -v "^\s*$\|----" | grep -v "OPT \|INF \|   [A-Z][a-z].*\.\s*$" | cut -c1-150; done
ncu -i $O.ncu-repz --page source --csv --print-source sass -k regex:IndexerScore --launch-skip 1 > $O.sass.csv 2>/dev/null; wc -l $O.sass.csv
