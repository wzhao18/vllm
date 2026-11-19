#!/bin/bash
# ncu of the candidate select kernel at the P3-16 shape. Usage: run_pidx_ncu_cs.sh <out prefix>
O=$1; mkdir -p $(dirname $O)
export CUTE_DSL_LINEINFO=1 CUTE_DSL_CACHE_DIR=$(dirname $O)/cute-lineinfo-cs CUTE_DSL_ARCH=sm_107a
ncu --set full --import-source yes -k regex:CandSelect -s 1 -c 1 -f -o $O \
  python3 benchmarks/kimi_k3/bench_pidx_fused.py --ncu 2>&1 | grep -v "jit-cache" | tail -2
for s in SpeedOfLight Occupancy LaunchStats WarpStateStats SchedulerStats; do
  ncu -i $O.ncu-repz --page details --section $s 2>&1 \
    | grep -v "^\s*$\|----" | grep -v "OPT \|INF \|   [A-Z][a-z].*\.\s*$" | cut -c1-110 | head -24
done
ncu -i $O.ncu-repz --page source --csv --print-source sass > $O.sass.csv 2>/dev/null; wc -l $O.sass.csv
