#!/bin/bash
# ncu of one selective-path kernel at the P3-16 shape (inside the image, cwd = overlay root):
# key sections + SASS opcode mix. Usage: run_pidx_ncu_k.sh <out prefix> [kernel regex] [skip]
O=$1; K=${2:-CandSelect}; S=${3:-1}; mkdir -p $(dirname $O)
export CUTE_DSL_LINEINFO=1 CUTE_DSL_ARCH=sm_107a
ncu --set full --import-source yes -k regex:$K -s $S -c 1 -f -o $O \
  python3 benchmarks/kimi_k3/bench_pidx_fused.py --ncu 2>&1 | grep -v "jit-cache" | tail -2
R=$(ls $O.ncu-rep* | head -1)
for s in SpeedOfLight WarpStateStats SchedulerStats Occupancy; do
  ncu -i $R --page details --section $s 2>&1 | grep -E "Duration|Issued Warp|Eligible|Warp Cycles Per Issued|Achieved Occ|Registers|Memory Throughput|Compute .SM. Thr|Active Warps Per" | cut -c1-100 | head -8
done
ncu -i $R --page source --csv --print-source sass > $O.sass.csv 2>/dev/null
python3 benchmarks/kimi_k3/sass_opmix.py $O.sass.csv 22
