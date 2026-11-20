#!/bin/bash
# D-KDA M3 split: KdaRing prep kernel alone / ring core alone / both (mx, die-local), KdaFlow reference.
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
export CUTE_DSL_CACHE_DIR=/tmp/dkda-ring4
for sk in "" prepk core; do
  echo "== skip=[$sk]"
  timeout 600 python3 $B --n 32 64 96 --kinds ring --gap 1 --pending 4 --place local --mx ${sk:+--skip $sk} 2>&1 | grep -E "us/layer|rror" | cut -c1-70
done
echo "== flow ref"
timeout 600 python3 $B --n 32 64 96 --kinds flow --gap 1 --pending 4 --place local --mx 2>&1 | grep -E "us/layer|rror" | cut -c1-70
echo "== ncu prep+core n96"
timeout 900 ncu -k regex:KdaRing -c 4 --section SpeedOfLight --section LaunchStats --section Occupancy python3 $B --n 96 --kinds ring --gap 1 --pending 4 --place local --mx --layers 2 --iters 1 2>&1 | grep -E "KdaRing|Duration|Registers Per|Achieved Occupancy|Theoretical Occupancy|Block Size|Grid Size|Shared Memory Config|Dynamic Shared" | cut -c1-120 | head -40
