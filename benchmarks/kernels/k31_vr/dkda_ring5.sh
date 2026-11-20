#!/bin/bash
# D-KDA M3 split: KdaRing prep kernel alone / ring core alone / both (mx, die-local), KdaFlow reference, + resource usage.
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
export CUTE_DSL_CACHE_DIR=/tmp/dkda-ring5
export CUTE_DSL_KEEP_CUBIN=1
for sk in "" prepk core; do
  echo "== skip=[$sk]"
  timeout 600 python3 $B --n 32 64 96 --kinds ring --gap 1 --pending 4 --place local --mx ${sk:+--skip $sk} 2>&1 | grep -E "us/layer|rror" | cut -c1-90
done
echo "== flow ref"
timeout 600 python3 $B --n 32 64 96 --kinds flow --gap 1 --pending 4 --place local --mx 2>&1 | grep -E "us/layer|rror" | cut -c1-90
echo "== cache files"
find /tmp/dkda-ring5 -type f | head -5
for f in $(find /tmp/dkda-ring5 /tmp -maxdepth 3 -name '*.cubin' 2>/dev/null | head -8); do cuobjdump -res-usage $f 2>/dev/null | grep -E "Function|REG" | sed 's/Function \(.\{60\}\).*/F \1/' | cut -c1-150 | head -6; done
