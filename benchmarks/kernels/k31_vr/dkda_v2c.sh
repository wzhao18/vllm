#!/bin/bash
# D-KDA v2c: KdaRing v2 cycle trace (per-role phase totals, CTAs 0/53/106/159) at n96 local mx.
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
export CUTE_DSL_CACHE_DIR=/tmp/dkda-v2c-$$
for n in 96; do
  echo "== trace n$n"
  timeout 600 python3 $B --n $n --kinds ring --gap 1 --pending 4 --place local striped --mx --trace --iters 1 --layers 2 > /tmp/dkda-v2c-$$.out 2>&1
  echo "rc=$? lines=$(wc -l < /tmp/dkda-v2c-$$.out)"
  grep -o "RTRACE [a-z]* cta [0-9]*" /tmp/dkda-v2c-$$.out | sort | uniq -c | head -12
  for k in pro scan prep; do grep "RTRACE $k" /tmp/dkda-v2c-$$.out | sort -t' ' -k4,4n -s | awk '!seen[$4]++' | cut -c1-230 | head -4; done
  grep -vE "RTRACE" /tmp/dkda-v2c-$$.out | grep -E "us/layer|rror|Traceback|Exception" | cut -c1-200 | head -6
done
