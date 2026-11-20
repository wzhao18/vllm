#!/bin/bash
# D-KDA: KdaRing cycle trace at n96 (CTAs 0/53), place ${1:-striped}; prints errors if the trace build fails.
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
export CUTE_DSL_CACHE_DIR=/tmp/dkda-tr-$$
O=/tmp/dkda-tr-$$.out
timeout 600 python3 $B --n ${2:-96} --kinds ring --gap 1 --pending 4 --place ${1:-striped} --mx --trace --iters 1 --layers 2 ${3:+--skip $3} > $O 2>&1
echo "rc=$? lines=$(wc -l < $O)"
for k in pro prep tail scan; do grep "RTRACE $k" $O | sort -t' ' -k4,4n -s | awk '!seen[$4]++' | cut -c1-260 | head -2; done
grep -vE "RTRACE" $O | grep -E "rror|Traceback|Exception|us/layer" | cut -c1-240 | head -8
