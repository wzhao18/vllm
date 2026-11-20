#!/bin/bash
# D-KDA v5s2: KdaRing v5 with SLOTS=2 (stream depth check): regs, bitwise test n33, bench, trace n96.
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
T=tests/models/kimi_k3/test_specialized_decode_sp_kda_flow_gpu.py
D=/tmp/dkda-dump-$$; mkdir -p $D
export CUTE_DSL_CACHE_DIR=/tmp/dkda-v5s2-$$ CUTE_DSL_KEEP_CUBIN=1 CUTE_DSL_DUMP_DIR=$D
echo "== bench n96 striped (compile + regs)"
timeout 900 python3 $B --n 96 --kinds ring --gap 1 --pending 4 --place striped --mx --iters 10 2>&1 | grep -E "us/layer|rror|FAILED" | cut -c1-120 | head -4
for f in $D/*.cubin; do cuobjdump -res-usage $f 2>/dev/null | grep -iE "REG" | cut -c1-120 | head -2; done | sort -u | head -4
unset CUTE_DSL_KEEP_CUBIN CUTE_DSL_DUMP_DIR
echo "== bench ring local+striped n32 64 96 128"
timeout 900 python3 $B --n 32 64 96 128 --kinds ring --gap 1 --pending 4 --place striped local --mx 2>&1 | grep -E "us/layer|rror|FAILED" | cut -c1-80
echo "== trace n96 striped"
timeout 600 python3 $B --n 96 --kinds ring --gap 1 --pending 4 --place striped --mx --trace --iters 1 --layers 2 > /tmp/dkda-v5s2-$$.out 2>&1
for k in pro prep tail scan; do grep "RTRACE $k" /tmp/dkda-v5s2-$$.out | sort -t' ' -k4,4n -s | awk '!seen[$4]++' | cut -c1-260 | head -2; done
