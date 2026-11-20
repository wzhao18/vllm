#!/bin/bash
# D-KDA R1: KdaRing core with register-staged state + early quarter release. bitwise test, bench, regs.
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
T=tests/models/kimi_k3/test_specialized_decode_sp_kda_flow_gpu.py
D=/tmp/dkda-dump-$$; mkdir -p $D
export CUTE_DSL_CACHE_DIR=/tmp/dkda-c-$$ CUTE_DSL_KEEP_CUBIN=1 CUTE_DSL_DUMP_DIR=$D
echo "== test"; timeout 900 python3 -m pytest --noconftest -q -rf -x $T -k "ring_is_bitwise" 2>&1 | grep -E "passed|failed|^FAILED|Error|Mismatch|assert" | cut -c1-200 | head -8
echo "== res"; for f in $D/*.cubin; do cuobjdump -res-usage $f 2>/dev/null | grep -iE "Function|REG" | sed -E "s/Function ([^:]{0,50}).*:/F \1/" | cut -c1-160 | head -4; done | sort -u | head -12
unset CUTE_DSL_KEEP_CUBIN CUTE_DSL_DUMP_DIR
for sk in "" prepk; do echo "== skip=[$sk]"; timeout 600 python3 $B --n 32 64 96 128 --kinds ring --gap 1 --pending 4 --place local --mx ${sk:+--skip $sk} 2>&1 | grep -E "us/layer|rror" | cut -c1-60; done
echo "== core no-store"; timeout 600 python3 $B --n 96 --kinds ring --gap 1 --pending 4 --place local --mx --skip prepk,rst 2>&1 | grep -E "us/layer|rror" | cut -c1-60
