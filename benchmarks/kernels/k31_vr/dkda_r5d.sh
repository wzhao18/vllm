#!/bin/bash
# D-KDA round 5 (d): KdaCore v2 = per-warp sub-block state ring (3 x 8 KiB, 2D TMA) +
# store ring: bitwise tests vs KdaFlow, timing (full / no store) striped + local mx.
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
T=tests/models/kimi_k3/test_specialized_decode_sp_kda_flow_gpu.py
C=/tmp/dkda-r5d-$$
echo "== tests"
CUTE_DSL_CACHE_DIR=$C-t timeout 1500 python3 -m pytest --noconftest -q -rf -x $T -k "core_is_bitwise and core" 2>&1 | grep -E "passed|failed|^FAILED|rror:|Mismatch|assert|differ" | cut -c1-300 | head -12
for pl in striped local; do
  for sk in "" store; do
    echo "-- core skip=[$sk] $pl"
    CUTE_DSL_CACHE_DIR=$C-$sk timeout 900 python3 $B --n 32 96 128 --kinds core --gap 1 --pending 4 --place $pl --mx ${sk:+--skip $sk} 2>&1 | grep -E "us/layer|rror|FAILED" | cut -c1-120
  done
done
