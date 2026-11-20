#!/bin/bash
# D-KDA v10a: KdaRing v10 (v5 + 2 slots + checkpoint via smem store buffer + per-warp bulk stores).
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
T=tests/models/kimi_k3/test_specialized_decode_sp_kda_flow_gpu.py
export CUTE_DSL_CACHE_DIR=/tmp/dkda-v10a-$$
echo "== tests n33"
timeout 900 python3 -m pytest --noconftest -q -rf $T -k "(ring_is_bitwise and 33) or (fold_matches and ring and 33)" 2>&1 | grep -E "passed|failed|^FAILED|Error|differ" | cut -c1-200 | head -6
for sk in "" prep store; do echo "== skip=[$sk]"; timeout 900 python3 $B --n 32 64 96 128 --kinds ring --gap 1 --pending 4 --place striped local --mx ${sk:+--skip $sk} 2>&1 | grep -E "us/layer|rror" | cut -c1-70; done
