#!/bin/bash
# D-KDA round 5 (b): KdaCore (no prep: operand blocks + KdaFlow scan, XOR butterfly,
# per-warp store ring + TMA tensor stores) - bitwise vs KdaFlow (prep stand-in), then
# timing vs KdaFlow v18 (striped / local, mx, --gap 1 --pending 4).
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
export CUTE_DSL_CACHE_DIR=/tmp/dkda-r5b-$$
echo "== tests"
timeout 1200 python3 -m pytest -x -q tests/models/kimi_k3/test_specialized_decode_sp_kda_flow_gpu.py -k "core_is_bitwise and core" 2>&1 | grep -E "passed|failed|Error|error|assert|Mismatch|differ" | head -20 | cut -c1-200
for pl in striped local; do
  echo "-- core/flow $pl"
  timeout 1200 python3 $B --n 32 96 128 --kinds core flow --gap 1 --pending 4 --place $pl --mx 2>&1 | grep -E "us/layer|rror|FAILED" | cut -c1-120
done
