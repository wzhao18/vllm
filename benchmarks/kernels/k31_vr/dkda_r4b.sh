#!/bin/bash
# D-KDA round 4 (b): KdaFlowP (prep warpgroup + two scan groups) vs KdaFlow v18:
# bitwise tests, then striped mx --gap 1 --pending 4 timing n32/64/96/128.
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
T=tests/models/kimi_k3/test_specialized_decode_sp_kda_flow_gpu.py
export CUTE_DSL_CACHE_DIR=/tmp/dkda-r4b-$$
echo "== test flowp bitwise vs flow"
timeout 900 python3 -m pytest --noconftest -q -rf -x $T -k "core_is_bitwise and flowp and (3 or 33)" 2>&1 | grep -E "passed|failed|^FAILED|Error|rror:|Mismatch|assert" | cut -c1-300 | head -12
echo "== timing"
for k in flow flowp; do
  timeout 900 python3 $B --n 32 64 96 128 --kinds $k --gap 1 --pending 4 --place striped --mx 2>&1 | grep -E "us/layer|rror|FAILED" | cut -c1-200
done
echo "== flowp skip prep,gate"
timeout 900 python3 $B --n 32 96 --kinds flowp --gap 1 --pending 4 --place striped --mx --skip prep,gate 2>&1 | grep -E "us/layer|rror|FAILED" | cut -c1-200
