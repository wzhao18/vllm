#!/bin/bash
# D-KDA round 3 (d): KdaFlow checkpoint written back into the slot + one bulk store per
# warp quarter, read-wait + refill deferred past the reduce/solve (R3-e). Tests, timing.
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
T=tests/models/kimi_k3/test_specialized_decode_sp_kda_flow_gpu.py
export CUTE_DSL_CACHE_DIR=/tmp/dkda-r3d-$$
echo "== tests (flow fold vs stock, striped + die-local)"
timeout 900 python3 -m pytest --noconftest -q -rf $T -k "fold_matches and not chain and not ring" 2>&1 | grep -E "passed|failed|^FAILED|Error|differ" | cut -c1-200 | head -8
for sk in "" prep,gate; do
  echo "== flow skip=[$sk]"
  timeout 900 python3 $B --n 32 64 96 128 --kinds flow --gap 1 --pending 4 --place striped local --mx ${sk:+--skip $sk} 2>&1 | grep -E "us/layer|rror" | cut -c1-62
done
