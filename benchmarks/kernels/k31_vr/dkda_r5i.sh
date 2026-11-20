#!/bin/bash
# D-KDA round 5 (i): on-SM store cost vs HBM write traffic - KdaCore v2 with every TMA
# store box sent to ONE page (L2-resident: same TMA/smem work, no HBM write stream);
# KdaCoreQ + KdaCore fold tests vs stock (graph variant rebinding the state map).
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
T=tests/models/kimi_k3/test_specialized_decode_sp_kda_flow_gpu.py
C=/tmp/dkda-r5i-$$
for sk in "" l2store; do
  echo "-- core skip=[$sk] striped"
  CUTE_DSL_CACHE_DIR=$C-$sk timeout 900 python3 $B --n 96 128 --kinds core --gap 1 --pending 4 --place striped --mx ${sk:+--skip $sk} 2>&1 | grep -E "us/layer|rror|FAILED" | cut -c1-120
done
echo "== fold tests vs stock (core, coreq Q3)"
CUTE_DSL_CACHE_DIR=$C-t timeout 1500 python3 -m pytest --noconftest -q -rf $T -k "fold_matches and (core or coreq) and (33 or 96)" 2>&1 | grep -E "passed|failed|^FAILED|rror:|Mismatch|differ" | cut -c1-250 | head -10
