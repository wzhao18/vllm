#!/bin/bash
# D-KDA round 4 (d): lazy-fold F=2 proxy on production KdaFlow v18: the checkpoint store
# skipped for half the items (slot parity) = the non-flush half; full / halfstore / no
# store, with prep on and with prep+gate off; striped mx --gap 1 --pending 4.
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
export CUTE_DSL_CACHE_DIR=/tmp/dkda-r4d-$$
for sk in "" halfstore store prep,gate prep,gate,halfstore prep,gate,store; do
  echo "-- flow skip=[$sk]"
  timeout 900 python3 $B --n 32 96 128 --kinds flow --gap 1 --pending 4 --place striped --mx ${sk:+--skip $sk} 2>&1 | grep -E "us/layer|rror|FAILED" | cut -c1-120
done
