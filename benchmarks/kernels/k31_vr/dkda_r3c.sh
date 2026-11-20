#!/bin/bash
# D-KDA round 3 (c): does KdaFlow's slot ring / checkpoint path bind? --slots 3 vs 4
# with prep+gate skipped (+ store skipped), and full; striped mx --gap 1 --pending 4.
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
export CUTE_DSL_CACHE_DIR=/tmp/dkda-r3c-$$
for sl in 3 4; do
  for sk in "" prep,gate prep,gate,store prep,gate,store,scan; do
    echo "== flow slots=$sl skip=[$sk]"
    timeout 900 python3 $B --n 32 96 128 --kinds flow --gap 1 --pending 4 --place striped --mx --slots $sl ${sk:+--skip $sk} 2>&1 | grep -E "us/layer|rror" | cut -c1-62
  done
done
