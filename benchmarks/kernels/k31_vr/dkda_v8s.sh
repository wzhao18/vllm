#!/bin/bash
# D-KDA v8s: KdaRing v8 skip decomposition (striped, mx): full / no prep math / no prep + no scan math / no stores.
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
export CUTE_DSL_CACHE_DIR=/tmp/dkda-v8s-$$
for sk in "" prep prep,scan store prep,store; do echo "== skip=[$sk]"; timeout 600 python3 $B --n 32 96 --kinds ring flow --gap 1 --pending 4 --place striped --mx ${sk:+--skip $sk} 2>&1 | grep -E "us/layer|rror" | cut -c1-70; done
