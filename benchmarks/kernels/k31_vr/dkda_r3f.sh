#!/bin/bash
# D-KDA round 3 (f): die-local vs striped once prep + gate are off KdaFlow's chain
# (production store path), and the same with the checkpoint store skipped.
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
export CUTE_DSL_CACHE_DIR=/tmp/dkda-r3f-$$
for sk in prep,gate prep,gate,store; do
  echo "== flow skip=[$sk]"
  timeout 900 python3 $B --n 32 96 128 --kinds flow --gap 1 --pending 4 --place striped local --mx --skip $sk 2>&1 | grep -E "us/layer|rror" | cut -c1-62
done
