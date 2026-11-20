#!/bin/bash
# D-KDA round 5 (j): KdaCore movement-only arms (scan math skipped) with and without the
# checkpoint stores, striped and die-localized memory (static claim), n96/n128 mx.
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
C=/tmp/dkda-r5j-$$
for sk in scan scan,store; do
  echo "-- core skip=[$sk] striped"
  CUTE_DSL_CACHE_DIR=$C-$sk timeout 900 python3 $B --n 96 128 --kinds core --gap 1 --pending 4 --place striped --mx --skip $sk 2>&1 | grep -E "us/layer|rror|FAILED" | cut -c1-120
done
