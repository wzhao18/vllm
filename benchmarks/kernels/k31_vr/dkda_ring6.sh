#!/bin/bash
# D-KDA M3 diag: KdaRing core phase marginals (skip flags) + prep gate marginal, n32/96 mx local.
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
export CUTE_DSL_CACHE_DIR=/tmp/dkda-ring6
for sk in prepk prepk,rfin prepk,rfin,rsolve prepk,rfin,rred prepk,rfin,rred,rbar prepk,rfin,rred,rbar,rst prepk,rfin,rred,rbar,rscan core,pgate; do
  echo "== skip=[$sk]"
  timeout 600 python3 $B --n 32 96 --kinds ring --gap 1 --pending 4 --place local --mx --skip $sk 2>&1 | grep -E "us/layer|rror" | cut -c1-60
done
