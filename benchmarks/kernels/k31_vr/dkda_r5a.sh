#!/bin/bash
# D-KDA round 5 (a): is KdaFlow v18 slot-starved? Page-slot sensitivity (2 vs 3 slots) of
# the full kernel and the prep+gate-off arms (with / without the checkpoint store), striped
# mx --gap 1 --pending 4, n96/n128; plus the S0c setmaxnreg switch probe (no grep filter).
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
export CUTE_DSL_CACHE_DIR=/tmp/dkda-r5a-$$
for sl in 2 3; do
  for sk in "" prep,gate prep,gate,store; do
    echo "-- flow slots=$sl skip=[$sk]"
    timeout 900 python3 $B --n 96 128 --kinds flow --gap 1 --pending 4 --place striped --mx --slots $sl ${sk:+--skip $sk} 2>&1 | grep -E "us/layer|rror|FAILED" | cut -c1-120
  done
done
echo "== S0c switch probe (flowp n32 striped mx)"
timeout 900 python3 $B --n 32 --kinds flowp --gap 1 --pending 4 --place striped --mx --switch --iters 1 --layers 2 2>&1 | grep -E "S0C|rror|us/layer" | sort | uniq -c | sort -rn | head -12 | cut -c1-140
