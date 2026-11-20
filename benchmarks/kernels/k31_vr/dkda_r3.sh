#!/bin/bash
# D-KDA R3: is the ring core smem-operand bound? core n32/96 with XV operand loads replaced by registers.
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
export CUTE_DSL_CACHE_DIR=/tmp/dkda-r3-$$
for sk in prepk prepk,rxv prepk,rfin,rred prepk,rfin,rred,rxv; do echo "== skip=[$sk]"; timeout 600 python3 $B --n 32 96 --kinds ring --gap 1 --pending 4 --place local --mx --skip $sk 2>&1 | grep -E "us/layer|rror" | cut -c1-60; done
