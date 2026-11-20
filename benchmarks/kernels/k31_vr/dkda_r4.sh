#!/bin/bash
# D-KDA R4: checkpoint store cache policy: default st.global.v4 vs st.global.cs.v4 (stcs).
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
export CUTE_DSL_CACHE_DIR=/tmp/dkda-r4-$$
for sk in prepk prepk,stcs prepk,rfin,rred prepk,rfin,rred,stcs "" stcs; do echo "== skip=[$sk]"; timeout 600 python3 $B --n 32 96 --kinds ring --gap 1 --pending 4 --place local --mx ${sk:+--skip $sk} 2>&1 | grep -E "us/layer|rror" | cut -c1-60; done
