#!/bin/bash
# D-KDA v14a: KdaRing v14 (v10 + two prep/tail pair warpgroups, NOPS=NSTAGE=3 with stage_empty).
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
T=tests/models/kimi_k3/test_specialized_decode_sp_kda_flow_gpu.py
export CUTE_DSL_CACHE_DIR=/tmp/dkda-v14a-$$
for sk in "" prep; do echo "== skip=[$sk]"; timeout 900 python3 $B --n 32 64 96 128 --kinds ring --gap 1 --pending 4 --place striped local --mx ${sk:+--skip $sk} 2>&1 | grep -E "us/layer|rror" | cut -c1-70; done
