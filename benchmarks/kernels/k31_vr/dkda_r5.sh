#!/bin/bash
# D-KDA R5: prep kernel occupancy (min blocks per SM caps registers) alone and in the full ring.
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
export CUTE_DSL_CACHE_DIR=/tmp/dkda-r5-$$
for sk in core core,pm3 core,pm4 core,pm6 core,pm8 pm4 pm8; do echo "== skip=[$sk]"; timeout 600 python3 $B --n 32 96 --kinds ring --gap 1 --pending 4 --place local --mx --skip $sk 2>&1 | grep -E "us/layer|rror" | cut -c1-60; done
echo "== test"; T=tests/models/kimi_k3/test_specialized_decode_sp_kda_flow_gpu.py; timeout 900 python3 -m pytest --noconftest -q $T -k "ring and (33 or 96 or die or eager or graph)" 2>&1 | grep -E "^E  .*(Error|assert)|passed|failed" | cut -c1-160 | head -6
