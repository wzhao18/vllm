#!/bin/bash
# D-KDA round 7 (h): KdaRelay tests (eager + graph, fold_matches vs stock) and the
# microbench vs KdaCore / KdaFlow v18 (MXFP8 outputs, pending 4, --gap 1), striped and
# die-local placement (relay: static head-major ranges). Clock stamped.
T=tests/models/kimi_k3/test_specialized_decode_sp_kda_flow_gpu.py
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
C=/tmp/dkda-r7h-$$
export CUTE_DSL_CACHE_DIR=$C
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw,clocks_throttle_reasons.active --format=csv,noheader 2>/dev/null | head -1
timeout 900 python3 -m pytest --noconftest -q $T -k "relay" 2>&1 | grep -E "^E |passed|failed|FAILED" | cut -c1-200 | tail -12
for place in striped local; do
  timeout 900 python3 $B --gap 1 --pending 4 --mx --place $place --n 32 64 96 128 --kinds relay core 2>&1 | grep -E "us/layer|rror|FAILED" | cut -c1-140
done
nvidia-smi --query-gpu=clocks.sm,power.draw,clocks_throttle_reasons.active --format=csv,noheader 2>/dev/null | head -1
