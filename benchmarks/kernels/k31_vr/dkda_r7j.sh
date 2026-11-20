#!/bin/bash
# D-KDA round 7 (j): KdaRelay per-phase clock64 stamps (layer 0, CTAs 0-3, every consumer
# warp and item): where does the per-item time go? Plus untraced timing, same run.
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
export CUTE_DSL_CACHE_DIR=/tmp/dkda-r7j-$$
nvidia-smi --query-gpu=clocks.sm,power.draw,clocks_throttle_reasons.active --format=csv,noheader 2>/dev/null | head -1
timeout 900 python3 $B --gap 1 --pending 4 --mx --place striped --n 32 96 --kinds relay --rtrace 2>&1 | grep -E "us/layer|RTRACE|rror" | sed "s/ (best.*R+W//" | cut -c1-260
timeout 900 python3 $B --gap 1 --pending 4 --mx --place striped --n 32 96 --kinds relay 2>&1 | grep -E "us/layer|rror" | sed "s/ (best.*R+W//" | cut -c1-160
