#!/bin/bash
T=tests/models/kimi_k3/test_specialized_decode_sp_kda_flow_gpu.py
export CUTE_DSL_CACHE_DIR=/tmp/dkda-r6-$$
timeout 900 python3 -m pytest --noconftest -q -x $T -k "ring_is_bitwise and 33 and mxfp8" 2>&1 | grep -E "^E |Error|passed|failed" | cut -c1-220 | head -14
timeout 900 python3 -m pytest --noconftest -q -rf $T -k "ring and not bitwise" 2>&1 | grep -E "^FAILED|passed|failed" | cut -c1-200 | head -12
