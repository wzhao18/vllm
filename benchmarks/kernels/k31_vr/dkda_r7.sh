#!/bin/bash
T=tests/models/kimi_k3/test_specialized_decode_sp_kda_flow_gpu.py
export CUTE_DSL_CACHE_DIR=/tmp/dkda-r7-$$
timeout 900 python3 -m pytest --noconftest -q -x $T -k "ring_is_bitwise and 33 and mxfp8" 2>&1 | grep -E "^E |sp.py:[0-9]+|test_spec.*py:[0-9]+|passed|failed" | cut -c1-200 | head -16
