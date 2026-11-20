#!/bin/bash
# P-KDA2: conv + persistent KDA prefill unit tests
cd /k31ov
export CUTE_DSL_CACHE_DIR=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-KDA2/cutecache
export VLLM_CACHE_ROOT=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-KDA2/vllmcache
echo "QKT start $(date +%T)"
timeout 900 python3 -m pytest --noconftest -q tests/models/kimi_k3/test_kda_qkv_conv.py 2>&1 | tail -3
timeout 900 python3 -m pytest --noconftest -q tests/models/kimi_k3/test_kda_frost_prefill.py -k persistent 2>&1 | tail -3
echo "QKT done $(date +%T)"
