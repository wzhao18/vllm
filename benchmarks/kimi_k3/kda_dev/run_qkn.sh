#!/bin/bash
# P-KDA2 QKN checks: producer norm in qkv_conv, fip QKN_ route on prenormalized q/k, persistent tests
cd /k31ov
export CUTE_DSL_CACHE_DIR=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-KDA2/cutecache
export VLLM_CACHE_ROOT=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-KDA2/vllmcache
echo "QKN start $(date +%T)"
timeout 900 python3 -u benchmarks/kimi_k3/kda_dev/qkn_conv_check.py 2>&1 | grep -E "qkn_conv|Error|error" | head -20
timeout 900 python3 -u benchmarks/kimi_k3/bench_fip_dev.py --prenorm --rounds 4 --lens 5461,5461,5462 8192,8192 --variants "{}" "{\"QKN_\": 1}" 2>&1 | grep -E "lens=|Error|error|Traceback"
timeout 900 python3 -m pytest --noconftest -q -x tests/models/kimi_k3/test_kda_frost_prefill.py -k persistent 2>&1 | tail -2
echo "QKN done $(date +%T)"
