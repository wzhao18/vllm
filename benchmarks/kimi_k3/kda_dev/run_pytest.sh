#!/bin/bash
cd /k31ov
export CUTE_DSL_CACHE_DIR=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-KDA2/cutecache
timeout 900 python3 -m pytest --noconftest -q -x tests/models/kimi_k3/test_kda_frost_prefill.py -k persistent 2>&1 | tail -3
echo PYT_DONE
