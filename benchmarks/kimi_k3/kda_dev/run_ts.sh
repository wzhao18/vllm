#!/bin/bash
cd /k31ov
export CUTE_DSL_CACHE_DIR=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-KDA2/cutecache
for c in 2720,2720,2752 2731,2731,2730; do timeout 600 python3 -u benchmarks/kimi_k3/kda_dev/check_timesplit.py 5461,5461,5462 $c 2>&1 | grep -E "timesplit|Error|Traceback" ; done
timeout 600 python3 -u benchmarks/kimi_k3/kda_dev/check_timesplit.py 16384 8192 2>&1 | grep -E "timesplit|Error|Traceback"
