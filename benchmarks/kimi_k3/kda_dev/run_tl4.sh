#!/bin/bash
# P-KDA2 timeline of CTA-0 for given variant JSONs: bash run_tl4.sh <json>...
cd /k31ov
export CUTE_DSL_CACHE_DIR=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-KDA2/cutecache
for X in "$@"; do timeout 600 python3 -u benchmarks/kimi_k3/bench_fip_tline.py --t0 10 --t1 40 --extra "$X" 2>&1 | grep -v -E "^W1120|warn" | tail -33; done
echo TL4_DONE
