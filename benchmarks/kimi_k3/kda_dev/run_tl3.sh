#!/bin/bash
cd /k31ov
export CUTE_DSL_CACHE_DIR=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-KDA2/cutecache
for X in "{}" "{\"RAWG_\": 1}"; do timeout 600 python3 -u benchmarks/kimi_k3/bench_fip_tline.py --t0 10 --t1 30 --extra "$X" 2>&1 | grep -v -E "^W1120|warn" | tail -22; done
echo TL3_DONE
