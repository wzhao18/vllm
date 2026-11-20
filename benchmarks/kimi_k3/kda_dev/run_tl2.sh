#!/bin/bash
cd /k31ov
export CUTE_DSL_CACHE_DIR=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-KDA2/cutecache
timeout 900 python3 -u benchmarks/kimi_k3/bench_fip_tline.py --t0 8 --t1 30 2>&1 | grep -v -E "^W1120|warn" | tail -26
echo TL2_DONE
