#!/bin/bash
# P-KDA2: GSWZ A/B + CTA-0 timeline
cd /k31ov
export CUTE_DSL_CACHE_DIR=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-KDA2/cutecache
echo "TL start $(date +%T)"
timeout 900 python3 -u benchmarks/kimi_k3/bench_fip_tline.py --t0 6 --t1 34 2>&1 | grep -v -i warn | tail -34
timeout 900 python3 -u benchmarks/kimi_k3/bench_fip_dev.py --rounds 4 --lens 5461,5461,5462 8192,8192 100,37,3000,1,64,33 --variants "{}" "{\"GSWZ_\": 1}" 2>&1 | grep --line-buffered -E "lens=|Error|error|Traceback"
echo "TL done $(date +%T)"
