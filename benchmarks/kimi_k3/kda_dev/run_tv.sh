#!/bin/bash
# P-KDA2 (b base): persistent KDA tests then dev timing: bash run_tv.sh <tag> "<lens list>" <variant json>...
cd /k31ov
export CUTE_DSL_CACHE_DIR=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-KDA2/cutecache
TAG=$1; LENS=$2; shift 2
echo "TV $TAG start $(date +%T)"
timeout 900 python3 -m pytest --noconftest -q -x tests/models/kimi_k3/test_kda_frost_prefill.py -k persistent 2>&1 | tail -3
timeout 900 python3 -u benchmarks/kimi_k3/bench_fip_devb.py --rounds 4 --lens $LENS --variants "$@" 2>&1 | grep --line-buffered -E "lens=|Error|error|Traceback"
echo "TV $TAG done $(date +%T)"
