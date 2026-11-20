#!/bin/bash
# P-KDA2: in-kernel gate projection (FAG) check + A/B: bash run_fag.sh <tag> "<lens list>" [test]
cd /k31ov
export CUTE_DSL_CACHE_DIR=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-KDA2/cutecache
TAG=$1; LENS=$2
echo "FAG $TAG start $(date +%T)"
if [ "$3" = test ]; then timeout 900 python3 -m pytest --noconftest -q -x tests/models/kimi_k3/test_kda_frost_prefill.py -k persistent 2>&1 | tail -3; fi
timeout 1200 python3 -u benchmarks/kimi_k3/bench_fip_fag.py --rounds 4 --lens $LENS --dev "${@:4}" 2>&1 | grep -v -E "^W1120|UserWarning|warn" | grep --line-buffered -E "lens=|Error|error|Traceback|File |line [0-9]" | head -60
echo "FAG $TAG done $(date +%T)"
