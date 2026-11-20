#!/bin/bash
# P-KDA2-S6 r8 dev A/B runner (copy of kda_dev/run_dev.sh): bash run_dev.sh <tag> <rounds> "<lens list>" <variant json>...
# Output streams (unbuffered) to stdout; hard timeout 20 min.
cd /k31ov
TAG=$1; R=$2; LENS=$3; shift 3
export CUTE_DSL_CACHE_DIR=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-KDA2-S6/cc8
mkdir -p $CUTE_DSL_CACHE_DIR
echo "DEV $TAG start $(date +%T)"
timeout 1200 python3 -u benchmarks/kimi_k3/bench_fip_devb.py --rounds $R --lens $LENS --variants "$@" 2>&1 | grep --line-buffered -E "lens=|Error|error|Traceback"
echo "DEV $TAG done rc=${PIPESTATUS[0]} $(date +%T)"
