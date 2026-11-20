#!/bin/bash
# bash run_var.sh <tag> <rounds> "<lens list>" <kernel file|-> ... : one process per kernel variant (timing arms)
cd /k31ov
TAG=$1; R=$2; LENS=$3; shift 3
export CUTE_DSL_CACHE_DIR=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-KDA2/cutecache
mkdir -p $CUTE_DSL_CACHE_DIR
echo "VAR $TAG start $(date +%T)"
for K in "$@"; do
  KF=$K; [ "$K" != "-" ] && KF=/k31ov/benchmarks/kimi_k3/kda_dev/kvb/$K
  timeout 900 python3 -u benchmarks/kimi_k3/kda_dev/bench_fip_var.py $KF --rounds $R --lens $LENS --variants "{}" 2>&1 | grep --line-buffered -E "KERNEL|lens=|Error|error|Traceback" | sed "s|^|[$K] |"
done
echo "VAR $TAG done $(date +%T)"
