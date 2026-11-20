#!/bin/bash
# generic runner: bash run_py.sh <tag> <script relative to /k31ov> [args...]; hard timeout 20 min
cd /k31ov
TAG=$1; shift
export CUTE_DSL_CACHE_DIR=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-KDA2/cutecache
mkdir -p $CUTE_DSL_CACHE_DIR
echo "PY $TAG start $(date +%T)"
timeout 1200 python3 -u "$@" 2>&1 | grep --line-buffered -v -E "^\s*$|Warning|warn\("
echo "PY $TAG done rc=${PIPESTATUS[0]} $(date +%T)"
