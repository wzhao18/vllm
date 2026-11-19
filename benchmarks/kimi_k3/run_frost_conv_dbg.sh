#!/bin/bash
# TEMP: fused-conv FROST debug variants (K31_FROST_CONV_DBG). usage: run_frost_conv_dbg.sh <overlay> <outdir> "<dbg values>" [bench args]
OV=$1; OUT=$2; V=$3; shift 3; mkdir -p $OUT
export PYTHONPATH=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/venvs/cudnnfe133/site:$PYTHONPATH
export CUDNN_FRONTEND_DISABLE_COMPILED_CACHE=1
for d in $V; do echo "=== dbg $d"; K31_FROST_CONV_DBG=$d timeout 900 python3 -u $OV/benchmarks/kimi_k3/bench_frost_conv.py ${*:---seqs 2} 2>&1 | grep -E "check out|impl=|frost_kda_prefill_K|Error|error" | cut -c1-120; done
