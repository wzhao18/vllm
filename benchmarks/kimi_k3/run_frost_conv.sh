#!/bin/bash
# Fused-conv FROST vs qkv_conv + FROST. usage: run_frost_conv.sh <overlay> <outdir> [bench args] [-- more bench args runs]
OV=$1; OUT=$2; shift 2; mkdir -p $OUT
export PYTHONPATH=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/venvs/cudnnfe133/site:$PYTHONPATH
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
IFS="|" read -ra RUNS <<< "${*:---seqs 2}"
for R in "${RUNS[@]}"; do echo "=== $R"; timeout 1500 python3 -u $OV/benchmarks/kimi_k3/bench_frost_conv.py $R 2>&1 | grep --line-buffered -v -E "^\s*$|Warning|warn|deprecated|jit-cache|flashinfer-cubin" | tail -40; done
