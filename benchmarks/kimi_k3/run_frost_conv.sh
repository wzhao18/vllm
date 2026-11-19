#!/bin/bash
# Fused-conv FROST vs qkv_conv + FROST. usage: run_frost_conv.sh <overlay> <outdir> ["bench args|bench args|..."]
OV=$1; OUT=$2; shift 2; mkdir -p $OUT
export PYTHONPATH=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/venvs/cudnnfe133/site:$PYTHONPATH
export CUDNN_FRONTEND_COMPILED_CACHE=$OUT/cc
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
IFS="|" read -ra RUNS <<< "${*:---seqs 2}"
for R in "${RUNS[@]}"; do echo "=== $R"; timeout 1500 python3 -u $OV/benchmarks/kimi_k3/bench_frost_conv.py $R 2>&1 | grep --line-buffered -v -E "^\s*$|Warning|warn|deprecated|jit-cache|flashinfer-cubin" | tail -40; done
for f in $(find $OUT/cc -type f -size +100k 2>/dev/null); do cuobjdump -res-usage $f 2>/dev/null | grep -A1 -E "Function .*(frost_kda_prefill_K|frost_kda_prep_K)" | grep -oE "(frost_kda_prefill_K|frost_kda_prep_K|REG:[0-9]+ STACK:[0-9]+ SHARED:[0-9]+ LOCAL:[0-9]+)" | paste - - ; done | sort | uniq -c
