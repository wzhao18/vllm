#!/bin/bash
# kc64_rec (chunk-64 chain, FP32 state in TMEM): checks, F16K time, stamps, resource usage.
# usage: run_kc64.sh <overlay> [kc64_diag.py args]
OV=$1; shift
ARGS=${*:---check --time --stamps}
nvidia-smi --query-gpu=name,clocks.sm --format=csv,noheader | head -1
timeout 900 python3 -u $OV/benchmarks/kimi_k3/kc64_diag.py $ARGS 2>&1 | grep --line-buffered -v -E "^\s*$|Warning|warn|deprecated|jit-cache|flashinfer-cubin" | tail -40
SO=$(ls -t ${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}/k3_flashkda_split/*/k3_flashkda_split.so 2>/dev/null | head -1)
[ -n "$SO" ] && cuobjdump -res-usage $SO 2>/dev/null | grep -A1 "kc64" | grep -oE "REG:[0-9]+ STACK:[0-9]+ SHARED:[0-9]+ LOCAL:[0-9]+" | sort | uniq -c
