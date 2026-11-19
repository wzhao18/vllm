#!/bin/bash
# kp64 (chunk-64 WY prepare, no conv) checks + F16K time. usage: run_kp64.sh <overlay> [args]
OV=$1; shift
ARGS=${*:---check --time}
nvidia-smi --query-gpu=name,clocks.sm --format=csv,noheader | head -1
python3 -u $OV/benchmarks/kimi_k3/kp64_diag.py $ARGS 2>&1 | grep --line-buffered -v -E "^\s*$|Warning|warn|deprecated|jit-cache|flashinfer-cubin" | tail -30
SO=$(ls -t ${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}/k3_flashkda_split/*/k3_flashkda_split.so 2>/dev/null | head -1)
[ -n "$SO" ] && cuobjdump -res-usage $SO 2>/dev/null | grep -A1 "kp64" | grep -oE "REG:[0-9]+ STACK:[0-9]+ SHARED:[0-9]+ LOCAL:[0-9]+" | sort | uniq -c
