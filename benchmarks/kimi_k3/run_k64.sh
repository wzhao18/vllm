#!/bin/bash
# chunk-64 tcgen05 KDA recurrence: correctness vs torch emulation, F16K time, clock stamps.
OV=$1; OUT=$2; mkdir -p $OUT
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
timeout 900 python3 -u $OV/benchmarks/kimi_k3/k64_diag.py --check --e2e --time --stamps 2>&1 | grep --line-buffered -v -E "^\s*$|Warning|warn|deprecated|jit-cache" | tail -60
SO=$(ls -t ${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}/k3_flashkda_split/*/k3_flashkda_split.so 2>/dev/null | head -1)
[ -n "$SO" ] && cuobjdump -res-usage $SO 2>/dev/null | grep -A1 "k64" | grep -oE "REG:[0-9]+ STACK:[0-9]+ SHARED:[0-9]+ LOCAL:[0-9]+" | sort | uniq -c
