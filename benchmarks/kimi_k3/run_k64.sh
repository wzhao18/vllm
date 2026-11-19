#!/bin/bash
# chunk-64 tcgen05 KDA recurrence: correctness vs torch emulation, F16K time, clock stamps.
# usage: run_k64.sh <overlay> <outdir> [k64_diag.py args]
OV=$1; OUT=$2; shift 2; mkdir -p $OUT
ARGS=${*:---check --e2e --time --stamps}
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
timeout 600 python3 -u $OV/benchmarks/kimi_k3/k64_diag.py $ARGS 2>&1 | grep --line-buffered -v -E "^\s*$|Warning|warn|deprecated|jit-cache|flashinfer-cubin"
SO=$(ls -t ${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}/k3_flashkda_split/*/k3_flashkda_split.so 2>/dev/null | head -1)
[ -n "$SO" ] && cuobjdump -res-usage $SO 2>/dev/null | grep -A1 -E "k64|prep64tc" | grep -oE "(k64tc|prep64tc|REG:[0-9]+ STACK:[0-9]+ SHARED:[0-9]+ LOCAL:[0-9]+)" | paste - - | sort | uniq -c
[ -n "$SO" ] && cuobjdump -sass $SO 2>/dev/null | awk '/Function : .*(k64|prep64tc)/{f=1} /Function : /&&!/(k64|prep64tc)/{f=0} f&&/LDL|STL/{n++} END{print "k64 SASS local ld/st:", n+0}'
