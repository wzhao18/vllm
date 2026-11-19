#!/bin/bash
# tcgen05 KDA recurrence: diagnostic (errors per kr-descriptor variant + F16K
# times) and the chain bench (stock / onorm / tc).
# usage (inside wsgpu.sh): run_k2tc.sh <overlay dir> <out dir> [diag args]
OV=$1; OUT=$2; shift 2; mkdir -p $OUT
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
timeout 600 python3 -u $OV/benchmarks/kimi_k3/k2tc_diag.py "$@" 2>&1 | grep --line-buffered -v -E "^\s*$|Warning|warn|deprecated|jit-cache"
SO=$(ls -t ${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}/k3_flashkda_split/*/k3_flashkda_split.so 2>/dev/null | head -1)
[ -n "$SO" ] && cuobjdump -res-usage $SO 2>/dev/null | grep -A1 -E "k2_tc" | grep -oE "Function [^ ]*|REG:[0-9]+|STACK:[0-9]+|LOCAL:[0-9]+" | paste - - - - | sed 's/Function _ZN4k2tc12k2_tc_kernelIL/k2tc</' 
[ -n "$SO" ] && echo "in-tree K2 res-usage:" && cuobjdump -res-usage $SO 2>/dev/null | grep -A1 "fwd_recurrence" | grep -oE "REG:[0-9]+ STACK:[0-9]+ SHARED:[0-9]+ LOCAL:[0-9]+" | sort | uniq -c
B=$OV/benchmarks/kimi_k3/bench_kda_prefill_chain.py
for impl in onorm tc; do
  python3 $B --json $OUT/${impl}_16384.json --impl $impl --tokens 16384 2>&1 | grep -E "^impl=|^ +[0-9.]+ us|Error|error" | \
    sed -E 's/(_Z25_flash_kda_fwd_recurrence|void _flash_kda_fwd_prepare|void k3conv::qkv_conv_kernel|\(anonymous namespace\)::o_norm_mx_kernel|void k2tc::k2_tc_kernel).*/\1/'
done
