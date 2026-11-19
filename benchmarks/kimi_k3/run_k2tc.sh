#!/bin/bash
# tcgen05 KDA recurrence: diagnostic (errors per kr-descriptor variant + F16K
# times) and the chain bench (stock / onorm / tc).
# usage (inside wsgpu.sh): run_k2tc.sh <overlay dir> <out dir> [diag args]
OV=$1; OUT=$2; shift 2; mkdir -p $OUT
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
python3 $OV/benchmarks/kimi_k3/k2tc_diag.py "$@" 2>&1 | grep -v -E "^\s*$|Warning|warn" | tail -60
B=$OV/benchmarks/kimi_k3/bench_kda_prefill_chain.py
for impl in onorm tc; do
  python3 $B --json $OUT/${impl}_16384.json --impl $impl --tokens 16384 2>&1 | grep -E "^impl=|^ +[0-9.]+ us|Error|error" | \
    sed -E 's/(_Z25_flash_kda_fwd_recurrence|void _flash_kda_fwd_prepare|void k3conv::qkv_conv_kernel|\(anonymous namespace\)::o_norm_mx_kernel|void k2tc::k2_tc_kernel).*/\1/'
done
