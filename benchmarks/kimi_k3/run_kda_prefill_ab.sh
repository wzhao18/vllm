#!/bin/bash
# KDA prefill chain: unit test + per-kernel microbench of stock / onorm at the
# deployed DEP16 shapes (F16K = 3 x 5.4K, F8K = 3 x 2.7K; 96 heads).
# usage (inside wsgpu.sh): run_kda_prefill_ab.sh <overlay dir> <out dir>
OV=$1; OUT=$2; mkdir -p $OUT
B=$OV/benchmarks/kimi_k3/bench_kda_prefill_chain.py
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
python3 -m pytest -q --noconftest -p no:cacheprovider -W ignore \
  $OV/tests/models/kimi_k3/test_kda_o_norm_mxfp8.py 2>&1 | grep -E "passed|failed|Error|assert" | tail -15
run() {  # run <tag> <bench args...>
  python3 $B --json $OUT/$1.json "${@:2}" 2>&1 | grep -E "^impl=|^ +[0-9.]+ us|Error|error" | \
    sed -E 's/(_Z25_flash_kda_fwd_recurrence|void _flash_kda_fwd_prepare|void k3conv::qkv_conv_kernel|_kda_rmsnorm_gated_mxfp8_kernel|\(anonymous namespace\)::o_norm_mx_kernel).*/\1/'
}
for t in 16384 8192; do
  for impl in stock onorm; do run ${impl}_$t --impl $impl --tokens $t; done
done
