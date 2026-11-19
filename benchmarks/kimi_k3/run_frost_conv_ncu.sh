#!/bin/bash
# ncu of the fused-conv FROST kernels. usage: run_frost_conv_ncu.sh <overlay> <outdir>
OV=$1; OUT=$2; mkdir -p $OUT
export PYTHONPATH=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/venvs/cudnnfe133/site:$PYTHONPATH
export CUDNN_FRONTEND_COMPILED_CACHE=$OUT/cc
B=$OV/benchmarks/kimi_k3/bench_frost_conv.py
timeout 900 python3 -u $B --seqs 3 2>&1 | grep -E "check|supported|impl=|  +[0-9.]+ us" | cut -c1-150
echo "=== ncu main (N=2)"
bash $OV/benchmarks/kimi_k3/run_ncu_cmd.sh $OV $OUT/main frost_kda_prefill_K 1 python3 -u $B --seqs 2 --no-time
echo "=== ncu prep (N=1)"
bash $OV/benchmarks/kimi_k3/run_ncu_cmd.sh $OV $OUT/prep frost_kda_prep_K 1 python3 -u $B --seqs 1 --no-time
