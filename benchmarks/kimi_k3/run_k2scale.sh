#!/bin/bash
# FlashKDA K2/K1 time vs (seq, head) items per GPU at 5461 tokens per sequence:
# separates the per-item chunk-chain latency from per-SM throughput.
# usage (inside wsgpu.sh): run_k2scale.sh <overlay dir> <out dir>
OV=$1; OUT=$2; mkdir -p $OUT
B=$OV/benchmarks/kimi_k3/bench_kda_prefill_chain.py
nvidia-smi --query-gpu=name,clocks.sm --format=csv,noheader | head -1
for nh in 1x106 2x53 2x106 3x96 4x106 6x106 8x106; do
  n=${nh%x*}; h=${nh#*x}
  timeout 300 python3 $B --impl stock --seqs $n --heads $h --tokens $((n*5461)) --row-stride 56000 --iters 5 \
    --json $OUT/s_${nh}.json 2>&1 | grep -E "^impl=|_flash_kda_fwd|Error|error" | \
    sed -E 's/(_Z25_flash_kda_fwd_recurrence|void _flash_kda_fwd_prepare).*/\1/'
done
