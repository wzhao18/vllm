#!/bin/bash
# FLA chunk-64 KDA (Triton) vs FlashKDA chunk-16 at the deployed F16K shape + item scaling.
OV=$1; OUT=$2; mkdir -p $OUT
B=$OV/benchmarks/kimi_k3/bench_kda_prefill_chain.py
for nh in 1x106 2x106 3x96 4x106; do
  n=${nh%x*}; h=${nh#*x}
  timeout 600 python3 $B --impl fla --seqs $n --heads $h --tokens $((n*5461)) --row-stride 56000 --iters 5 \
    --json $OUT/fla_${nh}.json 2>&1 | grep -E "^impl=|^ +[0-9.]+ us|Error|error" | cut -c1-140
done
