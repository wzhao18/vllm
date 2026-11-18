#!/bin/bash
# KDA prefill chain: unit test + per-kernel microbench of stock / split / fused at
# the deployed DEP16 shapes (F16K = 3 x 5.4K, F8K = 3 x 2.7K; 96 heads).
# usage (inside wsgpu.sh): run_kda_prefill_ab.sh <overlay dir> <out dir>
set -x
OV=$1; OUT=$2; mkdir -p $OUT
B=$OV/benchmarks/kimi_k3/bench_kda_prefill_chain.py
which ncu nsys; ls /usr/local/cuda/bin | grep -i "ncu\|nsys"
nvidia-smi --query-gpu=name,clocks.max.sm --format=csv
python -m pytest -x -q $OV/tests/models/kimi_k3/test_kda_prefill_fused_conv.py 2>&1 | tail -30
for impl in stock split fused; do
  python $B --impl $impl --tokens 16384 --json $OUT/f16k_$impl.json 2>&1 | grep -v Warning
  python $B --impl $impl --tokens 8192 --json $OUT/f8k_$impl.json 2>&1 | grep -v Warning
done
