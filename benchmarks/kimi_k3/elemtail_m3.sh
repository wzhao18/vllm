#!/bin/bash
# D-ELEM-TAIL M3: VocabParallelMarkov select fused into the keys kernel + int32 first-row gather.
# 4 GPUs: the vpmarkov torchrun test on /k31ov, then ABBA x2 of bench_vp_markov_fused.py (A = overlay $1, B = /k31ov).
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws
A=$K/ov/$1
BENCH=/k31ov/benchmarks/kernels/k31_vr/dsrel/bench_vp_markov_fused.py
BASEPP=$PYTHONPATH
nvidia-smi --query-gpu=name,clocks.sm --format=csv,noheader | head -1
echo "== test vpmarkov (B)"
torchrun --nproc-per-node 4 --master-port 29512 /k31ov/tests/models/kimi_k3/test_dspark_gqa_fastpath_gpu.py vpmarkov 2>&1 | grep -aE "checked|Error|error|FAIL|assert|Traceback" | head -20
port=29513
for arm in A B B A A B B A; do
  if [ $arm = A ]; then export PYTHONPATH=$A:$BASEPP; else export PYTHONPATH=$BASEPP; fi
  torchrun --nproc-per-node 4 --master-port $port $BENCH --tag $arm 2>&1 | grep -aE "^\[|Error|error" | head -12
  port=$((port + 1))
done
echo M3_DONE
