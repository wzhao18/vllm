#!/bin/bash
# D-ELEM-TAIL M1: VocabParallelMarkov A/B (ABBA) on 4 GPUs. A = overlay $1 (old path), B = /k31ov.
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws
A=$K/ov/$1
BENCH=/k31ov/benchmarks/kernels/k31_vr/dsrel/bench_vp_markov_fused.py
BASEPP=$PYTHONPATH
nvidia-smi --query-gpu=name,clocks.sm --format=csv,noheader | head -2
for arm in A B B A; do
  if [ $arm = A ]; then export PYTHONPATH=$A:$BASEPP; else export PYTHONPATH=$BASEPP; fi
  python3 -c "import vllm.models.kimi_k3.nvidia.specialized_decode.runtime.draft as d; print('$arm', d.__file__, hasattr(d, '_vp_markov_fused'))"
  torchrun --nproc-per-node 4 --master-port 29511 $BENCH --tag $arm 2>&1 | grep -E "^\[|Error|error" | head -30
done
