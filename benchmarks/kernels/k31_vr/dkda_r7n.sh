#!/bin/bash
# D-KDA round 7 (n): KdaRelay v5 = finisher warps with batched (ILP 8) MX epilogues + a
# die-matched item order on die-local state (bench: static metadata). Tests, stamps,
# (in_proj-like GEMM neighbour: L2 + I-cache thrash) arm. Clock stamped.
T=tests/models/kimi_k3/test_specialized_decode_sp_kda_flow_gpu.py
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
export CUTE_DSL_CACHE_DIR=/tmp/dkda-r7n-$$
nvidia-smi --query-gpu=clocks.sm,power.draw,clocks_throttle_reasons.active --format=csv,noheader 2>/dev/null | head -1
python3 -c "
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda import relay
print('relay attrs (regs, local, smem):', relay._module().relay_attrs())" 2>&1 | grep attrs
timeout 900 python3 $B --gap 1 --pending 4 --mx --place local --n 32 96 --kinds relay --rtrace 2>&1 | grep -E "RTRACE|rror" | cut -c1-260
  timeout 900 python3 $B --gap 1 --pending 4 --mx --place $place --n 32 64 96 128 --kinds relay core 2>&1 | grep -E "us/layer|rror" | sed "s/ (best.*R+W//" | cut -c1-160
nvidia-smi --query-gpu=clocks.sm,power.draw,clocks_throttle_reasons.active --format=csv,noheader 2>/dev/null | head -1
