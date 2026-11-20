#!/bin/bash
# D-KDA round 7 (f): KdaRelay (production state pass, kind 'relay') correctness vs the
# stock commit + KdaWS (fold_matches test, KdaFlow's prep as the stand-in), eager first.
T=tests/models/kimi_k3/test_specialized_decode_sp_kda_flow_gpu.py
export CUTE_DSL_CACHE_DIR=/tmp/dkda-r7f-$$
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw,clocks_throttle_reasons.active --format=csv,noheader 2>/dev/null | head -1
python3 -c "
import torch
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda import relay
print('relay attrs (regs, local, smem):', relay._module().relay_attrs())
" 2>&1 | grep -E "relay attrs|rror|error" | head -12
timeout 900 python3 -m pytest --noconftest -q -x $T -k "relay and eager and (3 or 33)" 2>&1 | grep -E "^E |relay.py:[0-9]+|sp.py:[0-9]+|test_spec.*py:[0-9]+|passed|failed|error" | cut -c1-220 | head -30
timeout 900 python3 -m pytest --noconftest -q $T -k "relay" 2>&1 | grep -E "passed|failed" | tail -3
