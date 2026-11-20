#!/bin/bash
# D-GEMM R7 Experiment 1 (09:10 audit): kda_o role-B loads in the shadow of D-KDA's state-pass skeleton
# ('z' = rmwt512o1z, MATH 13) vs z + role B (rmwt512o1B, MATH 14), die-local, n32 / n96; us/layer + role-B tail.
cd /k31ov
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw,clocks_throttle_reasons.active --format=csv,noheader | head -1
python3 benchmarks/kimi_k3/dg7_kdao_skel.py --n 32 96 --arms rmwt512o1B rmwt512o1z rmwt512o1B --place local 2>&1 \
  | grep -E "us/layer|role-B|rror|Trace|error|go_tma" | cut -c1-200
