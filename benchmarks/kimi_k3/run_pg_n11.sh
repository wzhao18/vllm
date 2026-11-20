#!/bin/bash
# P-GEMM: preferred 2x4 + fallback 2x1 clusters (all 212 SMs) vs c2x4 base
cd /k31ov
timeout 1200 python3 benchmarks/kimi_k3/pgemm_conv_bench.py --mode mix --ms 16384,8192 2>&1 | grep -E "^\{|Error" | cut -c1-400
