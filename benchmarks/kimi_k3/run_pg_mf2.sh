#!/bin/bash
# P-GEMM MF2: MoE dense front isolated + sustained cuBLAS kernels at the DEP16 per-rank shape
cd /k31ov
timeout 900 python3 benchmarks/kimi_k3/pgemm_moefront_bench.py 300 2>&1 | grep -E "^\{|Error" | cut -c1-900
