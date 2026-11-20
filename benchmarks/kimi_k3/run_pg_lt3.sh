#!/bin/bash
# P-GEMM LT3: power-aware cuBLASLt search, big prefill MXFP8 GEMMs at M16384 (burst + sustained J)
cd /k31ov
timeout 1700 python3 benchmarks/kimi_k3/pgemm_lt_search.py 16384 8 kda_in_proj,pack_n,pack_k,o_proj,dense_gate_up,dense_down 2>&1 | grep -E "^\{|Error|error" | cut -c1-1200
echo "== END"
