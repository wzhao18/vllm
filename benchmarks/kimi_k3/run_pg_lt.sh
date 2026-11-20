#!/bin/bash
# P-GEMM LT1: cuBLASLt algorithm search for the prefill MXFP8 GEMMs
cd /k31ov
timeout 1500 python3 benchmarks/kimi_k3/pgemm_lt_search.py ${ROWS:-16384,8192} ${SECS:-6} 2>&1 | grep -E "^\{|Error|error" | cut -c1-900
echo "== END"
