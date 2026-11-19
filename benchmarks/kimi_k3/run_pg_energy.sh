#!/bin/bash
# P-GEMM M2: sustained power-capped energy per GEMM of skeleton tactics vs cuBLASLt (KDA in_proj shape)
cd /k31ov
timeout 900 python3 benchmarks/kimi_k3/pgemm_clk_probe.py ${SECS:-10} energy 2>&1 | grep -E "^\{|Error|error" | cut -c1-300
echo "== END"
