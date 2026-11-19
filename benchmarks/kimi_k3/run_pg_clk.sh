#!/bin/bash
# P-GEMM HW-CLK: sustained pure tcgen05 2-CTA mxf8f6f4 loop (random + zero operands) and cuBLAS GEMM loops with NVML
cd /k31ov
B=$(mktemp -d /tmp/pgclk.XXXX)
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a -o $B/clk benchmarks/kimi_k3/pgemm_clk_probe.cu -ldl > $B/b.log 2>&1 || { head -30 $B/b.log; exit 1; }
nvidia-smi -q -d POWER,CLOCK 2>/dev/null | grep -iE "Power Limit|Max Clocks|  SM  |Applications" | head -12
timeout 60 $B/clk 1 ${SECS:-12}
sleep 3
timeout 60 $B/clk 0 ${SECS:-12}
sleep 3
timeout 400 python3 benchmarks/kimi_k3/pgemm_clk_probe.py ${SECS:-12} 2>&1 | grep -E "^\{|Error|error" | cut -c1-300
echo "== END"
