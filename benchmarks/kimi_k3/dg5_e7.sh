#!/bin/bash
# D-GEMM round 5 E7: what does an IDEAL weight-only stream kernel cost at one GEMM's size (92 MB) on VR200?
# hwdb hw61 ring (1 CTA/SM, PDL graph of 6 kernels, lean consumer-released ring) with per-kernel bytes 46/92/184 MB,
# die-local vs striped, 32K x6 I2 (hw61 W08) vs a GEMM-like 16K x12 I1; K04 = the 2.6 GB W08 calibration.
B=$(mktemp -d)
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a -o $B/hw61k /k31ov/benchmarks/kimi_k3/dgw_hw61k.cu -lcuda > $B/build.log 2>&1 || { grep -m12 -i -B1 -A2 error $B/build.log; exit 1; }
nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader 2>/dev/null | head -1
n=$($B/hw61k -1 | tail -1)
for i in $(seq 0 $((n-1))); do timeout 120 $B/hw61k $i 0 | grep "^==" | cut -c1-150; done
