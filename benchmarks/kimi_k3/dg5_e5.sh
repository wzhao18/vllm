#!/bin/bash
# D-GEMM round 5 E5: HBM rate of the strided K-major weight-box pattern (dgw_probe.cu), W only.
# Pre-registered: if A2d (die-local, 128 B x 128 rows at pitch 3584) streams < 13 TB/s while A3d (contiguous
# 16 KiB boxes) reaches >= 16 TB/s, the GEMM weight LAYOUT (not placement / ring / issue) binds kda_in cold.
B=$(mktemp -d)
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a -o $B/dgw /k31ov/benchmarks/kimi_k3/dgw_probe.cu -lcuda > $B/build.log 2>&1 || { grep -m12 -i -B1 -A2 error $B/build.log; exit 1; }
nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader 2>/dev/null | head -1
$B/dgw
