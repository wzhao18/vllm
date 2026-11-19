#!/bin/bash
# P-GEMM: ncu nvjet vs CuTe 512x256 c4x1 base, then swap-AB skeleton tracebacks.
R=/k31ov/benchmarks/kimi_k3/run_pgemm_conv.sh
bash $R ncuskel '[[512,256],[4,1],256,8,"n",0]'
bash $R skel --swap-only --ms 16384
