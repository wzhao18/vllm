#!/bin/bash
# D-GEMM round 6 E8: kda_in R384 swap 256x192 2x1 (landed cap 6 and uncapped) with ONE global claim counter in
# weight-major order (claimed = pair * 2 + token tile: both token tiles of a weight pair go to CTAs that start
# together, so the second W read can hit L2), W striped (no die placement), vs the static raster. Separates the
# R6-2 gain (stream 12.29 -> 10.56) into claim-order L2 reuse vs die locality.
cd /k31ov
python3 benchmarks/kimi_k3/bench_kda_ksplit.py --n 12448 --k 7168 --rows 384 \
  --arms 256x192:2x1:1:256:1:bf16:0:0:1:6,256x192:2x1:1:256:1:bf16:0:2:1:6,256x192:2x1:1:256:1:bf16:0:1:1:6,256x192:2x1:1:256:1:bf16:0:2:1:0 \
  2>&1 | grep -E '"us"|rror|Traceback' | cut -c1-700
