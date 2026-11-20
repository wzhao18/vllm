#!/bin/bash
# D-GEMM round 6 E2: kda_in R384 on the landed swap 256x192 2x1 form (2 token tiles per weight pair), striped
# vs die-local W with the 2x1 die claim generalized to several token tiles (claimed = (pair * n_tiles + n) * L
# + l; the two token tiles of a pair are consecutive claims). Cap 6 (landed) and uncapped stages, stamps.
# R5-3/R4-7: R384 is chip-delivery bound (~315 MB delivered at ~25-28 TB/s, deeper rings do not help); MR5:
# HBM + L2 delivery share one in-flight budget and a die-local HBM byte needs less of it than a striped one.
# Prediction: die-local stream 11.2 -> <= 9.5 us (-1.7), chain 16.1 -> <= 14.6 us/call. Stop if < 0.8 us.
cd /k31ov
python3 benchmarks/kimi_k3/bench_kda_ksplit.py --n 12448 --k 7168 --rows 384 \
  --arms 256x192:2x1:1:256:1:bf16:0:0:1:6,256x192:2x1:1:256:1:bf16:0:1:1:6,256x192:2x1:1:256:1:bf16:0:0:0:0,256x192:2x1:1:256:1:bf16:0:1:0:0 \
  2>&1 | grep -E '"us"|rror|Traceback|die split' | cut -c1-1200
