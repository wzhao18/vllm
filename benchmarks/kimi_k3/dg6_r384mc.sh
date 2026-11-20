#!/bin/bash
# D-GEMM round 6 E6: kda_in R384 swap 256x192: the stream does not speed up with ring depth (E5: 11.6-11.9 us at
# 6/8/10 stages, ~24.6 TB/s delivered): the two token-tile clusters of one weight pair read W twice from HBM
# (184 MB at the striped rate ~ 13-14 us; EX1 R4 raster). Cluster (2,2) puts both token tiles of a weight pair in
# one 4-CTA cluster with W (the A operand) multicast across them: W read once (92 MB). Prediction: stream
# 11.6 -> <= 8.5 us, chain 15.85 -> <= 13 us/call. Stop if < 1.5 us gain.
cd /k31ov
python3 benchmarks/kimi_k3/bench_kda_ksplit.py --n 12448 --k 7168 --rows 384 \
  --arms 256x192:2x1:1:256:1:bf16:0:0:1:6,256x192:2x2:1:256:1:bf16:0:0:1:0,256x192:2x2:1:256:1:bf16:0:0:1:6,256x192:2x2:1:256:1:bf16:1:0:1:0 \
  2>&1 | grep -E '"us"|rror|Traceback' | cut -c1-700
