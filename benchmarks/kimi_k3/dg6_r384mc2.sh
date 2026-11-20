#!/bin/bash
# D-GEMM round 6 E7: kda_in R384 swap 256x192 with 8-CTA clusters (4,2): two weight pairs x two token tiles per
# cluster (W multicast across the token tiles, X across the weight pairs); 25 clusters x 8 = 200 CTAs for 49
# weight pairs: one wave if >= 25 eight-CTA clusters are co-resident (E6: only 46 four-CTA clusters fit).
cd /k31ov
python3 benchmarks/kimi_k3/bench_kda_ksplit.py --n 12448 --k 7168 --rows 384 \
  --arms 256x192:4x2:1:256:1:bf16:0:0:1:0,256x192:4x2:1:256:1:bf16:1:0:1:0,256x192:4x1:1:256:1:bf16:0:0:1:0 \
  2>&1 | grep -E '"us"|rror|Traceback' | cut -c1-700
