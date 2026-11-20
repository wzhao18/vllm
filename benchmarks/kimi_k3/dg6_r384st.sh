#!/bin/bash
# D-GEMM round 6 E5: kda_in R384 swap 256x192 2x1 striped, smem stage cap 6 / 8 / 10 (uncapped) with per-CTA
# stamps: does the deeper ring stream faster (Little: 168 -> 280 KB in flight) and lose it in the epilogue tail
# (fewer C stages), or does the stream itself not speed up? R4-7 / R6-2 chain: cap 6 16.06-16.13, 8 16.51,
# uncapped 16.47 us/call.
cd /k31ov
python3 benchmarks/kimi_k3/bench_kda_ksplit.py --n 12448 --k 7168 --rows 384 \
  --arms 256x192:2x1:1:256:1:bf16:0:0:1:6,256x192:2x1:1:256:1:bf16:0:0:1:8,256x192:2x1:1:256:1:bf16:0:0:1:0,256x192:2x1:1:256:1:bf16:1:0:1:6,256x192:2x1:1:256:1:bf16:1:0:1:0 \
  2>&1 | grep -E '"us"|rror|Traceback' | cut -c1-700
