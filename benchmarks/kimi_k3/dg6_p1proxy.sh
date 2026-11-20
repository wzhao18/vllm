#!/bin/bash
# D-GEMM round 6 E1b: P1 byte-flow proxy. The M128-pair form moves 64 W rows + 64 tokens (917 KB) per CTA on
# ~196 CTAs at kda_in R128. The existing non-swap 128x64 1x1 tactic at R64 moves the SAME bytes per CTA
# (X 64 x 7168 + W 64 x 7168) on 195 CTAs. E1 (run 1120062628): striped R64 11.12 us (stream 7.74) vs R128
# 10.53 (stream 7.20): halving X per CTA gives nothing with striped W (W stream ~12 TB/s binds).
# E1b: die-local W (1x1 die claim, reset fix) at R64 and R128. Prediction (P1 premise): die-local R64
# stream ~5.4 us (W 92 MB at ~17 TB/s); R128 die-local stays ~7.0 (per-SM ingest 1.375 MB binds).
# Gate: die-local R64 >= 0.8 us/call faster than striped R64, else P1's projected gain does not exist.
cd /k31ov
python3 benchmarks/kimi_k3/bench_kda_ksplit.py --n 12448 --k 7168 --rows 64,128 \
  --arms 128x64:1x1:0:128:1:bf16:0:0:0,128x64:1x1:0:128:1:bf16:0:1:0,128x64:1x1:0:128:1:bf16:0:0:1,128x64:1x1:0:128:1:bf16:0:1:1 \
  2>&1 | grep -E '"us"|rror|Traceback|die split' | cut -c1-1200
