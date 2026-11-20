#!/bin/bash
# D-GEMM round 7: kda_o R384 tactic by the per-SM landing law (bytes per CTA: 128x128 1x1 786 KB on 168 CTAs,
# swap 256x128 2x1 590 KB on 168 CTAs, swap 256x192 2x1 688 KB on 112 CTAs); cold chain us/call.
cd /k31ov
python3 benchmarks/kimi_k3/bench_kda_ksplit.py --n 7168 --k 3072 --rows 384 \
  --arms 128x128:1x1:0:128:1:bf16,256x128:2x1:1:256:1:bf16,256x192:2x1:1:256:1:bf16 2>&1 | grep -E '"us"|rror|Traceback' | cut -c1-500
