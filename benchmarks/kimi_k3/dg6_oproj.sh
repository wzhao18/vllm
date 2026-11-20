#!/bin/bash
# D-GEMM round 6 E3: kda_o (7168 x 3072) stage counts + standalone chain for the pre-wait-depth quick win
# (today's tactics R128 128x64 1x1, R256 256x128 2x1, R384 128x128 1x1; candidate R128 swap 256x128 2x1 = 128 W
# rows per CTA, 16 KB of W per stage). The in-step win comes from W staged before griddepcontrol.wait behind
# KdaFlow's ragged tail (R5-8: kda_o CTAs resident ~1.3 us before the wait); standalone only checks no loss.
cd /k31ov
python3 benchmarks/kimi_k3/bench_kda_ksplit.py --n 7168 --k 3072 --rows 128 \
  --arms 128x64:1x1:0:128:1:bf16:0:0:1,256x128:2x1:1:256:1:bf16:0:0:1 2>&1 | grep -E '"us"|rror|Traceback' | cut -c1-700
python3 benchmarks/kimi_k3/bench_kda_ksplit.py --n 7168 --k 3072 --rows 256 \
  --arms 256x128:2x1:0:256:1:bf16:0:0:0,256x256:2x1:1:256:1:bf16:0:0:0 2>&1 | grep -E '"us"|rror|Traceback' | cut -c1-400
python3 benchmarks/kimi_k3/bench_kda_ksplit.py --n 7168 --k 3072 --rows 384 \
  --arms 128x128:1x1:0:128:1:bf16:0:0:0,256x192:2x1:1:256:1:bf16:0:0:0 2>&1 | grep -E '"us"|rror|Traceback' | cut -c1-400
