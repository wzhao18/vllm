#!/bin/bash
# D-GEMM round 5 E6: per-CTA %globaltimer stamps in the REAL vendored MXFP8 kernel (dbg_stamps) at kda_in R128,
# cold: where do the ~10.5 us of the L=1 / L=2 / non-swap forms go (entry skew, first issue, ramp, stream, tail)?
cd /k31ov
python3 benchmarks/kimi_k3/bench_kda_ksplit.py --n 12448 --k 7168 --rows 128 \
  --arms 128x64:1x1:0:128:1:bf16:0:0:1,256x128:2x1:1:256:1:bf16:0:0:1,256x128:2x1:1:256:2:f32:0:0:1,256x128:2x1:1:256:2:f32:1:0:1,256x128:2x1:1:256:2:f32:0:1:1 2>&1 | grep -E '"us"|rror' | cut -c1-900
