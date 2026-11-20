#!/bin/bash
# D-GEMM round 6 E4: split (weight / activation) smem rings in the vendored sm107 MXFP8 kernel (split_ring =
# (weight stages, activation stages); TMA warp 0 streams the weight ring from kernel entry, warp 1 the activation
# after griddepcontrol.wait). kda_in 12448 x 7168, cold chain of 16 PDL calls (the next call's weight ring fills
# behind the previous call's tail), us/call; arms: coupled baseline vs splits. R6-5 pre-registration (per call,
# in-step): R128 -2.0 / R256 -2.0 / R384 -3..-5 us with ~160-190 KB of weight staged before the wait.
cd /k31ov
B=benchmarks/kimi_k3/bench_ring.py
python3 $B --n 12448 --k 7168 --rows 128 --arms \
128x64:1x1:0:128:1:bf16:0:0:0:0,128x64:1x1:0:128:1:bf16:0:0:0:0:20:7,128x64:1x1:0:128:1:bf16:0:0:0:0:24:5,128x64:1x1:0:128:1:bf16:0:0:0:0:16:9 \
  2>&1 | grep -E '"us"|rror|Traceback' | cut -c1-300
python3 $B --n 12448 --k 7168 --rows 256 --arms \
256x128:2x1:0:256:1:bf16:0:0:0:0,256x128:2x1:0:256:1:bf16:0:0:0:0:20:7 \
  2>&1 | grep -E '"us"|rror|Traceback' | cut -c1-300
python3 $B --n 12448 --k 7168 --rows 384 --arms \
256x192:2x1:1:256:1:bf16:0:0:0:6,256x192:2x1:1:256:1:bf16:0:0:0:0:12:6,256x192:2x1:1:256:1:bf16:0:0:0:0:14:4 \
  2>&1 | grep -E '"us"|rror|Traceback' | cut -c1-300
