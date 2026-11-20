#!/bin/bash
# D-GEMM round 5 E3: which resource binds each kda_in form (12448x7168 MXFP8)?
# (a) R256/R384 today's tactics cold vs warm (W L2-resident), full vs loads-only;
# (b) K-split L=2 (FP32 partial planes) at R128/R256: cold vs warm, loads-only, BF16 vs FP32 epilogue.
cd /k31ov
B=benchmarks/kimi_k3
for w in "" --warm; do
  echo "== today R256/R384 $w"
  python3 $B/bench_oproj_mx.py --n 12448 --k 7168 --rows 256 $w \
    --arms own:256x128:2x1:0:256:1:4:1:128:0:0,own:256x128:2x1:0:256:1:4:1:128:1:0 2>&1 | grep -E '"us"' | sed -E 's/"rel"[^,]*, "bitwise"[^,]*, "build_s"[^,]*, //' | cut -c1-150
  python3 $B/bench_oproj_mx.py --n 12448 --k 7168 --rows 384 $w \
    --arms own:256x192:2x1:1:256:1:4:1:128:0:6,own:256x192:2x1:1:256:1:4:1:128:1:6 2>&1 | grep -E '"us"' | sed -E 's/"rel"[^,]*, "bitwise"[^,]*, "build_s"[^,]*, //' | cut -c1-150
  echo "== ksplit R128/R256 $w"
  python3 $B/bench_kda_ksplit.py --n 12448 --k 7168 --rows 128,256 $w \
    --arms 256x128:2x1:1:256:2:f32,256x128:2x1:1:256:2:f32:1,256x128:2x1:1:256:2:bf16,256x256:2x1:1:256:2:f32,256x256:2x1:1:256:2:f32:1,256x256:2x1:1:256:2:bf16 2>&1 | grep -E '"us"|error' | cut -c1-150
done
