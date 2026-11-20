#!/bin/bash
# D-GEMM round 5 E2: separate the per-kernel edge F from the per-CTA stream slope (vary K at a fixed grid),
# cold vs warm, full vs loads-only, and check whether the L=2 K-split form runs its 196 CTAs concurrently.
cd /k31ov
python3 -c "
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.sm107 import prewait as pw
print('max_active_clusters', {c: pw._max_active_clusters(c) for c in (1, 2, 4)})
"
T="own:128x64:1x1:0:128:1:4:1:128:0:12,own:128x64:1x1:0:128:1:4:1:128:1:12"
S="own:256x128:2x1:1:256:1:4:1:128:0:0,own:256x128:2x1:1:256:1:4:1:128:1:0"
for k in 1792 3584 7168 14336; do
  for w in "" --warm; do
    echo "== K $k $w"
    python3 benchmarks/kimi_k3/bench_oproj_mx.py --n 12448 --k $k --rows 128 --arms $T,$S $w 2>&1 | grep -E '"us"' | sed -E 's/"rel"[^,]*, "bitwise"[^,]*, "build_s"[^,]*, //' | cut -c1-150
  done
done
for k in 3584 7168; do
  echo "== ksplit K $k"
  python3 benchmarks/kimi_k3/bench_kda_ksplit.py --n 12448 --k $k --rows 128 \
    --arms 256x128:2x1:1:256:1:f32,256x128:2x1:1:256:2:f32,128x64:1x1:0:128:1:bf16 2>&1 | grep -E '"us"|error' | cut -c1-150
done
