#!/bin/bash
# D-GEMM round 5 E4: die-local W + die-matched 2x1 cluster claim on the K-split (L=2) form.
# Pre-registered (R128, cold): L=2 die-local ~7.9 us (F 2.4 + max(688 KB/CTA / 190 GB/s = 3.6, 92 MB / 17 TB/s = 5.4))
# vs L=2 striped 10.8; stop rule: dl gain < 1.4 us.
cd /k31ov
python3 -c "
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.sm107 import prewait as pw
print('max_active_clusters', {c: pw._max_active_clusters(c) for c in (1, 2)})
"
B=benchmarks/kimi_k3
for w in "" --warm; do
  echo "== ksplit dl R128 $w"
  python3 $B/bench_kda_ksplit.py --n 12448 --k 7168 --rows 128 $w \
    --arms 256x128:2x1:1:256:2:f32,256x128:2x1:1:256:2:f32:0:1,256x128:2x1:1:256:2:f32:1:1,256x128:2x1:1:256:1:bf16,256x128:2x1:1:256:1:bf16:0:1 2>&1 | grep -E '"us"|error|die split|Error' | cut -c1-170
done
echo "== ksplit dl R256 (cold)"
python3 $B/bench_kda_ksplit.py --n 12448 --k 7168 --rows 256 \
  --arms 256x256:2x1:1:256:2:f32,256x256:2x1:1:256:2:f32:0:1,256x256:2x1:1:256:2:bf16:0:1 2>&1 | grep -E '"us"|error|die split|Error' | cut -c1-170
