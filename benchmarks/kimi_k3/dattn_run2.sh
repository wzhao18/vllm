#!/bin/bash
# D-ATTN: sparse MLA (trtllm-gen) at D3-32 shapes + ncu of the R384 call. $1 = overlay dir, $2 = out dir
OV=$1; OUT=$2
cd $OV
python3 benchmarks/kimi_k3/bench_dattn_sparse.py --out $OUT/sparse1.json
NCU=$(command -v ncu || ls /usr/local/cuda/bin/ncu 2>/dev/null)
echo "ncu: $NCU"
[ -n "$NCU" ] && $NCU --target-processes all -k regex:fmha --launch-count 2 \
  --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section LaunchStats \
  --section ComputeWorkloadAnalysis --section WarpStateStats \
  python3 benchmarks/kimi_k3/bench_dattn_sparse.py --rows 384 --share 0,0.8 --ncu > $OUT/ncu_sparse_r384.txt 2>&1
tail -150 $OUT/ncu_sparse_r384.txt
