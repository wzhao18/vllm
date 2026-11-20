#!/bin/bash
# M3c IMA bisect (inside the image, cwd = overlay root): one process per case
for c in ${CASES:-"small 0 0" "small 0 4" "small 0 3" "small 0 6" "mid 0 0" "big 0 0"}; do
  echo "== $c"
  CUDA_LAUNCH_BLOCKING=1 timeout 180 python3 benchmarks/kimi_k3/pidx_bisect.py $c 2>&1 | grep -E 'GROUPS|^OK|Error|error:|indexer.py", line' | tail -4 | cut -c1-260
done
echo == done
