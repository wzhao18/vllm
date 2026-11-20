#!/bin/bash
# D-GEMM R7 Experiment 1 decomposition (stop rule hit: z+B +10.8 / +14.2 us/layer): role-B modes
# 0 full, 1 round sync only (no loads), 2 W_o slices only, 3 X only, 4 free-running loads without round sync.
cd /k31ov
for m in 1 2 3 4 0; do
  DG7_MODE=$m python3 benchmarks/kimi_k3/dg7_kdao_skel.py --n 32 96 --arms rmwt512o1B --place local 2>&1 \
    | grep -E "us/layer|role-B|rror|Trace" | sed "s/^/mode $m: /" | cut -c1-200
done
