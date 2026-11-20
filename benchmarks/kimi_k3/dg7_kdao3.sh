#!/bin/bash
# D-GEMM R7 Experiment 1 decomposition 2: mode 5 = z with the role-B smem (oversized mode, no role-B work) vs z.
cd /k31ov
python3 benchmarks/kimi_k3/dg7_kdao_skel.py --n 32 96 --arms rmwt512o1z --place local 2>&1 | grep -E "us/layer|rror" | cut -c1-120
DG7_MODE=5 python3 benchmarks/kimi_k3/dg7_kdao_skel.py --n 32 96 --arms rmwt512o1B --place local 2>&1 | grep -E "us/layer|rror|Trace" | sed "s/^/mode 5: /" | cut -c1-130
DG7_MODE=4 python3 benchmarks/kimi_k3/dg7_kdao_skel.py --n 32 96 --arms rmwt512o1B --place local 2>&1 | grep -E "us/layer|rror|Trace" | sed "s/^/mode 4: /" | cut -c1-130
