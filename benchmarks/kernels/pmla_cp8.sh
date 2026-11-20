#!/bin/bash
# P-MLA M17: cp.async gather with 8 lanes per row (cpa 10: 4 rows x 128 B per warp
# instruction) vs cpa 3 (8 rows x 64 B): bitwise on physical mixed lists + F16K L78K ABBA of
# full, park (gather with a minimal consumer) and gather-only no-store.
# usage: pmla_cp8.sh <run>
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/$1
mkdir -p $R
cat > $R/t.py <<'PY'
import os, sys, torch
sys.path.insert(0, os.path.join(os.getcwd(), "benchmarks/kernels"))
from pmla10_check import abba, inputs, run
C10 = (0,) * 10 + (10,)
for t, short, seed in ((300, True, 1), (149, True, 2), (2048, False, 3), (640, True, 4)):
    inp = inputs(t, 9000, seed=seed, short=short)
    ref = run(*inp, ())
    o = run(*inp, C10)
    torch.cuda.synchronize()
    print(f"phys t={t} short={short} cpa10: equal={torch.equal(o, ref)}", flush=True)
inp = inputs(16384, 78 * 1024, seed=5)
P = (1, 0, 1, 1) + (0,) * 11 + (1,)
abba(lambda a: run(*inp, a), [(), C10, P, P[:10] + (10,) + P[11:], (1, 0, 1, 1)], "phys F16K L78K")
PY
timeout 1200 python3 $R/t.py > $R/ab.log 2>&1
echo "ab rc=$?" > $R/summary.txt
grep -E "equal|F16K|Error|error|Traceback|assert" $R/ab.log | grep -v Warning | cut -c1-140 | tail -12 >> $R/summary.txt
echo DONE >> $R/summary.txt
