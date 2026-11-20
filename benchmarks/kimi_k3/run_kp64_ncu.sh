#!/bin/bash
# ncu --set full of one kp64 launch at F16K + per-line stalls. usage: run_kp64_ncu.sh <overlay> <outdir>
OV=$1; OUT=$2
bash $OV/benchmarks/kimi_k3/run_ncu_cmd.sh $OV $OUT kp64 2 python3 -u $OV/benchmarks/kimi_k3/kp64_diag.py --time
python3 $OV/benchmarks/kimi_k3/ncu_lines.py $OUT/sass.csv 2>/dev/null | head -40
