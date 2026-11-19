#!/bin/bash
# ncu of the persistent KDA kernel at F16K with CuTe DSL line info; per-source-line executed warp instructions.
# usage: run_fip_ncu_lines.sh <overlay> <outdir>
OV=$1; OUT=$2; mkdir -p $OUT; cd $OV/benchmarks/kimi_k3
export CUTE_DSL_LINEINFO=1
ncu --set full --import-source yes -k regex:_pkd -s 2 -c 1 -f -o $OUT/r python3 bench_kda_fi_persistent.py --impls fip --iters 1 > $OUT/app.log 2>&1
grep -E "==ERR|Error" $OUT/app.log | head -5
ncu -i $OUT/r.ncu-repz --page source --csv --print-source cuda,sass > $OUT/src.csv 2>/dev/null
ls -la $OUT/src.csv
python3 $OV/benchmarks/kimi_k3/ncu_lines.py $OUT/src.csv kernel.py 2>&1 | head -60
