#!/bin/bash
# ncu of one F16K prep64_tc launch (the k64_diag.py --ptime timing launch).
# usage: run_ncu_prep.sh <overlay> <outdir> [kernel regex] [skip]
OV=$1; OUT=$2; KR=${3:-prep64_tc}; SKIP=${4:-0}; mkdir -p $OUT
ncu --set full -k regex:$KR -s $SKIP -c 1 -f -o $OUT/r python3 -u $OV/benchmarks/kimi_k3/k64_diag.py --ptime \
  > $OUT/app.log 2>&1; grep -E "F16K|phases|==ERR|==WARN|Error|error" $OUT/app.log | head -8; tail -4 $OUT/app.log
ncu -i $OUT/r.ncu-repz --page details --section WarpStateStats --section SchedulerStats \
  --section SpeedOfLight --section Occupancy --section LaunchStats 2>&1 | grep -vE '^\s*$|----' | head -80
ncu -i $OUT/r.ncu-repz --page source --csv --print-source sass > $OUT/sass.csv 2>/dev/null
ncu -i $OUT/r.ncu-repz --page source --csv --print-source cuda > $OUT/cuda.csv 2>/dev/null
python3 $OV/benchmarks/kimi_k3/ncu_top.py $OUT/sass.csv
python3 $OV/benchmarks/kimi_k3/ncu_lines.py $OUT/cuda.csv
