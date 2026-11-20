#!/bin/bash
# ncu of one kernel launch of any command.
# usage: run_ncu_cmd.sh <overlay> <outdir> <kernel regex> <skip> CMD...
OV=$1; OUT=$2; KR=$3; SKIP=$4; shift 4; mkdir -p $OUT; cd $OV
ncu --set full -k regex:$KR -s $SKIP -c 1 -f -o $OUT/r "$@" > $OUT/app.log 2>&1
grep -E "==ERR|Error|error" $OUT/app.log | head -5
ncu -i $OUT/r.ncu-repz --page details --section SpeedOfLight --section MemoryWorkloadAnalysis \
  --section SchedulerStats --section Occupancy --section LaunchStats 2>&1 \
  | grep -E "Duration|DRAM Through|Memory Through|L1/TEX Cache|L2 Cache Through|Issued Warp|Achieved Occ|Theoretical Occ|Registers Per|Grid Size|Block Size|Waves Per|Max Bandwidth|Mem Busy|Memory Throughput|L2 Hit|L1/TEX Hit|Dynamic Shared"
ncu -i $OUT/r.ncu-repz --page source --csv --print-source sass > $OUT/sass.csv 2>/dev/null
python3 $OV/benchmarks/kimi_k3/ncu_top.py $OUT/sass.csv | head -30
