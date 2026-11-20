#!/bin/bash
# P-MLA: ncu source-level profile of h8: pmla_ncu.sh <run> <tag> [knobs]
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/$1; T=$2; shift 2
mkdir -p $R
export CUTE_DSL_LINEINFO=1 CUTE_DSL_CACHE_DIR=$R/dslcache-$T
timeout 1200 ncu --set full --import-source yes -k regex:cutlass --launch-skip 1 --launch-count 1 -f -o $R/$T python3 benchmarks/kernels/pmla_ncu.py "$@" > $R/$T.ncu.log 2>&1
echo "ncu rc=$?" >> $R/summary.txt
ncu -i $R/$T.ncu-repz --page details --section SpeedOfLight --section WarpStateStats --section SchedulerStats 2>&1 | grep -E "Duration|Issued|Eligible|Active Warps|No Eligible|Stall|Throughput|Cycles" | head -24 | cut -c1-160 >> $R/summary.txt
ncu -i $R/$T.ncu-repz --page source --csv --print-source cuda > $R/$T.src.csv 2>&1
python3 benchmarks/kernels/pmla_ncu_parse.py $R/$T.src.csv 30 >> $R/summary.txt 2>&1
echo DONE >> $R/summary.txt
