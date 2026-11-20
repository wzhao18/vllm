#!/bin/bash
# P-KDA2: ncu (full set, CuTe line info) of one persistent-KDA launch, then a per-line /
# per-phase instruction table.  bash run_ncu.sh <tag> "<lens>" '<variant json>' [ranges file]
cd /k31ov
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws
TAG=$1; LENS=$2; VAR=$3; RANGES=${4:-/k31ov/benchmarks/kimi_k3/kda_dev/ncu_ranges.txt}
export CUTE_DSL_CACHE_DIR=$K/runs/P-KDA2/cutecache_li
export CUTE_DSL_LINEINFO=1
mkdir -p $CUTE_DSL_CACHE_DIR $K/runs/P-KDA2/ncu
OUT=$K/runs/P-KDA2/ncu/$TAG
NCU=$K/tools/nsight/bin/ncu
echo "NCU $TAG start $(date +%T)"
timeout 1500 $NCU --set full --import-source yes -k regex:pkd -c 1 -f -o $OUT \
  python3 -u benchmarks/kimi_k3/bench_fip_devb.py --rounds 1 --iters 1 --lens $LENS --variants "$VAR" 2>&1 \
  | grep -E "lens=|Error|error|Traceback|==PROF==|==ERROR==" | tail -8
$NCU -i $OUT.ncu-repz --page source --csv --print-source cuda > $OUT.cuda.csv 2>&1
$NCU -i $OUT.ncu-repz --page source --csv --print-source sass > $OUT.sass.csv 2>&1
$NCU -i $OUT.ncu-repz --page details --csv > $OUT.details.csv 2>&1
ls -la $OUT.* | awk '{print $5, $9}'
$NCU -i $OUT.ncu-repz --page source --csv --print-source cuda,sass > $OUT.cs.csv 2>&1; python3 -u benchmarks/kimi_k3/kda_dev/ncu_agg.py $OUT.cs.csv $RANGES 30 2>&1 | head -60
echo "NCU $TAG done $(date +%T)"
