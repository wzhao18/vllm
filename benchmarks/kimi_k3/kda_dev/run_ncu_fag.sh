#!/bin/bash
# P-KDA2: ncu WarpStateStats of the k-th persistent-KDA launch of bench_fip_fag.py: bash run_ncu_fag.sh <tag> <skip> [dev json...]
cd /k31ov
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws; TAG=$1; SKIP=$2; shift 2
export CUTE_DSL_CACHE_DIR=$K/runs/P-KDA2/cutecache
OUT=$K/runs/P-KDA2/ncu/$TAG; NCU=$K/tools/nsight/bin/ncu
timeout 900 $NCU --section WarpStateStats --section SchedulerStats --section InstructionStats --section LaunchStats -k regex:pkd -s $SKIP -c 1 -f -o $OUT python3 -u benchmarks/kimi_k3/bench_fip_fag.py --rounds 1 --iters 1 --lens 5461,5461,5462 --dev "$@" > /dev/null 2>&1
$NCU -i $OUT.ncu-repz --page details --print-details all 2>&1 | grep -E "Duration|Stall (No Instruction|Barrier|Long|Short|Wait|Not Selected|Math|MIO)|Warp Cycles Per Issued|Issued Warp Per|Executed Instructions  |Shared Memory Configuration|Dynamic Shared" | head -20
