#!/bin/bash
# D-KDA v2b: KdaRing v2 decomposition at n32/n96 (local, mx): full, no scan math, no checkpoint stores,
# no prep math, pure stream (no scan math, no stores); + ncu summary of the full kernel at n96.
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws; O=$K/runs/D-KDA/v2b; mkdir -p $O
export CUTE_DSL_CACHE_DIR=/tmp/dkda-v2b-$$
for sk in "" scan store prep scan,store scan,store,prep; do
  echo "== skip=[$sk]"
  timeout 600 python3 $B --n 32 96 --kinds ring --gap 1 --pending 4 --place local --mx ${sk:+--skip $sk} 2>&1 | grep -E "us/layer|rror|FAILED" | cut -c1-80
done
export CUTE_DSL_LINEINFO=1
timeout 900 ncu --set full --import-source yes -k regex:KdaRing -s 6 -c 1 --clock-control none -f -o $O/ring96 python3 $B --n 96 --kinds ring --gap 1 --pending 4 --place local --mx --iters 2 --layers 8 2>&1 | grep -E "rror|==PROF==" | head -4
ncu -i $O/ring96.ncu-repz --page details --section SpeedOfLight --section WarpStateStats --section SchedulerStats --section LaunchStats --section MemoryWorkloadAnalysis 2>/dev/null | grep -E "Registers|Warp Cycles|Eligible|Issued Warp|Active Warps|No Eligible|Duration|DRAM Throughput|Memory Throughput|Mem Busy|Max Bandwidth|Shared Memory|Stall" | cut -c1-120 | head -30
