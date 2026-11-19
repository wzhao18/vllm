#!/bin/bash
# P-GEMM KDA conv-epilogue GEMM: $1 = base | conv | test
cd /k31ov
F='^\{|Error|error|Traceback|File "/k31ov|raise|Exception'
case "$1" in
  base) timeout 1500 python3 benchmarks/kimi_k3/pgemm_conv_bench.py --mode base 2>&1 | grep -E "$F" | head -40 ;;
  conv) shift
    export CUTE_DSL_KEEP_PTX=1 CUTE_DSL_DUMP_DIR=/tmp/pgc_dump
    rm -rf /tmp/pgc_dump; mkdir -p /tmp/pgc_dump
    timeout 1200 python3 benchmarks/kimi_k3/pgemm_conv_bench.py --mode conv --seqlens 5000,1,2,3,4000,7000 "$@" > /tmp/pgc1.log 2>&1
    grep -E "$F" /tmp/pgc1.log | grep -v "^\s*$" | cut -c1-700 | tail -30
    for x in $(ls /tmp/pgc_dump/*kda_conv*.ptx 2>/dev/null | head -2); do echo "== $x"; ptxas -arch=sm_107a -v $x -o /tmp/pgc.cubin 2>&1 | grep -E "registers|spill" | head -4; done
    ls /tmp/pgc_dump | head -5 ;;
  skel) shift; timeout 1500 python3 benchmarks/kimi_k3/pgemm_conv_bench.py --mode skel "$@" 2>&1 | grep -E "$F" | cut -c1-400 | head -40 ;;
  b3) shift; timeout 1500 python3 benchmarks/kimi_k3/pgemm_conv_bench.py --mode b3 "$@" 2>&1 | grep -E "$F" | cut -c1-400 | head -40 ;;
  test) timeout 900 python3 -m pytest --noconftest -q tests/models/kimi_k3/test_kda_conv_gemm.py 2>&1 | grep -E 'passed|failed|Error|assert|^E ' | head -30 ;;
esac
if [ "$1" = ncuskel ]; then
  # $2 = JSON tactic; profiles nvjet and the CuTe kernel once each at M16384
  cd /k31ov; S="--section SpeedOfLight --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis --section Occupancy --section LaunchStats --section WarpStateStats --section SchedulerStats"
  for KR in nvjet cutlass; do
    timeout 900 ncu -k regex:$KR -c 1 $S python3 benchmarks/kimi_k3/pgemm_conv_bench.py --mode skel --ms 16384 --tactic "$2" > /tmp/pgn_$KR.log 2>&1
    echo "=== ncu $KR"; grep -E '^  [a-zA-Z_]|Duration|Frequency|Throughput|Pipe|Busy|Issue|Eligible|Hit Rate|Registers|Shared Memory Config|Block Size|Grid Size|Warp Cycles|Stall|ERR|rror' /tmp/pgn_$KR.log | grep -v '^\s*$' | cut -c1-150 | head -45
  done
fi
if [ "$1" = ncu ]; then
  cd /k31ov; shift
  timeout 1500 ncu -k regex:kda_conv_gemm -c 1 --launch-skip 1 --section WarpStateStats --section SchedulerStats --section ComputeWorkloadAnalysis --section SpeedOfLight --section Occupancy --section LaunchStats --section SourceCounters python3 benchmarks/kimi_k3/pgemm_conv_bench.py --mode ncu "$@" > /tmp/pgc_ncu.log 2>&1
  grep -E "Stall|stall|Issue|Eligible|Active Warps|Registers|Duration|Throughput|Local|Warp Cycles|smsp|Executed Ipc|Branch|ERR|Error" /tmp/pgc_ncu.log | grep -v "^\s*$" | head -60
fi
