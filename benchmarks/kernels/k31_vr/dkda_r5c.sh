#!/bin/bash
# D-KDA round 5 (c): KdaCore v1 - bitwise tests vs KdaFlow (prep stand-in), the n128
# striped launch failure under compute-sanitizer, ncu of core n96 (stalls, instructions),
# store-path arms (no store / STS without TMA store) at n96/n128 striped mx.
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
T=tests/models/kimi_k3/test_specialized_decode_sp_kda_flow_gpu.py
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws; O=$K/runs/D-KDA/r5c; mkdir -p $O
C=/tmp/dkda-r5c-$$
echo "== tests"
CUTE_DSL_CACHE_DIR=$C-t timeout 1500 python3 -m pytest --noconftest -q -rf -x $T -k "core_is_bitwise and core" 2>&1 | grep -E "passed|failed|^FAILED|rror:|Mismatch|assert|differ" | cut -c1-300 | head -12
echo "== sanitizer core n128 striped"
CUTE_DSL_CACHE_DIR=$C-s timeout 1500 compute-sanitizer --tool memcheck --print-limit 4 python3 $B --n 128 --kinds core --gap 1 --pending 4 --place striped --mx --layers 2 --iters 1 2>&1 | grep -E "=========|us/layer|FAILED" | head -24 | cut -c1-200
for sk in "" store tma; do
  echo "-- core skip=[$sk] striped"
  CUTE_DSL_CACHE_DIR=$C-b$sk timeout 900 python3 $B --n 96 128 --kinds core --gap 1 --pending 4 --place striped --mx ${sk:+--skip $sk} 2>&1 | grep -E "us/layer|rror|FAILED" | cut -c1-120
done
echo "== ncu core n96 striped"
CUTE_DSL_CACHE_DIR=$C-b timeout 900 ncu --set full -k regex:KdaCore -s 2 -c 1 -o $O/core96 -f python3 $B --n 96 --kinds core --gap 1 --pending 4 --place striped --mx --iters 1 --layers 4 > /dev/null 2>&1
ncu -i $O/core96.ncu-rep --csv --page raw --metrics smsp__inst_executed.sum,sm__cycles_elapsed.avg,gpu__time_duration.sum,smsp__issue_active.avg.pct_of_peak_sustained_active,dram__bytes_read.sum,dram__bytes_write.sum,launch__registers_per_thread 2>/dev/null | tail -2 | cut -c1-600
ncu -i $O/core96.ncu-rep --page details --section WarpStateStats 2>/dev/null | grep -E "Stall|Warp Cycles" | head -24 | cut -c1-140
