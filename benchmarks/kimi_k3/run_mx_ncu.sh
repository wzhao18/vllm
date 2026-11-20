#!/bin/bash
# D-GEMM: ncu metrics for vendored sm107 MXFP8 tactics (one profiled launch per arm).
# usage: run_mx_ncu.sh N K ROWS ARM [ARM ...]
N=$1; K=$2; R=$3; shift 3
M=gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,lts__t_sector_hit_rate.pct,lts__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__m_xbar2l1tex_read_bytes.sum,sm__cycles_active.avg,gpc__cycles_elapsed.max,smsp__inst_executed.sum,launch__grid_size,launch__shared_mem_per_block_dynamic
for A in "$@"; do
  echo "=== $A"
  ncu --metrics $M -k regex:Sm107 --launch-skip 6 --launch-count 1 --csv \
    python3 benchmarks/kimi_k3/bench_mx_ncu.py --n $N --k $K --rows $R --arm $A 2>&1 \
    | grep -E '^"[0-9]' | awk -F'","' '{print $(NF-2), $(NF-1), $NF}' | tr -d '"'
done
