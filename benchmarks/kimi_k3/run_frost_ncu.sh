#!/bin/bash
# ncu DRAM bytes / duration / tensor-pipe of the FROST KDA and FlashKDA kernels at F16K.
# usage: run_frost_ncu.sh <overlay> <outdir> <impls> <kernel-regex> <count>
OV=$1; OUT=$2; mkdir -p $OUT; K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws
export PYTHONPATH=$K/venvs/cudnnfe133/site:$PYTHONPATH
M=gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed,launch__grid_size,launch__block_size,launch__registers_per_thread,launch__shared_mem_per_block_dynamic,sm__warps_active.avg.pct_of_peak_sustained_active
timeout 1500 $K/tools/nsight/bin/ncu --metrics $M -k regex:"$4" -c $5 python3 -u $OV/benchmarks/kimi_k3/bench_kda_frost_ab.py --impls $3 --iters 1 > $OUT/ncu.txt 2>&1
grep -E "^  [a-zA-Z_]|^    [a-z_]+__" $OUT/ncu.txt | sed -E "s/ +/ /g" | cut -c1-160 | head -120
