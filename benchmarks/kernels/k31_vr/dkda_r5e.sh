#!/bin/bash
# D-KDA round 5 (e): where KdaCore v2's time goes - per-phase cycle trace (n96 striped,
# full), the store-wait-free arm (TMA stores without wait_group.read: timing only), ncu.
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws; O=$K/runs/D-KDA/r5e; mkdir -p $O
C=/tmp/dkda-r5e-$$
echo "== trace n96 striped (last launch)"
CUTE_DSL_CACHE_DIR=$C-tr timeout 900 python3 $B --n 96 --kinds core --gap 1 --pending 4 --place striped --mx --trace --layers 2 --iters 1 2>&1 | grep -E "CTRACE|rror" > $O/trace96.txt
tail -32 $O/trace96.txt | sort -t' ' -k3,3n -k5,5n -k7,7n | cut -c1-200
echo "-- core skip=[nowait] striped"
CUTE_DSL_CACHE_DIR=$C-nw timeout 900 python3 $B --n 96 128 --kinds core --gap 1 --pending 4 --place striped --mx --skip nowait 2>&1 | grep -E "us/layer|rror|FAILED" | cut -c1-120
echo "== ncu core n96 striped"
CUTE_DSL_CACHE_DIR=$C-nc timeout 1200 ncu --set full -k regex:Kda -s 2 -c 1 -o $O/core96 -f python3 $B --n 96 --kinds core --gap 1 --pending 4 --place striped --mx --iters 1 --layers 4 > $O/ncu.log 2>&1
tail -3 $O/ncu.log | cut -c1-200
ncu -i $O/core96.ncu-rep --csv --page raw --metrics smsp__inst_executed.sum,sm__cycles_elapsed.avg,gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,launch__registers_per_thread,smsp__warps_active.avg.per_cycle_active 2>/dev/null | tail -2 | cut -c1-700
ncu -i $O/core96.ncu-rep --page details --section WarpStateStats 2>/dev/null | grep -E "Stall|Warp Cycles" | head -24 | cut -c1-140
