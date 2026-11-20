#!/bin/bash
# D-KDA round 5 (f): KdaCore v2 scan cost split - cycle traces with the checkpoint store
# skipped (store) and with STS but no TMA store (tma), n96 striped; ncu (stalls, IPC).
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws; O=$K/runs/D-KDA/r5f; mkdir -p $O
C=/tmp/dkda-r5f-$$
for sk in store tma; do
  echo "== trace n96 striped skip=$sk (last launch, g0 w0 and g1 w0 of 4 CTAs)"
  CUTE_DSL_CACHE_DIR=$C-$sk timeout 900 python3 $B --n 96 --kinds core --gap 1 --pending 4 --place striped --mx --trace --skip $sk --layers 2 --iters 1 2>&1 | grep -E "CTRACE|rror" > $O/trace96-$sk.txt
  tail -32 $O/trace96-$sk.txt | grep " w 0 " | cut -c1-200
done
echo "== ncu core n96 striped"
CUTE_DSL_CACHE_DIR=$C-nc timeout 1200 ncu --set full -k regex:Kda -s 2 -c 1 -o $O/core96 -f python3 $B --n 96 --kinds core --gap 1 --pending 4 --place striped --mx --iters 1 --layers 4 > $O/ncu.log 2>&1
R=$(ls $O/core96.ncu-rep* | head -1); echo "report $R"
ncu -i $R --csv --page raw --metrics smsp__inst_executed.sum,sm__cycles_elapsed.avg,gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,launch__registers_per_thread,smsp__warps_active.avg.per_cycle_active,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,smsp__inst_executed_pipe_fma.sum 2>/dev/null | tail -2 | cut -c1-900
ncu -i $R --page details --section WarpStateStats 2>/dev/null | grep -E "Stall|Warp Cycles" | head -24 | cut -c1-140
