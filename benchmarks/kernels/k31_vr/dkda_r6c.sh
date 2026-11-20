#!/bin/bash
# D-KDA round 6 (c): operands shared through tensor memory (kda_skel MATH 8 't': one warp per
# TMEM lane quarter LDS + tcgen05.st's the item's 104 operand floats per lane, the 4 warps of
# the quarter tcgen05.ld them) vs smem operands ('m') and the operand-LDS/4 proxy ('p'), RMW
# and read-only, die-local; lazy F=2 arms (state written every 2nd step) for the SOL path.
B=benchmarks/kernels/k31_vr/kda_skel.py
for arm in rmwt512o1 rmwt512o1t rmwt512o1m rmwt512o1p rot512o1t; do
  timeout 600 python3 $B --n 32 64 96 128 --place local --sin 0 --arms $arm 2>&1 | grep -E "us/layer|regs|rror" | sed "s/ (R .*ok=/ ok=/" | cut -c1-120
done
echo "-- lazy F=2 (8 KiB ring per item)"
for arm in lazyt512o1 lazyt512o1t lazyt512o1m; do
  timeout 600 python3 $B --n 32 64 96 128 --place local --sin 0 --fold 2 --arms $arm 2>&1 | grep -E "us/layer|rror" | sed "s/ (R .*ok=/ ok=/" | cut -c1-120
done
M="gpu__time_duration.sum,smsp__inst_executed.sum,smsp__issue_active.avg.pct_of_peak_sustained_active,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,smsp__average_warps_issue_stalled_tmem_per_issue_active.ratio"
for arm in rmwt512o1t; do
  timeout 300 ncu --metrics $M -k regex:tma_k -s 10 -c 1 --clock-control none --csv python3 $B --n 96 --place local --sin 0 --arms $arm --iters 2 --layers 24 2>/dev/null | grep '"' | awk -F'","' '{print $(NF-2)" "$NF}' | tr -d '"' | sed "s/^/$arm /" | tail -10
done
