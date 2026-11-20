#!/bin/bash
# D-KDA round 6 (b): which SM resource binds the release-fast RMW form with KdaFlow's math
# (kda_skel 'm' = fold + 8 dots, 4 rows x 8 keys per thread, 512 consumers, 3 x 64 KiB slots,
# STG write-back of the folded state, die-local). Attribution arms: p = operand loads / 4,
# b = no butterfly, q = both; ro = no write-back. + ncu (issue, smem wavefronts, stalls).
B=benchmarks/kernels/k31_vr/kda_skel.py
for arm in rmwt512o1 rmwt512o1m rmwt512o1p rmwt512o1b rmwt512o1q rot512o1 rot512o1m rot512o1q; do
  timeout 600 python3 $B --n 32 96 --place local --sin 0 --arms $arm 2>&1 | grep -E "us/layer|regs|rror" | sed "s/ (R .*ok=/ ok=/" | cut -c1-120
done
echo "-- m with 12 KiB per-item operand block on the slot barrier"
timeout 600 python3 $B --n 32 96 --place local --sin 12288 --arms rmwt512o1m rmwt512o1 2>&1 | grep -E "us/layer|rror" | sed "s/ (R .*ok=/ ok=/" | cut -c1-120
M="gpu__time_duration.sum,smsp__inst_executed.sum,smsp__issue_active.avg.pct_of_peak_sustained_active,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,l1tex__data_pipe_lsu_wavefronts.sum,smsp__inst_executed_op_shfl.sum,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active"
for arm in rmwt512o1m rmwt512o1q rot512o1m; do
  timeout 300 ncu --metrics $M -k regex:tma_k -s 10 -c 1 --clock-control none --csv python3 $B --n 96 --place local --sin 0 --arms $arm --iters 2 --layers 24 2>/dev/null | grep '"' | awk -F'","' '{print $(NF-2)" "$NF}' | tr -d '"' | sed "s/^/$arm /" | tail -13
done
