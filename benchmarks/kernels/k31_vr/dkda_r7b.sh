#!/bin/bash
# D-KDA round 7 (b): tensor-memory operand path with the loads pipelined by phase ('j',
# MATH 11): R2K16 x 16 consumer warps (96 regs) and R4K16 x 8 consumer warps (168 regs),
# vs the unpipelined R2K16 ('k') and the stream with the same 8 KiB operand block. Local.
B=benchmarks/kernels/k31_vr/kda_skel.py
timeout 900 python3 $B --n 32 64 96 128 --place local --sin 8192 --arms rmwt512o1j rmwt256o1j rmwt512o1k rmwt512o1 2>&1 | grep -E "us/layer|regs|rror|check" | sed "s/ (R .*ok=/ ok=/" | cut -c1-150
M="smsp__inst_executed.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,smsp__issue_active.avg.pct_of_peak_sustained_active,smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,smsp__average_warp_latency_issue_stalled_long_scoreboard.ratio,smsp__average_warp_latency_issue_stalled_short_scoreboard.ratio,smsp__average_warp_latency_issue_stalled_wait.ratio,smsp__average_warp_latency_issue_stalled_mio_throttle.ratio,smsp__average_warp_latency_issue_stalled_math_pipe_throttle.ratio,smsp__average_warp_latency_issue_stalled_not_selected.ratio,smsp__average_warp_latency_issue_stalled_selected.ratio,gpu__time_duration.sum"
for arm in rmwt512o1j rmwt256o1j; do
  timeout 300 ncu --metrics $M -k regex:tma_k -s 10 -c 1 --clock-control none --csv python3 $B --n 96 --place local --sin 8192 --arms $arm --iters 2 --layers 24 2>/dev/null | grep '"' | awk -F'","' '{print $(NF-2)" "$NF}' | tr -d '"' | sed "s/^/$arm /" | tail -12
done
