#!/bin/bash
# D-KDA round 7 (a): KdaStream operand path v2 = R2K16 layout, per-key operands in tensor
# memory via producer tcgen05.cp.128x256b with SBO 0 (lane L <- key group L % 8), no
# storer warp, no quarter barrier ('k', MATH 10). References in the same run: stream and
# 'm' (R4K8, operands by LDS fan-out). All die-local.
B=benchmarks/kernels/k31_vr/kda_skel.py
timeout 900 python3 $B --n 32 64 96 128 --place local --sin 8192 --arms rmwt512o1k rmwt512o1 2>&1 | grep -E "us/layer|regs|rror|check|SMs" | sed "s/ (R .*ok=/ ok=/" | cut -c1-150
timeout 600 python3 $B --n 32 64 96 128 --place local --sin 0 --arms rmwt512o1m rmwt512o1 2>&1 | grep -E "us/layer|rror" | sed "s/ (R .*ok=/ ok=/" | cut -c1-150
M="smsp__inst_executed.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,smsp__issue_active.avg.pct_of_peak_sustained_active,smsp__average_warp_latency_issue_stalled_long_scoreboard.ratio,smsp__average_warp_latency_issue_stalled_mio_throttle.ratio,smsp__average_warp_latency_issue_stalled_short_scoreboard.ratio,smsp__average_warp_latency_issue_stalled_wait.ratio,smsp__average_warp_latency_issue_stalled_barrier.ratio,smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,gpu__time_duration.sum"
timeout 300 ncu --metrics $M -k regex:tma_k -s 10 -c 1 --clock-control none --csv python3 $B --n 96 --place local --sin 8192 --arms rmwt512o1k --iters 2 --layers 24 2>/dev/null | grep '"' | awk -F'","' '{print $(NF-2)" "$NF}' | tr -d '"' | tail -10
