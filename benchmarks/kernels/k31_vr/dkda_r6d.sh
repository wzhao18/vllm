#!/bin/bash
# D-KDA round 6 (d): dot operands by 64-bit shared loads ('x', MATH 9) vs 128-bit ('m'):
# do the two half-warps' identical addresses merge in a 2-phase LDS.64 (x-operand wavefronts /2)?
B=benchmarks/kernels/k31_vr/kda_skel.py
for arm in rmwt512o1x rmwt512o1m; do
  timeout 600 python3 $B --n 32 64 96 128 --place local --sin 0 --arms $arm 2>&1 | grep -E "us/layer|regs|rror" | sed "s/ (R .*ok=/ ok=/" | cut -c1-120
done
M="smsp__inst_executed.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,smsp__issue_active.avg.pct_of_peak_sustained_active"
for arm in rmwt512o1x rmwt512o1m; do
  timeout 300 ncu --metrics $M -k regex:tma_k -s 10 -c 1 --clock-control none --csv python3 $B --n 96 --place local --sin 0 --arms $arm --iters 2 --layers 24 2>/dev/null | grep '"' | awk -F'","' '{print $(NF-2)" "$NF}' | tr -d '"' | sed "s/^/$arm /" | tail -3
done
