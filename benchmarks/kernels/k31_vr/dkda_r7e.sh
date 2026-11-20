#!/bin/bash
# D-KDA round 7 (e): operand blocks in their own 3-slot ring (TMA'd ~3 items ahead, copied
# to TMEM one buffer ahead) instead of riding on the page barrier. Arms: 'z' plain stream
# with the new producer (no TMEM), 'y' TMEM handshake without math, 'j' R2K16 math,
# old stream. 8 KiB operand block, die-local.
B=benchmarks/kernels/k31_vr/kda_skel.py
timeout 900 python3 $B --n 32 64 96 128 --place local --sin 8192 --arms rmwt512o1z rmwt512o1y rmwt512o1j 2>&1 | grep -E "us/layer|regs|rror|check" | sed "s/ (R .*ok=/ ok=/" | cut -c1-150
M="smsp__inst_executed.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,smsp__issue_active.avg.pct_of_peak_sustained_active,smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,gpu__time_duration.sum"
timeout 300 ncu --metrics $M -k regex:tma_k -s 10 -c 1 --clock-control none --csv python3 $B --n 96 --place local --sin 8192 --arms rmwt512o1j --iters 2 --layers 24 2>/dev/null | grep '"' | awk -F'","' '{print $(NF-2)" "$NF}' | tr -d '"' | tail -8
