#!/bin/bash
# D-KDA round 4 (e): S0c probe - cycles of setmaxnreg role switches inside KdaFlowP
# (4 rounds x 4 dependent switches between the scan groups and the prep warpgroup), and
# the register / local-memory (spill) footprint of KdaFlow and KdaFlowP (ncu launch stats).
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws; O=$K/runs/D-KDA/r4e; mkdir -p $O
export CUTE_DSL_CACHE_DIR=/tmp/dkda-r4e-$$
echo "== S0c switch probe (flowp n32 striped mx)"
timeout 900 python3 $B --n 32 --kinds flowp --gap 1 --pending 4 --place striped --mx --switch --iters 1 --layers 2 2>&1 | grep -E "^S0C|rror" | sort | uniq -c | sort -rn | head -24 | cut -c1-120
echo "== registers / spills (n96)"
for k in flow flowp; do
  timeout 900 ncu --metrics launch__registers_per_thread,launch__shared_mem_per_block_dynamic,l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum,gpu__time_duration.sum -k regex:KdaFlow -s 2 -c 1 --csv --page raw python3 $B --n 96 --kinds $k --gap 1 --pending 4 --place striped --mx --iters 1 --layers 4 2>/dev/null > $O/$k.csv
  python3 -c "
import csv
rows=[r for r in csv.reader(open('$O/$k.csv')) if len(r)>5]; h=rows[0]; v=rows[-1]
keep=('launch__registers_per_thread','launch__shared_mem_per_block_dynamic','l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum','l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum','gpu__time_duration.sum')
print('$k', ' '.join(f'{h[i]}={v[i]}' for i in range(len(h)) if h[i] in keep))
"
done
