#!/bin/bash
# D-KDA round 4 (c): I-cache lever. ncu r4a: KdaFlow full stalls no_instruction 1.41/issue
# (noprep arm 0.16): ~112 KB of straight-line SASS. Arm: the scan's chunk loop as a
# runtime loop (one copy of the chunk code). Bitwise tests (rolled flow, flowp vs flow),
# then striped mx --gap 1 --pending 4 timing, then ncu stall summary of the rolled arm.
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
T=tests/models/kimi_k3/test_specialized_decode_sp_kda_flow_gpu.py
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws; O=$K/runs/D-KDA/r4c; mkdir -p $O
export CUTE_DSL_CACHE_DIR=/tmp/dkda-r4c-$$
echo "== test rolled: flow/flowp fold vs stock (n33 graph), flowp bitwise vs flow"
timeout 900 python3 -c "
import sys, pytest
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda import flow
flow.SCAN_ROLLED = True
sys.exit(pytest.main(['--noconftest', '-q', '-rf', '-x', '$T', '-k', '(fold_matches and 33 and graph and not local and not chain and not ring) or (core_is_bitwise and flowp and 33)']))
" 2>&1 | grep -E "passed|failed|^FAILED|rror:|Mismatch|assert" | cut -c1-300 | head -12
echo "== timing (unrolled, rolled)"
for k in flow flowp; do
  for r in "" --rolled; do
    echo "-- $k $r"
    timeout 900 python3 $B --n 32 96 128 --kinds $k --gap 1 --pending 4 --place striped --mx $r 2>&1 | grep -E "us/layer|rror|FAILED" | cut -c1-200
  done
done
echo "== ncu stalls: flow rolled, flowp rolled (n96)"
M=smsp__issue_active.avg.pct_of_peak_sustained_active,gpu__time_duration.sum,smsp__inst_executed.sum,launch__registers_per_thread,l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum
for k in flow flowp; do
  timeout 900 ncu --metrics regex:smsp__average_warps_issue_stalled_.*_per_issue_active.ratio,$M -k regex:KdaFlow -s 6 -c 1 --clock-control none --csv --page raw python3 $B --n 96 --kinds $k --gap 1 --pending 4 --place striped --mx --rolled --iters 2 --layers 8 2>/dev/null > $O/$k.raw.csv
  python3 -c "
import csv
rows=[r for r in csv.reader(open('$O/$k.raw.csv')) if len(r)>10]; h=rows[0]; v=rows[-1]
f=lambda x: float(x.replace(',','')) if x.replace(',','').replace('.','',1).isdigit() else 0.0
st=sorted([(f(v[i]),h[i].replace('smsp__average_warps_issue_stalled_','').replace('_per_issue_active.ratio','')) for i in range(len(h)) if 'stalled' in h[i]],reverse=True)
print('$k rolled stalls/issue:',' '.join(f'{n}={x:.2f}' for x,n in st[:8]))
print('$k rolled other:',' '.join(f'{h[i]}={v[i]}' for i in range(len(h)) if 'stalled' not in h[i] and ('__' in h[i])))
"
done
