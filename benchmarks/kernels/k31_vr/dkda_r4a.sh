#!/bin/bash
# D-KDA round 4 (a): ncu of production KdaFlow v18 at n96 striped mx: full vs prep+gate
# skipped (the scan-side floor KdaFlow-P runs into): stall reasons, smem wavefronts,
# DRAM bytes, per-function and per-line stalls.
B=benchmarks/kernels/k31_vr/kda_sp_bench.py; A=benchmarks/kernels/k31_vr/dkda_srcagg.py; L=benchmarks/kernels/k31_vr/dkda_lines.py
R=vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/kda/flow.py
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws; O=$K/runs/D-KDA/r4a; mkdir -p $O
export CUTE_DSL_CACHE_DIR=/tmp/dkda-r4a-$$ CUTE_DSL_LINEINFO=1
M=smsp__issue_active.avg.pct_of_peak_sustained_active,dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,smsp__inst_executed.sum,smsp__warps_active.avg.per_cycle_active
for arm in full noprep; do
  sk=""; [ $arm = noprep ] && sk="--skip prep,gate"
  echo "=== arm $arm"
  timeout 900 ncu --set full --import-source yes -k regex:KdaFlow -s 6 -c 1 --clock-control none -f -o $O/$arm python3 $B --n 96 --kinds flow --gap 1 --pending 4 --place striped --mx --iters 2 --layers 8 $sk 2>&1 | grep -E "rror|==PROF== Report" | head -3
  ncu -i $O/$arm.ncu-repz --page raw --csv --metrics regex:smsp__average_warps_issue_stalled_.*_per_issue_active.ratio,$M 2>/dev/null > $O/$arm.raw.csv
  python3 -c "
import csv
rows=list(csv.reader(open('$O/$arm.raw.csv'))); h=rows[0]; v=rows[-1]
f=lambda x: float(x.replace(',','')) if x.replace(',','').replace('.','',1).isdigit() else 0.0
st=sorted([(f(v[i]),h[i].replace('smsp__average_warps_issue_stalled_','').replace('_per_issue_active.ratio','')) for i in range(len(h)) if 'stalled' in h[i]],reverse=True)
print('stalls/issue:',' '.join(f'{n}={x:.2f}' for x,n in st[:10]))
print('other:',' '.join(f'{h[i]}={v[i]}' for i in range(len(h)) if 'stalled' not in h[i] and ('__' in h[i])))
"
  ncu -i $O/$arm.ncu-repz --page source --csv --print-source cuda,sass > $O/$arm.mix.csv 2>/dev/null
  python3 $A $O/$arm.mix.csv $R | head -14
  echo "== top lines by stall samples ($arm)"
  python3 $L $O/$arm.mix.csv 16 1 | cut -c1-150
done
