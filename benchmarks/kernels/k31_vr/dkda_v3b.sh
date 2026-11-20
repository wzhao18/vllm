#!/bin/bash
# D-KDA v3b: ncu of KdaRing v3 at n96 striped: stall reasons, per-function and per-line stalls.
B=benchmarks/kernels/k31_vr/kda_sp_bench.py; A=benchmarks/kernels/k31_vr/dkda_srcagg.py; L=benchmarks/kernels/k31_vr/dkda_lines.py
R=vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/kda/ring.py
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws; O=$K/runs/D-KDA/v3b; mkdir -p $O
export CUTE_DSL_CACHE_DIR=/tmp/dkda-v3b-$$ CUTE_DSL_LINEINFO=1
timeout 900 ncu --set full --import-source yes -k regex:KdaRing -s 6 -c 1 --clock-control none -f -o $O/ring96 python3 $B --n 96 --kinds ring --gap 1 --pending 4 --place striped --mx --iters 2 --layers 8 2>&1 | grep -E "rror|==PROF== Report" | head -3
ncu -i $O/ring96.ncu-repz --page raw --csv --metrics regex:smsp__average_warps_issue_stalled_.*_per_issue_active.ratio,smsp__issue_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,smsp__inst_executed.sum 2>/dev/null > $O/raw.csv
python3 -c "
import csv
rows=list(csv.reader(open('$O/raw.csv'))); h=rows[0]; v=rows[-1]
f=lambda x: float(x.replace(',','')) if x.replace(',','').replace('.','',1).isdigit() else 0.0
st=sorted([(f(v[i]),h[i].replace('smsp__average_warps_issue_stalled_','').replace('_per_issue_active.ratio','')) for i in range(len(h)) if 'stalled' in h[i]],reverse=True)
print('stalls/issue:',' '.join(f'{n}={x:.2f}' for x,n in st[:10]))
print('other:',' '.join(f'{h[i]}={v[i]}' for i in range(len(h)) if 'stalled' not in h[i] and ('__' in h[i])))
"
ncu -i $O/ring96.ncu-repz --page source --csv --print-source cuda,sass > $O/ring96.mix.csv 2>/dev/null
python3 $A $O/ring96.mix.csv $R | head -16
echo "== top lines by stall samples"
python3 $L $O/ring96.mix.csv 24 1 | cut -c1-150
