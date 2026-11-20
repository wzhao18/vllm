#!/bin/bash
# D-KDA R2: ncu of the KdaRing core (n96, mx, local, prep skipped): per-function instr/stalls, warp state, + ring test detail.
B=benchmarks/kernels/k31_vr/kda_sp_bench.py; A=benchmarks/kernels/k31_vr/dkda_srcagg.py
R=vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/kda/ring.py
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws; O=$K/runs/D-KDA/r2; mkdir -p $O
export CUTE_DSL_CACHE_DIR=/tmp/dkda-r2-$$ CUTE_DSL_LINEINFO=1
timeout 900 ncu --set full --import-source yes -k regex:KdaRing -s 6 -c 1 --clock-control none -f -o $O/core python3 $B --n 96 --kinds ring --gap 1 --pending 4 --place local --mx --skip prepk --iters 2 --layers 8 2>&1 | grep -E "rror|==PROF==" | head -4
ncu -i $O/core.ncu-repz --page source --csv --print-source cuda,sass > $O/core.mix.csv 2>/dev/null; python3 $A $O/core.mix.csv $R
ncu -i $O/core.ncu-repz --page details --section WarpStateStats --section SchedulerStats --section LaunchStats --section Occupancy 2>/dev/null | grep -E "Registers|Warp Cycles|Eligible|Issued Warp|Active Warps|Stall|No Eligible|Duration|Achieved Occ" | head -24
echo "== test"; T=tests/models/kimi_k3/test_specialized_decode_sp_kda_flow_gpu.py; timeout 900 python3 -m pytest --noconftest -q -x $T -k "ring_is_bitwise" 2>&1 | grep -E "^E |passed|failed" | cut -c1-200 | head -12
