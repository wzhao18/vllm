#!/bin/bash
# P-MLA r7i: M41b rung finder: (1) ladder (pmla12 form) on 4 index/KV data modes; (2) SASS of the h8 C0 copier loop vs the ladder
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/r7i
mkdir -p $R/dump
cd /k31ov/benchmarks/kernels
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a -Xptxas -v -o $R/pmla18 pmla18_ladder.cu -lcuda > $R/build.log 2>&1 || { grep -m5 error $R/build.log; exit 1; }
for m in 0 1 2 3; do echo "== mode $m"; timeout 300 $R/pmla18 $m | grep -v '^SMs'; done > $R/ladder.log 2>&1
cat $R/ladder.log
cd /k31ov
CUTE_DSL_KEEP_CUBIN=1 CUTE_DSL_DUMP_DIR=$R/dump timeout 600 python3 -c "
import torch
from vllm.models.kimi_k3.nvidia.specialized_prefill import sparse_mla_h8 as m
m._compiled(212, (0,)*27 + (1,), 0)
" > $R/dump.log 2>&1
C=$(ls -t $R/dump/*.cubin 2>/dev/null | head -1); echo "cubin: $C"
[ -n "$C" ] && cuobjdump -sass $C > $R/h8c0.sass 2>&1
cuobjdump -sass $R/pmla18 > $R/ladder.sass 2>&1
for f in h8c0 ladder; do echo "== $f: $(grep -c LDGSTS $R/$f.sass) LDGSTS, $(grep -cE '^\s+/\*[0-9a-f]{4}\*/' $R/$f.sass) instr"; done
