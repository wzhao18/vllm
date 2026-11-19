#!/bin/bash
# Compile score (mode $2, default 0) with PTX kept and print the loop opcode histogram. Usage: run_ptx.sh <dir> [mode]
D=$1; M=${2:-0}; mkdir -p $D
export CUTE_DSL_KEEP_PTX=1 CUTE_DSL_DUMP_DIR=$D CUTE_DSL_CACHE_DIR=$D/cache
python3 -c "
from vllm.models.kimi_k3.nvidia.specialized_prefill import indexer as p
p._score_kernel(212, $M)
" 2>&1 | grep -v "flashinfer\|deprecated" | tail -2
bash /lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-IDX/tools/ptx_hist.sh $D
