#!/bin/bash
cd /lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/D-GEMM/sg
python3 cg_bench.py $1 $2 2>&1 | grep -v -E "Warning|deprecated"
