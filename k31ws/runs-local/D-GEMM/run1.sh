#!/bin/bash
# D-GEMM bench run 1: all families, base overlay.
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/D-GEMM
cd $R
ls /usr/local/cuda/bin | grep -E "^(ncu|nsys)$" > $R/tools_avail.txt 2>&1
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.max.mem --format=csv >> $R/tools_avail.txt 2>&1
python3 -c "import flashinfer, torch; print(flashinfer.__version__, torch.__version__)" >> $R/tools_avail.txt 2>&1
python3 $R/gemm_bench.py $R/bench1.json "$@"
