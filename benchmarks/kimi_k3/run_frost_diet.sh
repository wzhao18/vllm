#!/bin/bash
# FROST diet / augmented-cut / head-split experiments. usage: run_frost_diet.sh <overlay> [bench args]
OV=$1; shift
nvidia-smi --query-gpu=name,clocks.sm --format=csv,noheader | head -1
python3 -c "import cudnn; print(\"cudnn\", cudnn.__version__)" 2>&1 | tail -1
timeout 2400 python3 -u $OV/benchmarks/kimi_k3/bench_frost_diet.py "$@" 2>&1 | grep --line-buffered -v -E "^\s*$|Warning|warn|deprecated|jit-cache|flashinfer-cubin" | tail -120
