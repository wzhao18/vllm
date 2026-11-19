#!/bin/bash
# cuDNN FROST KDA vs FlashKDA at F16K. usage: run_frost_ab.sh <overlay> <outdir> [bench args]
OV=$1; OUT=$2; shift 2; mkdir -p $OUT
export PYTHONPATH=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/venvs/cudnnfe133/site:$PYTHONPATH
nvidia-smi --query-gpu=name,clocks.sm --format=csv,noheader | head -1
python3 -c "import cudnn, tvm_ffi, cutlass; print(\"cudnn\", cudnn.__version__, cudnn.__file__, \"tvm_ffi\", getattr(tvm_ffi,\"__version__\",\"?\"), \"cutlass\", getattr(cutlass,\"__version__\",\"?\"))" 2>&1 | tail -2
timeout 1200 python3 -u $OV/benchmarks/kimi_k3/bench_kda_frost_ab.py "$@" 2>&1 | grep --line-buffered -v -E "^\s*$|Warning|warn|deprecated|jit-cache|flashinfer-cubin" | tail -60
