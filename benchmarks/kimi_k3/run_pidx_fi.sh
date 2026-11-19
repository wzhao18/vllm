#!/bin/bash
# Bench helper (inside the K3.1 image, overlay cwd): FlashInfer main FP8MQATopKWrapper selective
# prefill top-k at the P3-16 per-GPU shape. Usage: run_pidx_fi.sh <fi checkout> <out_prefix> [bench args]
FI=$1; O=$2; shift 2
export PYTHONPATH=$FI:$PYTHONPATH
export FLASHINFER_WORKSPACE_BASE=$(dirname $O)/fi-ws
export CUTE_DSL_ARCH=${CUTE_DSL_ARCH:-sm_107a}
[ $# -eq 0 ] && set -- --check --perf --math-mode native_bf16
python3 -c "import flashinfer, cutlass; print('flashinfer', flashinfer.__file__, getattr(flashinfer, '__version__', '?')); print('cutlass-dsl', getattr(cutlass, '__version__', '?'))"
timeout -s KILL 1500 python3 benchmarks/kimi_k3/bench_pidx_fi.py --json $O.json "$@" 2>&1 \
  | grep -v "deprecated\|ExperimentalWarning" | grep -v "^\s*$" | tail -60
echo "== relu_fast (FP32 head math, perf only)"
timeout -s KILL 900 python3 benchmarks/kimi_k3/bench_pidx_fi.py --json $O.fast.json --perf --math-mode relu_fast 2>&1 \
  | grep -v "deprecated\|ExperimentalWarning" | grep -v "^\s*$" | tail -20
