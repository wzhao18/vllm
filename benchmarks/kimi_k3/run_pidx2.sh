#!/bin/bash
# Bench helper (inside the K3.1 image, overlay cwd): sm_107 prefill indexer check + perf at the
# P3-16 per-GPU shape (no stock reference: stock = 2.222 ms/layer, runs/P-IDX/stock1.log), plus the
# register / stack usage of the compiled kernels. Usage: run_pidx2.sh <out_prefix> [bench args]
O=$1; shift
mkdir -p $O.dump
[ $# -eq 0 ] && set -- --check --perf --modes 2 0
CUTE_DSL_KEEP_CUBIN=1 CUTE_DSL_DUMP_DIR=$O.dump timeout -s KILL 900 python3 \
  benchmarks/kimi_k3/bench_pidx_fused.py --json $O.json "$@" 2>&1 \
  | grep -v "flashinfer\|deprecated" | grep -v "^\s*$"
for f in $(find $O.dump -name "*.cubin"); do
  echo "== $(basename $f | cut -c1-90): $(/usr/local/cuda/bin/cuobjdump -res-usage $f 2>&1 | grep -o 'Function [^:]*:\|REG:[0-9]* STACK:[0-9]*' | tr '\n' ' ' | cut -c1-400)"
done
