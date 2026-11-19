#!/bin/bash
# Bench helper (inside the K3.1 image, overlay cwd): stock reference + the sm_107
# prefill indexer check and perf at the P3-16 per-GPU shape. Usage: run_pidx.sh <out_prefix>
O=$1
timeout -s KILL 300 python3 benchmarks/kimi_k3/bench_pidx_prefill_indexer.py --json $O.stock.json 2>&1 \
  | grep -v "flashinfer\|deprecated" | grep -v "^\s*$" | head -4
timeout -s KILL 600 python3 benchmarks/kimi_k3/bench_pidx_fused.py --check --perf --modes 2 3 0 \
  --reps 1 --json $O.json 2>&1 | grep -v "flashinfer\|deprecated" | grep -v "^\s*$"
