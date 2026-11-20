#!/bin/bash
# Effective SM clock of the IndexerScore launches (needs an overlay whose IndexerScore
# prints "IDXCLK mode M epi E items N clk C ns T" from CTA 0): microbench at the P3-16
# shape, summarized per (mode, items) as launches / mean us / mean GHz.
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,power.draw,power.limit --format=csv,noheader | head -1
python3 benchmarks/kimi_k3/bench_pidx_fused.py --ref --refmod pidx_ref_e480.py --iters 10 2>&1 \
  | grep -a IDXCLK | awk '{k=$3" items "$7; n[k]++; c[k]+=$9; t[k]+=$11}
      END {for (k in n) printf "mode %s: %d launches, %.1f us, %.3f GHz\n", k, n[k], t[k]/n[k]/1000, c[k]/t[k]}'
