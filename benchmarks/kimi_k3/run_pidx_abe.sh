#!/bin/bash
# IndexerScore A/B at the P3-16 shape (inside the image, cwd = overlay root): same-process ABBA + NVML energy of this
# tree vs REFS (pidx_abe.py, bitwise candidate equality), then exactness (logits vs DeepGEMM + exact top-k, unit test).
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,power.limit --format=csv,noheader | head -1
timeout 600 python3 benchmarks/kimi_k3/pidx_abe.py 2>&1 | grep -E '^ABE|Error|error|Traceback' | cut -c1-2500 | tail -4
[ -n "$ABE_ONLY" ] && { echo == done; exit 0; }
timeout 600 python3 benchmarks/kimi_k3/bench_pidx_fused.py --check 2>&1 | grep -E 'bitwise|bad_rows|Error|error|Traceback' | cut -c1-400 | tail -4
timeout 600 python3 -m pytest --noconftest -q tests/models/kimi_k3/test_specialized_prefill_indexer_gpu.py 2>&1 | tail -1
echo == done
