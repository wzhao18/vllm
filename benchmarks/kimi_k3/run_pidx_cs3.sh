#!/bin/bash
# Candidate select A/B at the P3-16 shape (inside the image, cwd = overlay root):
# exactness of the selective path vs stock logits, same-process ABBA of this tree's
# CandSelect vs ${REF:-pidx_ref_e480.py} on the same candidates, unit tests.
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm --format=csv,noheader | head -1
python3 benchmarks/kimi_k3/bench_pidx_fused.py --cs --iters ${ITERS:-20} --refmod ${REF:-pidx_ref_e480.py} 2>&1 \
  | grep -E 'bad_rows|perf_cs|Error|error|Traceback|^row ' | cut -c1-700 | tail -12
python3 -m pytest --noconftest -q tests/models/kimi_k3/test_specialized_prefill_indexer_gpu.py 2>&1 | tail -2
echo == done
