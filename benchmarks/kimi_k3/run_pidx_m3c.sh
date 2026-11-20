#!/bin/bash
# M3c (split-N de-lockstep IndexerScore) microbench A/B + exactness at the P3-16 shape.
# Usage (inside the image, cwd = overlay root): run_pidx_m3c.sh [ref module]
REF=${1:-pidx_ref_e480.py}
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm --format=csv,noheader | head -1
python3 benchmarks/kimi_k3/bench_pidx_fused.py --check 2>&1 | grep -E 'bitwise|bad_rows|Error|error|Traceback' | cut -c1-600 | tail -5
python3 benchmarks/kimi_k3/bench_pidx_fused.py --sel --iters 10 2>&1 | grep -E 'perf_|check_sel|Error|error|Traceback' | cut -c1-900 | tail -6
python3 benchmarks/kimi_k3/bench_pidx_fused.py --ref --refmod $REF --iters 10 2>&1 | grep -E 'perf_|Error|error|Traceback' | cut -c1-900 | tail -4
python3 -m pytest --noconftest -q -s tests/models/kimi_k3/test_specialized_prefill_indexer_gpu.py 2>&1 | tail -2
echo == done
