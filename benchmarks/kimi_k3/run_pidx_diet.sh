#!/bin/bash
# Instruction diet A/B at the P3-16 shape (inside the image, cwd = overlay root):
# exactness of the production kernel, then same-process ABBA of each reference module vs production.
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm --format=csv,noheader | head -1
python3 benchmarks/kimi_k3/bench_pidx_fused.py --check 2>&1 | grep -E 'bitwise|bad_rows|Error|error|Traceback' | cut -c1-600 | tail -5
python3 -m pytest --noconftest -q tests/models/kimi_k3/test_specialized_prefill_indexer_gpu.py 2>&1 | tail -1
for R in ${REFS:-pidx_ref_e480.py pidx_var_ti.py}; do
  echo "== ref $R"
  python3 benchmarks/kimi_k3/bench_pidx_fused.py --ref --refmod $R --iters 10 2>&1 | grep -E 'perf_|Error|error|Traceback' | cut -c1-600 | tail -3
done
echo == done
