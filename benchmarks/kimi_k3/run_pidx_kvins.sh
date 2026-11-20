#!/bin/bash
# MLA FP8 KV insert microbench + unit tests (inside the image, cwd = overlay root)
nvidia-smi --query-gpu=name,clocks.sm --format=csv,noheader | head -1
python3 benchmarks/kimi_k3/bench_pidx_kv_insert.py 2>&1 | grep -E "kv_insert|Error|Traceback" | tail -6
python3 -m pytest --noconftest -q tests/models/kimi_k3/test_specialized_prefill_kv_insert.py 2>&1 | tail -4
echo == done
