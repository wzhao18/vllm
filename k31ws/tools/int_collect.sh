#!/bin/bash
# int_collect.sh: show collection errors of the SP MLP / coverage tests on the current overlay (cwd).
python3 -m pytest -q -p no:cacheprovider --noconftest --collect-only \
  tests/models/kimi_k3/test_specialized_decode_sp_mlp_gpu.py tests/models/kimi_k3/test_specialized_decode_sp_coverage.py 2>&1 \
  | grep -v "Warning\|warn\|^  \|^$" | tail -40
