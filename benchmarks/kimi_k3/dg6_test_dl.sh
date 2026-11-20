#!/bin/bash
# D-GEMM round 6: unit tests of the die-local kda_in branch (bitwise vs _scaled_mm, graph replays, rejects).
cd /k31ov
python3 -m pytest -q -x tests/models/kimi_k3/test_specialized_decode_sm107_pre_gpu.py 2>&1 | tail -15
