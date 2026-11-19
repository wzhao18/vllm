#!/bin/bash
# D-ATTN: unit tests + q-path microbench inside wsgpu.sh (1 GPU). $1 = overlay dir, $2 = out dir
OV=$1; OUT=$2
cd $OV
nvidia-smi --query-gpu=name,clocks.max.sm --format=csv,noheader
python3 -m pytest -x -q tests/models/kimi_k3/test_specialized_decode_sp_local_query_gpu.py 2>&1 | tail -15
python3 -m pytest -x -q tests/models/kimi_k3/test_specialized_decode_sp_dsa_gpu.py 2>&1 | tail -8
python3 benchmarks/kimi_k3/bench_dattn_qpath.py --out $OUT/qpath2.json
