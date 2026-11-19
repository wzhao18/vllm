#!/bin/bash
# D-ATTN: unit tests (no root conftest: the image lacks tblib) + LocalQuery breakdown. $1 = overlay, $2 = out dir
OV=$1; OUT=$2
cd $OV
python3 -m pytest --noconftest -x -q tests/models/kimi_k3/test_specialized_decode_sp_local_query_gpu.py 2>&1 | tail -25
python3 -m pytest --noconftest -x -q tests/models/kimi_k3/test_specialized_decode_sp_dsa_gpu.py 2>&1 | tail -8
python3 benchmarks/kimi_k3/bench_dattn_qpath.py --local-only --out $OUT/qpath3.json
