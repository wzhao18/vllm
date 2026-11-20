#!/bin/bash
# D-KDA M3a: KdaRing (prep kernel + 512-thread page ring) vs KdaFlow: bitwise test, then microbench.
B=benchmarks/kernels/k31_vr/kda_sp_bench.py
T=tests/models/kimi_k3/test_specialized_decode_sp_kda_flow_gpu.py
echo "== test ring bitwise"
timeout 1500 python3 -m pytest -x -q $T -k "ring_is_bitwise and bf16 and 33" 2>&1 | tail -25
echo "== bench"
for rep in 1 2; do timeout 900 python3 $B --n 32 64 96 128 --kinds flow ring --gap 1 --pending 4 --place local 2>&1 | grep -E "us|rror|Error" | tail -12; done
echo "== bench mx"
timeout 900 python3 $B --n 32 96 --kinds flow ring --gap 1 --pending 4 --place local --mx 2>&1 | grep -E "us|rror" | tail -6
echo "== tests full"
timeout 1500 python3 -m pytest -x -q $T -k "ring" 2>&1 | tail -15
