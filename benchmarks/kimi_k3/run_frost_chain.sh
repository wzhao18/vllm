#!/bin/bash
# FROST wiring: unit tests + chain bench onorm vs frost. usage: run_frost_chain.sh <overlay> <outdir>
OV=$1; OUT=$2; mkdir -p $OUT; cd $OV
timeout 600 python3 -m pytest -q -x tests/models/kimi_k3/test_kda_frost_prefill.py tests/models/kimi_k3/test_kda_o_norm_mxfp8.py 2>&1 | tail -n 4
for i in onorm frost onorm frost; do timeout 600 python3 -u benchmarks/kimi_k3/bench_kda_prefill_chain.py --impl $i 2>&1 | grep -A6 "^impl=" | cut -c1-140; done
