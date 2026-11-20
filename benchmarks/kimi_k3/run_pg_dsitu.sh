#!/bin/bash
# P-GEMM R2-DS: prefill dense SiTU fast BF16 form: unit tests + microbench
cd /k31ov
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws
python3 -m pytest -q -x tests/models/kimi_k3/test_k31_situ_fast.py tests/models/kimi_k3/test_k31_vec_situ.py 2>&1 | tail -6
python3 benchmarks/kimi_k3/pgemm_dense_situ_bench.py $K/models/k31-nvfp4-view-L24/config.json 2>&1 | grep -E "^\{|Error|error" | cut -c1-300
echo "== END"
