#!/bin/bash
# D-GEMM: MXFP8 projection unit test + the existing SP GEMM / MLP unit tests on the overlay.
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/D-GEMM
OV=$(python3 -c "import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))")
echo "vllm from $OV"
cd $OV
timeout 900 python3 -m pytest -x -q --noconftest -p no:cacheprovider tests/models/kimi_k3/test_specialized_decode_sp_mxfp8_gpu.py 2>&1 | tail -30
timeout 900 python3 -m pytest -x -q --noconftest -p no:cacheprovider tests/models/kimi_k3/test_specialized_decode_sp_mlp_gpu.py tests/models/kimi_k3/test_specialized_decode_sp_coverage.py 2>&1 | tail -15
