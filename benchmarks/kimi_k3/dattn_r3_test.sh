#!/bin/bash
# D-ATTN round 3 tests (4 GPUs): the sparse MLA kernel twins vs trtllm-gen, then the SP DSA
# block vs the stock layer. dattn_r3_test.sh [pytest -k expression for the SP test]
cd /k31ov
python3 -m pytest -q -p no:cacheprovider --noconftest \
  tests/models/kimi_k3/test_specialized_decode_sparse_mla_km_gpu.py 2>&1 | tail -n 6
python3 -m pytest -q -p no:cacheprovider --noconftest \
  tests/distributed/test_kimi_k3_sdec_sp_layers.py \
  -k "${1:-test_sp_dsa_block_matches_stock_layer and not all_rows}" 2>&1 \
  | grep -E 'passed|failed|rror|core replay|same-set|FAILED|assert|Mismatch|Greatest' | cut -c1-260 | tail -n 40
