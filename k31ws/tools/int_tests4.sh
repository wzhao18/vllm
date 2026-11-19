#!/bin/bash
# int_tests4.sh (inside the image via wsgpu.sh -g 4, cwd = overlay): multi-GPU tests of the workstreams merged into
# wzhao/k31ws-int, plus the 1-GPU suite (int_tests.sh). Append each landed workstream's multi-GPU tests here.
set -o pipefail
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws
bash $K/tools/int_tests.sh; rc=$?
t() { echo "== pytest $*"; timeout 1500 python3 -m pytest -x -q -p no:cacheprovider --noconftest "$@" 2>&1 | grep -vE "Warning|deprecat" | tail -8; r=${PIPESTATUS[0]}; [ $r = 0 ] || rc=$r; }
# EN (e49d1d3a87): GPU numerics of native NVFP4 MegaMoE layers + the export's MXFP8 linears
[ -f tests/models/kimi_k3/test_megamoe_rubin_nvfp4_native_gpu.py ] && t tests/models/kimi_k3/test_megamoe_rubin_nvfp4_native_gpu.py
[ -f tests/models/kimi_k3/test_k31_nvfp4_mxfp8_linear_gpu.py ] && t tests/models/kimi_k3/test_k31_nvfp4_mxfp8_linear_gpu.py
# D-TOPK (b162ec54a6): DCP exchange top-k v2 (bitwise vs topk_merge) + policy + sdec DSA block vs stock
if [ -f tests/models/kimi_k3/test_k31_dcp_exchange.py ]; then  # torchrun script (prints ALL PASS)
  echo "== torchrun tests/models/kimi_k3/test_k31_dcp_exchange.py"
  timeout 1200 torchrun --nproc-per-node 4 tests/models/kimi_k3/test_k31_dcp_exchange.py 2>&1 | grep -E "ALL PASS|FAILURES|Error|FAIL" | tail -5
  [ ${PIPESTATUS[0]} = 0 ] || rc=1
fi
[ -f tests/models/kimi_k3/test_k31_dcp_exchange_policy.py ] && t tests/models/kimi_k3/test_k31_dcp_exchange_policy.py
t tests/distributed/test_kimi_k3_sdec_sp_layers.py -k "test_sp_dsa_block_matches_stock_layer and not all_rows"
# D-FUSE (6758683397): SP fused AttnRes / comm (TP4)
[ -f tests/distributed/test_kimi_k3_sdec_sp_fused.py ] && t tests/distributed/test_kimi_k3_sdec_sp_fused.py
# D-ATTN (72fdf53350): SP DSA local query vs stock
[ -f tests/models/kimi_k3/test_specialized_decode_sp_local_query_gpu.py ] && t tests/models/kimi_k3/test_specialized_decode_sp_local_query_gpu.py
echo "== int_tests4 rc=$rc"
exit $rc
