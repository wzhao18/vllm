#!/bin/bash
# int_tests.sh (inside the image via wsgpu.sh, cwd = overlay): unit tests of every workstream merged into wzhao/k31ws-int.
# Append each landed workstream's test files here when merging it.
set -o pipefail
echo "== overlay $(pwd) $(head -1 SOURCE_REV.ws 2>/dev/null)"
rc=0
t() { echo "== pytest $*"; timeout 900 python3 -m pytest -q -p no:cacheprovider --noconftest "$@" 2>&1 | tail -8; r=${PIPESTATUS[0]}; [ $r = 0 ] || rc=$r; }
# EN (6f164db270)
t tests/models/kimi_k3/test_megamoe_rubin_conversion.py tests/models/kimi_k3/test_megamoe_rubin_integration.py tests/models/kimi_k3/test_k31_nvfp4_quant_layout.py
# D-GEMM (520afc7416)
t tests/models/kimi_k3/test_specialized_decode_sp_mxfp8_gpu.py
t tests/models/kimi_k3/test_specialized_decode_sp_mlp_gpu.py tests/models/kimi_k3/test_specialized_decode_sp_coverage.py
# P-MOE (d0aaaa1a8e)
[ -f tests/models/kimi_k3/test_prefill_moe_subcalls.py ] && t tests/models/kimi_k3/test_prefill_moe_subcalls.py
# P-KDA (47bc2f317f)
[ -f tests/models/kimi_k3/test_kda_o_norm_mxfp8.py ] && t tests/models/kimi_k3/test_kda_o_norm_mxfp8.py
echo "== int_tests rc=$rc"
exit $rc
