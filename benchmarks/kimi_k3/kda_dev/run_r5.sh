#!/bin/bash
# P-KDA2 R5 dev A/B: V direct loads (VLDG_), MB_TAB-split prep deps (TABW_), V L2 prefetch (VPF_).
cd /k31ov
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm --format=csv,noheader
python3 benchmarks/kimi_k3/bench_fip_dev.py --rounds 4 --lens 5461,5461,5462 8192,8192 100,37,3000,1,64,33 16384 --variants "{}" "{\"VLDG_\": 1}" "{\"TABW_\": 1}" "{\"VLDG_\": 1, \"TABW_\": 1}" "{\"VLDG_\": 1, \"TABW_\": 1, \"VPF_\": 2}" 2>&1 | grep -E "lens=|Error|error|Traceback|File \"" | tail -40
echo R5_DONE
