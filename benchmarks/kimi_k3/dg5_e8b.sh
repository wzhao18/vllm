#!/bin/bash
# D-GEMM round 5 E8b: layout checks only (f16 D layout, SFA, SFB) of the cta_group::2 M128 probe on VR200.
cd /k31ov && python3 benchmarks/kimi_k3/dg_m128pair_probe.py f16 sfa sfb 2>&1 | grep -E "rel err|SFA|SFB|rror" | cut -c1-260
