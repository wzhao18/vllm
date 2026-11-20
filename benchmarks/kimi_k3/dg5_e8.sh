#!/bin/bash
# D-GEMM round 5 E8: does cta_group::2 M128 (64 rows per CTA) kind::mxf8f6f4 block-scaled MMA work on VR200 with
# the GB300-measured D/SFA/SFB TMEM layouts (memory note)? Port of the K3.1 prefill-port probe (sm_100 verified).
cd /k31ov && python3 benchmarks/kimi_k3/dg_m128pair_probe.py f16 sfa sfb qk rate 2>&1 | grep -v -E "^W1120|deprecated" | tail -30
