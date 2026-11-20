#!/bin/bash
# S6-1 check: reference (parent overlay) vs new (pkda2s6 overlay), bitwise + timing
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws
R=$K/runs/P-KDA2-S6; mkdir -p $R
CHK=$K/ov/pkda2s6/benchmarks/kimi_k3/fip_s6_check.py
if [ ! -f $R/ref_c3363c7.pt ]; then
  (cd $K/ov/pkda2r-c3363c7-2 && python3 $CHK --save $R/ref_c3363c7.pt 2>&1 | grep -v Warning | tail -8)
fi
cd /k31ov && python3 $CHK --cmp $R/ref_c3363c7.pt 2>&1 | grep -v Warning | tail -12
cd /k31ov && python3 -m pytest --noconftest -q -x tests/models/kimi_k3/test_kda_frost_prefill.py -k persistent 2>&1 | tail -4
cd /k31ov && python3 $CHK --phantom 1 --shapes f16k,2x8k --time f16k,2x8k 2>&1 | grep -v Warning | tail -4
cd /k31ov && python3 $CHK --shapes f16k,2x8k --time f16k,2x8k 2>&1 | grep -v Warning | tail -4
