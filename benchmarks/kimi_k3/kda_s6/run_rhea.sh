#!/bin/bash
# S6 check on Rhea: reference overlay (pkda2s6ref = 22d9c40693) vs new (/k31ov): bitwise + graph timing
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws; R=$K/runs/P-KDA2-S6; mkdir -p $R
CHK=/k31ov/benchmarks/kimi_k3/fip_s6_check.py; REF=$R/ref_22d9c40.pt; T=${TMO:-400}
SH=${SHAPES:-small,mixed,1x8k,2x8k,f16k}; TM=${TIMES:-2x8k,f16k}
if [ ! -f $REF ]; then (export PYTHONPATH=$K/ov/pkda2s6ref:$PYTHONPATH; timeout $T python3 $CHK --save $REF --shapes $SH --time $TM 2>&1 | grep -v -i warn | tail -7); fi
timeout $T python3 $CHK --cmp $REF --shapes $SH --time $TM 2>&1 | grep -v -i warn | tail -8; echo "new rc=${PIPESTATUS[0]}"
for p in ${PHANTOM:-}; do timeout $T python3 $CHK --phantom $p --shapes 2x8k,f16k --time 2x8k,f16k 2>&1 | grep -v -i warn | tail -3; echo "phantom rc=${PIPESTATUS[0]}"; done
if [ -n "${TESTS:-}" ]; then cd /k31ov && timeout 600 python3 -m pytest --noconftest -q -x tests/models/kimi_k3/test_kda_frost_prefill.py -k persistent 2>&1 | tail -3; fi
