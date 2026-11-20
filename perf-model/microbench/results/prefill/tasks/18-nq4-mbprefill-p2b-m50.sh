#!/bin/bash
# mb lane prefill retry: P2 (DP4 x TP4=DCP4 ShardQ, EP16) F32K per DP group (production MNBT 32768) at L 78K.
# The first P2 run OOMed in the 32K case at gpu-memory-utilization 0.92 (bench state snapshot + 32K transients on top of
# the production KV pool): bench-only GMU 0.85. MXFP8-pinned MegaMoE, then release policy; timing only.
source /lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/mb/prefill/tools/pf_common.sh
R=$L/results/p2b; mkdir -p $R
pst "p2b start job $SLURM_JOB_ID hosts $NQ_HOSTS deadline $NQ_DEADLINE"
guard 25 || exit 0
export PF_GMU=0.85
source $T/pf_common.sh
PF_TIMEOUT=1500 PF_MOE_QUANT=mxfp8 PF_CNAME=k31pf-a bash $T/pf_run.sh $NQ_HOSTS 4 $R/p2_mx32 "${P2[@]}" "${C32[@]}" \
  --output-json $R/p2_mx32.json > $R/p2_mx32.out 2>&1
pst "p2b A: $(grep -a '^PFEND' $R/p2_mx32.log) $(grep -a '^\[forward\].*median' $R/p2_mx32.log | sed 's/, sample spread.*//' | tr '\n' ';')"
if guard 20; then
  PF_TIMEOUT=1500 PF_MOE_QUANT=release PF_CNAME=k31pf-a bash $T/pf_run.sh $NQ_HOSTS 4 $R/p2_rel32 "${P2[@]}" "${C32[@]}" \
    --output-json $R/p2_rel32.json > $R/p2_rel32.out 2>&1
  pst "p2b B: $(grep -a '^PFEND' $R/p2_rel32.log) $(grep -a '^\[forward\].*median' $R/p2_rel32.log | sed 's/, sample spread.*//' | tr '\n' ';')"
fi
pst "p2b done"
